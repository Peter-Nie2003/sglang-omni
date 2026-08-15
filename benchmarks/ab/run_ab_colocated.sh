#!/usr/bin/env bash
# 单卡 colocated 三臂 A/B：base / mixed / inter，c = 1/8/16/32/64。
# 每臂每个并发重启一次 server，丢弃前若干轮，其余取中位数。
#
#   GPU=4 bash benchmarks/ab/run_ab_colocated.sh
#
# 三臂对应 perf/talker-mixed-chunk-interleave 的两个 commit：
#   base   68abc7ee            对照
#   mixed  0d20f5f6 默认参数    只有 enable_mixed_chunk 生效（interleave 默认 off）
#   inter  0d20f5f6 + flag on   两个 commit 都生效
set -uo pipefail

BASE_WT="${BASE_WT:-$HOME/bench/wt-base}"
TIP_WT="${TIP_WT:-$HOME/bench/wt-tip}"
OUT_ROOT="${OUT_ROOT:-$HOME/bench/results/$(date +%Y%m%d-%H%M%S)}"
PORT="${PORT:-8000}"
PY="${PY:-python3}"
GPU="${GPU:-0}"
COLOCATED_CONFIG="${COLOCATED_CONFIG:-examples/configs/qwen3_omni_colocated_h200.yaml}"

read -ra CONCURRENCIES <<< "${CONCURRENCIES:-1 8 16 32 64}"
REPEATS="${REPEATS:-3}"
DISCARD="${DISCARD:-1}"
SAMPLES_PER_CONC="${SAMPLES_PER_CONC:-8}"
MAX_NEW_TOKENS=256
IDLE_MIB=2000
FORCE_GPU_CLEANUP="${FORCE_GPU_CLEANUP:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

ARM_NAMES=(base mixed inter)
ARM_WT=("$BASE_WT" "$TIP_WT" "$TIP_WT")
ARM_EXTRA=("" "" "--talker-prefill-decode-interleave on")

SERVER_LOG_DIR="$OUT_ROOT/server-logs"
mkdir -p "$SERVER_LOG_DIR"
SERVER_PID=""
SAMPLER_PID=""

log() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_used_mib() {
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}

wait_gpu_idle() {
  local timeout=${1:-300} waited=0 used
  while (( waited < timeout )); do
    used=$(gpu_used_mib)
    (( used < IDLE_MIB )) && return 0
    sleep 5; waited=$((waited+5))
  done
  log "!!! GPU $GPU 显存未释放（${used} MiB）"
  if [[ "$FORCE_GPU_CLEANUP" == "1" ]]; then
    log "!!! 强杀 GPU $GPU 上的进程"
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 15
    return 0
  fi
  return 1
}

# sglang_omni 可能被 pip install -e 装到别的 clone；worktree 必须排在它前面，
# 否则各臂会加载同一份代码，A/B 静默失效。
assert_imports_from_worktree() {
  local wt=$1 name origin
  for name in sglang_omni benchmarks; do
    origin=$( cd "$wt" && PYTHONPATH="$wt" "$PY" -c \
      "import importlib.util as u
s = u.find_spec('$name')
if not s: print('MISSING')
elif s.origin: print(s.origin)
else: print(next(iter(s.submodule_search_locations), 'MISSING'))" 2>/dev/null )
    case "$origin" in
      "$wt"/*) ;;
      *) log "!!! $wt: '$name' 解析到 $origin，不在该 worktree 内 —— 中止"; exit 1 ;;
    esac
  done
  log "    $wt: sglang_omni / benchmarks 均来自本 worktree ✓"
}

start_gpu_sampler() {
  ( while :; do
      echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader | tr '\n' ';')"
      sleep 10
    done ) > "$OUT_ROOT/gpu-sample.log" 2>&1 &
  SAMPLER_PID=$!
}

preflight() {
  command -v nvidia-smi >/dev/null || { log "找不到 nvidia-smi"; exit 1; }
  [[ -d "$BASE_WT" && -d "$TIP_WT" ]] || { log "worktree 不存在: $BASE_WT / $TIP_WT"; exit 1; }
  assert_imports_from_worktree "$BASE_WT"
  assert_imports_from_worktree "$TIP_WT"
  for wt in "$BASE_WT" "$TIP_WT"; do
    [[ -f "$wt/$COLOCATED_CONFIG" ]] || { log "!!! $wt/$COLOCATED_CONFIG 不存在"; exit 1; }
  done
  # inter 臂依赖 tip 上新增的 CLI flag；base 上没有，只有 tip 需要有
  if ! grep -q "talker-prefill-decode-interleave" "$TIP_WT/sglang_omni/cli/serve.py"; then
    log "!!! $TIP_WT 的 serve.py 没有 --talker-prefill-decode-interleave，inter 臂无法测"
    exit 1
  fi
  log "同机其他 GPU 占用（非空即有别的租户）："
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F, -v g="$GPU" '$1+0!=g && $2+0>500'
  local used; used=$(gpu_used_mib)
  if (( used >= IDLE_MIB )); then
    log "!!! 开跑前 GPU $GPU 已占用 ${used} MiB —— 换一张空闲的卡"
    exit 1
  fi
}

start_server() {
  local wt=$1 extra=$2 logfile=$3
  log ">>> 启动 colocated server: $wt ${extra:-（默认参数）}"
  ( cd "$wt" && PYTHONPATH="$wt" exec "$PY" -m sglang_omni.cli serve \
      --config "$wt/$COLOCATED_CONFIG" --colocate --port "$PORT" \
      $extra ) >"$logfile" 2>&1 &
  SERVER_PID=$!
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  log "<<< 停止 server (pid $SERVER_PID)"
  kill -TERM "$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  wait_gpu_idle 300 || log "!!! 继续，但下一格结果可能受污染"
}

cleanup() { [[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null; stop_server; }
trap 'cleanup; exit 130' INT TERM

run_cell() {
  local arm=$1 wt=$2 extra=$3 conc=$4
  local n=$(( SAMPLES_PER_CONC * conc ))
  (( n < 32 )) && n=32

  start_server "$wt" "$extra" "$SERVER_LOG_DIR/${arm}-c${conc}.log"

  local total=$(( DISCARD + REPEATS )) i tag out
  for (( i=1; i<=total; i++ )); do
    if (( i <= DISCARD )); then tag="discard$i"; else tag="run$(( i - DISCARD ))"; fi
    out="$OUT_ROOT/c${conc}/${arm}/${tag}"
    mkdir -p "$out"
    log ">>> [$arm] c=$conc N=$n $tag"
    ( cd "$wt" && PYTHONPATH="$wt" "$PY" -m benchmarks.eval.benchmark_omni_seedtts \
        --port "$PORT" \
        --lang en --voice-clone --stream --generate-only \
        --max-new-tokens "$MAX_NEW_TOKENS" --temperature 0 \
        --warmup 2 --max-samples "$n" --max-concurrency "$conc" \
        --disable-tqdm --output-dir "$out" ) 2>&1 | tee "$out/bench.log"

    if [[ ! -f "$out/speed_results.json" ]]; then
      log "!!! $tag 没产出 speed_results.json，见 $out/bench.log —— 中止"
      cleanup
      exit 1
    fi
  done

  stop_server
}

preflight
start_gpu_sampler
log "输出目录: $OUT_ROOT"
for conc in "${CONCURRENCIES[@]}"; do
  for idx in "${!ARM_NAMES[@]}"; do
    run_cell "${ARM_NAMES[$idx]}" "${ARM_WT[$idx]}" "${ARM_EXTRA[$idx]}" "$conc"
  done
done

[[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null
log "全部完成，汇总： $PY benchmarks/ab/aggregate.py $OUT_ROOT"
