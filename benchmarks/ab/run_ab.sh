#!/usr/bin/env bash
# baseline vs PR 背靠背 A/B：c = 1/8/16/32/64，每格重启 server，丢弃第一轮，取 3 轮平均。
#
#   GPU_THINKER=2 GPU_TALKER=3 bash benchmarks/ab/run_ab.sh
#
# 先用 `nvidia-smi topo -m` 挑一对 NVLink 直连的卡：encoder 输出跨 thinker->talker
# 卡传输，正是被测改动影响的那条边，走 PCIe 会放大收益。
set -uo pipefail

BASE_WT="${BASE_WT:-$HOME/bench/wt-base}"
PR_WT="${PR_WT:-$HOME/bench/wt-pr}"
OUT_ROOT="${OUT_ROOT:-$HOME/bench/results/$(date +%Y%m%d-%H%M%S)}"
PORT="${PORT:-8000}"
PY="${PY:-python3}"

GPU_THINKER="${GPU_THINKER:-0}"   # thinker + image_encoder + audio_encoder
GPU_TALKER="${GPU_TALKER:-1}"     # talker_ar + code2wav
GPUS="$GPU_THINKER,$GPU_TALKER"

read -ra CONCURRENCIES <<< "${CONCURRENCIES:-1 8 16 32 64}"
REPEATS="${REPEATS:-3}"                    # 计入平均的轮数
DISCARD="${DISCARD:-1}"                    # 每次 server 重启后丢弃的轮数
SAMPLES_PER_CONC="${SAMPLES_PER_CONC:-8}"  # N = max(32, SAMPLES_PER_CONC * C)
MAX_NEW_TOKENS=256
IDLE_MIB=2000         # 两张卡空闲时的显存上限
FORCE_GPU_CLEANUP="${FORCE_GPU_CLEANUP:-0}"

# server 只看得到这两张卡，因此 --gpu-thinker/--gpu-talker 用 mask 内的逻辑号
export CUDA_VISIBLE_DEVICES="$GPUS"
SERVER_ARGS=(--gpu-thinker 0 --gpu-talker 1)

SERVER_LOG_DIR="$OUT_ROOT/server-logs"
mkdir -p "$SERVER_LOG_DIR"
SERVER_PID=""
SAMPLER_PID=""

log() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_used_mib() {
  nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}

wait_gpu_idle() {
  local timeout=${1:-300} waited=0 used
  while (( waited < timeout )); do
    used=$(gpu_used_mib)
    (( used < IDLE_MIB )) && return 0
    sleep 5; waited=$((waited+5))
  done
  log "!!! GPU $GPUS 显存未释放（${used} MiB）"
  if [[ "$FORCE_GPU_CLEANUP" == "1" ]]; then
    log "!!! 强杀 GPU $GPUS 上的进程"
    nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 15
    return 0
  fi
  return 1
}

# sglang_omni 可能被 pip install -e 装到别的 clone；worktree 必须排在它前面，
# 否则两个分支会加载同一份代码，A/B 静默失效。
assert_imports_from_worktree() {
  local wt=$1 name origin
  for name in sglang_omni benchmarks; do
    # benchmarks 是 namespace package，origin 为 None，退回搜索路径
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

preflight() {
  command -v nvidia-smi >/dev/null || { log "找不到 nvidia-smi"; exit 1; }
  [[ -d "$BASE_WT" && -d "$PR_WT" ]] || { log "worktree 不存在: $BASE_WT / $PR_WT"; exit 1; }
  assert_imports_from_worktree "$BASE_WT"
  assert_imports_from_worktree "$PR_WT"
  log "GPU $GPU_THINKER (thinker) <-> GPU $GPU_TALKER (talker) 互联方式："
  nvidia-smi topo -m 2>/dev/null | grep -E "^\s+GPU0|^GPU${GPU_THINKER}\b"
  log "同机其他 GPU 占用（非空即有别的租户，会挤 NVSwitch 带宽）："
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F, -v a="$GPU_THINKER" -v b="$GPU_TALKER" '$1+0!=a && $1+0!=b && $2+0>500'
  local used; used=$(gpu_used_mib)
  if (( used >= IDLE_MIB )); then
    log "!!! 开跑前 GPU $GPUS 已占用 ${used} MiB —— 有别的任务在这两张卡上"
    log "!!! 换一对空闲的卡，否则结果不可信"
    exit 1
  fi
}

start_server() {
  local wt=$1 logfile=$2
  log ">>> 启动 server: $wt"
  ( cd "$wt" && PYTHONPATH="$wt" exec "$PY" examples/run_qwen3_omni_speech_server.py \
      --port "$PORT" "${SERVER_ARGS[@]}" ) >"$logfile" 2>&1 &
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

# 全程记录未预留 GPU 的负载：噪声若来自同机其他租户，事后能对上时间段
start_gpu_sampler() {
  ( while :; do
      echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader | tr '\n' ';')"
      sleep 10
    done ) > "$OUT_ROOT/gpu-sample.log" 2>&1 &
  SAMPLER_PID=$!
}

cleanup() { [[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null; stop_server; }
trap 'cleanup; exit 130' INT TERM

run_cell() {
  local branch=$1 wt=$2 conc=$3
  local n=$(( SAMPLES_PER_CONC * conc ))
  (( n < 32 )) && n=32

  start_server "$wt" "$SERVER_LOG_DIR/${branch}-c${conc}.log"

  local total=$(( DISCARD + REPEATS )) i tag out
  for (( i=1; i<=total; i++ )); do
    if (( i <= DISCARD )); then tag="discard$i"; else tag="run$(( i - DISCARD ))"; fi
    out="$OUT_ROOT/c${conc}/${branch}/${tag}"
    mkdir -p "$out"
    log ">>> [$branch] c=$conc N=$n $tag"
    ( cd "$wt" && PYTHONPATH="$wt" "$PY" -m benchmarks.eval.benchmark_omni_seedtts \
        --port "$PORT" \
        --lang en --voice-clone --stream --generate-only \
        --max-new-tokens "$MAX_NEW_TOKENS" --temperature 0 \
        --warmup 2 --max-samples "$n" --max-concurrency "$conc" \
        --disable-tqdm --output-dir "$out" ) 2>&1 | tee "$out/bench.log"

    if [[ ! -f "$out/speed_results.json" ]]; then
      log "!!! $tag 没产出 speed_results.json，见 $out/bench.log —— 中止"
      stop_server
      exit 1
    fi
  done

  stop_server
}

preflight
start_gpu_sampler
log "输出目录: $OUT_ROOT"
for conc in "${CONCURRENCIES[@]}"; do
  run_cell base "$BASE_WT" "$conc"
  run_cell pr   "$PR_WT"   "$conc"
done

[[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null
log "全部完成，汇总： python3 benchmarks/ab/aggregate.py $OUT_ROOT"
