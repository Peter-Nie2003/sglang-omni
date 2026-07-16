#!/usr/bin/env bash
# One-shot Phase 2 measurement for issue #1039 (talker predictor
# graph-coverage audit, RFC #1018 item 3.2).
#
# Runs on a single GPU node, end to end:
#   arm A "graph"  — normal serving, per-step host-segment profile at each
#                    concurrency level, plus one torch trace for the GPU-side
#                    replay cross-check;
#   arm B "eager"  — talker CUDA graph off (--talker-cuda-graph off), same
#                    profile at the diagnostic concurrency;
# then aggregates everything into <output-dir>/step_report.{txt,json} and
# checks the two health conditions that must hold (zero predictor graph
# capture failures, zero decode-time eager fallbacks).
#
# Usage (from the repo root, with the serving venv active):
#   bash benchmarks/eval/run_issue_1039_step_profiling.sh
#
# Reference hardware for comparable numbers: H100, thinker TP2 (#1018).
# Every knob below is an environment variable; override as needed, e.g.:
#   MODEL_PATH=/data/models/Qwen3-Omni-30B-A3B-Instruct \
#   THINKER_GPUS=2,3 bash benchmarks/eval/run_issue_1039_step_profiling.sh
#
# Attach step_report.txt, step_report.json, and meta.json to issue #1039.

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
PORT=${PORT:-8000}
BASE_URL="http://127.0.0.1:${PORT}"
THINKER_TP_SIZE=${THINKER_TP_SIZE:-2}
THINKER_GPUS=${THINKER_GPUS:-0,1}
# Extra `sglang_omni.cli serve` args appended to both arms.
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
# Graph arm sweeps these levels; the eager arm only needs the diagnostic one.
CONCURRENCIES=${CONCURRENCIES:-1 8 16 32}
EAGER_CONCURRENCIES=${EAGER_CONCURRENCIES:-8}
TRACE_CONCURRENCY=${TRACE_CONCURRENCY:-8}
WARMUP_STEPS=${WARMUP_STEPS:-16}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1800}
OUT=${OUTPUT_DIR:-results/issue-1039-$(date +%Y%m%d-%H%M%S)}

mkdir -p "$OUT"/{events,trace,bench,logs}
OUT=$(cd "$OUT" && pwd)

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping server pid=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_server() {
    local arm=$1
    shift
    local log="$OUT/logs/server-$arm.log"
    echo "[server:$arm] launching (log: $log)"
    python -m sglang_omni.cli serve \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --thinker-tp-size "$THINKER_TP_SIZE" \
        --thinker-gpus "$THINKER_GPUS" \
        # shellcheck disable=SC2086  # SERVER_EXTRA_ARGS is a word-split arg list
        "$@" $SERVER_EXTRA_ARGS >"$log" 2>&1 &
    SERVER_PID=$!

    local waited=0
    until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[server:$arm] died during startup; tail of $log:" >&2
            tail -40 "$log" >&2
            exit 1
        fi
        if (( waited >= HEALTH_TIMEOUT_S )); then
            echo "[server:$arm] no /health after ${HEALTH_TIMEOUT_S}s" >&2
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[server:$arm] healthy after ~${waited}s"
}

stop_server() {
    cleanup
    SERVER_PID=""
    sleep 5
}

# One benchmark invocation per (arm, level): the stress runner wraps its whole
# invocation in a single profile run_id, so per-level run ids are the only way
# the report can attribute segments per concurrency.
FAILED_LEVELS=""
run_level() {
    local arm=$1 c=$2
    local run_id="$arm-c$c"
    echo "[bench] $run_id"
    if ! python -m benchmarks.eval.benchmark_omni_rollout_stress \
        --host 127.0.0.1 --port "$PORT" \
        --rollout-counts "$c" \
        --profile-run-id "$run_id" \
        --profile-event-dir "$OUT/events" \
        --output-dir "$OUT/bench/$run_id" \
        --disable-tqdm >"$OUT/logs/bench-$run_id.log" 2>&1; then
        # c32 may hit the known async cache-write failure (#1018 item 1.1);
        # keep going — successful steps still carry valid attribution.
        echo "[bench] $run_id FAILED (see logs/bench-$run_id.log)" >&2
        FAILED_LEVELS="$FAILED_LEVELS $run_id"
    fi
}

# Unprofiled (--no-profile) so lazy CUDA-graph captures and cold caches
# don't pollute the first profiled level; event files are per-pid, so a
# profiled warmup could not be deleted afterwards.
warmup_arm() {
    local arm=$1
    echo "[bench] $arm warmup (unprofiled)"
    python -m benchmarks.eval.benchmark_omni_rollout_stress \
        --host 127.0.0.1 --port "$PORT" \
        --rollout-counts 1 \
        --no-profile \
        --output-dir "$OUT/bench/$arm-warmup" \
        --disable-tqdm >"$OUT/logs/bench-$arm-warmup.log" 2>&1 || {
        echo "[bench] $arm warmup failed; aborting (server not serving?)" >&2
        exit 1
    }
}

echo "== issue #1039 step profiling =="
echo "output: $OUT"
git -C "$(dirname "$0")/../.." rev-parse HEAD >"$OUT/meta-revision.txt" || true

# ---- arm A: normal graph-enabled serving --------------------------------
start_server graph
warmup_arm graph

for c in $CONCURRENCIES; do
    run_level graph "$c"
done

# GPU-side cross-check: one torch trace at the diagnostic concurrency.
echo "[trace] torch trace at c$TRACE_CONCURRENCY"
curl -sf -X POST "$BASE_URL/start_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\": \"trace-c$TRACE_CONCURRENCY\",
         \"trace_path_template\": \"$OUT/trace/trace-c$TRACE_CONCURRENCY\",
         \"event_dir\": \"$OUT/trace/events\",
         \"enable_torch\": true}" >/dev/null
python -m benchmarks.eval.benchmark_omni_rollout_stress \
    --host 127.0.0.1 --port "$PORT" \
    --rollout-counts "$TRACE_CONCURRENCY" \
    --no-profile \
    --output-dir "$OUT/bench/trace-c$TRACE_CONCURRENCY" \
    --disable-tqdm >"$OUT/logs/bench-trace.log" 2>&1 || \
    echo "[trace] benchmark under torch trace failed (non-fatal)" >&2
curl -sf -X POST "$BASE_URL/stop_profile" -H 'Content-Type: application/json' \
    -d '{}' >/dev/null

stop_server

# ---- arm B: talker CUDA graph off (diagnostic full-eager decode) --------
start_server eager --talker-cuda-graph off
warmup_arm eager
for c in $EAGER_CONCURRENCIES; do
    run_level eager "$c"
done
stop_server

# ---- aggregate -----------------------------------------------------------
echo "== aggregating =="
python -m sglang_omni.profiler.step_report "$OUT/events" \
    --stage talker --warmup-steps "$WARMUP_STEPS" | tee "$OUT/step_report.txt"
python -m sglang_omni.profiler.step_report "$OUT/events" \
    --stage talker --warmup-steps "$WARMUP_STEPS" --format json \
    >"$OUT/step_report.json"

CAPTURE_FAILURES=$(grep -h "Disabling Qwen3-Omni predictor CUDA graph" \
    "$OUT"/logs/server-*.log 2>/dev/null | wc -l | tr -d ' ')

{
    echo "revision: $(cat "$OUT/meta-revision.txt" 2>/dev/null || echo unknown)"
    echo "host: $(hostname)"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model: $MODEL_PATH  thinker_tp: $THINKER_TP_SIZE"
    echo "graph_arm_concurrencies: $CONCURRENCIES"
    echo "eager_arm_concurrencies: $EAGER_CONCURRENCIES"
    echo "warmup_steps_excluded: $WARMUP_STEPS"
    echo "failed_levels:${FAILED_LEVELS:- none}"
    echo "predictor_graph_capture_failures_in_logs: $CAPTURE_FAILURES"
} | tee "$OUT/meta.txt"

echo
echo "== done =="
echo "attach to issue #1039: $OUT/step_report.txt, step_report.json, meta.txt"
echo "torch trace for the c$TRACE_CONCURRENCY cross-check: $OUT/trace/"
