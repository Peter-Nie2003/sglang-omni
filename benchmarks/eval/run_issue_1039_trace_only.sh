#!/usr/bin/env bash
# Re-run of the issue #1039 torch-trace cross-check only. The full runner's
# trace arm came back with zero CUDA kernel events on the first H100 run
# (CUPTI silently degraded to CPU-only inside the container), so this script
# probes CUPTI with a tiny standalone profile BEFORE launching the server —
# a failed probe means the container must be fixed first, not re-run:
#   - docker: add --cap-add=SYS_ADMIN (or --privileged) and make sure
#     NVIDIA_DRIVER_CAPABILITIES includes "compute,utility";
#   - host driver: NVreg_RestrictProfilingToAdminUsers=0 if kernels still
#     don't show up as a non-root user.
#
# Usage (from the repo root, with the serving venv active):
#   bash benchmarks/eval/run_issue_1039_trace_only.sh
#
# Attach the resulting trace/*.trace.json.gz to issue #1039 next to the
# step_report from the full run.

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
PORT=${PORT:-8000}
BASE_URL="http://127.0.0.1:${PORT}"
THINKER_TP_SIZE=${THINKER_TP_SIZE:-2}
THINKER_GPUS=${THINKER_GPUS:-0,1}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
TRACE_CONCURRENCY=${TRACE_CONCURRENCY:-8}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1800}
OUT=${OUTPUT_DIR:-results/issue-1039-trace-$(date +%Y%m%d-%H%M%S)}

echo "== CUPTI probe =="
python - <<'EOF'
import sys
import torch
from torch.profiler import ProfilerActivity, profile

a = torch.randn(1024, 1024, device="cuda")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(10):
        a = a @ a
    torch.cuda.synchronize()
gpu_us = sum(
    getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
    for e in p.key_averages()
)
if gpu_us <= 0:
    print(
        "CUPTI captured no GPU kernels — fix the container first "
        "(see the header of this script), then retry.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"CUPTI ok: {gpu_us / 1e3:.2f}ms of GPU kernel time captured")
EOF

mkdir -p "$OUT"/{trace,bench,logs}
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

echo "[server] launching (log: $OUT/logs/server.log)"
# shellcheck disable=SC2086  # SERVER_EXTRA_ARGS is a word-split arg list
python -m sglang_omni.cli serve \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --thinker-tp-size "$THINKER_TP_SIZE" \
    --thinker-gpus "$THINKER_GPUS" \
    $SERVER_EXTRA_ARGS >"$OUT/logs/server.log" 2>&1 &
SERVER_PID=$!

waited=0
until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[server] died during startup; tail of $OUT/logs/server.log:" >&2
        tail -40 "$OUT/logs/server.log" >&2
        exit 1
    fi
    if (( waited >= HEALTH_TIMEOUT_S )); then
        echo "[server] no /health after ${HEALTH_TIMEOUT_S}s" >&2
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done
echo "[server] healthy after ~${waited}s"

# Unprofiled warmup so lazy CUDA-graph captures don't land inside the trace.
echo "[bench] warmup (unprofiled)"
python -m benchmarks.eval.benchmark_omni_rollout_stress \
    --host 127.0.0.1 --port "$PORT" \
    --rollout-counts 1 \
    --no-profile \
    --output-dir "$OUT/bench/warmup" \
    --disable-tqdm >"$OUT/logs/bench-warmup.log" 2>&1

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
    --disable-tqdm >"$OUT/logs/bench-trace.log" 2>&1
curl -sf -X POST "$BASE_URL/stop_profile" -H 'Content-Type: application/json' \
    -d '{}' >/dev/null

cleanup
SERVER_PID=""

# A trace without kernel events is the failure this re-run exists to fix;
# refuse to report success on one.
echo "== verifying kernels in the trace =="
python - "$OUT/trace" <<'EOF'
import gzip
import json
import pathlib
import sys

trace_dir = pathlib.Path(sys.argv[1])
files = sorted(trace_dir.glob("*.trace.json*"))
if not files:
    print(f"no trace files under {trace_dir}", file=sys.stderr)
    sys.exit(1)
ok = False
for f in files:
    opener = gzip.open if f.suffix == ".gz" else open
    with opener(f, "rt") as fh:
        events = json.load(fh).get("traceEvents", [])
    kernel_ms = sum(
        e.get("dur", 0) for e in events if e.get("cat") == "kernel"
    ) / 1e3
    print(f"{f.name}: {len(events)} events, {kernel_ms:.1f}ms kernel time")
    ok = ok or kernel_ms > 0
sys.exit(0 if ok else 1)
EOF

echo
echo "== done =="
echo "attach to issue #1039: $OUT/trace/*.trace.json.gz"
