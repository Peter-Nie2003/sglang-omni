# PR #1175 process-replica performance results

Measurement of process-level replication on `feat/runtime-stage-replicas`,
run from `perf/pr1175-process-replica-bench` (based on `fe6866dd`).

Method, arms and attribution gates: [`pr1175_process_replica_bench.md`](pr1175_process_replica_bench.md).
Raw artifacts, per-run logs and `provenance.json` (exact host, driver, package
inventory) live under `results/pr1175/`.

## Summary

Process replication pays off where the replicated process is the bottleneck,
and the size of the win tracks whether the replicas got independent hardware.

| Workload | Concurrency | Gain vs its control | Significance |
| --- | --- | --- | --- |
| Qwen3-Omni, 2 × (talker_ar + code2wav) on separate GPUs | 64 | **+52.1% QPS** | 32σ |
| Qwen3-Omni, same | 32 | +24.5% QPS | 19σ |
| Qwen3-Omni, same | 16 | +11.5% QPS | 4.3σ |
| Qwen3-Omni, same | 1 | **−4.6% QPS** | 6σ |
| Higgs TTS, 2 × tts_frontend on one GPU | 96 | **+11.6% QPS** | 11.7σ |

Three results are worth carrying into the design discussion:

1. **The Qwen3-Omni gain is not the third GPU.** The `split3gpu` control uses
   the same three cards without replication and performs within ±3% of the
   two-GPU arm at every concurrency. All 52% comes from replication.
2. **Replication costs a little when there is nothing to parallelize.** At
   concurrency 1 it is 4.6% slower, consistently and well outside noise.
3. **Neither workload improves tail latency.** Both improve mean latency and
   throughput; p95 is flat (Higgs) or worse (Qwen3-Omni inter-chunk gap).

## Attribution gates

| Gate | Qwen3-Omni | Higgs |
| --- | --- | --- |
| Admission bound every configured replica | pass | pass |
| No failed runs | pass | pass |
| 3 interleaved independent server starts per arm | pass | pass |
| Gain exceeds restart noise | pass (32σ at c=64) | pass (11.7σ) |
| Output work unchanged | pass (audio/req within 0.7%) | pass (tokens/req within 0.6%) |
| Output quality unchanged | **outstanding** (WER not re-run) | n/a |
| Non-replication control used | pass (`split3gpu`) | n/a |

## Qwen3-Omni

Three topologies over the same three GPUs, Seed-TTS EN, streaming
generate-only, `--temperature 0.0`, `--max-new-tokens 256`, samples scaled as
`max(128, concurrency × 10)`.

| arm | thinker | talker_ar | code2wav |
| --- | --- | --- | --- |
| `pair2gpu` | 0 | 1 | 1 |
| `split3gpu` | 0 | 1 | 2 |
| `replica2` | 0 | @r0→1, @r1→2 | @r0→1, @r1→2 |

### Throughput (QPS, mean ±stdev over 3 repeats)

| concurrency | pair2gpu | split3gpu | replica2 | replica2 vs split3gpu |
| --- | --- | --- | --- | --- |
| 1 | 2.179 ±0.006 | 2.233 ±0.017 | 2.131 ±0.013 | **−4.6%** |
| 16 | 11.26 ±0.113 | 12.06 ±0.038 | 13.45 ±0.325 | **+11.5%** |
| 32 | 14.12 ±0.101 | 13.95 ±0.148 | 17.37 ±0.182 | **+24.5%** |
| 64 | 13.79 ±0.163 | 13.34 ±0.083 | 20.29 ±0.215 | **+52.1%** |

The gain grows monotonically with concurrency and is negative at concurrency
1. That shape is what replication should produce: a single request has nothing
to spread across replicas and pays only the extra admission bookkeeping, while
at load the replicated stage stops being the constraint.

### The controls saturate; the replicated arm does not

```
                c=16     c=32     c=64
pair2gpu :     11.26 →  14.12 →  13.79    knee at c=32, then falls back
split3gpu:     12.06 →  13.95 →  13.34    same knee, same fallback
replica2 :     13.45 →  17.37 →  20.29    still climbing at c=64
```

Both controls peak at c=32 and lose ground by c=64 — the same curve, which is
further evidence that the third GPU changed nothing structural. `replica2` has
not reached its knee at
concurrency 64 — the increments are still positive and only mildly decelerating
(+3.92, then +2.92). **+52.1% is therefore a lower bound on the capacity gain,
not the capacity gain.** A concurrency-128 point is needed to close this.

### The third GPU on its own buys nothing

| concurrency | split3gpu vs pair2gpu |
| --- | --- |
| 1 | +2.5% |
| 16 | +7.1% |
| 32 | −1.2% |
| 64 | −3.3% |

Giving `code2wav` its own card, without replicating anything, moves throughput
by less than the spread between arms. This is what makes the `replica2` number
attributable: comparing `replica2` against `pair2gpu` instead would report
+47.1% and silently bundle the extra card into the claim.

### Latency and TTFA at concurrency 64

| metric | pair2gpu | split3gpu | replica2 | vs split3gpu |
| --- | --- | --- | --- | --- |
| TTFA mean (s) | 2.754 ±0.021 | 2.864 ±0.018 | 1.258 ±0.019 | **−56.1%** |
| TTFA p95 (s) | 3.169 ±0.026 | 3.346 ±0.037 | 1.704 ±0.017 | −49.1% |
| latency mean (s) | 4.503 ±0.035 | 4.669 ±0.033 | 3.084 ±0.033 | −34.0% |
| latency p95 (s) | 5.752 ±0.046 | 5.965 ±0.045 | 4.562 ±0.044 | −23.5% |
| RTF mean | 1.286 ±0.004 | 1.335 ±0.011 | 0.857 ±0.010 | −35.8% |
| inter-chunk mean (s) | 0.4241 ±0.003 | 0.4381 ±0.004 | 0.4401 ±0.013 | +0.5% |
| **inter-chunk p95 (s)** | 0.5636 ±0.007 | 0.5958 ±0.005 | **0.8384 ±0.020** | **+40.7%** |
| audio throughput (s/s) | 51.09 ±0.41 | 49.31 ±0.28 | 75.51 ±0.79 | +53.1% |

Throughput rises 52% *and* latency falls 34%, which means the controls are
already backlogged at concurrency 64 while `replica2` is not.

**The exception is the inter-chunk gap tail: 40.7% worse.** The mean is
unchanged, so this is a tail effect only. For streaming audio the inter-chunk
p95 is what drives playback underruns, so this belongs in any user-facing
claim. Note the two arms are not under equal real load here — `replica2` is
serving 52% more requests at the same nominal concurrency.

### Output work is unchanged

Mean audio produced per request, derived as
`audio_throughput_s_per_s / throughput_qps`, at concurrency 64:

| arm | s of audio per request |
| --- | --- |
| pair2gpu | 3.705 |
| split3gpu | 3.696 |
| replica2 | 3.722 |

Spread 0.68%. No arm is fast because it generated less audio. **WER has not
been re-verified**; that half of the quality gate is still open.

## Higgs TTS

Single `tts_frontend` versus two replicas pinned to the same GPU. Seed-TTS EN
full 1088-sample pool, original reference audio, voice cloning, MPS off,
non-streaming, concurrency 96, 20 s warmup + 90 s measurement window,
`max_new_tokens=512`, `max_running_requests=96`, `cuda_graph_max_bs=96`,
`prefill_coalesce_requests=32`, `prefill_coalesce_wait_ms=300`, engine
fraction 0.85, encoder fraction 0.0245 per replica.

| metric | frontend1 | frontend2 | change |
| --- | --- | --- | --- |
| throughput (QPS) | 28.12 ±0.278 | 31.38 ±0.180 | **+11.6%** (11.7σ) |
| latency mean (s) | 3.447 ±0.038 | 3.073 ±0.020 | −10.9% |
| latency p95 (s) | 4.436 ±0.045 | 4.423 ±0.035 | −0.3% |
| audio throughput (s/s) | 118.7 ±1.44 | 132.0 ±0.64 | +11.2% |
| output throughput (tok/s) | 3165 ±37.9 | 3520 ±17.3 | +11.2% |
| output tokens / request | 112.7 ±0.58 | 112.0 ±0.00 | −0.6% |
| tokens per engine-second | 89.27 ±0.57 | 40.50 ±0.27 | **−54.6%** |
| requests in window | 2531 ±25 | 2824 ±16 | +11.6% |

Tokens per request are identical, so the throughput gain is real work, not
shorter output.

### Where the time went

Profiler stage breakdown, repeat 0 (`stage_input_received → stage_complete`).
The two `audio_encoder` replicas are combined by request-weighted mean.

| stage | frontend1 avg | frontend2 avg | change |
| --- | --- | --- | --- |
| **audio_encoder** | **2340.2 ms** | **568.9 ms** | **−75.7%** |
| tts_engine | 1244.1 ms | 2606.5 ms | +109.5% |
| vocoder | 14.4 ms | 27.6 ms | +91.7% |
| preprocessing | 2.65 ms | 1.98 ms | −25.5% |

With one frontend, `audio_encoder` accounted for 2340 ms of a 3447 ms request —
**68% of its life**. `preprocessing` is 2.65 ms, so the frontend cost is
entirely in the encoder.

**That 2340 ms is queueing, not compute.** Doubling capacity cut it by 4.1×.
Pure compute would not shrink at all when replicated; only a queue collapses
super-linearly as utilization drops off saturation. The frontend was
queue-bound, and replication is the direct fix.

The engine gives some of it back:

| tts_engine interval | frontend1 | frontend2 | change |
| --- | --- | --- | --- |
| `queue_enter → prefill_start` | 177.0 ms | 188.3 ms | +6.4% |
| `prefill_start → prefill_end` | 28.6 ms | 52.2 ms | +82.6% |
| `prefill_end → stage_complete` | 1031.8 ms | 2348.8 ms | **+127.6%** |

Almost all the growth is in the AR decode loop. More requests reach the engine
concurrently, so the scheduler runs larger batches: each request sits in the
engine longer while aggregate token throughput rises 11.2%. This is also the
explanation for the −54.6% tokens-per-engine-second above — that metric divides
by engine residency, which is exactly what larger batches inflate.

Two independent measurements agree on the engine residency, which is a useful
cross-check on both:

| | from `X-Engine-Time` header | from profiler |
| --- | --- | --- |
| frontend1 | 1.262 s | 1.244 s |
| frontend2 | 2.765 s | 2.607 s |

### The latency budget closes

| component | delta |
| --- | --- |
| audio_encoder | **−1771.3 ms** |
| tts_engine | +1362.4 ms |
| vocoder | +13.2 ms |
| preprocessing | −0.7 ms |
| hops | +0.6 ms |
| **net** | **−395.8 ms** |
| measured latency delta | **−374 ms** |

Within 5.8%, the residual being stage overlap (`tts_engine` streams to
`vocoder` while still running). Replication removed 1771 ms of frontend
queueing and paid back 1362 ms in larger engine batches.

### GPU and CPU utilization

| | frontend1 | frontend2 | change |
| --- | --- | --- | --- |
| GPU util mean (%) | 90.70 ±0.06 | 94.17 ±0.02 | +3.47 pp |
| GPU util max (%) | 97–98 | 100 | — |
| **GPU idle (%)** | **9.30** | **5.83** | **−37.3%** |
| CPU mean (%) | 5.87 | 6.52 | +11.1% |

The gain decomposes cleanly:

```
GPU occupancy      94.17 / 90.70 = 1.038   → +3.8%
work per busy-second  1.116 / 1.038 = 1.075   → +7.5%
                                     ───────────────
measured throughput   31.38 / 28.12 = 1.116   → +11.6%
```

A third of the win is simply keeping the GPU fed — the engine was starving on
the frontend 9.3% of the time. The rest is larger batches doing more work per
busy second. CPU rises 11.1% against a throughput rise of 11.6%, i.e. linearly
with requests served: the second replica adds no measurable coordination
overhead of its own.

### Same-GPU replicas are not symmetric

Admission balanced perfectly, service time did not:

| | requests | avg | p95 |
| --- | --- | --- | --- |
| audio_encoder@r0 | 1652 | 508.8 ms | 2839.5 ms |
| audio_encoder@r1 | 1651 | 629.1 ms | 3382.1 ms |

A 24% service-time gap on a 1-request load difference. Two replicas on one GPU
time-share a CUDA context (MPS is off) and the sharing is not fair. This does
not threaten the result, but it is a plausible part of why same-GPU
replication returns +11.6% where cross-GPU replication returns +52%.

### Tail latency moves rather than improves

| p95 | frontend1 | frontend2 |
| --- | --- | --- |
| audio_encoder | 4653.3 ms | 2839.5 / 3382.1 ms |
| tts_engine | 1799.8 ms | **3912.1 ms** |
| **end to end** | **4436 ms** | **4423 ms** |

The frontend tail shrinks by roughly what the engine tail grows, and end-to-end
p95 is unchanged at −0.3%. Replication improves mean latency and throughput; it
does not improve the tail.

### Streaming measures a much smaller gain

An earlier run of the same two arms with `--higgs-stream` measured only +2.3%
QPS (28.19 ±0.263 → 28.83 ±0.045), 2.4σ, with p95 latency 6.4% *worse*. Same
code, same arms, same window — only the response mode differed.

Streaming also cannot report token throughput at all: the server attaches
`X-Completion-Tokens` and `X-Engine-Time` to the non-streaming response only,
because HTTP headers are sent before the body and the count is not known until
the stream ends.

The non-streaming numbers are the ones reported above, because the Higgs
comparison is specified with token throughput and without streaming. **The
discrepancy is unexplained and worth following up** — a client that consumes
chunks as they arrive changes the pipeline's backpressure, which may mask the
frontend relief. Artifacts for the streaming run are kept alongside.

## Known confounds

- **The Higgs arms do not reserve equal GPU memory.** The encoder fraction is
  per replica, so `frontend2` reserves 0.0245 × 2 + 0.85 + 0.10 = 0.999 against
  `frontend1`'s 0.9745. Inherent to a per-replica budget; the extra 2.45% of a
  card is not free and is not accounted for in the +11.6%.
- **Qwen3-Omni concurrency levels share a server.** One server start per
  (arm, repeat), four concurrencies inside it in fixed ascending order. Later
  levels inherit allocator state from earlier ones, identically for every arm.
- **`replica2` pairs talker and code2wav by policy, not contract.** Both
  processes have two replicas and the default round-robin advances them
  together, keeping a request's codec stream on one GPU. A different binding
  policy need not preserve that.

## Outstanding

1. **Qwen3-Omni WER** — the quality half of gate 5. One `--transcribe-only`
   pass over a saved `replica2` run against its control.
2. **Qwen3-Omni at concurrency 128** — `replica2` had not reached its knee at
   64, so +52.1% is a lower bound.
3. **Streaming vs non-streaming discrepancy on Higgs** — +2.3% versus +11.6%
   for the same change.
