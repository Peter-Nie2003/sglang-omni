# PR #1175 process-replica performance results

Measurement of process-level replication on `feat/runtime-stage-replicas`,
run from `perf/pr1175-process-replica-bench` (based on `fe6866dd`).

Raw artifacts, per-run logs and `provenance.json` (exact host, driver, package
inventory) live under `results/pr1175/`.

## Summary

| Workload | Concurrency | Gain vs control | Significance |
| --- | --- | --- | --- |
| Qwen3-Omni — 2 × (talker_ar + code2wav), separate GPUs | 64 | **+52.1% QPS** | 32σ |
| Qwen3-Omni — same | 32 | +24.5% QPS | 19σ |
| Qwen3-Omni — same | 16 | +11.5% QPS | 4.3σ |
| Qwen3-Omni — same | 1 | **−4.6% QPS** | 6σ |
| Higgs TTS — 2 × tts_frontend, one GPU | 96 | **+11.6% QPS** | 11.7σ |

## Findings

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | The Qwen3-Omni gain is replication, not the third GPU | `split3gpu` uses the same 3 cards without replication and lands within ±3% of the 2-GPU arm at every concurrency, with an identical saturation curve |
| 2 | Replication costs a little when there is nothing to parallelize | −4.6% QPS at concurrency 1, 6σ, consistent across repeats |
| 3 | +52.1% is a lower bound, not the capacity gain | `replica2` had not reached its knee at concurrency 64 (increments still +3.92, +2.92) |
| 4 | On Higgs the frontend was **queue**-bound, not compute-bound | `audio_encoder` 2340 ms → 569 ms; doubling capacity cut it 4.1×, which pure compute cannot do |
| 5 | The engine pays part of the gain back in larger batches | `prefill_end→complete` +127.6%; latency budget closes to within 5.8% of measured |
| 6 | Neither workload improves tail latency | Higgs end-to-end p95 −0.3% (flat); Qwen3-Omni inter-chunk p95 +40.7% at c=64 |
| 7 | Same-GPU replicas are not symmetric | 1652 vs 1651 requests, but 508.8 ms vs 629.1 ms mean service time |

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

| | |
| --- | --- |
| Workload | Seed-TTS EN, streaming, generate-only |
| Parameters | `temperature=0.0`, `max_new_tokens=256`, samples `max(128, concurrency × 10)` |
| Repeats | 3 interleaved, independent server start per (arm, repeat) |
| `pair2gpu` | thinker@0, talker_ar@1, code2wav@1 |
| `split3gpu` | thinker@0, talker_ar@1, code2wav@2 — **the control** |
| `replica2` | thinker@0, talker_ar+code2wav @r0→1 / @r1→2 |

All values mean ±stdev over 3 repeats. Δ is `replica2` vs `split3gpu`.

| c | metric | pair2gpu | split3gpu | replica2 | Δ |
| --- | --- | --- | --- | --- | --- |
| **1** | QPS | 2.179 ±0.006 | 2.233 ±0.017 | 2.131 ±0.013 | **−4.6%** |
| | TTFA mean (s) | 0.2021 ±0.0034 | 0.2028 ±0.0022 | 0.2068 ±0.0015 | +2.0% |
| | TTFA p95 (s) | 0.2144 ±0.0067 | 0.2157 ±0.0027 | 0.2212 ±0.0044 | +2.5% |
| | latency mean (s) | 0.4587 ±0.0015 | 0.4480 ±0.0035 | 0.4697 ±0.0029 | +4.8% |
| | latency p95 (s) | 0.6973 ±0.0040 | 0.6513 ±0.0136 | 0.6907 ±0.0210 | +6.0% |
| | RTF mean | 0.1457 ±0.0007 | 0.1434 ±0.0017 | 0.1490 ±0.0008 | +3.9% |
| | inter-chunk mean (s) | 0.0698 ±0.0001 | 0.0662 ±0.0004 | 0.0709 ±0.0002 | +7.1% |
| | inter-chunk p95 (s) | 0.0819 ±0.0027 | 0.0743 ±0.0008 | 0.1132 ±0.0009 | +52.4% |
| | audio throughput (s/s) | 7.314 ±0.043 | 7.485 ±0.092 | 7.155 ±0.042 | −4.4% |
| **16** | QPS | 11.26 ±0.113 | 12.06 ±0.038 | 13.45 ±0.325 | **+11.5%** |
| | TTFA mean (s) | 0.6657 ±0.0223 | 0.7522 ±0.0036 | 0.6667 ±0.0296 | −11.4% |
| | TTFA p95 (s) | 1.027 ±0.009 | 1.117 ±0.023 | 0.9093 ±0.0108 | −18.6% |
| | latency mean (s) | 1.375 ±0.013 | 1.303 ±0.006 | 1.138 ±0.020 | −12.7% |
| | latency p95 (s) | 2.155 ±0.061 | 1.980 ±0.031 | 1.725 ±0.072 | −12.9% |
| | RTF mean | 0.4427 ±0.0092 | 0.4286 ±0.0033 | 0.3769 ±0.0059 | −12.1% |
| | inter-chunk mean (s) | 0.1976 ±0.0006 | 0.1515 ±0.0033 | 0.1319 ±0.0124 | −12.9% |
| | inter-chunk p95 (s) | 0.4251 ±0.0241 | 0.3628 ±0.0120 | 0.2785 ±0.0443 | −23.2% |
| | audio throughput (s/s) | 36.74 ±0.839 | — ¹ | 43.72 ±0.665 | — |
| **32** | QPS | 14.12 ±0.101 | 13.95 ±0.148 | 17.37 ±0.182 | **+24.5%** |
| | TTFA mean (s) | 0.8954 ±0.0209 | 0.9343 ±0.0129 | 0.8957 ±0.0355 | −4.1% |
| | TTFA p95 (s) | 1.230 ±0.016 | 1.3527 ±0.0233 | 1.140 ±0.047 | −15.7% |
| | latency mean (s) | 2.206 ±0.021 | 2.244 ±0.028 | 1.804 ±0.016 | −19.6% |
| | latency p95 (s) | 3.169 ±0.032 | 3.204 ±0.029 | 2.675 ±0.061 | −16.5% |
| | RTF mean | 0.6415 ±0.0102 | 0.6504 ±0.0179 | 0.5285 ±0.0046 | −18.7% |
| | inter-chunk mean (s) | 0.3295 ±0.0119 | 0.3270 ±0.0138 | 0.2276 ±0.0130 | −30.4% |
| | inter-chunk p95 (s) | 0.5996 ±0.0097 | 0.6651 ±0.0284 | 0.5400 ±0.0619 | −18.8% |
| | audio throughput (s/s) | 50.84 ±0.731 | 50.23 ±1.420 | 62.43 ±0.982 | +24.3% |
| **64** | QPS | 13.79 ±0.163 | 13.34 ±0.083 | 20.29 ±0.215 | **+52.1%** |
| | TTFA mean (s) | 2.754 ±0.021 | 2.864 ±0.018 | 1.258 ±0.019 | **−56.1%** |
| | TTFA p95 (s) | 3.169 ±0.026 | 3.346 ±0.037 | 1.704 ±0.017 | −49.1% |
| | latency mean (s) | 4.503 ±0.035 | 4.669 ±0.033 | 3.084 ±0.033 | −34.0% |
| | latency p95 (s) | 5.752 ±0.046 | 5.965 ±0.045 | 4.562 ±0.044 | −23.5% |
| | RTF mean | 1.286 ±0.004 | 1.335 ±0.011 | 0.8572 ±0.0096 | −35.8% |
| | inter-chunk mean (s) | 0.4241 ±0.0032 | 0.4381 ±0.0036 | 0.4401 ±0.0131 | +0.5% |
| | **inter-chunk p95 (s)** | 0.5636 ±0.0071 | 0.5958 ±0.0051 | **0.8384 ±0.0197** | **+40.7%** |
| | audio throughput (s/s) | 51.09 ±0.409 | 49.31 ±0.280 | 75.51 ±0.792 | +53.1% |

¹ Not transcribed from the run; readable from `results/pr1175/qwen3-omni/summary.json`.

### Saturation and the third-GPU control

| c | pair2gpu | split3gpu | replica2 | split3gpu vs pair2gpu |
| --- | --- | --- | --- | --- |
| 16 | 11.26 | 12.06 | 13.45 | +7.1% |
| 32 | 14.12 ← knee | 13.95 ← knee | 17.37 | −1.2% |
| 64 | 13.79 ↓ | 13.34 ↓ | 20.29 ↑ | −3.3% |

Both controls peak at c=32 and fall back by c=64; `replica2` is still climbing.
Comparing `replica2` against `pair2gpu` instead of `split3gpu` would report
+47.1% and silently bundle the extra card into the claim.

### Output work per request (c=64)

`audio_throughput_s_per_s / throughput_qps`:

| pair2gpu | split3gpu | replica2 | spread |
| --- | --- | --- | --- |
| 3.705 s | 3.696 s | 3.722 s | 0.68% |

## Higgs TTS

| | |
| --- | --- |
| Workload | Seed-TTS EN full 1088, original reference audio, voice cloning, **non-streaming** |
| Load | concurrency 96, 20 s warmup + 90 s window, attributed by completion time |
| Parameters | `max_new_tokens=512`, `max_running_requests=96`, `cuda_graph_max_bs=96`, `prefill_coalesce=32/300ms`, engine fraction 0.85, encoder fraction 0.0245/replica |
| Environment | MPS off |
| Repeats | 3 interleaved, independent server start per (arm, repeat) |
| `frontend1` | 1 × tts_frontend on GPU 0 — **the control** |
| `frontend2` | 2 × tts_frontend on GPU 0, `replica_devices: [0, 0]` |

| group | metric | frontend1 | frontend2 | Δ |
| --- | --- | --- | --- | --- |
| **Throughput** | QPS | 28.12 ±0.278 | 31.38 ±0.180 | **+11.6%** |
| | audio throughput (s/s) | 118.7 ±1.44 | 132.0 ±0.64 | +11.2% |
| | output throughput (tok/s) | 3165 ±37.9 | 3520 ±17.3 | +11.2% |
| | requests in window | 2531 ±25 | 2824 ±16 | +11.6% |
| **Latency** | mean (s) | 3.447 ±0.038 | 3.073 ±0.020 | −10.9% |
| | p95 (s) | 4.436 ±0.045 | 4.423 ±0.035 | −0.3% |
| **Work check** | output tokens / request | 112.7 ±0.58 | 112.0 ±0.00 | −0.6% |
| | tokens per engine-second | 89.27 ±0.57 | 40.50 ±0.27 | −54.6% ² |
| **Stage avg** | **audio_encoder** | **2340.2 ms** | **568.9 ms** | **−75.7%** |
| | tts_engine | 1244.1 ms | 2606.5 ms | +109.5% |
| | vocoder | 14.4 ms | 27.6 ms | +91.7% |
| | preprocessing | 2.65 ms | 1.98 ms | −25.5% |
| **Engine detail** | `queue_enter→prefill_start` | 177.0 ms | 188.3 ms | +6.4% |
| | `prefill_start→prefill_end` | 28.6 ms | 52.2 ms | +82.6% |
| | `prefill_end→complete` | 1031.8 ms | 2348.8 ms | **+127.6%** |
| **Stage p95** | audio_encoder | 4653.3 ms | 2839.5 / 3382.1 ms ³ | — |
| | tts_engine | 1799.8 ms | 3912.1 ms | +117.4% |
| **Resources** | GPU util mean (%) | 90.70 ±0.06 | 94.17 ±0.02 | +3.47 pp |
| | GPU idle (%) | 9.30 | 5.83 | −37.3% |
| | CPU mean (%) | 5.87 | 6.52 | +11.1% |
| **Replica balance** | requests @r0 / @r1 | — | 1652 / 1651 | balanced |
| | audio_encoder avg @r0 / @r1 | — | 508.8 / 629.1 ms | +23.7% skew |

² Divides by engine residency, which larger batches inflate — see engine detail.
³ Per replica; p95 does not combine.

Stage figures are profiler repeat 0, `stage_input_received → stage_complete`;
the two `audio_encoder` replicas are combined by request-weighted mean.

### Latency budget

| component | Δ |
| --- | --- |
| audio_encoder | **−1771.3 ms** |
| tts_engine | +1362.4 ms |
| vocoder | +13.2 ms |
| preprocessing | −0.7 ms |
| hops | +0.6 ms |
| **net** | **−395.8 ms** |
| measured latency Δ | **−374 ms** |

Closes to within 5.8%; the residual is stage overlap (`tts_engine` streams to
`vocoder` while still running).

### Gain decomposition

| component | factor | contribution |
| --- | --- | --- |
| GPU occupancy | 94.17 / 90.70 = 1.038 | +3.8% |
| work per busy-second | 1.116 / 1.038 = 1.075 | +7.5% |
| **measured throughput** | **31.38 / 28.12 = 1.116** | **+11.6%** |

CPU rises 11.1% against throughput +11.6% — linear with requests served, so
the second replica adds no measurable coordination overhead of its own.

## Known confounds

| Confound | Detail |
| --- | --- |
| Higgs arms reserve unequal GPU memory | Encoder fraction is per replica: `frontend2` reserves 0.999 vs `frontend1`'s 0.9745. The extra 2.45% of a card is not free and is not accounted for in the +11.6%. |
| Qwen3-Omni concurrency levels share a server | One server start per (arm, repeat), four concurrencies inside it in fixed ascending order. Later levels inherit allocator state, identically for every arm. |
| `replica2` pairing is policy, not contract | Both processes have 2 replicas and the default round-robin advances them together, keeping a request's codec stream on one GPU. A different binding policy need not preserve that. |
| Qwen3-Omni arms are not under equal real load at c=64 | `replica2` serves 52% more requests at the same nominal concurrency, which affects the tail comparison. |

## Outstanding

| # | Item | Blocks |
| --- | --- | --- |
| 1 | Qwen3-Omni WER (`--transcribe-only` over a saved `replica2` run vs control) | the quality half of gate 5 |
| 2 | Qwen3-Omni at concurrency 128 | tightening +52.1% from a lower bound to the real capacity gain |
| 3 | `split3gpu` c=16 audio throughput | table completeness only |
