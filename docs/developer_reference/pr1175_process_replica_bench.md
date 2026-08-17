# PR #1175 process-replica performance runbook

How to measure whether process-level replication actually buys anything, on
top of `feat/runtime-stage-replicas`.

Two independent comparisons:

- **Qwen3-Omni** — three speech topologies over the same three GPUs.
- **Higgs TTS** — one `tts_frontend` process versus two pinned to the same GPU.

Everything is driven by `benchmarks/eval/bench_process_replicas.py`, which
owns server lifecycle, interleaving, artifact retention and the replica
residency checks. Running the arms by hand is possible but gives up the
interleaving and the binding verification, which are the two things that make
the result attributable.

## Arms

### Qwen3-Omni (3 GPUs)

| key | config | thinker | talker_ar | code2wav |
| --- | --- | --- | --- | --- |
| `pair2gpu` | `examples/configs/qwen3_omni_speech_pair2gpu.yaml` | 0 | 1 | 1 |
| `split3gpu` | `examples/configs/qwen3_omni_speech_split3gpu.yaml` | 0 | 1 | 2 |
| `replica2` | `examples/configs/qwen3_omni_speech_replica2.yaml` | 0 | @r0→1, @r1→2 | @r0→1, @r1→2 |

`split3gpu` is the control that matters. `replica2` uses three GPUs, so a
delta against `pair2gpu` alone conflates "we replicated" with "we got another
card". Only the `replica2` − `split3gpu` delta isolates replication.

Note that the pipeline default places `code2wav` on the *thinker* GPU, not the
talker GPU (see `_speech_stages` in `sglang_omni/models/qwen3_omni/config.py`).
`pair2gpu` therefore is not the stock default — it is an explicit config, which
is also why all three arms are declared the same way.

### Higgs TTS (1 GPU)

| key | config | tts_frontend | tts_engine | vocoder |
| --- | --- | --- | --- | --- |
| `frontend1` | `examples/configs/higgs_frontend_single.yaml` | 1 × GPU 0 | GPU 0 | GPU 0 |
| `frontend2` | `examples/configs/higgs_frontend_replica2.yaml` | 2 × GPU 0 | GPU 0 | GPU 0 |

## Preconditions

- ≥3 GPUs (H100/H200 class). The Higgs suite only uses the first one.
- **MPS off.** `examples/mps_dp/launch.sh` starts an MPS control daemon and
  same-GPU replicas behave differently under it. The driver refuses to start
  if it finds `nvidia-cuda-mps-control` running:
  `echo quit | nvidia-cuda-mps-control`.
- `CUDA_VISIBLE_DEVICES` **unset** in your shell. Pass `--gpus` instead — the
  driver sets it per server and needs the raw physical ids for NVML sampling.
  It refuses to start otherwise.
- Seed-TTS EN staged locally (step 1).

## Step 0 — compile the arms before spending GPU time

```bash
pytest tests/unit_test/pipeline/test_pr1175_bench_configs.py -q
```

Sub-second, CPU only. It compiles all five arm configs and pins the placements
this runbook claims. A typo in a YAML fails here instead of after a
multi-minute server start.

## Step 1 — dataset

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
```

Both suites read `zhaochenyang20/seed-tts-eval-arrow`, EN split. The Higgs arm
uses the full 1088-sample pool with its original reference audio and voice
cloning (`references[]` payload shape).

## Step 2 — Qwen3-Omni

```bash
python -m benchmarks.eval.bench_process_replicas \
    --suite qwen3-omni \
    --gpus 0,1,2 \
    --repeats 3 \
    --concurrency 1,16,32,64 \
    --qwen-samples 128 \
    --out results/pr1175
```

`--gpus` takes physical ids, so `--gpus 3,5,7` runs the same topologies on
three otherwise-idle cards without editing any YAML — the configs address
GPUs logically as 0/1/2 and the driver maps them via `CUDA_VISIBLE_DEVICES`.

Load per measurement: `benchmark_omni_seedtts --generate-only --stream`,
Seed-TTS EN, `--temperature 0.0`, `--max-new-tokens 256`.

Reported per (arm, concurrency), meaned over repeats with stdev:
`throughput_qps`, `audio_ttfp_mean_s` / `audio_ttfp_p95_s` (TTFA),
`rtf_mean`, `latency_mean_s` / `latency_p95_s`,
`inter_chunk_mean_s` / `inter_chunk_p95_s` (inter-chunk gap), and
`audio_throughput_s_per_s` (audio seconds per second).

Structure: one server start per (arm, repeat) — 9 starts — with the four
concurrencies run inside it in a fixed ascending order. That satisfies "at
least three independent server starts per arm" while keeping the run to a
sane length. The tradeoff is that later concurrencies inherit allocator state
from earlier ones; the order is identical across arms, so it biases all arms
the same way. Pass a single value to `--concurrency` and raise `--repeats` if
you want that confound gone entirely.

## Step 3 — Higgs TTS

```bash
python -m benchmarks.eval.bench_process_replicas \
    --suite higgs \
    --gpus 0 \
    --repeats 3 \
    --higgs-concurrency 96 \
    --higgs-warmup-s 20 \
    --higgs-measure-s 90 \
    --out results/pr1175
```

The Higgs load is duration-driven, not sample-count driven: a closed loop of
`--higgs-concurrency` workers cycles the 1088-sample pool, the first 20 s are
discarded as warmup, and the reported window is the next 90 s. Requests are
attributed by **completion** time — workers stay busy for the whole window, so
what is still in flight when it closes is balanced by what was in flight when
it opened. Attributing by issue time instead would drop every request started
near the deadline, and would drop more of them from the slower arm. Whatever
was still running at the deadline is reported as `truncated_requests` and is
not folded into the metrics.

Reported: `throughput_qps`, `latency_mean_s` / `latency_p95_s`,
`audio_throughput_s_per_s`, `output_throughput_tok_s`, `output_tok_per_req_s`,
plus per-run GPU/CPU utilization (`resources`) and the profiler stage
breakdown, from which the frontend stage time is read.

Both arms fix `max_new_tokens=512`, `max_running_requests=96`,
`cuda_graph_max_bs=96`, `prefill_coalesce_requests=32`,
`prefill_coalesce_wait_ms=300`, engine fraction `0.85`, encoder fraction
`0.0245` per replica.

`--higgs-concurrency` defaults to 96 to match the scheduler capacity; it is
not part of the original spec, so record whatever you use.

## Artifacts

```
results/pr1175/<suite>/
  provenance.json          commit, dirty flag, nvidia-smi, packages, all knobs
  runs.json                every run record, rewritten after each one
  summary.json             per (arm, concurrency) mean/stdev/min/max
  summary.md               the same as a table
  r<repeat>/<arm>/
    arm_config.yaml        the exact config that was served
    server.log             full server log (the bindings evidence lives here)
    result.json            binding verdict, compute-app snapshot, startup time
    c<N>/
      measurement.json     summary + profiler report for this concurrency
      speed_results.json   raw per-request records (Qwen3-Omni)
      client.log
      events/              raw profiler JSONL
```

Failed runs keep their directory, their log and their error, and the driver
continues to the next arm. `summary.json` lists them under `failed_runs`.

To re-read a profiler event directory later:

```bash
python -m sglang_omni.profiler results/pr1175/higgs/r0/frontend2/c96/events --format table
```

## Attribution gates

A gain may be attributed to process replication only when all of the
following hold. The driver checks the first three mechanically.

1. **Bindings match the config.** For each replicated process, admission must
   have bound every configured replica id, and every replica instance
   (`talker_ar@r0`, `tts_frontend@r1`, …) must appear in the server log.
   Control arms must produce no bindings at all. Reported as
   `summary.json → binding_ok`, with per-run detail in `result.json →
   binding`. A `false` here invalidates the comparison — the replicas were
   configured but not actually used.
2. **No failed runs.** `failed_runs` empty.
3. **Every arm completed three interleaved repeats.** `repeats: 3` on every
   row of `summary.json`.
4. **The gain exceeds restart noise.** Compare the mean delta against the
   stdev across repeats within each arm. A delta smaller than the larger of
   the two arms' stdevs is restart noise, not a result.
5. **Output work and quality are unchanged.** `completed_requests` and
   `audio_duration_mean_s` must be comparable across arms — an arm that
   produced less audio is not faster. For Qwen3-Omni, re-run one
   configuration with the transcribe phase and confirm WER has not moved:
   ```bash
   python -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only \
       --meta zhaochenyang20/seed-tts-eval-arrow \
       --output-dir results/pr1175/qwen3-omni/r0/replica2/c16 \
       --model qwen3-omni --lang en --port <asr-port>
   ```
6. **For Qwen3-Omni, use the `split3gpu` control.** See the arms table.

## Known confounds

- **The Higgs arms do not reserve the same amount of GPU memory.** The encoder
  fraction is per replica, so `frontend2` reserves `0.0245 × 2 + 0.85 + 0.10 =
  0.999` against `frontend1`'s `0.9745`. That is inherent to a per-replica
  budget and belongs in the writeup. It also sits right on the validation cap
  of 1.0, leaving almost nothing for CUDA contexts and fragmentation — if
  `frontend2` OOMs at startup, trim `tts_engine` **in both arms**, never in
  just one.
- **Qwen3-Omni concurrencies share a server.** See step 2.
- **`replica2` pairs talker and code2wav by policy, not by contract.** Both
  processes have two replicas and the default round-robin policy advances them
  together, which keeps a request's codec stream on one GPU. A different
  binding policy need not preserve that, so do not report it as a guarantee.
