# TTS Process Topology

`StageConfig.process` is the only source of truth for process topology. Stages
that name the same process share one OS process; stages with different names run
in different processes. There is no CLI override — a deployment that needs a
different topology declares it in YAML.

For example, a config can make vocoder isolation persistent:

```yaml
stages:
  - name: vocoder
    process: vocoder
```

Or it can keep the vocoder in a shared process:

```yaml
stages:
  - name: vocoder
    process: pipeline
```

## Changing Placement Without Editing the Model Config

A YAML config or a dotted override sets `process` on individual stages. The
following reproduces the topology the built-in Higgs-TTS config already
declares:

```bash
python -m sglang_omni.cli serve \
  --model-path bosonai/higgs-tts-3-4b \
  --stages.preprocessing.process tts_frontend \
  --stages.audio_encoder.process tts_frontend
```

```text
tts_frontend : preprocessing, audio_encoder
pipeline     : tts_engine
vocoder      : vocoder
```

Running a stage alone means giving it a process name nothing else uses.

## How a Topology Is Validated

The compiler enumerates every cross-process edge of the final topology from
`next`, `stream_to`, and `wait_for`, and applies two independent model contracts:

- `process_local_edges()` — which handoffs must stay inside one process because
  the payload does not carry required process-local state. Edges are splittable
  by default. The exception is declared per **edge**, not per stage, because
  grouping `preprocessing` with `audio_encoder` leaves their shared handoff local
  while still permitting `audio_encoder -> tts_engine` to cross processes.
- `process_edge_resources()` — which GPU memory fractions to apply when that
  edge crosses processes. A recommendation, not a capability, applied only to
  stages that declare no fraction of their own. An edge with no recommendation is
  still splittable when the config already declares fractions or nothing else
  shares its GPU; otherwise placement validation names the stages whose fractions
  are missing.

The constraint is checked once while compiling the config, including for edges
that tensor parallelism creates by putting a TP stage in its own process. A
process-local handoff is reported before a missing fraction, because declaring
fractions would not make that split correct.

## Applicability by Model

| Model | Process-local edges | Recommended fractions |
| --- | --- | --- |
| Higgs-TTS | — | none needed; the config already declares 0.03 / 0.85 / 0.10 |
| FishAudio S2-Pro | — | `tts_engine -> vocoder` |
| Voxtral TTS | — | `tts_generation -> vocoder` |
| Ming-Omni-TTS | — | `tts_engine -> audio_decode` |
| MOSS-TTS Local (single-GPU) | `preprocessing -> tts_engine` — preprocessing publishes into a process-local `PreparedRequestQueue` the AR stage pops | `tts_engine -> vocoder` |
| MOSS-TTS Local (split) | all pipeline edges; placement declares GPU 0 while the codec runs on `cuda:1` | — |
| Qwen3-TTS | `preprocessing -> tts_engine` — prepared requests live in `_PREPROCESSING_CONTEXT` / `_PREPARED_REQUESTS`, read in-process by the AR engine builder | `tts_engine -> vocoder` |
| MOSS-TTS Delay | `preprocessing -> tts_engine` — same process-local `PreparedRequestQueue` handoff | `tts_engine -> vocoder` |
| Audar-TTS | — | none yet — declare fractions before splitting |
| Zonos2 | — | none yet — declare fractions before splitting |

Higgs-TTS already groups `preprocessing` and `audio_encoder` in a
`tts_frontend` process and places `vocoder` in its own process by default.
Redeclaring either placement is a no-op; the stages can still be fully separated
or regrouped under another process name.

Audar-TTS and Zonos2 carry stage state in `StagePayload.data`, but neither ships
recommended fractions, so a split on a shared GPU fails with the missing-fraction
error until the operator declares
`runtime.resources.total_gpu_memory_fraction` for every stage on that GPU.

## Process Replicas

`PipelineConfig.processes` gives a Process more than one instance. A replica
copies the whole Process, so members never end up in different replicas:

```yaml
processes:
  vocoder:
    num_replicas: 2
    replica_devices: [1, 2]
```

Each request picks one replica per replicated Process at admission and keeps it
for its lifetime. See
[`config.md`](../developer_reference/config.md) for the naming, device, and
memory-fraction rules.

## Resource and Performance Trade-offs

Splitting a stage out creates another OS process and usually another CUDA
context. It can improve throughput by overlapping vocoder scheduling and GPU
work with generation, but it also changes IPC and serialization paths, can
increase idle VRAM, and may duplicate process-local caches or runtime state.
Grouping stages that share a cache or a local handoff keeps that cost down,
which is what a shared process name expresses.

When multiple processes share one GPU, all affected GPU stages must declare
compatible `runtime.resources.total_gpu_memory_fraction` values, and their total
must fit the placement limit. A model may opt out of that requirement with
`require_memory_fraction_for_colocation: false`, including for sharing introduced
by `replica_devices`. Explicitly configured fractions still count toward the
placement limit. Recommended fractions fill in only where the config declares
none, so explicitly configured values are preserved, and conflicting
recommendations for one stage are rejected.

These fractions are placement-accounting declarations, not proof of an
allocator-enforced runtime limit. A factory receives
`total_gpu_memory_fraction` only when its signature accepts that argument, and
an SGLang `mem_fraction_static` override can represent a different runtime
value. Keep runtime overrides consistent with the placement declaration.
Unsafe declared same-GPU topologies are rejected before startup.

Performance depends on the model, hardware, concurrency, request shape, and
streaming mode. Measure the target workload before making a topology change
persistent in model or YAML configuration.
