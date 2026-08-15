# perf/preprocessing-concurrency（tip 2282f948，base 68abc7ee）
#
# fbd40981 把 preprocessing 改成 max_concurrency=4，默认生效。
# 2282f948 只是把硬编码的 max_batch_wait_ms=50 暴露成环境变量，默认仍是 50 ——
# 也就是说 tip 跑默认值等于没测第二个 commit，必须扫 wait 值才有意义。
ARM_NAMES=(base wait50 wait10 wait0)
ARM_WT=("$BASE_WT" "$TIP_WT" "$TIP_WT" "$TIP_WT")
ARM_EXTRA=("" "" "" "")
ARM_ENV=(
  ""
  ""
  "SGLANG_OMNI_ENCODER_BATCH_WAIT_MS=10"
  "SGLANG_OMNI_ENCODER_BATCH_WAIT_MS=0"
)

arms_preflight() {
  if ! grep -q "SGLANG_OMNI_ENCODER_BATCH_WAIT_MS" "$TIP_WT/sglang_omni/models/qwen3_omni/stages.py"; then
    log "!!! $TIP_WT 没有 SGLANG_OMNI_ENCODER_BATCH_WAIT_MS，wait10/wait0 臂会静默退化成 wait50"
    return 1
  fi
  if ! grep -q "max_concurrency" "$TIP_WT/sglang_omni/models/qwen3_omni/stages.py"; then
    log "!!! $TIP_WT 的 preprocessing 没有 max_concurrency"
    return 1
  fi
}
