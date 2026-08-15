# perf/talker-mixed-chunk-interleave（tip 0d20f5f6，base 68abc7ee）
#
# 第二个 commit 只加了 --talker-prefill-decode-interleave，默认 off，
# 所以 tip 跑默认参数只能测到 enable_mixed_chunk。
ARM_NAMES=(base mixed inter)
ARM_WT=("$BASE_WT" "$TIP_WT" "$TIP_WT")
ARM_EXTRA=("" "" "--talker-prefill-decode-interleave on")
ARM_ENV=("" "" "")

arms_preflight() {
  if ! grep -q "talker-prefill-decode-interleave" "$TIP_WT/sglang_omni/cli/serve.py"; then
    log "!!! $TIP_WT 的 serve.py 没有 --talker-prefill-decode-interleave，inter 臂无法测"
    return 1
  fi
}
