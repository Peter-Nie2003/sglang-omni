# perf/talker-tensor-path，只测后 3 个 commit
#
# 分支有 4 个 commit，第一个 74cfa1be 是 mm_aggregate 摘除（另一个 PR），
# 所以 base 取 74cfa1be 而不是 68abc7ee。三个 commit 都无 flag 无环境变量，
# 默认即生效，逐个叠加成臂来定位效果来自哪一个。
#
#   aux     34e469ab  thinker->talker 只发 layer_hidden，不再把 embed 一起塞进 metadata
#   inline  8885563d  <=16KB 的 CPU chunk 直接内联进控制消息（阈值硬编码）
#   tip     b2e4f53f  prefill embeds 保持 tensor，去掉 cpu().tolist()
#
# tip vs inline 在语音链路上是空对照：唯一的生产调用点
# request_builders.py 传 input_embeds_are_projected=True，base 里那条
# `None if input_embeds_are_projected else ...tolist()` 永远走 None 分支，
# b2e4f53f 删掉的热点在这条路径上根本不执行。
ARM_NAMES=(base aux inline tip)
ARM_WT=("$BASE_WT" "$AUX_WT" "$INLINE_WT" "$TIP_WT")
ARM_EXTRA=("" "" "" "")
ARM_ENV=("" "" "" "")

arms_preflight() {
  local missing=0 v
  for v in AUX_WT INLINE_WT; do
    [[ -n "${!v:-}" ]] || { log "!!! 需要设置 $v"; missing=1; }
  done
  (( missing )) && return 1

  # 每条臂必须真的是它声称的那个 commit，否则叠加关系就废了
  local -a want=(74cfa1be 34e469ab 8885563d b2e4f53f)
  local i got
  for i in "${!ARM_NAMES[@]}"; do
    got=$(git -C "${ARM_WT[$i]}" rev-parse --short=8 HEAD 2>/dev/null)
    if [[ "$got" != "${want[$i]}" ]]; then
      log "!!! ${ARM_NAMES[$i]} 臂在 ${ARM_WT[$i]} 上是 ${got:-未知}，应为 ${want[$i]}"
      return 1
    fi
  done

  if ! grep -q "_INLINE_STREAM_CHUNK_BYTES_LIMIT" "$INLINE_WT/sglang_omni/comm/stage_io.py"; then
    log "!!! $INLINE_WT 没有内联 chunk 实现"
    return 1
  fi
  log "    四条臂的 HEAD 与预期 commit 一致 ✓"
}
