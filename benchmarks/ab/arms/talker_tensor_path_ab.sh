# perf/talker-tensor-path：base vs 分支最新状态（后 3 个 commit 整体）
#
# base 取 74cfa1be —— 分支第一个 commit 是 mm_aggregate 摘除，属于另一个 PR，
# 两侧都带着它，测出来的才是后 3 个 commit 的净效果。
# 三个 commit 都无 flag 无环境变量，默认即生效，所以两条臂就够。
ARM_NAMES=(base tip)
ARM_WT=("$BASE_WT" "$TIP_WT")
ARM_EXTRA=("" "")
ARM_ENV=("" "")

arms_preflight() {
  local -a want=(74cfa1be b2e4f53f)
  local i got
  for i in "${!ARM_NAMES[@]}"; do
    got=$(git -C "${ARM_WT[$i]}" rev-parse --short=8 HEAD 2>/dev/null)
    if [[ "$got" != "${want[$i]}" ]]; then
      log "!!! ${ARM_NAMES[$i]} 臂在 ${ARM_WT[$i]} 上是 ${got:-未知}，应为 ${want[$i]}"
      return 1
    fi
  done
  log "    两条臂的 HEAD 与预期 commit 一致 ✓"
}
