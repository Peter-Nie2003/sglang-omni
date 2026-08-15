# perf/admission-staggering（tip 68e7c2e3，base 68abc7ee）
#
# 68e7c2e3 在 coordinator 入口加了 admission_min_gap_ms 闸门，默认 0.0 = 关闭。
# 也就是说 tip 跑默认值和 base 行为完全一致，必须扫 gap 值才测得到东西。
#
# gap0 是空对照：与 base 只差一次函数调用和一个 if，任何"显著"差异都是噪声，
# 用来标定分离判据的假阳性率。
# gap 值上限受吞吐约束：闸门把准入速率钉死在 1000/gap req/s，而 base 峰值约
# 11.8 QPS（c=64），所以 gap 超过 ~85ms 会直接卡住吞吐；10/25 留足余量。
ARM_NAMES=(base gap0 gap10 gap25)
ARM_WT=("$BASE_WT" "$TIP_WT" "$TIP_WT" "$TIP_WT")
ARM_EXTRA=(
  ""
  ""
  "--admission-min-gap-ms=10"
  "--admission-min-gap-ms=25"
)
ARM_ENV=("" "" "" "")

arms_preflight() {
  if ! grep -q "admission_min_gap_ms" "$TIP_WT/sglang_omni/pipeline/coordinator.py"; then
    log "!!! $TIP_WT 没有 admission_min_gap_ms，gap 臂会静默退化成 base"
    return 1
  fi
  if ! grep -q "allow_extra_args" "$TIP_WT/sglang_omni/cli/__init__.py"; then
    log "!!! $TIP_WT 的 serve 不接受额外参数，--admission-min-gap-ms 会被丢弃"
    return 1
  fi
  # 直接走一遍 CLI 的覆盖路径，确认值真的落到 config 上
  local got
  got=$( cd "$TIP_WT" && PYTHONPATH="$TIP_WT" "$PY" -c "
from sglang_omni.config.manager import ConfigManager
m = ConfigManager.from_file('$TIP_WT/$COLOCATED_CONFIG')
c = m.merge_config(m.parse_extra_args(['--admission-min-gap-ms=25']))
print(c.admission_min_gap_ms)
" 2>/dev/null )
  case "$got" in
    25|25.0) ;;
    *) log "!!! --admission-min-gap-ms 没落到 config（得到 '${got:-无输出}'）—— gap 臂无效"; return 1 ;;
  esac
  log "    admission_min_gap_ms 覆盖路径校验通过 ✓"
}
