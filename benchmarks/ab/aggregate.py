#!/usr/bin/env python3
"""汇总 A/B 结果：每格多轮取中位数，各臂与 base 对比。

臂名从目录结构推断，所以两臂（base/pr）和三臂（base/mixed/inter）通用。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# (key, 标签, 方向)。中位数优先：极差大时均值被离群点主导。
METRICS = [
    ("throughput_qps", "吞吐 QPS", "higher"),
    ("text_ttft_median_s", "首字 TTFT 中位 (s)", "lower"),
    ("audio_ttfp_median_s", "首包 TTFP 中位 (s)", "lower"),
    ("latency_median_s", "端到端延迟中位 (s)", "lower"),
    ("latency_p95_s", "端到端延迟 P95 (s)", "lower"),
    ("latency_p99_s", "端到端延迟 P99 (s)", "lower"),
    ("inter_chunk_p95_s", "音频块间隔 P95 (s)", "lower"),
    ("rtf_median", "RTF 中位", "lower"),
]

# 不比性能，只用来判断这次对比是否成立
VALIDITY = [
    ("completed_requests", "完成请求数"),
    ("failed_requests", "失败请求数"),
    ("output_tokens_mean", "输出 token 均值"),
    ("audio_duration_mean_s", "音频时长均值 (s)"),
]

REFERENCE_ARM = "base"
# 已知臂按递进关系排；其余臂按字母序接在后面
ARM_ORDER = [
    "base",
    "pr",
    "mixed",
    "inter",
    "wait50",
    "wait10",
    "wait0",
    "gap0",
    "gap10",
    "gap25",
    "aux",
    "inline",
    "tip",
]


def load_runs(cell: Path) -> tuple[list[dict], list[str]]:
    """读该格所有计入统计的轮次（run*/，discard*/ 忽略）。"""
    summaries: list[dict] = []
    warnings: list[str] = []
    if not cell.is_dir():
        return [], []
    runs = sorted(cell.glob("run*"))
    if not runs:
        return [], [f"{cell}: 没有 run* 目录"]
    for run in runs:
        path = run / "speed_results.json"
        if not path.exists():
            warnings.append(f"{run}: 缺 speed_results.json")
            continue
        summary = json.loads(path.read_text())["summary"]
        if summary.get("failed_requests", 0):
            warnings.append(
                f"{run}: {summary['failed_requests']} 个请求失败，该轮不可信"
            )
        summaries.append(summary)
    return summaries, warnings


def series(summaries: list[dict], key: str) -> list[float]:
    return [float(s[key]) for s in summaries if s.get(key) is not None]


def spread_pct(values: list[float]) -> float | None:
    """轮次之间的相对极差，用来判断噪声是否盖过了改动。"""
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    return (max(values) - min(values)) / mean * 100 if mean else None


def emit_comparison(
    ref_runs: list[dict],
    arm_runs: list[dict],
    ref_name: str,
    arm_name: str,
) -> tuple[int, int]:
    """打印一臂对参照臂的对比表，返回 (可判读数, 不可判读数)。"""
    resolvable = unresolved = 0
    n_rounds = min(len(ref_runs), len(arm_runs))
    print(f"\n**{arm_name} vs {ref_name}**\n")
    print(
        f"| 指标 | {ref_name} | {arm_name} | Δ | {n_rounds} 轮极差 | "
        f"取值区间 {ref_name} / {arm_name} | 可判读 |"
    )
    print("|---|---|---|---|---|---|---|")

    for key, label, direction in METRICS:
        rs, as_ = series(ref_runs, key), series(arm_runs, key)
        if not rs or not as_:
            continue
        ref_v, arm_v = statistics.median(rs), statistics.median(as_)
        if not ref_v:
            continue
        delta = (arm_v - ref_v) / ref_v * 100
        good = (delta > 0) if direction == "higher" else (delta < 0)
        mark = "✅" if abs(delta) >= 1 and good else ("🔴" if abs(delta) >= 1 else "·")

        sr, sa = spread_pct(rs), spread_pct(as_)
        spread_txt = "-" if sr is None or sa is None else f"{sr:.1f}%/{sa:.1f}%"
        # 区间完全不重叠才算测得出来；单个离群轮只会撑大区间，不会伪造分离。
        if len(rs) < 2 or len(as_) < 2:
            verdict = "?"
        elif max(rs) < min(as_) or max(as_) < min(rs):
            verdict = "是"
            resolvable += 1
        else:
            verdict = "**否**"
            unresolved += 1
        print(
            f"| {label} | {ref_v:.4g} | {arm_v:.4g} | {delta:+.2f}% {mark} | "
            f"{spread_txt} | [{min(rs):.3g},{max(rs):.3g}] / "
            f"[{min(as_):.3g},{max(as_):.3g}] | {verdict} |"
        )
    return resolvable, unresolved


def emit_validity(runs: dict[str, list[dict]], arms: list[str]) -> None:
    print("\n有效性校验（各臂应当一致，不一致则对比不成立）：\n")
    print("| 项 | " + " | ".join(arms) + " |")
    print("|---" * (len(arms) + 1) + "|")
    for key, label in VALIDITY:
        values = {}
        for arm in arms:
            vs = series(runs[arm], key)
            if vs:
                values[arm] = statistics.median(vs)
        if len(values) < len(arms):
            continue
        ref = values[arms[0]]
        cells = []
        for arm in arms:
            cell = f"{values[arm]:.4g}"
            if (
                key in ("output_tokens_mean", "audio_duration_mean_s")
                and ref
                and abs(values[arm] - ref) / ref > 0.05
            ):
                cell += " ⚠️"
            cells.append(cell)
        print(f"| {label} | " + " | ".join(cells) + " |")


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    conc_dirs = sorted(
        (p for p in root.glob("c*") if p.is_dir()),
        key=lambda p: int(p.name[1:]),
    )
    all_warnings: list[str] = []
    resolvable = unresolved = 0

    for conc_dir in conc_dirs:
        found = sorted(p.name for p in conc_dir.iterdir() if p.is_dir())
        if REFERENCE_ARM not in found:
            all_warnings.append(f"{conc_dir}: 没有 {REFERENCE_ARM!r} 臂，跳过")
            continue
        others = sorted(
            (a for a in found if a != REFERENCE_ARM),
            key=lambda a: (
                ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER),
                a,
            ),
        )
        arms = [REFERENCE_ARM] + others

        runs: dict[str, list[dict]] = {}
        for arm in arms:
            summaries, warns = load_runs(conc_dir / arm)
            runs[arm] = summaries
            all_warnings.extend(warns)

        compare = [a for a in arms[1:] if runs[a]]
        if not runs[REFERENCE_ARM] or not compare:
            continue

        print(f"\n### concurrency = {conc_dir.name[1:]}")
        for arm in compare:
            r, u = emit_comparison(runs[REFERENCE_ARM], runs[arm], REFERENCE_ARM, arm)
            resolvable += r
            unresolved += u
        emit_validity(runs, [REFERENCE_ARM] + compare)

    total = resolvable + unresolved
    if total:
        print("\n### 判读\n")
        print(f"{total} 项对比中 **{resolvable} 项** 取值区间完全分离，可以采信；")
        print(f"其余 {unresolved} 项区间重叠，不支持任何方向的结论。")

    if all_warnings:
        print("\n### 告警")
        for warning in all_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
