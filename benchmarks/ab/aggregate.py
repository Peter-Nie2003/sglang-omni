#!/usr/bin/env python3
"""汇总 run_ab.sh 的结果：每格多轮取中位数，输出 baseline vs PR 对比表。"""

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
BRANCHES = ["base", "pr"]


def load_runs(cell: Path) -> tuple[list[dict], list[str]]:
    """读该格所有计入统计的轮次（run*/，discard*/ 忽略）。"""
    summaries: list[dict] = []
    warnings: list[str] = []
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


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    conc_dirs = sorted(
        (p for p in root.glob("c*") if p.is_dir()),
        key=lambda p: int(p.name[1:]),
    )
    all_warnings: list[str] = []
    resolvable = unresolved = 0

    for conc_dir in conc_dirs:
        runs = {}
        for branch in BRANCHES:
            summaries, warns = load_runs(conc_dir / branch)
            runs[branch] = summaries
            all_warnings.extend(warns)
        if not runs["base"] or not runs["pr"]:
            continue

        print(f"\n### concurrency = {conc_dir.name[1:]}")
        n_rounds = min(len(runs["base"]), len(runs["pr"]))
        print(f"| 指标 | baseline | PR | Δ | {n_rounds} 轮极差(base/pr) | 可判读 |")
        print("|---|---|---|---|---|---|")

        for key, label, direction in METRICS:
            bs, ps = series(runs["base"], key), series(runs["pr"], key)
            if not bs or not ps:
                continue
            base_v, pr_v = statistics.median(bs), statistics.median(ps)
            if not base_v:
                continue
            delta = (pr_v - base_v) / base_v * 100
            good = (delta > 0) if direction == "higher" else (delta < 0)
            mark = "✅" if abs(delta) >= 1 and good else ("🔴" if abs(delta) >= 1 else "·")

            sb, sp = spread_pct(bs), spread_pct(ps)
            if sb is None or sp is None:
                spread_txt, verdict = "-", "?"
            else:
                spread_txt = f"{sb:.1f}%/{sp:.1f}%"
                # Δ 必须超过两侧各自的轮间波动，否则读到的是噪声
                if abs(delta) > max(sb, sp):
                    verdict = "是"
                    resolvable += 1
                else:
                    verdict = "**否**"
                    unresolved += 1
            print(
                f"| {label} | {base_v:.4g} | {pr_v:.4g} | "
                f"{delta:+.2f}% {mark} | {spread_txt} | {verdict} |"
            )

        print(f"\n有效性校验（两分支应当一致，不一致则对比不成立）：")
        print("| 项 | baseline | PR |")
        print("|---|---|---|")
        for key, label in VALIDITY:
            bs, ps = series(runs["base"], key), series(runs["pr"], key)
            if not bs or not ps:
                continue
            bm, pm = statistics.median(bs), statistics.median(ps)
            flag = ""
            if key in ("output_tokens_mean", "audio_duration_mean_s") and bm:
                if abs(pm - bm) / bm > 0.05:
                    flag = "  ⚠️ 相差 >5%，延迟对比失效"
            print(f"| {label} | {bm:.4g} | {pm:.4g}{flag} |")

    total = resolvable + unresolved
    if total:
        print(f"\n### 判读\n")
        print(f"{total} 项对比中 **{resolvable} 项** 的 Δ 超过了轮间波动，可以采信；")
        print(f"其余 {unresolved} 项落在噪声里，不支持任何方向的结论。")

    if all_warnings:
        print("\n### 告警")
        for warning in all_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
