#!/usr/bin/env python3
"""汇总 run_ab.sh 的结果：每格 3 轮取平均，输出 baseline vs PR 对比表。"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

METRICS = [
    ("throughput_qps", "吞吐 QPS", "higher"),
    ("audio_ttfp_mean_s", "首包 TTFP 均值 (s)", "lower"),
    ("latency_mean_s", "端到端延迟均值 (s)", "lower"),
    ("latency_p95_s", "端到端延迟 P95 (s)", "lower"),
    ("rtf_mean", "RTF 均值", "lower"),
]
BRANCHES = ["base", "pr"]


def load_cell(cell: Path) -> tuple[dict[str, float], list[str]]:
    """返回 (指标 -> 多轮均值, 告警列表)。只读 run*/，discard*/ 忽略。"""
    per_metric: dict[str, list[float]] = {k: [] for k, _, _ in METRICS}
    warnings: list[str] = []
    runs = sorted(cell.glob("run*"))
    if not runs:
        return {}, [f"{cell}: 没有 run* 目录"]
    for run in runs:
        path = run / "speed_results.json"
        if not path.exists():
            warnings.append(f"{run}: 缺 speed_results.json")
            continue
        summary = json.loads(path.read_text())["summary"]
        failed = summary.get("failed_requests", 0)
        if failed:
            warnings.append(f"{run}: {failed} 个请求失败，该轮数据不可信")
        for key, _, _ in METRICS:
            value = summary.get(key)
            if value is not None:
                per_metric[key].append(float(value))
    return (
        {k: statistics.mean(v) for k, v in per_metric.items() if v},
        warnings,
    )


def spread(cell: Path, key: str) -> float | None:
    """同格 3 轮的相对极差，用来判断噪声是否盖过了改动。"""
    values = []
    for run in sorted(cell.glob("run*")):
        path = run / "speed_results.json"
        if path.exists():
            value = json.loads(path.read_text())["summary"].get(key)
            if value is not None:
                values.append(float(value))
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    return (max(values) - min(values)) / mean * 100 if mean else None


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    concurrencies = sorted(
        (p for p in root.glob("c*") if p.is_dir()),
        key=lambda p: int(p.name[1:]),
    )
    all_warnings: list[str] = []

    for conc_dir in concurrencies:
        conc = conc_dir.name[1:]
        cells = {}
        for branch in BRANCHES:
            values, warns = load_cell(conc_dir / branch)
            cells[branch] = values
            all_warnings.extend(warns)

        print(f"\n### concurrency = {conc}")
        print(f"| 指标 | baseline | PR | Δ | 3 轮极差(base/pr) |")
        print(f"|---|---|---|---|---|")
        for key, label, direction in METRICS:
            base_v, pr_v = cells["base"].get(key), cells["pr"].get(key)
            if base_v is None or pr_v is None:
                continue
            delta = (pr_v - base_v) / base_v * 100 if base_v else 0.0
            good = (delta > 0) if direction == "higher" else (delta < 0)
            mark = "✅" if abs(delta) >= 1 and good else ("🔴" if abs(delta) >= 1 else "·")
            sb = spread(conc_dir / "base", key)
            sp = spread(conc_dir / "pr", key)
            spread_txt = f"{sb:.1f}%/{sp:.1f}%" if sb is not None and sp is not None else "-"
            print(
                f"| {label} | {base_v:.4g} | {pr_v:.4g} | "
                f"{delta:+.2f}% {mark} | {spread_txt} |"
            )

    if all_warnings:
        print("\n### 告警")
        for warning in all_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
