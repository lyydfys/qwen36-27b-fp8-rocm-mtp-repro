#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROUND4 = [
    ("baseline", 21.455850, 47.726203, 1.00),
    ("MTP t2", 42.413046, 24.144029, 1.98),
    ("MTP t3", 48.675316, 21.039062, 2.27),
    ("MTP t4", 54.481873, 18.795832, 2.54),
]


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1220",
            "axes.facecolor": "#0f172a",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#dbeafe",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#e2e8f0",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
        }
    )


def save_round4(out_dir: Path):
    labels = [x[0] for x in ROUND4]
    tok_s = [x[1] for x in ROUND4]
    latency = [x[2] for x in ROUND4]
    colors = ["#94a3b8", "#38bdf8", "#2dd4bf", "#f97316"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    fig.suptitle("Qwen3.6-27B-FP8 on AMD ROCm: Forced 1024-token Decode")

    axes[0].bar(labels, tok_s, color=colors)
    axes[0].set_ylabel("Median decode throughput (tok/s)")
    axes[0].set_title("Higher is better")
    for i, value in enumerate(tok_s):
        axes[0].text(i, value + 1.2, f"{value:.2f}", ha="center", va="bottom")

    axes[1].bar(labels, latency, color=colors)
    axes[1].set_ylabel("Median latency (s)")
    axes[1].set_title("Lower is better")
    for i, value in enumerate(latency):
        axes[1].text(i, value + 1.0, f"{value:.2f}s", ha="center", va="bottom")

    for ax in axes:
        ax.grid(axis="y", color="#334155", alpha=0.45)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_dir / "round4_forced_1024_decode.png", bbox_inches="tight")
    plt.close(fig)


def read_round5(summary_path: Path):
    if not summary_path.exists():
        return []
    rows = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 10 and parts[0] in {"baseline", "mtp_t4"}:
            rows.append(
                {
                    "label": parts[0],
                    "concurrency": int(parts[1]),
                    "aggregate_tok_s": float(parts[5]),
                    "p50_latency": float(parts[6]),
                    "p95_latency": float(parts[7]),
                    "all_full": parts[9] == "True",
                }
            )
    return rows


def save_round5(out_dir: Path, summary_path: Path):
    rows = read_round5(summary_path)
    if not rows:
        return
    concurrencies = sorted({r["concurrency"] for r in rows})
    baseline = {r["concurrency"]: r for r in rows if r["label"] == "baseline"}
    mtp = {r["concurrency"]: r for r in rows if r["label"] == "mtp_t4"}

    x = range(len(concurrencies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    fig.suptitle("Small-concurrency Serving Throughput")
    b_vals = [baseline[c]["aggregate_tok_s"] for c in concurrencies]
    m_vals = [mtp[c]["aggregate_tok_s"] for c in concurrencies]
    ax.bar([i - width / 2 for i in x], b_vals, width, label="baseline", color="#94a3b8")
    ax.bar([i + width / 2 for i in x], m_vals, width, label="MTP t4", color="#f97316")
    ax.set_xticks(list(x), [f"c={c}" for c in concurrencies])
    ax.set_ylabel("Aggregate completion tok/s")
    ax.grid(axis="y", color="#334155", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend()
    for i, c in enumerate(concurrencies):
        speedup = m_vals[i] / b_vals[i] if b_vals[i] else 0
        ax.text(i, max(b_vals[i], m_vals[i]) + 1.5, f"{speedup:.2f}x", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "round5_concurrency_serving.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--round5-summary")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style()
    save_round4(out_dir)
    if args.round5_summary:
        save_round5(out_dir, Path(args.round5_summary))


if __name__ == "__main__":
    main()

