"""Compare two evaluation results (e.g. cross-domain vs. single-domain baseline).

Inputs are JSON files written by ``geoshift-evaluate`` or ``test_metrics.json``
files written by ``geoshift-train``. Produces a Markdown table and, if
matplotlib is installed, a bar chart.

Example::

    python scripts/analyze_results.py \
        --model experiments/cross_domain/eval_all_rmd17-aspirin.json \
        --baseline experiments/single_domain/eval_all_rmd17-aspirin.json \
        --metric energy_mae --output results/
"""

import argparse
import json
from pathlib import Path

SKIP = {"equivariance", "meta", "best_epoch", "training_time_hours"}


def load(path):
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if k not in SKIP and isinstance(v, dict)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Results of the model under test.")
    parser.add_argument("--baseline", required=True, help="Results of the baseline.")
    parser.add_argument("--metric", default="energy_mae")
    parser.add_argument("--labels", nargs=2, default=["Cross-domain", "Baseline"])
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    model, baseline = load(args.model), load(args.baseline)
    groups = [g for g in model if g in baseline and args.metric in model[g] and args.metric in baseline[g]]
    if not groups:
        raise SystemExit(f"No dataset groups with metric '{args.metric}' in both files.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        f"| Dataset | {args.labels[1]} | {args.labels[0]} | Relative change |",
        "|---|---|---|---|",
    ]
    for g in groups:
        b, m = baseline[g][args.metric], model[g][args.metric]
        rows.append(f"| {g} | {b:.4f} | {m:.4f} | {100 * (m - b) / b:+.1f}% |")
    table = "\n".join(rows)
    (out_dir / f"comparison_{args.metric}.md").write_text(table + "\n")
    print(table)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    x = range(len(groups))
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(groups)), 4))
    ax.bar([i - 0.2 for i in x], [baseline[g][args.metric] for g in groups], 0.4, label=args.labels[1])
    ax.bar([i + 0.2 for i in x], [model[g][args.metric] for g in groups], 0.4, label=args.labels[0])
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(args.metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_{args.metric}.pdf")
    fig.savefig(out_dir / f"comparison_{args.metric}.png", dpi=200)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
