"""F2 / F3 — Pareto analysis of the M1, M2 and M3 benchmark.

Reads results/benchmark_results.json (written by benchmark.py) and writes:
  results/benchmark_results.csv  the five-metric table (3 variants x 5 metrics = 15 cells),
                                 Critical recall, and whether each variant is Pareto-optimal
  results/pareto_chart.png       accuracy against mean latency, file size and energy, with
                                 each panel's Pareto frontier and the F3 pick annotated

A variant is Pareto-optimal when no other variant is at least as good on every metric
and strictly better on at least one (accuracy: higher is better; latency, file size and
energy: lower is better).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSON_PATH = RESULTS_DIR / "benchmark_results.json"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
CHART_PATH = RESULTS_DIR / "pareto_chart.png"

RECOMMENDED = "M3_PRUNED_INT8"  # Task F3 decision, deployment/deployment_recommendation.md
ACCURACY_GATE_PCT = 88.0

# Display name, colour and marker per variant. Colours are fixed per variant in every panel.
STYLE = {
    "M1_FP32": ("M1 FP32", "#2a78d6", "o"),
    "M2_PTQ_INT8": ("M2 PTQ INT8", "#eb6834", "s"),
    "M3_PRUNED_INT8": ("M3 Pruned + INT8", "#1baf7a", "D"),
}
SURFACE, INK, INK_MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"

# (result key, +1 when higher is better / -1 when lower is better)
ACCURACY = ("val_accuracy", 1)
OBJECTIVES = [ACCURACY, ("mean_latency_ms", -1), ("p95_latency_ms", -1),
              ("file_size_kb", -1), ("energy_mj", -1)]
PANELS = [  # (cost key, panel title, axis label, value format)
    ("mean_latency_ms", "Accuracy vs mean latency", "Mean latency per inference (ms, log scale)", "{:.4f} ms"),
    ("file_size_kb", "Accuracy vs file size", "Model file size (KB, log scale)", "{:.2f} KB"),
    ("energy_mj", "Accuracy vs energy", "Energy per inference (mJ, log scale)", "{:.5f} mJ"),
]
# Label placement (x, y offset in points, alignment): M2 and M3 sit close together on every
# cost axis, so M2's label goes below and M3's (leftmost) label runs rightwards from its marker.
LABEL_PLACEMENT = {"M1_FP32": (0, 12, "center"), "M2_PTQ_INT8": (0, -30, "center"),
                   "M3_PRUNED_INT8": (-8, 12, "left")}


def dominates(a, b, objectives):
    """True if a is at least as good as b on every objective and strictly better on one."""
    at_least_as_good = all(sign * a[key] >= sign * b[key] for key, sign in objectives)
    strictly_better = any(sign * a[key] > sign * b[key] for key, sign in objectives)
    return at_least_as_good and strictly_better


def pareto_set(results, objectives):
    return {name for name, r in results.items()
            if not any(dominates(other, r, objectives) for o, other in results.items() if o != name)}


def write_csv(results, optimal):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "model_file", "mean_latency_ms", "p95_latency_ms", "file_size_kb",
                         "accuracy_pct", "energy_mj", "critical_recall_pct", "pareto_optimal",
                         "savedmodel_size_kb"])
        for name, r in results.items():
            # M1 only: the same model as a TF SavedModel (file_size_kb is the .keras file the service loads)
            savedmodel_kb = f"{r['savedmodel_size_kb']:.2f}" if "savedmodel_size_kb" in r else ""
            writer.writerow([
                name, r["model_file"], f"{r['mean_latency_ms']:.4f}", f"{r['p95_latency_ms']:.4f}",
                f"{r['file_size_kb']:.2f}", f"{r['val_accuracy'] * 100:.2f}", f"{r['energy_mj']:.5f}",
                f"{r['critical_recall'] * 100:.2f}", "yes" if name in optimal else "no", savedmodel_kb,
            ])


def summary(results, optimal):
    """One data-driven caption: why the frontier looks the way it does."""
    accuracies = {r["val_accuracy"] for r in results.values()}
    lines = []
    if len(accuracies) == 1:
        lines.append(f"All three variants reach {accuracies.pop() * 100:.0f}% validation accuracy, "
                     "so cost alone decides the frontier.")
    best = {key: STYLE[min(results, key=lambda n: results[n][key])][0] for key, *_ in PANELS}
    lines.append(f"Lowest mean latency: {best['mean_latency_ms']}. Smallest file: {best['file_size_kb']}. "
                 f"Lowest energy: {best['energy_mj']}.")
    names = ", ".join(STYLE[n][0] for n in results if n in optimal)
    dominated = ", ".join(STYLE[n][0] for n in results if n not in optimal) or "none"
    lines.append(f"Pareto-optimal across all five F2 metrics: {names}. Dominated: {dominated}. "
                 f"Task F3 deploys {STYLE[RECOMMENDED][0]}.")
    return "\n".join(lines)


def draw_chart(results, optimal):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True, facecolor=SURFACE)

    for ax, (key, title, xlabel, fmt) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        frontier = pareto_set(results, [ACCURACY, (key, -1)])
        costs = [r[key] for r in results.values()]
        x_min, x_max = min(costs) / 3, max(costs) * 3

        # Frontier as an attainment step: nothing to the right of / below it is worth choosing
        points = sorted((results[n][key], results[n]["val_accuracy"] * 100) for n in frontier)
        xs = [x for x, _ in points] + [x_max]
        ys = [y for _, y in points] + [points[-1][1]]
        ax.step(xs, ys, where="post", color=INK_MUTED, linewidth=2, linestyle="--", zorder=1)

        ax.axhline(ACCURACY_GATE_PCT, color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.text(x_min * 1.15, ACCURACY_GATE_PCT + 0.4, f"{ACCURACY_GATE_PCT:.0f}% accuracy gate",
                color=INK_MUTED, fontsize=8, va="bottom")

        for name, r in results.items():
            label, colour, marker = STYLE[name]
            x, y = r[key], r["val_accuracy"] * 100
            filled = name in frontier
            # Frontier marks drawn above dominated ones, so a near-tie never hides the winner
            ax.scatter(x, y, s=110, marker=marker, zorder=4 if filled else 3, linewidths=2,
                       facecolors=colour if filled else SURFACE,
                       edgecolors=SURFACE if filled else colour)
            dx, dy, ha = LABEL_PLACEMENT[name]
            text = f"{label}{' (F3 pick)' if name == RECOMMENDED else ''}\n{fmt.format(x)}"
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=8, color=INK,
                        fontweight="bold" if name == RECOMMENDED else "normal")

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(85, 104)
        ax.set_yticks(range(86, 101, 2))
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", color=RULE, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation accuracy (%)")

    handles = [Line2D([], [], marker=m, linestyle="none", markersize=8, markerfacecolor=c,
                      markeredgecolor=c, label=n) for n, c, m in STYLE.values()]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=INK_MUTED,
               markeredgecolor=INK_MUTED, label="Filled: on this panel's frontier"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=SURFACE,
               markeredgecolor=INK_MUTED, markeredgewidth=2, label="Hollow: dominated"),
        Line2D([], [], color=INK_MUTED, linewidth=2, linestyle="--", label="Pareto frontier"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("F2 Pareto analysis: M1 / M2 / M3 on the held-out validation split",
                 x=0.01, y=0.99, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, summary(results, optimal), ha="left", va="bottom", color=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.88))
    fig.savefig(CHART_PATH, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    if not JSON_PATH.is_file():
        print(f"{JSON_PATH} is missing. Run optimisation/benchmark.py first.", file=sys.stderr)
        return 1
    results = json.loads(JSON_PATH.read_text())["results"]
    optimal = pareto_set(results, OBJECTIVES)

    write_csv(results, optimal)
    draw_chart(results, optimal)
    print(summary(results, optimal))
    print(f"\nSaved {CSV_PATH.name} and {CHART_PATH.name} in {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
