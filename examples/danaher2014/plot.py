"""Plot the five Figure 2 comparisons from a completed Julia simulation run.

Usage: python3 examples/danaher2014/plot.py path/to/output
Requires matplotlib. Uses actual precision-matrix truth by default; add
--generating-graph to plot recovery against the sparse generating skeleton.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--generating-graph", action="store_true")
    args = parser.parse_args()
    with (args.directory / "summary.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError("summary.tsv has no completed fits")
    metadata = dict(line.strip().split(" = ", 1) for line in (args.directory / "metadata.txt").read_text().splitlines() if " = " in line)
    graph = "graph_" if args.generating_graph else ""
    panels = [
        (graph + "fp", graph + "tp", "False positive edges", "True positive edges"),
        ("edges", "sse", "Edges selected", "Off-diagonal squared error"),
        (graph + "fp_diff", graph + "tp_diff", "False differential edges", "True differential edges"),
        ("l1", "dkl_paper", "Off-diagonal L1 norm", "Paper dKL (includes Kp/2)"),
        ("edges", "seconds", "Edges selected", "Fit time (seconds)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7), constrained_layout=True)
    groups = sorted({(r["method"], float(r["lambda2"])) for r in rows}, key=lambda v: (v[0] != "GL", v[1]))
    colors = plt.get_cmap("Reds")
    for group_index, (method, lambda2) in enumerate(groups):
        series = [r for r in rows if r["method"] == method and float(r["lambda2"]) == lambda2]
        series.sort(key=lambda r: -float(r["lambda1"]))
        label = "Separate graphical lasso" if method == "GL" else f"FGL, $\\lambda_2$ = {lambda2:g}"
        color = "#222222" if method == "GL" else colors(0.45 + 0.5 * group_index / max(1, len(groups)-1))
        style = "-" if method == "GL" else ["-", "--", ":", "-."][(group_index-1) % 4]
        marker = "o" if method == "GL" else ["o", "s", "^", "D"][(group_index-1) % 4]
        for panel_index, (axis, (x, y, xlabel, ylabel)) in enumerate(zip(axes.flat, panels)):
            xv, yv = [[float(r[field + "_mean"]) for r in series] for field in (x, y)]
            axis.plot(xv, yv, marker=marker, linestyle=style, ms=4, lw=1.7, label=label, color=color)
            # Show Monte Carlo standard errors when at least two replications
            # were run. These are uncertainty bars, not published-paper bands.
            if all(int(r["replicates"]) > 1 for r in series):
                se = [float(r[y + "_se"]) for r in series]
                axis.errorbar(xv, yv, yerr=se, fmt="none", capsize=2, alpha=0.4, color=color)
            axis.set(xlabel=xlabel, ylabel=ylabel, title=f"({chr(97+panel_index)})")
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
    axes[1, 1].set_yscale("log")
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="upper left", frameon=False)
    replications = int(rows[0]["replicates"])
    truth = "generating graph" if args.generating_graph else "actual precision matrices"
    axes[1, 2].text(0.05, 0.42, f"{replications} Monte Carlo replicate(s)\nTruth: {truth}\nBars: +/- 1 Monte Carlo SE\n\nRegenerated simulation design;\nnot the original paper's data.", transform=axes[1, 2].transAxes, va="top", fontsize=10)
    fig.suptitle(f"Danaher, Wang and Witten (2014): three-group FGL simulation\np = {metadata['p']}, n = {metadata['n']} per group; {metadata['mode']} run", fontsize=14)
    name = "figure2_generating_graph" if args.generating_graph else "figure2"
    for extension in ("png", "svg"):
        fig.savefig(args.directory / f"{name}.{extension}", dpi=170)
    print(args.directory / f"{name}.png")


if __name__ == "__main__":
    main()
