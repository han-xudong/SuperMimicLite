"""Rebuild the README learning curve from the accepted 64-point evaluation CSV.

Run from the repository root: uv run --with matplotlib python docs/build_figures.py
No training, policy inference, physics, smoothing or invented zero-point data.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {"P-L": "#c5a136", "P-M": "#d9826b", "S-L": "#9eb875", "S-M": "#2f6236"}
LABELS = {
    "P-L": "PPO / LaFAN40",
    "P-M": "PPO / Mixed",
    "S-L": "SuperMimic / LaFAN40",
    "S-M": "SuperMimic / Mixed",
}


def main():
    with (ROOT / "reference/learning-curves.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 64
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#171913", "axes.labelcolor": "#171913",
        "xtick.color": "#666860", "ytick.color": "#666860",
        "svg.fonttype": "none", "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(12, 5.25), facecolor="#f4f2ec")
    ax.set_facecolor("#f4f2ec")
    for cell in PALETTE:
        series = sorted((r for r in rows if r["model"] == cell), key=lambda r: int(r["iteration"]))
        xs = [int(r["iteration"]) for r in series]
        ys = [float(r["normalized_tracking_return"]) for r in series]
        assert xs == list(range(2000, 32001, 2000))
        ax.plot(xs, ys, color=PALETTE[cell], lw=2.8, marker="o", ms=3.8,
                label=LABELS[cell], zorder=3)
    ax.set_xlim(1500, 32500)
    ax.set_ylim(1.2, 1.92)
    ax.set_xticks([2000, 8000, 16000, 24000, 32000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1000)}k"))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.grid(axis="y", color="#d8d7cd", lw=.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for edge in ["left", "bottom"]:
        ax.spines[edge].set_color("#babcb1")
    ax.set_xlabel("Training iteration", labelpad=12)
    ax.set_ylabel("Benchmark tracking return ↑", labelpad=12)
    ax.set_title("One evaluation protocol across all 64 checkpoints", loc="left", pad=55, fontsize=17)
    ax.legend(loc="lower left", bbox_to_anchor=(-.005, 1.025), ncols=4,
              frameon=False, fontsize=11, columnspacing=2.0, handlelength=2.4)
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.76)
    out = ROOT / "docs/assets"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "learning-curves.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
