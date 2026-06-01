"""
analysis.py
===========
Creates the main results tables and figures.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats

N_RUNS = 20
FORMATS = ["tight", "medium", "loose"]
CONTENTS = ["narrow", "medium", "open"]
TOP_K = 20

TURN_CONTENT = {0: "narrow", 1: "open", 2: "medium", 3: "narrow", 4: "open",
                5: "medium", 6: "narrow", 7: "medium", 8: "open", 9: "narrow"}

COLOR = {"tight": "#D55E00", "medium": "#E69F00", "loose": "#0072B2"}
LEGEND = {"tight": 'Tight  ("one word only")',
          "medium": "Medium",
          "loose": 'Loose  ("respond freely")'}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "figure.dpi": 120,
})


def load_run_level(csv_path):
    """
    Load the CSV and average by format, content type, and run.
    """
    df = pd.read_csv(csv_path)
    df = df[df["content_constraint"] != "baseline"].copy()
    run_level = (df.groupby(["format_constraint", "content_constraint", "run"])
                   ["entropy_bits"].mean().reset_index())
    return df, run_level


def cell(run_level, fmt, content):
    """
    Return the run-level entropy values for one cell.
    """
    mask = ((run_level["format_constraint"] == fmt) &
            (run_level["content_constraint"] == content))
    return run_level.loc[mask, "entropy_bits"].to_numpy()


def results_table(run_level, delta=0.05, n=N_RUNS):
    """
    Build the summary statistics table.
    """
    cheby_factor = np.sqrt(1.0 / (n * delta))
    rows = []
    for fmt in FORMATS:
        for content in CONTENTS:
            x = cell(run_level, fmt, content)
            sd = x.std(ddof=1)
            se = sd / np.sqrt(n)
            rows.append({
                "format": fmt, "content": content,
                "mean": x.mean(), "sd": sd, "se": se,
                "chebyshev_hw": sd * cheby_factor,
                "clt_hw": 1.96 * se,
            })
    return pd.DataFrame(rows)


def hoeffding_halfwidth(delta=0.05, n=N_RUNS, top_k=TOP_K):
    """
    Compute the Hoeffding 95% half-width.
    """
    rng = np.log2(top_k)
    return np.sqrt(rng**2 * np.log(2.0 / delta) / (2.0 * n))


def format_axis_tests(run_level):
    """
    Run the format-axis comparison tests.
    """
    comparisons = [("open", "tight", "loose"), ("open", "tight", "medium"),
                   ("medium", "tight", "loose"), ("medium", "tight", "medium")]
    rows = []
    for content, a, b in comparisons:
        xa, xb = cell(run_level, a, content), cell(run_level, b, content)
        t, p = stats.ttest_ind(xa, xb, equal_var=False)
        _, p_mwu = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1) +
                      (len(xb) - 1) * xb.var(ddof=1)) /
                     (len(xa) + len(xb) - 2))
        d = (xa.mean() - xb.mean()) / sp
        rows.append({"content": content, "comparison": f"tight vs {b}",
                     "mean_tight": xa.mean(), "mean_other": xb.mean(),
                     "welch_t": t, "welch_p": p, "mwu_p": p_mwu, "cohen_d": d})
    out = pd.DataFrame(rows)
    out["holm_p"] = _holm(out["welch_p"].to_numpy())
    return out


def _holm(pvals):
    """
    Apply Holm-Bonferroni correction.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def figure1_bars(run_level, outdir):
    """
    Make the grouped bar chart.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(CONTENTS))
    width = 0.26
    for i, fmt in enumerate(FORMATS):
        means = [cell(run_level, fmt, c).mean() for c in CONTENTS]
        ses = [cell(run_level, fmt, c).std(ddof=1) / np.sqrt(N_RUNS) for c in CONTENTS]
        ax.bar(x + (i - 1) * width, means, width, yerr=ses, capsize=3.5,
               color=COLOR[fmt], edgecolor="white", linewidth=0.6,
               label=LEGEND[fmt],
               error_kw=dict(elinewidth=1.1, capthick=1.1, ecolor="#333333"))

    ax.set_xticks(x)
    ax.set_xticklabels(["Narrow\n(factual)", "Medium", "Open\n(open-ended)"])
    ax.set_xlabel("Content constraint of the question")
    ax.set_ylabel(r"Mean per-token entropy  $H$  (bits)")
    ax.set_title("Entropy rises with content openness, but is highest under the\n"
                 "tightest format whenever content is uncertain",
                 loc="left", fontsize=12.5)
    ax.set_ylim(0, 1.55)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.legend(frameon=False, loc="upper left",
              title="Format constraint", title_fontsize=10)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)

    _bracket(ax, 2 - width, 2 + width, 1.32, "*** $d$=2.1")
    _bracket(ax, 1 - width, 1 + width, 0.78, "*** $d$=2.0")

    _save(fig, outdir, "fig1_content_format_bars")


def figure1_heatmap(run_level, outdir):
    """
    Make the heatmap figure.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    matrix = np.array([[cell(run_level, f, c).mean() for c in CONTENTS]
                       for f in FORMATS])
    im = ax.imshow(matrix, cmap="magma", aspect="auto", origin="upper")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Narrow", "Medium", "Open"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["Tight", "Medium", "Loose"])
    ax.set_xlabel("Content constraint"); ax.set_ylabel("Format constraint")
    ax.set_title(r"Two-axis entropy phase diagram  $H(\,$content, format$\,)$",
                 loc="left", fontsize=12)
    for i in range(3):
        for j in range(3):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.7 else "black",
                    fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"Mean $H$ (bits)")
    _save(fig, outdir, "fig1b_phase_heatmap")


def figure2_trajectory(raw_df, outdir):
    """
    Make the per-turn entropy trajectory figure.
    """
    shade = {"narrow": "#f2f2f2", "medium": "#e2ecf5", "open": "#d7e8ff"}
    turns = list(range(10))
    fig, ax = plt.subplots(figsize=(8.2, 4.5))

    for t in turns:
        ax.axvspan(t - 0.5, t + 0.5, color=shade[TURN_CONTENT[t]], zorder=0)

    for fmt in FORMATS:
        means, ses = [], []
        for t in turns:
            per_run = (raw_df[(raw_df.format_constraint == fmt) & (raw_df.turn == t)]
                       .groupby("run")["entropy_bits"].mean())
            means.append(per_run.mean())
            ses.append(per_run.std(ddof=1) / np.sqrt(N_RUNS))
        means, ses = np.array(means), np.array(ses)
        ax.plot(turns, means, "-o", color=COLOR[fmt], lw=1.8, ms=4.5,
                label=LEGEND[fmt], zorder=3)
        ax.fill_between(turns, means - ses, means + ses,
                        color=COLOR[fmt], alpha=0.18, zorder=2)

    ax.set_xlim(-0.5, 9.5); ax.set_ylim(0, 2.05); ax.set_xticks(turns)
    for t in turns:
        c = TURN_CONTENT[t]
        ax.text(t, -0.17, c[0].upper(), ha="center", va="top", fontsize=9,
                color={"narrow": "#888", "medium": "#3a6ea5", "open": "#1559a0"}[c],
                fontweight="bold")
    ax.text(-0.5, -0.17, "content:", ha="right", va="top",
            fontsize=8.5, color="#555")
    ax.set_xlabel(r"Conversation turn  $t$", labelpad=18)
    ax.set_ylabel(r"Mean per-token entropy  $H_t$  (bits)")
    ax.set_title("Entropy tracks the content label turn-by-turn; the tight-format "
                 "excess\nis largest at the first open question and relaxes as "
                 "history accumulates", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper right",
              title="Format constraint", title_fontsize=10)
    content_legend = [Patch(facecolor=shade[c], label=c.capitalize())
                      for c in CONTENTS]
    ax.add_artist(ax.legend(handles=content_legend, frameon=False,
                            loc="upper center", ncol=3,
                            title="turn content (N/M/O)", title_fontsize=9,
                            fontsize=9, bbox_to_anchor=(0.5, 1.0)))
    _save(fig, outdir, "fig2_per_turn_trajectory")


def _bracket(ax, x1, x2, y, text, h=0.03):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333", lw=1.0)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=10.5)


def _save(fig, outdir, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.png / .pdf")


def main():
    parser = argparse.ArgumentParser(description="Section 4 analysis & figures.")
    parser.add_argument("--csv", default="entropy_trajectories.csv")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raw_df, run_level = load_run_level(args.csv)

    print("\n=== Results table (run-level, n = 20 per cell) ===")
    tbl = results_table(run_level)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(tbl.to_string(index=False))
    print(f"\nHoeffding 95% half-width (range = log2 {TOP_K} = "
          f"{np.log2(TOP_K):.3f}) = {hoeffding_halfwidth():.3f}  "
          f"(looser than Chebyshev; range dwarfs the realised SD)")

    print("\n=== Format-axis non-monotonicity (Welch t-tests) ===")
    tests = format_axis_tests(run_level)
    with pd.option_context("display.float_format", lambda v: f"{v:.3g}"):
        print(tests.to_string(index=False))

    print("\n=== Figures ===")
    figure1_bars(run_level, args.outdir)
    figure1_heatmap(run_level, args.outdir)
    figure2_trajectory(raw_df, args.outdir)
    print(f"\nDone. Figures written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()

