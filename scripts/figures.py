"""Paper figures, drawn only from the `summary.json` files written by `cascade report` (no model calls).

Usage: uv run --extra figures python scripts/figures.py
Writes paper/figures/fig*.pdf (vector, for LaTeX) and fig*.png (300 dpi, for Word).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("paper/figures")
MAIN = json.loads(Path("results/main/summary.json").read_text())
C4 = json.loads(Path("results/change4/summary.json").read_text())

NAMES = {
    "qwen/qwen3-14b": "Qwen3 14B",
    "qwen/qwen3-32b": "Qwen3 32B",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "mistralai/mistral-large-2512": "Mistral Large 2512",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
}
MODELS = list(NAMES)
# Okabe-Ito, safe for colour-blind readers and readable in greyscale by marker shape.
BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#999999",
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf and .png")


def err(row, key="value"):
    v = row[key]
    return v, [[v - row["ci_low"]], [row["ci_high"] - v]]


def fig_design():
    """Design: one planted wrong answer in an otherwise gold history; later turns split by the dependency graph."""
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    xs = [0.9, 2.9, 4.9, 6.9, 8.9]
    labels = ["Turn 1", "Turn 2", "Turn 3", "Turn 4", "Turn 5"]
    styles = [
        ("gold answer", "#FFFFFF", "black"),
        ("planted\nwrong answer", "#FBE3D6", VERMILION),
        ("uses turn 2", "#DCEBF7", BLUE),
        ("independent", "#F2F2F2", GREY),
        ("uses turn 2", "#DCEBF7", BLUE),
    ]
    for x, lab, (txt, face, edge) in zip(xs, labels, styles, strict=True):
        ax.add_patch(FancyBboxPatch((x - 0.75, 1.55), 1.5, 0.95, boxstyle="round,pad=0.03", fc=face, ec=edge, lw=1.3))
        ax.text(x, 2.25, lab, ha="center", va="center", fontweight="bold")
        ax.text(x, 1.85, txt, ha="center", va="center", fontsize=7.5)
    # dependency-graph value links (gold programs): 3 and 5 use 2
    for a, b, rad, dx in ((1, 2, -0.45, 0.25), (1, 4, -0.25, -0.25)):
        ax.add_patch(
            FancyArrowPatch(
                (xs[a] + dx, 2.55), (xs[b], 2.55), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=9, color=BLUE, lw=1.1
            )
        )
    ax.text(10.0, 3.65, "dependency graph: value links\nfrom the gold programs", ha="right", va="top", color=BLUE, fontsize=7.5)
    ax.text(
        0.15,
        1.05,
        "Each later turn is asked on its own, with gold answers for every earlier turn except turn 2. Extra error = "
        "error with\nthis history minus error with the gold history, for dependent and independent turns (vs. a "
        "run-to-run noise floor).",
        fontsize=7.5,
        va="center",
    )
    ax.text(
        0.15,
        0.35,
        "Circuit breakers (self-check, cross-model check, program re-execution) check planted wrong answers "
        "and\nthe model's own natural answers; catch rate on wrong answers, false-alarm rate on right ones.",
        fontsize=7.5,
        va="center",
    )
    save(fig, "fig1_design")


def fig_cascade():
    """Extra error on dependent and independent later turns, per model, with the noise floor."""
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    order = sorted(
        MODELS, key=lambda m: next(r["extra_error"] for r in MAIN["models"][m]["cascade"] if r["type"] == "all" and r["turns"] == "depends")
    )
    for i, m in enumerate(order):
        d = MAIN["models"][m]
        nf = d["noise_floor"]
        ax.add_patch(plt.Rectangle((0, i - 0.32), nf["ci_high"], 0.64, color=GREY, alpha=0.18, lw=0))
        ax.plot([nf["flip_rate"]] * 2, [i - 0.32, i + 0.32], color=GREY, lw=1.2)
        for turns, off, col, mk in (("depends", 0.13, BLUE, "o"), ("independent", -0.13, ORANGE, "s")):
            r = next(r for r in d["cascade"] if r["type"] == "all" and r["turns"] == turns)
            v, e = err(r, "extra_error")
            ax.errorbar(
                v,
                i + off,
                xerr=e,
                fmt=mk,
                color=col,
                ms=5,
                capsize=2,
                lw=1.2,
                label=f"{turns.replace('depends', 'dependent')} turns" if i == 0 else None,
            )
    ax.plot([], [], color=GREY, lw=1.2, label="noise floor (flip rate; band to upper CI)")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(order)), [NAMES[m] for m in order])
    ax.set_xlabel("Extra error after one planted wrong answer (95% CI)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig2_cascade")


def fig_calibration():
    """Transmission model: predicted against observed error by number of dependency paths (held-out half)."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cols = [BLUE, ORANGE, GREEN, VERMILION, PURPLE]
    marks = ["o", "s", "^", "D", "v"]
    for m, c, mk in zip(MODELS, cols, marks, strict=True):
        cal = MAIN["models"][m]["transmission_fit"]["calibration_by_paths"]
        ax.scatter(
            [p["predicted"] for p in cal],
            [p["observed"] for p in cal],
            s=[8 + p["n"] / 25 for p in cal],
            color=c,
            marker=mk,
            alpha=0.75,
            edgecolor="white",
            lw=0.4,
            label=NAMES[m],
        )
    # Wide enough for every prediction (largest 0.870, DeepSeek V3 at 8 paths); checked below.
    top = max(max(p["predicted"], p["observed"]) for m in MODELS for p in MAIN["models"][m]["transmission_fit"]["calibration_by_paths"])
    lim = [0, 0.1 * (int(top * 10) + 1)]
    ax.plot(lim, lim, color="black", lw=0.6, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted error (path model)")
    ax.set_ylabel("Observed error")
    handles = [plt.Line2D([], [], marker=mk, color=c, ls="", ms=5, label=NAMES[m]) for m, c, mk in zip(MODELS, cols, marks, strict=True)]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.text(lim[1] - 0.01, 0.02, "marker area ~ n turns", ha="right", fontsize=7, color=GREY)
    save(fig, "fig3_calibration")


def fig_mechanisms():
    """Answer signatures of later turns that became wrong after an injection (post-hoc rule, CHANGELOG R50)."""
    sigs = [
        ("planted_input_reused", "planted value used in arithmetic", VERMILION),
        ("equals_planted", "equals planted value", PURPLE),
        ("power_of_ten", "power-of-ten offset", BLUE),
        ("loop", "unfinished reasoning", ORANGE),
        ("no_answer", "no answer", YELLOW),
        ("other", "other", GREY),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    rows, labels = [], []
    for m in MODELS:
        for turns in ("depends", "independent"):
            table = MAIN["posthoc_spread"]["models"][m]["signatures"]
            rows.append(next(r for r in table if r["type"] == "all" and r["turns"] == turns))
            labels.append(f"{NAMES[m]}, {'dependent' if turns == 'depends' else 'independent'}")
    y = list(range(len(rows)))[::-1]
    left = [0.0] * len(rows)
    for key, lab, col in sigs:
        share = [r[key] / r["newly_wrong"] if r["newly_wrong"] else 0 for r in rows]
        ax.barh(y, share, left=left, color=col, edgecolor="white", lw=0.4, label=lab, height=0.75)
        left = [a + b for a, b in zip(left, share, strict=True)]
    for yi, r in zip(y, rows, strict=True):
        ax.text(1.01, yi, f"n = {r['newly_wrong']}", va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of newly wrong later turns")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.2))
    save(fig, "fig4_mechanisms")


def _metric(scores, group):
    return next(s for s in scores if s["group"] == group)


def fig_breakers():
    """Planted against natural catch rate, and false alarms, for every breaker run."""
    rows = []  # (label, planted row, natural row, gold-right row, own-right row, style)
    kinds = (("self", "Self"), ("cross", "Cross (Llama)"), ("program", "Program"))
    for ans, tag in (("qwen32b", "Qwen3 32B"), ("deepseek", "DeepSeek V3")):
        for kind, kname in kinds:
            name = f"{kind}_{ans}" if kind != "cross" else f"cross_{ans}_by_llama"
            src = MAIN["breakers"] if ans == "qwen32b" else MAIN["exploratory_breakers"]
            sc = src[name]
            rows.append((f"{kname}, {tag}", *(_metric(sc, g) for g in ("inject:wrong", "free:wrong", "gold:right", "free:right")), "run"))
            if kind == "program":
                st = {r["subset"]: r["strict"] for r in MAIN["posthoc_breakers"][name]["strict_rescore"]}
                rows.append(
                    (
                        f"{kname}, {tag}, strict (post-hoc)",
                        st["inject:wrong"],
                        st["free:wrong"],
                        st["gold:right"],
                        st["free:right"],
                        "strict",
                    )
                )
    for name, tag in (("program_qwen32b", "Qwen3 32B"), ("program_deepseek", "DeepSeek V3")):
        sc = C4["breakers"][name]
        groups = ("inject:wrong", "free:wrong", "gold:right", "free:right")
        rows.append((f"Program, {tag}, fresh sample", *(_metric(sc, g) for g in groups), "fresh"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 4.0), sharey=True, gridspec_kw={"width_ratios": [1.4, 1]})
    y = list(range(len(rows)))[::-1]
    for yi, (_, planted, natural, gold, own, style) in zip(y, rows, strict=True):
        hollow = style == "strict"
        a1.plot([natural["value"], planted["value"]], [yi, yi], color=GREY, lw=1, zorder=1)
        for r, col, mk in ((planted, VERMILION, "o"), (natural, BLUE, "s")):
            v, e = err(r)
            a1.errorbar(v, yi, xerr=e, fmt=mk, color=col, ms=5, capsize=2, lw=1, mfc="white" if hollow else col, zorder=2)
        for r, col, mk, off in ((gold, PURPLE, "^", 0.15), (own, GREEN, "v", -0.15)):
            v, e = err(r)
            a2.errorbar(v, yi + off, xerr=e, fmt=mk, color=col, ms=4.5, capsize=2, lw=1, mfc="white" if hollow else col)
    for a in (a1, a2):  # separate Qwen3 32B, DeepSeek V3 and the fresh-sample rows
        for cut in (len(rows) - 4.5, len(rows) - 8.5):
            a.axhline(cut, color="#BBBBBB", lw=0.6, ls=":")
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlim(0, 1)
    a1.set_xlabel("Catch rate on wrong answers (95% CI)")
    a2.set_xlim(0, 0.3)
    a2.set_xlabel("False-alarm rate on right answers")
    for a in (a1, a2):
        a.grid(axis="x", color="#DDDDDD", lw=0.5)
    a1.errorbar([], [], fmt="o", color=VERMILION, label="planted errors")
    a1.errorbar([], [], fmt="s", color=BLUE, label="natural errors")
    a1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    a2.errorbar([], [], fmt="^", color=PURPLE, label="gold answers")
    a2.errorbar([], [], fmt="v", color=GREEN, label="model's own right answers")
    a2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    fig.text(
        0.01,
        -0.02,
        "Filled: scoring as run (pre-registered for Qwen3 32B, exploratory for DeepSeek V3 and the fresh sample "
        "of 600 training conversations). Hollow: post-hoc strict rescoring of the stored program values.",
        fontsize=7,
        color="#444444",
    )
    save(fig, "fig5_breakers")


def fig_revised():
    """Revised minus frozen program breaker: paired differences with the pre-registered decision lines."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), sharey=True)
    pairs = C4["change4"]["pairs"]
    post = C4["change4_posthoc"]
    items = []
    for name, p in pairs.items():
        for comp, lab in (("strict", "vs frozen, strict (primary)"), ("lenient", "vs frozen, as run")):
            items.append((f"{NAMES[p['answerer']]}\n{lab}", p["comparisons"][comp], post[name]["diffs"], comp))
    y = list(range(len(items)))[::-1]
    for ax, (out, title) in zip(
        axes, (("recall", "Natural recall difference"), ("false_alarms", "Natural false-alarm difference")), strict=True
    ):
        ph_key = "natural recall" if out == "recall" else "natural false alarms"
        for yi, (_, comps, ph, comp) in zip(y, items, strict=True):
            r = comps[out]
            v = r["diff"]
            ax.errorbar(
                v,
                yi + 0.12,
                xerr=[[v - r["ci_low"]], [r["ci_high"] - v]],
                fmt="o",
                color=BLUE,
                ms=5,
                capsize=2,
                mfc=BLUE if comp == "strict" else "white",
                label="as run" if yi == y[0] else None,
            )
            q = ph[f"{ph_key}|{comp}"]
            ax.errorbar(
                q["diff"],
                yi - 0.12,
                xerr=[[q["diff"] - q["ci_low"]], [q["ci_high"] - q["diff"]]],
                fmt="D",
                color=ORANGE,
                ms=4,
                capsize=2,
                mfc=ORANGE if comp == "strict" else "white",
                label='post-hoc, "-" column accepted' if yi == y[0] else None,
            )
        ax.axvline(0, color="black", lw=0.6)
        if out == "false_alarms":
            ax.axvline(C4["change4"]["margin"], color=VERMILION, lw=0.8, ls="--")
            ax.text(
                C4["change4"]["margin"],
                0.97,
                "+0.02 margin ",
                color=VERMILION,
                fontsize=7,
                ha="right",
                va="top",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_title(title)
        ax.set_xlabel("Revised minus frozen (95% CI)")
    axes[0].set_yticks(y, [i[0] for i in items])
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.32), ncol=2, fontsize=8)
    save(fig, "fig6_revised")


if __name__ == "__main__":
    fig_design()
    fig_cascade()
    fig_calibration()
    fig_mechanisms()
    fig_breakers()
    fig_revised()
