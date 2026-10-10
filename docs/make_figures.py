#!/usr/bin/env python3
"""Regenerate the showcase from committed CSVs and rational model bundles.

No optimizer, target generator, network access, or intermediate runs are used.
CSV decimals are display values; each selected row is checked against its
certificate's exact rational distances and hashes before plotting.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"  # Okabe–Ito source colour
ORANGE = "#D55E00"  # Okabe–Ito detector colour
GREY = "#7B8792"
INK = "#182B3A"
TRANSITIONS = {
    "renou-source": 0.89,
    "renou-detector": 0.91,
    "elegant-source": 0.80,
    "elegant-detector": 0.86,
}


def rational(value):
    return Fraction(value.replace("_", "") if isinstance(value, str) else value)


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, str) and re.fullmatch(r"-?[\d_]+(?:/[\d_]+)?", value):
        return value.replace("_", "")
    return value


def digest(value):
    text = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return sha256(text.encode()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select(rows, family, parameter=None):
    found = [r for r in rows if r["family"] == family and
             (parameter is None or rational(r["parameter"]) == rational(parameter))]
    if parameter is not None:
        if len(found) != 1:
            raise ValueError(f"Expected one saved row: {family}, {parameter}")
        return found[0]
    return sorted(found, key=lambda r: rational(r["parameter"]))


def check_selected(rows):
    """Bind plotted distances to certificates; the independent audit checks models."""
    bundles = {}
    for row in rows:
        path = ROOT / "benchmarks" / row["model_file"]
        if path not in bundles:
            with gzip.open(path, "rt") as stream:
                bundles[path] = json.load(stream)
        doc = bundles[path]
        cert = doc["certificate"]
        if digest(doc["model"]) != cert["model_sha256"]:
            raise ValueError(f"Model hash mismatch: {path.name}")
        target = {"outputs": doc["target"]["outputs"],
                  "probabilities": [str(rational(p)) for p in doc["target"]["probabilities"]]}
        if digest(target) != cert["target_sha256"]:
            raise ValueError(f"Target hash mismatch: {path.name}")
        if row["target_sha256"].replace(":", "") != cert["target_sha256"]:
            raise ValueError(f"CSV target hash mismatch: {path.name}")
        errors = [rational(p) - rational(t) for p, t in
                  zip(cert["model_probabilities"], target["probabilities"])]
        if sum(abs(e) for e in errors) != rational(cert["L1"]):
            raise ValueError(f"Certificate L1 mismatch: {path.name}")
        if sum(e * e for e in errors) != rational(cert["L2_squared"]):
            raise ValueError(f"Certificate squared L2 mismatch: {path.name}")
        distance = float(rational(cert["L2_squared"])) ** 0.5
        if not np.isclose(float(row["L2_rational"]), distance, rtol=1e-9, atol=1e-25):
            raise ValueError(f"CSV distance mismatch: {path.name}")
        if (row["exact_match"] == "True") != cert["exact_match"]:
            raise ValueError(f"CSV exact-match mismatch: {path.name}")
    return bundles


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.titleweight": "semibold", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9CA6AE", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "grid.color": "#D7DEE3", "grid.alpha": 0.6,
        "svg.fonttype": "none", "svg.hashsalt": "triangle-local-showcase-v1",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)  # Only one figure remains in memory at a time.


def distances(rows, family):
    selected = select(rows, family)
    return (np.array([float(rational(r["parameter"])) for r in selected]),
            np.array([float(r["L2_rational"]) for r in selected]),
            np.array([float(r["published_reference_L2"]) for r in selected]))


def noise_panel(ax, rows, prefix, log=False):
    for suffix, colour, marker in (("source", BLUE, "o"), ("detector", ORANGE, "s")):
        family = f"{prefix}-{suffix}"
        x, y, reference = distances(rows, family)
        # Zeros are omitted on log axes, never moved to an artificial floor.
        ax.plot(x, np.where(y > 0, y, np.nan) if log else y, color=colour,
                marker=marker, markersize=5, linewidth=1.9, zorder=3)
        ax.plot(x, np.where(reference > 0, reference, np.nan) if log else reference,
                color=colour, linestyle="--", linewidth=1.7, alpha=0.8)
        threshold = TRANSITIONS[family]
        ax.axvline(threshold, color=colour, linestyle=":", linewidth=1.3, alpha=0.8)
        if not log:
            ax.text(threshold, 0.98, f" {threshold:.2f}", color=colour, fontsize=10,
                    transform=ax.get_xaxis_transform(), ha="left", va="top")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y")
    ax.set_xlabel(r"Source visibility / detector efficiency $v$")
    ax.set_ylabel("Distance (log scale)" if log else "Euclidean distance")


def headline(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.7))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.805, bottom=0.14,
                        wspace=0.27, hspace=0.54)
    fig.text(0.5, 0.967, "Certified near-local models beyond the 2020 estimated noise transitions",
             ha="center", fontsize=17, weight="semibold")
    fig.text(0.5, 0.925,
             "Explicit finite models with independent sources; exact rational model-to-target distances",
             ha="center", fontsize=11.5)
    handles = [
        Line2D([], [], color=BLUE, marker="o", label="Source noise · certified model"),
        Line2D([], [], color=ORANGE, marker="s", label="Detector noise · certified model"),
        Line2D([], [], color=GREY, linestyle="--", label="2020 Eq. (4) estimate"),
        Line2D([], [], color=GREY, linestyle=":", label="2020 estimated transition"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.878),
               ncol=2, frameon=False, columnspacing=3.5, fontsize=10.5)
    for col, prefix in enumerate(("renou", "elegant")):
        noise_panel(axes[0, col], rows, prefix)
        noise_panel(axes[1, col], rows, prefix, log=True)
    axes[0, 0].set(title=r"A   Renou · $u^2=0.85$", xlim=(0.79, 1.005), ylim=(-0.00018, 0.0067))
    axes[0, 1].set(title="B   Elegant joint measurement (EJM)", xlim=(0.74, 1.005), ylim=(-0.0016, 0.058))
    axes[1, 0].set(title="C   Renou · resolved residuals", xlim=(0.875, 0.958), ylim=(1e-9, 1e-2))
    axes[1, 1].set(title="D   EJM · resolved residuals", xlim=(0.775, 0.905), ylim=(1e-11, 4e-2))
    axes[1, 0].set_xticks([0.89, 0.91, 0.93, 0.95])
    axes[1, 1].set_xticks([0.78, 0.80, 0.84, 0.86, 0.90])
    ren93 = float(select(rows, "renou-detector", "93/100")["L2_rational"])
    ren95 = float(select(rows, "renou-source", "19/20")["L2_rational"])
    axes[1, 0].annotate(r"Detector: $8.21\times10^{-9}$ at $v=0.93$",
                        xy=(0.93, ren93), xytext=(0.884, 1.2e-6), fontsize=10,
                        color=ORANGE, arrowprops={"arrowstyle": "-", "color": ORANGE})
    axes[1, 0].annotate(r"Source: $3.04\times10^{-5}$ at $v=0.95$",
                        xy=(0.95, ren95), xytext=(0.884, 1.5e-4), fontsize=10,
                        color=BLUE, arrowprops={"arrowstyle": "-", "color": BLUE})
    axes[1, 1].text(0.055, 0.47,
                    "At estimated transitions:\nsource: $5.77\\times10^{-11}$\ndetector: $1.51\\times10^{-9}$",
                    transform=axes[1, 1].transAxes, fontsize=10.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92})
    fig.text(0.5, 0.061,
             "Certified distances are upper bounds on distance to the local set; small residuals do not prove exact locality.",
             ha="center", fontsize=10)
    fig.text(0.5, 0.03,
             "Eq. (4) values are approximate analytic estimates, not digitized data. Lines guide the eye; log panels omit zeros.",
             ha="center", fontsize=10)
    save(fig, out, "headline")


def local_model(rows, bundles, out):
    row = select(rows, "renou-detector", "93/100")
    doc = bundles[ROOT / "benchmarks" / row["model_file"]]
    model = doc["model"]
    weights = {s: np.array([float(rational(w)) for w in model["weights"][s]])
               for s in ("alpha", "beta", "gamma")}
    for ws in model["weights"].values():
        if min(map(rational, ws)) < 0 or sum(map(rational, ws)) != 1:
            raise ValueError("Invalid source simplex")
    for table in model["responses"].values():
        for slab in table:
            for probabilities in slab:
                if min(map(rational, probabilities)) < 0 or sum(map(rational, probabilities)) != 1:
                    raise ValueError("Invalid response simplex")
    fig = plt.figure(figsize=(13.8, 11.9))
    fig.text(0.5, 0.974, "Inside a certified local model", ha="center", fontsize=20, weight="semibold")
    fig.text(0.5, 0.942,
             r"Renou detector noise · $u^2=0.85$, $v=0.93$ · certified distance $8.207624541\times10^{-9}$",
             ha="center", fontsize=12)
    fig.text(0.5, 0.912,
             "Three independent source laws and all four stochastic output probabilities for each party",
             ha="center", fontsize=11)
    source_grid = fig.add_gridspec(1, 3, left=0.072, right=0.895, top=0.862, bottom=0.748, wspace=0.3)
    for col, (name, letter) in enumerate((("alpha", r"\alpha"), ("beta", r"\beta"), ("gamma", r"\gamma"))):
        ax = fig.add_subplot(source_grid[0, col])
        ws = weights[name]
        ax.bar(np.arange(len(ws)), ws, color=BLUE, width=0.82)
        ax.set_title(rf"$q_{letter}$ · {np.count_nonzero(ws)} active / {len(ws)} symbols", fontsize=12)
        ax.set(xlim=(-0.6, len(ws)-0.4), ylim=(0, 0.56), xticks=[0, 3, 6, 9, 11],
               yticks=[0, 0.25, 0.5], xlabel="Hidden-variable symbol", ylabel="Probability")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    grid = fig.add_gridspec(3, 4, left=0.072, right=0.895, top=0.678, bottom=0.098,
                           wspace=0.3, hspace=0.63)
    cmap = matplotlib.colormaps["cividis"].copy()
    cmap.set_bad("#E8EDF0")
    specs = (("A", "a", "beta", "gamma", r"\beta", r"\gamma"),
             ("B", "b", "gamma", "alpha", r"\gamma", r"\alpha"),
             ("C", "c", "alpha", "beta", r"\alpha", r"\beta"))
    for r, (party, output, wy, wx, yletter, xletter) in enumerate(specs):
        tensor = np.array([[[float(rational(p)) for p in ps] for ps in slab]
                           for slab in model["responses"][party]])
        inactive = (weights[wy][:, None] == 0) | (weights[wx][None, :] == 0)
        for c in range(4):
            ax = fig.add_subplot(grid[r, c])
            im = ax.imshow(np.ma.array(tensor[:, :, c], mask=inactive), vmin=0, vmax=1,
                           cmap=cmap, origin="lower", interpolation="nearest", aspect="equal")
            ax.set_title(rf"$P_{party}({output}={c}\mid {yletter},{xletter})$", fontsize=11)
            ax.set(xticks=[0, 3, 6, 9, 11], yticks=[0, 3, 6, 9, 11],
                   xlabel=rf"${xletter}$ symbol")
            if c == 0:
                ax.set_ylabel(rf"${yletter}$ symbol")
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9, length=2)
    cax = fig.add_axes((0.923, 0.2, 0.014, 0.39))
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("Response probability", labelpad=10)
    fig.text(0.5, 0.051,
             r"Source law: $q_\alpha q_\beta q_\gamma$. Pale grey cells have a zero-weight incident symbol and never contribute.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.025,
             "Original symbol order retained. Colours display floating values of the stored rational tables; no responses are rounded to argmax.",
             ha="center", fontsize=10)
    save(fig, out, "local-model")


def label_distance(ax, value, y, colour=INK):
    ax.text(value * 1.4, y, f"{value:.3g}", fontsize=10, color=colour, va="center")


def improvement(rows, old, ablation, out):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.5))
    fig.subplots_adjust(left=0.107, right=0.965, top=0.76, bottom=0.19, wspace=0.73)
    fig.text(0.5, 0.948, "What improved the finite-model search?", ha="center", fontsize=19, weight="semibold")
    fig.text(0.5, 0.894, "Recorded distances at the 2020 estimated transitions and a selected component comparison",
             ha="center", fontsize=11.5)
    ax = axes[0]
    controls = (("elegant-source", "4/5", "EJM source\nv = 0.80"),
                ("elegant-detector", "43/50", "EJM detector\nv = 0.86"),
                ("renou-source", "89/100", "Renou source\nv = 0.89"),
                ("renou-detector", "91/100", "Renou detector\nv = 0.91"))
    floor = 1e-11
    for i, (family, parameter, _) in enumerate(controls):
        for data, offset, colour in ((old, -0.17, GREY), (rows, 0.17, BLUE)):
            value = float(select(data, family, parameter)["L2_rational"])
            ax.barh(i + offset, value - floor, left=floor, height=0.28, color=colour)
            label_distance(ax, value, i + offset, colour)
    ax.set_yticks(range(4), [label for _, _, label in controls], fontsize=10)
    ax.set_ylim(3.55, -0.7)
    ax.set_title("A   Round 1 → round 2", loc="left")
    ax.legend(handles=[Line2D([], [], color=GREY, linewidth=7, label="Round 1"),
                       Line2D([], [], color=BLUE, linewidth=7, label="Round 2")],
              loc="lower left", bbox_to_anchor=(0, 1.07), ncol=2,
              frameon=False, fontsize=10, borderaxespad=0)
    ax = axes[1]
    selected = (("blind-lbfgs-none-12", "Blind L-BFGS · 12³"),
                ("blind-lbfgs-none-24", "Blind L-BFGS · 24³"),
                ("seeded-lbfgs-12", "Seeded L-BFGS · 12³"),
                ("polish-fixed-weights", "+ polish, fixed weights"),
                ("polish-free-weights", "+ polish, free weights"),
                ("seeded-hybrid-hops-12", "Seeded hybrid + hops · 12³"))
    for i, (configuration, _) in enumerate(selected):
        row = next(r for r in ablation if r["family"] == "elegant-source" and
                   r["configuration"] == configuration)
        value = float(row["L2_rational"])
        colour = BLUE if i == len(selected) - 1 else GREY
        ax.barh(i, value - floor, left=floor, height=0.52, color=colour)
        label_distance(ax, value, i, colour)
    ax.set_yticks(range(len(selected)), [label for _, label in selected], fontsize=10)
    ax.set_ylim(len(selected) - 0.45, -0.7)
    ax.set_title("B   EJM source · v = 0.80", loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(floor, 0.3)
        ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
        ax.set_xlabel("Certified Euclidean distance (log scale)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.text(0.5, 0.092,
             "Different alphabet sizes and restart/iteration caps: descriptive comparisons, not equal-budget statistical estimates.",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.05,
             "Both polishing variants start from the same seeded L-BFGS candidate. Bar origins are 10⁻¹¹; residuals do not prove nonlocality.",
             ha="center", fontsize=10)
    save(fig, out, "search-improvement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument("--headline-copy", type=Path, help="Also copy the headline PNG to this path")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(ROOT / "benchmarks" / "best-fits.csv")
    old = read_csv(ROOT / "benchmarks" / "round1" / "best-fits.csv")
    ablation = read_csv(ROOT / "benchmarks" / "ablation.csv")
    controls = read_csv(ROOT / "benchmarks" / "controls.csv")
    bundles = check_selected(rows + controls)
    style()
    headline(rows, args.out_dir)
    local_model(rows, bundles, args.out_dir)
    improvement(rows, old, ablation, args.out_dir)
    if args.headline_copy:
        args.headline_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.out_dir / "headline.png", args.headline_copy)
    print(f"Checked {len(rows)} target summaries and {len(controls)} control rows against {len(bundles)} selected bundles.")
    print(f"Wrote three PNG/SVG figure pairs to {args.out_dir}.")


if __name__ == "__main__":
    main()
