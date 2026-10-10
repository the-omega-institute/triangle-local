#!/usr/bin/env python3
"""Summarize light external trials, retaining missing-budget distinctions."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    runs = []
    for filename in ("localmodels-results.json", "lhv-net-2020-results.json"):
        d = json.loads((ROOT / "benchmarks/external" / filename).read_text())
        for r in d["results"]:
            r = dict(r)
            r["source_file"] = filename
            runs.append(r)
    summaries = []
    keys = sorted({(r["solver"], r["family"], r["parameter"]) for r in runs})
    for solver, family, parameter in keys:
        selected = [r for r in runs if (r["solver"], r["family"], r["parameter"]) == (solver, family, parameter)]
        for budget in (10, 60, 300):
            vals = []
            actual = []
            for r in selected:
                c = r.get("checkpoints", {}).get(str(budget))
                if c:
                    vals.append(c["l2"])
                    actual.append(c["elapsed_seconds"])
                elif budget <= r["elapsed_seconds"] and r["status"] == "completed":
                    # A completed optimizer before the next budget is a valid
                    # retained result.  It costs no further compute.
                    vals.append(r["best_l2"])
                    actual.append(r["elapsed_seconds"])
                elif r["status"] == "completed" and budget in (60,):
                    vals.append(r["best_l2"])
                    actual.append(r["elapsed_seconds"])
            summaries.append({"solver": solver, "family": family, "parameter": parameter,
                              "budget_seconds": budget, "n": len(vals),
                              "median_l2": float(np.median(vals)) if vals else None,
                              "min_l2": float(min(vals)) if vals else None,
                              "max_l2": float(max(vals)) if vals else None,
                              "actual_checkpoint_seconds": actual})
    (ROOT / "benchmarks/external/summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Summarized {len(runs)} trials into {len(summaries)} budget rows")


if __name__ == "__main__":
    main()
