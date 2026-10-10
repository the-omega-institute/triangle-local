#!/usr/bin/env python3
"""Compute original-quantum-target enclosures for selected certificates.

This script is intentionally a read-only audit: it reads committed bundles and
CSV rows, evaluates the corresponding published formula with interval arithmetic,
and writes a small JSON receipt.  It never changes a model or runs an optimizer.
"""
from __future__ import annotations

import csv
from decimal import Decimal
import gzip
import json
import numpy as np
from pathlib import Path

from triangle_local.bounds import certificate_upper, distance_bound
from triangle_local.io import from_document
from triangle_local.exact import fraction
from triangle_local.bounds import quantum_interval
from triangle_local.families import family

ROOT = Path(__file__).resolve().parents[1]
FEATURED = {"elegant-source", "elegant-detector", "renou-source",
            "renou-detector", "renou-u2"}


def main():
    rows = list(csv.DictReader((ROOT / "benchmarks" / "best-fits.csv").open()))
    records = []
    for row in rows:
        if row["family"] not in FEATURED:
            continue
        path = ROOT / "benchmarks" / row["model_file"]
        with gzip.open(path, "rt") as stream:
            bundle = json.load(stream)
        target = from_document(bundle["target"])
        u_squared = "17/20" if row["family"] in {"renou-source", "renou-detector"} else row["parameter"]
        interval = quantum_interval(row["family"], row["parameter"], u_squared=u_squared)
        # EJM is an exact rational polynomial.  The known Renou controls and
        # fully mixed endpoints are exact too.  The interval evaluator still
        # runs as a consistency check, but endpoint-width noise must not be
        # reported as physics.
        exact_formula = (row["family"].startswith("elegant-") or
                         (row["family"] == "renou-u2" and row["parameter"] in {"1/2", "1"}) or
                         (row["family"] in {"renou-source", "renou-detector"} and row["parameter"] == "0"))
        if exact_formula:
            generator_name = "renou" if row["family"] == "renou-u2" else row["family"]
            identified = family(generator_name, row["parameter"])
            if not np.array_equal(target, identified):
                raise ValueError("Stored target does not equal its claimed exact formula")
        delta = Decimal(0) if exact_formula else distance_bound(interval, target)
        u = certificate_upper(bundle["certificate"]["L2_squared"], delta)
        records.append({
            "family": row["family"],
            "parameter": row["parameter"],
            "u_squared": u_squared if row["family"].startswith("renou") else None,
            "model_file": row["model_file"],
            "stored_target_sha256": bundle["certificate"]["target_sha256"],
            "interval_method": "mpmath.iv outward-rounded interval arithmetic",
            "interval_decimal_digits": 100,
            "endpoint_rounding": "exact binary endpoints; rational squared bounds; integer square-root ceiling",
            "delta_target_L2_upper": str(delta),
            "stored_model_target_L2": row["L2_rational"],
            "U_sqrtE_plus_delta_target_upper": u,
            "target_formula_exact": exact_formula,
        })
    payload = {
        "schema": "triangle-local-target-bounds/v1",
        "description": "Upper enclosures for ||P_quantum-P_stored||_2 and U=sqrt(E)+delta_target.",
        "target_convention": "Renou detector noise is K_v tensor K_v tensor K_v applied independently to the ideal distribution; source noise is the six-qubit Born contraction.",
        "records": records,
    }
    out = ROOT / "benchmarks" / "target-bounds.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
