#!/usr/bin/env python3
"""Create and verify the compact Renou detector witness."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from time import perf_counter

from triangle_local.exact import trim_zero_weight_symbols, verify
from triangle_local.exact import fraction
from triangle_local.io import dump_json, from_document, load_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/models/117-renou-detector-93-100-hybrid-reverse.json.gz"
DEST = ROOT / "benchmarks/models/renou-detector-93-100-trimmed-8-9-2.json.gz"


def main():
    original = load_json(SOURCE)
    compact_model = trim_zero_weight_symbols(original["model"])
    target = from_document(original["target"])
    started = perf_counter()
    certificate = verify(compact_model, target, tolerance=original["certificate"]["L1_tolerance"])
    verification_seconds = perf_counter() - started
    if not certificate["exact_match"] and certificate["L1"] != original["certificate"]["L1"]:
        raise SystemExit("Trimming changed the certified distribution")
    bundle = {
        "schema": "triangle-local-bundle/v1",
        "target": original["target"],
        "model": compact_model,
        "certificate": certificate,
        "search": {
            "kind": "zero-weight-trim-and-stable-reorder",
            "source_bundle": SOURCE.name,
            "verification_seconds": verification_seconds,
        },
    }
    dump_json(DEST, bundle)
    size = DEST.stat().st_size
    summary = {
        "source_bundle": str(SOURCE.relative_to(ROOT)),
        "compact_bundle": str(DEST.relative_to(ROOT)),
        "active_cardinalities": [len(compact_model["weights"][name]) for name in ("alpha", "beta", "gamma")],
        "stored_cardinalities": [len(original["model"]["weights"][name]) for name in ("alpha", "beta", "gamma")],
        "L2": str(float(fraction(certificate["L2_squared"])) ** 0.5),
        "L2_squared": certificate["L2_squared"],
        "L1": certificate["L1"],
        "total_variation": certificate["total_variation"],
        "verification_seconds": verification_seconds,
        "compressed_file_bytes": size,
        "permutations": compact_model["trimmed_permutations"],
        "certificate_status": certificate["status"],
    }
    (ROOT / "research/trimmed-witness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
