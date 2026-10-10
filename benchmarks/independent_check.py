"""Independent standard-library certificate audit by rational variable elimination.

This file deliberately imports no triangle_local code and uses a different
contraction order: eliminate gamma before contracting alpha, beta and C.
"""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import re
import gzip
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
import platform
import resource
import time


def F(value, denominator=None):
    value = value.replace("_", "") if isinstance(value, str) else value
    return Fraction(value) if denominator is None else Fraction(value, denominator)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def canonical(doc):
    def ungroup(value):
        if isinstance(value, dict):
            return {k: ungroup(v) for k,v in value.items()}
        if isinstance(value, list):
            return [ungroup(v) for v in value]
        if isinstance(value, str) and re.fullmatch(r"-?[0-9]+(?:_[0-9]+)*(?:/[0-9]+(?:_[0-9]+)*)?", value):
            return value.replace("_", "")
        return value
    return sha256(json.dumps(ungroup(doc), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()).hexdigest()


def simplex(values):
    assert all(not isinstance(x, bool) for x in values)
    row = tuple(F(x) for x in values)
    assert row and min(row) >= 0 and sum(row) == 1
    return row


def audit(path):
    doc = read(path)
    assert doc["schema"] == "triangle-local-bundle/v1"
    model, saved = doc["model"], doc["certificate"]
    assert model["schema"] == "triangle-local-model/v1"
    wa, wb, wg = [simplex(model["weights"][s]) for s in ("alpha", "beta", "gamma")]
    A, B, C = [tuple(tuple(simplex(row) for row in slab)
                     for slab in model["responses"][s]) for s in ("A", "B", "C")]
    na, nb, ng = len(wa), len(wb), len(wg)
    shape = tuple(len(r[0][0]) for r in (A, B, C))
    assert len(A) == nb and all(len(s) == ng for s in A)
    assert len(B) == ng and all(len(s) == na for s in B)
    assert len(C) == na and all(len(s) == nb for s in C)
    assert all(len(row) == m for r, m in zip((A, B, C), shape) for slab in r for row in slab)
    axes = tuple(product(*(range(m) for m in shape)))
    p = dict.fromkeys(axes, F(0))
    for i, j in product(range(na), range(nb)):
        if not wa[i] or not wb[j]:
            continue
        ab = [[F(0) for _ in range(shape[1])] for _ in range(shape[0])]
        for k in range(ng):
            if not wg[k]:
                continue
            for a, xa in enumerate(A[j][k]):
                if not xa:
                    continue
                weight = wg[k] * xa
                for b, xb in enumerate(B[k][i]):
                    if xb:
                        ab[a][b] += weight * xb
        weight = wa[i] * wb[j]
        for a, b in product(range(shape[0]), range(shape[1])):
            if not ab[a][b]:
                continue
            mass = weight * ab[a][b]
            for c, xc in enumerate(C[i][j]):
                if xc:
                    p[a, b, c] += mass * xc
    assert sum(p.values()) == 1
    target = simplex(doc["target"]["probabilities"])
    assert tuple(doc["target"]["outputs"]) == shape and len(target) == len(axes)
    dif = tuple(p[abc] - t for abc, t in zip(axes, target))
    l1, l2, linf = sum(map(abs, dif)), sum(x*x for x in dif), max(map(abs, dif))
    symmetric = len(set(shape)) == 1 and all(
        p[x] == p[tuple(x[i] for i in perm)] for x in axes for perm in permutations(range(3)))
    for key, expected in (("L1", l1), ("L2_squared", l2), ("Linf", linf), ("total_variation", l1/2)):
        assert F(saved[key]) == expected, (path.name, key)
    assert tuple(map(F, saved["model_probabilities"])) == tuple(p.values())
    assert saved["outputs"] == list(shape)
    assert saved["sources_independent"] is True and saved["normalized_exactly"] is True
    assert saved["model_S3_symmetric_exactly"] == symmetric
    tolerance = F(saved["L1_tolerance"])
    assert tolerance >= 0
    assert saved["exact_match"] == (l1 == 0)
    assert saved["within_L1_tolerance"] == (l1 <= tolerance)
    assert saved["status"] == ("exact_match" if l1 == 0 else "nearby_local_model" if l1 <= tolerance else "approximate_fit")
    assert saved["model_sha256"] == canonical(model)
    assert saved["target_sha256"] == canonical({"outputs": list(shape), "probabilities": [str(x) for x in target]})
    if "elegant-source" in path.name or "elegant-detector" in path.name:
        # Compare to the paper's polynomial, independently of package generation.
        import re
        match = re.search(r"elegant-(source|detector)-(\d+)(?:-(\d+))?-", path.name)
        kind, n, d = match.groups()
        v = F(int(n), int(d or 1))
        if kind == "source":
            entries = ((4+9*v+9*v*v+3*v**3)/256, (4+v-3*v*v-v**3)/256,
                       (4-3*v+3*v*v+v**3)/256)
        else:
            entries = ((4+9*v*v+12*v**3)/256, (4+v*v-4*v**3)/256,
                       (4-3*v*v+4*v**3)/256)
        assert target == tuple(entries[len(set(x))-1] for x in axes)
    return dict(file=path.name, source_sizes=[na, nb, ng],
                Euclidean=format(float(l2)**0.5, ".9g"), L1=format(float(l1), ".9g"),
                status=saved["status"], all_probabilities_and_distances_match=True)


def witness():
    symbols = list(permutations(range(4), 2))
    def response(x, y):
        return x[1] if x[1] in y else x[0]
    counts = Counter((response(b, c), response(c, a), response(a, b))
                     for a, b, c in product(symbols, repeat=3))
    assert all(counts[x] == {1: 123, 2: 19, 3: 23}[len(set(x))]
               for x in product(range(4), repeat=3))
    assert sum(counts[k, k, k] for k in range(4)) == 492
    return "41/144"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    results = []
    for path in sorted((args.repo / "benchmarks" / "models").glob("*.json.gz")):
        result = audit(path)
        results.append(result)
        print(path.name, result["Euclidean"], flush=True)
    assert results, "No selected bundles found"
    # Bind every displayed target summary to its selected exact certificate.
    with (args.repo / "benchmarks" / "best-fits.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    for row in summaries:
        doc = read(args.repo / "benchmarks" / row["model_file"])
        cert = doc["certificate"]
        assert row["target_sha256"].replace(":", "") == cert["target_sha256"]
        for key, exact in (("L1_rational", F(cert["L1"])),
                           ("L2_rational", float(F(cert["L2_squared"]))**0.5)):
            displayed, actual = float(row[key]), float(exact)
            assert abs(displayed-actual) <= max(abs(actual)*1e-9, 1e-25), (row["family"], key)
        assert (row["exact_match"] == "True") == cert["exact_match"]
        assert (row["S3_exact"] == "True") == cert["model_S3_symmetric_exactly"]
        assert row["sizes"] == "x".join(str(len(doc["model"]["weights"][name]))
                                         for name in ("alpha", "beta", "gamma"))
    report = dict(independent_method="Fraction variable elimination; no package imports",
                  checked_bundles=len(results), checked_target_summaries=len(summaries),
                  exact_matches=sum(r["status"] == "exact_match" for r in results),
                  witness_all_equal=witness(), seconds=round(time.perf_counter()-start, 3),
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1 if platform.system() == "Darwin" else 1024), results=results)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2)+"\n")
    print("All independent checks passed", flush=True)


if __name__ == "__main__":
    main()
