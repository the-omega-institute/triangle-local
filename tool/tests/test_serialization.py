"""Check that readable serialization preserves certificate arithmetic and identity."""
from fractions import Fraction
import gzip
import json
import re

import numpy as np

from triangle_local.cli import bundle, main
from triangle_local.exact import canonical_hash, verify
from triangle_local.io import dump_csv, dump_json, from_document, load_csv, load_json
from triangle_local.model import Model


def test_grouped_rationals_keep_all_digits_hashes_and_distances(tmp_path):
    target = np.array([[[Fraction(1, 10**15), Fraction(10**15-1, 10**15)]]], object)
    model = Model((np.ones(1),)*3,
                  (np.ones((1,1,1)), np.ones((1,1,1)), np.array([[[0.,1.]]])))
    document = bundle(model, target)
    document["search"] = {"digest": "a" + "1234567890"*3 + "b"*33, "cap": 10**12}
    for suffix in (".json", ".json.gz"):
        path = tmp_path / ("candidate" + suffix)
        dump_json(path, document)
        raw = gzip.decompress(path.read_bytes()) if suffix.endswith("gz") else path.read_bytes()
        assert not re.search(rb"[0-9]{13,}", raw)
        loaded = load_json(path)
        assert loaded == document
        direct = json.loads(raw)
        assert canonical_hash(direct["model"]) == document["certificate"]["model_sha256"]
        assert from_document(direct["target"])[0,0,0] == Fraction(1, 10**15)
        assert verify(direct["model"], from_document(direct["target"])) == document["certificate"]
        assert main(["verify", str(path)]) == 0


def test_csv_digest_grouping_restores_original_identifier(tmp_path):
    digest = "a" + "1234567890"*2 + "b"*43
    path = tmp_path / "measurements.csv"
    value = 0.000_123_456_789_012_345
    dump_csv(path, [{"target_sha256": digest, "L2_rational": value}])
    assert not re.search(r"[0-9]{13,}", path.read_text())
    restored = load_csv(path)[0]
    assert restored["target_sha256"] == digest
    assert abs(float(restored["L2_rational"]) - value) < 1e-13


def test_direct_grouped_model_keeps_exact_simplexes_and_model_hash(tmp_path):
    denominator = 10**15
    model = {"schema": "triangle-local-model/v1",
             "weights": {"alpha": [str(Fraction(1, denominator)),
                                    str(Fraction(denominator-1, denominator))],
                         "beta": ["1"], "gamma": ["1"]},
             "responses": {"A": [[["1"]]], "B": [[["1"], ["1"]]],
                           "C": [[["1"]], [["1"]]]}}
    target = np.array([[[Fraction(1)]]], object)
    certificate = verify(model, target)
    path = tmp_path / "model.json"
    dump_json(path, model)
    raw = json.loads(path.read_text())
    assert not re.search(r"[0-9]{13,}", path.read_text())
    assert canonical_hash(raw) == canonical_hash(model)
    assert verify(raw, target) == certificate
