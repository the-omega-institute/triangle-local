"""Explicit rational JSON and sparse CSV input; never silently normalize."""
import csv
from decimal import Decimal
import json
import gzip
import re
from pathlib import Path

import numpy as np

from .exact import fraction
from .serialization import canonical_document, public_json


def load_csv(path):
    """Read benchmark summaries, restoring colon-grouped SHA256 digests."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{k: v.replace(":", "") if k.endswith("sha256") else v
             for k, v in row.items()} for row in rows]


def dump_csv(path, rows):
    """Write readable benchmark diagnostics; exact distances remain in bundles."""
    if not rows:
        return
    def field(key, value):
        if key.endswith("sha256"):
            text = str(value).replace(":", "")
            return ":".join(text[i:i+8] for i in range(0, len(text), 8))
        text = str(value)
        if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", text):
            text = format(float(text), ".10g")
            if re.search(r"[0-9]{13,}", text):
                text = format(float(text), ".9e")
        return text
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: field(k, v) for k, v in row.items()} for row in rows)


def distribution_document(target, **metadata):
    t = np.asarray(target, dtype=object)
    return {"outputs": list(t.shape),
            "probabilities": [str(fraction(x)) for x in t.ravel()], **metadata}


def from_document(doc, normalize=False):
    if isinstance(doc, dict):
        shape = tuple(doc["outputs"])
        if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError("outputs must contain three positive integers")
        p = np.asarray(doc["probabilities"], dtype=object)
        if p.size != int(np.prod(shape)):
            raise ValueError("Probability count disagrees with outputs")
        p = p.reshape(shape)
    else:
        p = np.asarray(doc, dtype=object)
    if p.ndim != 3 or any(n < 1 for n in p.shape):
        raise ValueError("A nonempty three-dimensional distribution is required")
    vals = [fraction(x) for x in p.ravel()]
    if any(x < 0 for x in vals) or sum(vals) <= 0:
        raise ValueError("Input probabilities must be nonnegative with positive total")
    if normalize:
        total = sum(vals)
        vals = [x / total for x in vals]
    if sum(vals) != 1:
        raise ValueError("Input must sum exactly to one; use rational strings or --normalize-input explicitly")
    return np.asarray(vals, dtype=object).reshape(p.shape)


def load_distribution(path, outputs=None, normalize=False):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["a", "b", "c", "p"]:
                raise ValueError("CSV header must be a,b,c,p")
            entries = {}
            for row in reader:
                abc = tuple(int(row[k]) for k in ("a", "b", "c"))
                if min(abc) < 0 or abc in entries:
                    raise ValueError("CSV indices must be nonnegative and unique")
                entries[abc] = fraction(row["p"])
        if not entries:
            raise ValueError("Empty CSV")
        shape = outputs or tuple(max(x[i] for x in entries) + 1 for i in range(3))
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError("outputs must contain three positive integers")
        p = np.full(shape, 0, dtype=object)
        for abc, val in entries.items():
            if any(x >= n for x, n in zip(abc, shape)):
                raise ValueError("CSV index lies outside output alphabet")
            p[abc] = val
        return from_document(p, normalize)
    doc = json.loads(path.read_text(), parse_float=Decimal)
    p = from_document(doc, normalize)
    if outputs is not None and tuple(outputs) != p.shape:
        raise ValueError("Requested outputs differ from JSON shape")
    return p


def dump_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (public_json(document) + "\n").encode()
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                f.write(data)
    else:
        path.write_bytes(data)


def load_json(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return canonical_document(json.loads(data))
