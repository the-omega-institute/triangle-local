"""Readable JSON diagnostics without shortening exact rational values."""
from decimal import Decimal
import json
import re

RATIONAL = re.compile(r"-?[0-9]+(?:_[0-9]+)*(?:/[0-9]+(?:_[0-9]+)*)?\Z")
LONG_DIGITS = re.compile(r"[0-9]{13,}")


def canonical_document(value):
    """Remove optional rational digit grouping before hashes and comparison."""
    if isinstance(value, dict):
        return {k: canonical_document(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_document(v) for v in value]
    if isinstance(value, str) and RATIONAL.fullmatch(value):
        return value.replace("_", "")
    return value


def public_document(value):
    if isinstance(value, dict):
        return {k: public_document(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_document(v) for v in value]
    if isinstance(value, str) and RATIONAL.fullmatch(value):
        value = value.replace("_", "")
        def grouped(match):
            digits = match[0]
            first = len(digits) % 3 or 3
            return digits[:first] + "_" + "_".join(
                digits[i:i+3] for i in range(first, len(digits), 3))
        return LONG_DIGITS.sub(grouped, value)
    if isinstance(value, (float, Decimal)):
        # These are search diagnostics, never exact certificate probabilities.
        return float(format(value, ".10g"))
    return value


def public_json(document):
    data = json.dumps(public_document(document), indent=2, ensure_ascii=False, allow_nan=False)
    # A long all-digit run in a hash retains its value through a JSON escape.
    # Large JSON integers (e.g. the target denominator cap) retain their value
    # in decimal scientific notation. Rational strings have already been grouped.
    token = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w.])-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
    def readable(match):
        text = match[0]
        if text.startswith('"'):
            return LONG_DIGITS.sub(lambda m: m[0][:12] + "".join(
                "\\u003" + m[0][i] + m[0][i+1:i+9]
                for i in range(12, len(m[0]), 9)), text)
        if LONG_DIGITS.search(text) and "." not in text and "e" not in text.lower():
            sign = "-" if text.startswith("-") else ""
            digits = text.lstrip("-")
            mantissa = digits[0] + ("." + digits[1:].rstrip("0") if digits[1:].rstrip("0") else "")
            return sign + mantissa + "e" + str(len(digits)-1)
        if LONG_DIGITS.search(text):
            return format(float(text), ".9e")
        return text
    return token.sub(readable, data)
