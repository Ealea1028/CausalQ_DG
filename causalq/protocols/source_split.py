"""Deterministic source membership; no training RNG or data loading."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import re
from typing import Iterable


SPLIT_SALT = "causalq-gta5-source-split-v1"


def validate_ids(ids: Iterable[str]) -> list[str]:
    values = list(ids)
    if not values or any(not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value)
                         for value in values):
        raise ValueError("Filename fallback requires nonempty flat ASCII numeric IDs")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate source IDs")
    # Preserve spelling and leading zeros: IDs are paths, not integer values.
    return sorted(values)


def source_split(ids: Iterable[str], source_val_fraction: str = "0.1") -> tuple[list[str], list[str]]:
    """Rank by SHA256(salt + LF + exact ID); no seed or iteration-order effects."""
    values = validate_ids(ids)
    try:
        fraction = Decimal(source_val_fraction)
    except InvalidOperation as exc:
        raise ValueError("source_val_fraction must be a decimal number") from exc
    if not fraction.is_finite() or not Decimal(0) < fraction < Decimal(1):
        raise ValueError("source_val_fraction must be finite and strictly between 0 and 1")
    # Exact rational arithmetic avoids process-wide Decimal precision changing
    # membership at a rounding boundary.
    ratio = Fraction(fraction)
    count = (len(values) * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    if not 0 < count < len(values):
        raise ValueError("Split must leave nonempty train and validation membership")
    ranked = sorted(values, key=lambda value: (
        hashlib.sha256(f"{SPLIT_SALT}\n{value}".encode("ascii")).hexdigest(), value))
    validation = set(ranked[:count])
    train = [value for value in values if value not in validation]
    val = sorted(validation)
    verify_membership(values, train, val)
    return train, val


def manifest_bytes(ids: Iterable[str]) -> bytes:
    """Canonical UTF-8/ASCII, lexical order, LF endings, one exact ID per line."""
    return ("\n".join(validate_ids(ids)) + "\n").encode("ascii")


def verify_membership(all_ids: Iterable[str], train: Iterable[str], val: Iterable[str]) -> None:
    all_values, train_values, val_values = map(validate_ids, (all_ids, train, val))
    if set(train_values) & set(val_values):
        raise ValueError("Train/validation overlap")
    if set(train_values) | set(val_values) != set(all_values):
        raise ValueError("Split is not an exact partition of the source inventory")
