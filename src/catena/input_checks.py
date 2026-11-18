"""Intrinsic input-Dataset compatibility checks.

A small library of predicates over input `Dataset`s,
`check_X(datasets, **kwargs) -> None` (raise on failure, return `None` on
pass), plus a registry (`CHECKS`) and a runner (`run_input_checks`) that applies
a parsed `[validation].checks` config block and aggregates failures.

`datasets` maps a name to each dataset; the name is what a failure message
calls it. The first entry is the reference the others are compared against.

Every check here is a statement about the inputs *alone*. Anything coupled to a
DAG operation ("does a resample node land on the right anchor?") is a Freq
contract instead, not a check.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from .errors import CatenaError, CatenaValueError
from .io import time_dims
from .specs import CheckSpec

logger = logging.getLogger(__name__)


class InputCheckError(CatenaValueError):
    """Raised (by `run_input_checks`) when one or more input checks fail."""


Datasets = Mapping[str, xr.Dataset]
"""Name -> dataset, in the order the check compares them."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _single_time_index(name: str, ds: xr.Dataset) -> pd.DatetimeIndex:
    """Return the dataset's single time index, raising if there is not exactly one.

    Uses `catena.io.time_dims` so arbitrarily-labelled time dimensions are
    handled; the "at most one time dimension per input" invariant (enforced in
    `load_inputs`) guarantees `len(tdims) <= 1`.
    """
    tdims = time_dims(ds)
    if not tdims:
        raise CatenaValueError(
            f"{name} has no time dimension; a time check cannot be applied to it"
        )
    return ds.get_index(tdims[0])  # type: ignore[return-value]


def _require_arity(datasets: Datasets, name: str, arity: int) -> None:
    """Backstop arity guard for fixed-arity checks (the parser catches this earlier)."""
    if len(datasets) != arity:
        raise CatenaValueError(
            f"check '{name}' takes exactly {arity} datasets, got {len(datasets)}"
        )


# ---------------------------------------------------------------------------
# Time checks
# ---------------------------------------------------------------------------


def time_equal(datasets: Datasets) -> None:
    """Assert every dataset shares an identical time index."""
    if len(datasets) < 2:
        return
    (ref_name, ref_ds), *others = datasets.items()
    ref = _single_time_index(ref_name, ref_ds)
    for name, ds in others:
        idx = _single_time_index(name, ds)
        if not ref.equals(idx):
            raise CatenaValueError(
                f"{name} time index differs from {ref_name} "
                f"({len(idx)} vs {len(ref)} timestamps)"
            )


def time_subset(datasets: Datasets) -> None:
    """Assert the second dataset's timestamps are a subset of the first's.

    Directional: the first dataset is the reference (superset), the second the
    candidate subset. Raw timestamps, no resampling.
    """
    _require_arity(datasets, "time_subset", 2)
    (sup_name, sup_ds), (sub_name, sub_ds) = datasets.items()
    sup = _single_time_index(sup_name, sup_ds)
    sub = _single_time_index(sub_name, sub_ds)
    missing = sub[~sub.isin(sup)]
    if len(missing) > 0:
        raise CatenaValueError(
            f"{sub_name} has {len(missing)} timestamp(s) absent from {sup_name}: "
            f"{missing.tolist()}"
        )


# ---------------------------------------------------------------------------
# Coordinate and CRS checks
# ---------------------------------------------------------------------------


def coords_equal(datasets: Datasets, *, coords: list[str], atol: float = 1e-6) -> None:
    """Assert the named `coords` match across every dataset.

    Float coordinates are compared with `atol`; other dtypes (datetime,
    string, integer) are compared exactly.
    """
    if len(datasets) < 2:
        return
    (ref_name, ref_ds), *others = datasets.items()
    for coord in coords:
        ref = ref_ds.coords.get(coord)
        if ref is None:
            raise CatenaValueError(f"coordinate '{coord}' missing from {ref_name}")
        for name, ds in others:
            other = ds.coords.get(coord)
            if other is None:
                raise CatenaValueError(f"coordinate '{coord}' missing from {name}")
            if other.shape != ref.shape:
                raise CatenaValueError(
                    f"coordinate '{coord}' has shape {other.shape} in {name} "
                    f"but {ref.shape} in {ref_name}"
                )
            if np.issubdtype(ref.dtype, np.floating):
                if not np.allclose(ref.values, other.values, atol=atol):
                    raise CatenaValueError(
                        f"coordinate '{coord}' values differ between {ref_name} and "
                        f"{name} (atol={atol})"
                    )
            elif not ref.equals(other):
                raise CatenaValueError(
                    f"coordinate '{coord}' values differ between {ref_name} and {name}"
                )


def _grid_mapping_names(attribute: str) -> list[str]:
    """Return the variable names in a CF `grid_mapping` attribute.

    The short form is one name (`"crs"`); the extended form pairs each name with
    the coordinates it applies to (`"crs: x y"`, or several such pairs).
    """
    if ":" not in attribute:
        return attribute.split()
    return [token[:-1] for token in attribute.split() if token.endswith(":")]


def _grid_mapping_text(attrs: dict) -> str:
    """Return a grid-mapping variable's CRS: its WKT, else its CF parameters."""
    text = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    if text is not None:
        return str(text)
    return "; ".join(f"{key}={attrs[key]}" for key in sorted(attrs))


def _crs_text(name: str, ds: xr.Dataset) -> str:
    """Return the dataset's CRS as stored in the file; raise unless it has exactly one.

    Read from the CF grid-mapping variables each data variable names: their
    `crs_wkt` (or rioxarray's `spatial_ref`) attribute, else their CF parameters.
    A dataset with no grid mapping falls back to a global `crs` attribute.
    """
    found: set[str] = set()
    for var in ds.data_vars.values():
        attribute = var.encoding.get("grid_mapping") or var.attrs.get("grid_mapping")
        if attribute is None:
            continue
        names = [n for n in _grid_mapping_names(str(attribute)) if n in ds.variables]
        if names:
            found.add(" | ".join(_grid_mapping_text(ds[n].attrs) for n in names))
    if not found and ds.attrs.get("crs"):
        found.add(str(ds.attrs["crs"]))
    if not found:
        raise CatenaValueError(
            f"{name} has no CRS (no grid mapping or 'crs' attribute)"
        )
    if len(found) > 1:
        raise CatenaValueError(f"{name} has {len(found)} different CRSs")
    return found.pop()


def crs_equal(datasets: Datasets) -> None:
    """Assert every dataset declares the same CRS.

    The CRS is compared as stored in the file: the WKT where there is one, else
    the CF grid-mapping parameters. Two equivalent CRSs written differently (by
    different tools, say) therefore count as different.
    """
    if len(datasets) < 2:
        return
    (ref_name, ref_ds), *others = datasets.items()
    ref = _crs_text(ref_name, ref_ds)
    for name, ds in others:
        if _crs_text(name, ds) != ref:
            raise CatenaValueError(f"{name} CRS differs from {ref_name}")


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """A registry entry: the predicate plus its expected arity.

    `arity` is `"variadic"` (any number, `>=1`) or an exact integer. The
    config parser uses it to reject an ill-sized `inputs` list at parse time,
    and `func`'s signature to reject keys the check does not take.
    """

    func: Callable[..., None]
    arity: "Literal['variadic'] | int"


CHECKS: dict[str, Check] = {
    "time_equal": Check(time_equal, "variadic"),
    "time_subset": Check(time_subset, 2),
    "coords_equal": Check(coords_equal, "variadic"),
    "crs_equal": Check(crs_equal, "variadic"),
}


def run_input_checks(
    raw_datasets: dict[str, xr.Dataset], checks: list[CheckSpec]
) -> None:
    """Apply each `CheckSpec` to the loaded raw Datasets, aggregating failures.

    Each dataset is named `[inputs.<label>]` in a failure message. A variadic
    check that receives fewer than two datasets is a quiet no-op. Individual check
    failures are collected and raised together as a single `InputCheckError`
    naming every failed check.
    """
    failures: list[tuple[CheckSpec, CatenaError]] = []
    for spec in checks:
        entry = CHECKS[spec.check]
        datasets = {f"[inputs.{label}]": raw_datasets[label] for label in spec.inputs}
        if entry.arity == "variadic" and len(datasets) < 2:
            logger.info(
                "check '%s' over %s is a no-op (fewer than two datasets)",
                spec.check,
                spec.inputs,
            )
            continue
        try:
            entry.func(datasets, **spec.kwargs)
        except CatenaError as exc:
            failures.append((spec, exc))

    if failures:
        lines = [
            f"  - check '{spec.check}' over {spec.inputs}: {exc}"
            for spec, exc in failures
        ]
        raise InputCheckError(
            f"{len(failures)} input check(s) failed:\n" + "\n".join(lines)
        )
