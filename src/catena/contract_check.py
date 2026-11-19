r"""Build-time (static) and dry-run contract checks for the Hamilton DAG.

`xarray-annotated` validates a *single function's* declared contract when that
function runs. This module lifts that guarantee to the *whole graph*, and to a
point *before any compute*: `check_dag_contracts` verifies that every internal
edge whose producer and consumer both declare a contract is mutually consistent,
so a mismatch surfaces as soon as the driver is built rather than part way through
a run; `check_input_contracts` validates the metadata of the actually-loaded input
`DataArray`\\ s against the contracts declared by their consumers, without
executing any node (the basis of `catena run --dry-run`).

Each check raises one `CatenaValueError` naming everything it found.
`dag_contract_conflicts` and `input_contract_violations` return the same as a list
of `catena.faults.Fault` records instead, one per fault.

**Facets.** The checks cover every `xarray-annotated` facet — **units** (pint/CF),
**dims** / **coords** / **dtype** (schema), and **freq** (the spacing and phase of
the time axis) — and apply the rules `xarray_annotated.check_annotations` applies at
run time, through `xarray_annotated.declaration_conflicts` for edges and
`xarray_annotated.declaration_violations` for inputs. A malformed declaration, on a
function no `xarray_annotated.check_annotations` has validated, raises
`CatenaValueError`.

**Switching off.** Both checks do nothing when `xarray-annotated` is switched off,
with `XARRAY_ANNOTATED_ENABLED=false` or inside `xarray_annotated.enabled(False)`.

Two distinctions apply:

- **Edge vs input.** `coords` declarations are lower bounds ("at least these"), so
  two of them can never be shown inconsistent; coords participates in the input
  check only. `freq` is inferred from coordinate *values*, but reading a 1-D
  datetime coordinate executes no node, so it stays a legitimate `--dry-run`
  pre-flight.
- **Provable-only.** An edge is flagged only when the two declarations are *provably*
  inconsistent (different units, disjoint dim sets, different dtypes). An edge with a
  declaration missing at either end is not flagged, which lets a partially-annotated
  pipeline adopt the check without a wave of false positives. Units that could be
  converted still count as different, since nothing converts them between nodes.
  An edge that passes is not proven safe: `Freq("D", gaps_ok=True)` is compatible
  with `Freq("D")`, which still raises on an axis with holes when the data arrives.

**Generated nodes.** A `[[node]]` or `[[resample]]` node is checked like any
other, by what it declares. An edge into or out of a node declaring nothing is not
checked.
"""

from dataclasses import is_dataclass
from itertools import combinations
from typing import Any, get_type_hints, is_typeddict

from hamilton import driver
from hamilton.node import Node
from xarray_annotated import (
    Declared,
    Violation,
    assert_valid_declaration,
    declaration_conflicts,
    declaration_from_hint,
    declaration_violations,
    resolve_enabled,
)

from .errors import CatenaValueError
from .faults import Fault

# Node name -> (declaration, producer label).
_Produced = dict[str, tuple[Declared, str]]
# Node name -> a (declaration, consumer label) per consumer.
_Consumed = dict[str, list[tuple[Declared, str]]]


# ---------------------------------------------------------------------------
# Declaration collection off the built graph
# ---------------------------------------------------------------------------


def _label(node: Node) -> str:
    """Name a node, with its originating function when that is named differently."""
    fns = node.originating_functions or ()
    fn_names = [getattr(fn, "__name__", repr(fn)) for fn in fns]
    others = [name for name in dict.fromkeys(fn_names) if name != node.name]
    return f"{node.name} ({', '.join(others)})" if others else node.name


def _output_hint(node: Node) -> Any:
    """Return the hint a node's output is declared with.

    `@extract_fields` types each extracted node as the bare field type, so for a
    node fed only by a `TypedDict` or dataclass the field's hint is read from
    that type instead.
    """
    if declaration_from_hint(node.type) is None and len(node.input_types) == 1:
        (parent_type, _) = next(iter(node.input_types.values()))
        if is_typeddict(parent_type) or is_dataclass(parent_type):
            hints = get_type_hints(parent_type, include_extras=True)
            if node.name in hints:
                return hints[node.name]
    return node.type


def _collect_declarations(
    dr: driver.Driver,
) -> tuple[_Produced, _Consumed]:
    """Read declared contracts off the built DAG's nodes.

    Returns `(produced, consumed)`, the shared source for both
    the build-time (`check_dag_contracts`) and runtime (`check_input_contracts`)
    checks. Declarations are keyed by node name, read from each node's output type
    and from the hint on each of its dependencies, so nodes that a decorator renames
    or rebinds (`@parameterize_sources`, `@parameterize`, `@subdag`) are checked on
    their real edges.
    """
    produced: _Produced = {}
    consumed: _Consumed = {}
    for node in dr.graph.get_nodes():
        if node.user_defined:
            # Hamilton types an external input with its consumer's hint.
            continue
        label = _label(node)
        out = declaration_from_hint(_output_hint(node))
        if out is not None:
            produced[node.name] = (out, label)
        for dep, (hint, _) in node.input_types.items():
            decl = declaration_from_hint(hint)
            if decl is not None:
                consumed.setdefault(dep, []).append((decl, label))
    return produced, consumed


# ---------------------------------------------------------------------------
# The two checks
# ---------------------------------------------------------------------------


def dag_contract_conflicts(dr: driver.Driver) -> list[Fault]:
    """Return one `Fault` per provable conflict across the built DAG's edges.

    Covers units, dims, dtype and freq; coords is skipped, as its declarations are
    lower bounds. Each fault is a `contract_conflict`, naming the variable, the two
    ends of the edge, the facet and both declarations. Returns an empty list when
    `xarray-annotated` is switched off.

    Raises
    ------
    CatenaValueError
        If a declaration is malformed.
    """
    if not resolve_enabled():
        return []
    produced, consumed = _collect_declarations(dr)
    faults: list[Fault] = []
    for name, consumers in consumed.items():
        candidates: list[tuple[Declared, str]] = []
        if name in produced:
            decl, who = produced[name]
            candidates.append((decl, f"output of {who}"))
        candidates.extend((decl, f"input of {who}") for decl, who in consumers)
        for decl, source in candidates:
            try:
                assert_valid_declaration(decl, f"{name!r}, {source}")
            except ValueError as exc:
                raise CatenaValueError(
                    f"malformed contract declaration: {exc}"
                ) from exc
        # All pairs, not star-wise against candidates[0]: three of the four facets are
        # *non-transitive*, because a loose declaration is compatible with everything.
        # Dims("x","y") is compatible with both Dims("x","y", ordered=True) and
        # Dims("y","x", ordered=True), which provably conflict with each other — a star
        # check anchored on the loose one would pass. (dtype widths and freq anchors
        # have the same shape; units alone is transitive, being an equivalence
        # relation.) Candidates are producer-first, so combinations() still leads with
        # "output of ...".
        for (a_decl, a_src), (b_decl, b_src) in combinations(candidates, 2):
            for conflict in declaration_conflicts(a_decl, b_decl):
                record = conflict.to_dict()
                faults.append(
                    Fault(
                        "contract_conflict",
                        f"{name!r}: {a_src} and {b_src} conflict on {conflict}",
                        node=name,
                        source=a_src,
                        target=b_src,
                        facet=conflict.facet,
                        reason=conflict.kind,
                        expected=record["b"],
                        found=record["a"],
                    )
                )
    return faults


def check_dag_contracts(dr: driver.Driver) -> None:
    """Verify declared contracts are consistent across every built-DAG edge.

    Raises `CatenaValueError` naming every conflict `dag_contract_conflicts`
    finds, or for a malformed declaration. This is a no-op when `xarray-annotated`
    is switched off.
    """
    if faults := dag_contract_conflicts(dr):
        raise CatenaValueError(
            "contract declaration mismatch(es) in DAG:\n"
            + "\n".join(f"  {fault}" for fault in faults)
        )


def input_contract_violations(dr: driver.Driver, inputs: dict[str, Any]) -> list[Fault]:
    """Return one `Fault` per way a loaded input fails the contracts declared for it.

    The runtime leg that cannot be done statically, run for every facet: an input's
    `units` attribute (units), its dims / coords / dtype (schema), and the inferred
    frequency of its time axis (temporal), are checked against the contract declared
    by its consumer(s). Dims/coords/dtype live in the file header and a frequency
    needs only the 1-D datetime coordinate, so — like units — this executes no node
    and is suitable as a `run --dry-run` pre-flight. A declared input that is not
    a `DataArray` fails unless its hint allows the value, as `| None` allows
    `None`.

    Each fault is a `contract_violation`. Two consumers whose declarations fail the
    same way share one fault, which names both. Returns an empty list when
    `xarray-annotated` is switched off.

    Raises
    ------
    CatenaValueError
        If a declaration is malformed.
    """
    if not resolve_enabled():
        return []
    _, consumed = _collect_declarations(dr)
    faults: list[Fault] = []
    for name, value in inputs.items():
        # Keyed by message, as `Declared` equality ignores `others`: `| None` on one
        # consumer does not excuse a None for another.
        found: dict[str, tuple[Violation, list[str]]] = {}
        for decl, who in consumed.get(name, ()):
            try:
                violations = declaration_violations(value, decl, name)
            except ValueError as exc:
                raise CatenaValueError(
                    f"malformed contract declaration: {exc}"
                ) from exc
            for violation in violations:
                found.setdefault(violation.message, (violation, []))[1].append(who)
        for violation, consumers in found.values():
            record = violation.to_dict()
            faults.append(
                Fault(
                    "contract_violation",
                    violation.message,
                    node=name,
                    target=f"input of {', '.join(dict.fromkeys(consumers))}",
                    facet=violation.facet,
                    reason=violation.kind,
                    expected=record["declared"],
                    found=record["found"],
                    hint=violation.hint,
                )
            )
    return faults


def check_input_contracts(dr: driver.Driver, inputs: dict[str, Any]) -> None:
    """Validate loaded inputs' metadata against the contracts declared for them.

    Raises `CatenaValueError` naming every fault `input_contract_violations`
    finds, or for a malformed declaration. This is a no-op when `xarray-annotated`
    is switched off.
    """
    if faults := input_contract_violations(dr, inputs):
        raise CatenaValueError(
            "input contract mismatch(es):\n"
            + "\n".join(f"  {fault}" for fault in faults)
        )
