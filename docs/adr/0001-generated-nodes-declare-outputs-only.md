# 1. Generated nodes declare their outputs only

- **Status:** accepted
- **Date:** 2026-11-13
- **Discussion:** [#39](https://github.com/NERC-CEH/catena/pull/39)

## Context

A `[[node]]` or `[[resample]]` entry generates a Hamilton function whose contract comes from the entry's declared `units`, `dims`, `dtype`, `coords` and `freq`.
The whole-DAG contract check (`contract_check.py`) compares the declarations at the two ends of each edge, and flags an edge only when both ends declare a facet.

Before #39, a `[[resample]]` declared nothing of its own beyond `freq`.
Instead it was tagged *passthrough*, and the contract check propagated a declared unit across it, forward for the DAG check and backward for the input check.
This was the one place a contract was inferred rather than declared, and it needed its own tag, graph walk and fixed-point loop.

#39 replaced propagation with declarations on the `[[resample]]` entry.
A first version declared `units` and `dims` on each generated function's input as well as its output, so the edge into a resample was checked at build time too.
That caused two problems:

- Hamilton type-checks a parameter annotation against the producer's return type, so a producer typed `xr.DataArray | None` failed to wire as soon as the resample declared `units`, with Hamilton's bare `ValueError`.
- `xra.check_annotations` then validated the input at run time, so `units` on a resample could not stamp an unlabelled input, unlike `units` on a `[[node]]`.

A resample also keeps its input's units label whatever its `aggfunc`, and `overwrite_units` was refused on a `[[resample]]`.
So a sum that changes the unit, such as daily `mm d-1` summed to weekly totals, had no way to declare `mm week-1`.

## Decision

1. **A generated node declares its output contract only.**
   `[[node]]` and `[[resample]]` follow the same rule, and every input of a generated function is typed `Any`.
2. **Contracts are declared, never propagated or inferred.**
   A generated node that declares nothing for a facet is not checked for it before compute.
3. **The output unit is declared by the user, never inferred from `aggfunc`.**
   `overwrite_units` means the same on a `[[resample]]` as on a `[[node]]`: it replaces the label the output kept from its input with the declared `units`.

## Consequences

- The edge out of a generated node is checked when the driver is built; the edge into it is checked when the node runs.
- A resample with no declared `units` leaves the edge out of it unchecked before compute, as an undeclared `[[node]]` does.
  No declared contract is skipped, only checked later: the kept label still has to match the resample's declared `units` at run time, and a downstream consumer's own contract still applies.
- A loosely typed producer wires into any generated node, and `units` on a resample stamps an unlabelled input.
- A wrong `aggfunc` is not caught, since its units are dimensionally consistent either way; `docs/guides/nodes/resampling-and-units.md` documents this.
  `overwrite_units` relabels without looking at the values, so the declared unit is a claim the user is responsible for.

## Alternatives considered

- **Restore passthrough propagation, for units only, on resamples that declare none.**
  This would close the build-time gap for undeclared resamples, but brings back the machinery #39 removed, and the only contract catena would infer rather than read.
- **Annotate inputs as `Annotated[Any, ...]`.**
  Hamilton accepts this, but `xra.declaration_from_hint` reads it as no declaration, so the input side would be lost silently.
  Supporting it in `xarray-annotated` is not wanted.
- **Infer the output unit of a `sum` from the time step** (`mm d-1` summed weekly gives `mm week-1`).
  This guesses intent from `aggfunc`, and would make units depend on an operation rather than a declaration.
- **Refuse `aggfunc = "sum"` with a per-time `units`.**
  This catches only declared cases and blocks the legitimate one, a sum relabelled with the unit the user intends.
