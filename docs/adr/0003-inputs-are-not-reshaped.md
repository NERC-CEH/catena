# 3. Inputs are not reshaped

- **Status:** proposed
- **Date:** 2026-11-17

## Context

When an input carried CRS metadata, catena stacked its spatial dimensions `(y, x)` into one `pixel` dimension as it loaded, computed `latitude` and `longitude` on that stacked index, and unstacked every output back to `(y, x)` before writing it.

The behaviour came from viresco, the terrestrial ecosystem modelling package catena was factored out of, whose models treat every grid cell independently and take arrays with at most one spatial dimension.
As a general library, catena inherited that assumption, and it costs more than it gives:

- A model that couples grid cells, such as lateral flow or dispersal, needs the neighbourhood that stacking destroys.
- Users had to write a name catena invented into their contracts (`dims = ["time", "pixel"]`), though catena otherwise reads dimension names off the data and never assumes `time`.
- After ADR 0002, `[subset].dim` named a dimension of the files (`y`) and `[blocking].dim` one of the loaded arrays (`pixel`): two names for one grid, which that ADR recorded as its main cost.
- Whether an input was reshaped depended on metadata a user does not see, the presence of a CRS.
- Stacking a grid chunked along both axes gives irregular chunks along `pixel`.

## Decision

1. **Inputs reach nodes with the dimensions their files have.** catena does not stack, transpose or rename them.
2. **CRS metadata still activates the geospatial layer**, which now only checks that gridded inputs share one grid and adds `latitude` and `longitude` as 2-D arrays on the grid's own spatial dimensions, in `(y, x)` order.
3. **Outputs are written as the nodes return them.** A result with a stacked (MultiIndex) dimension is refused with a catena error naming it, since no supported format can store one.
   catena does not unstack it.

## Consequences

- `[subset].dim` and `[blocking].dim` both name a dimension of the inputs, `y` for a grid; the finest block of a grid is one row.
- Contracts on gridded data declare the files' dimensions, such as `("time", "y", "x")`.
- A model that wants one spatial dimension stacks inside its own node and unstacks before returning, which is also where it can drop cells it does not need, such as open sea.
  viresco's models need this change.
- `stack_if_gridded`, `unstack_if_gridded` and `catena.gridded.spatial` are removed.
- Code that used `.rio` on a dataset relied on stacking having imported `rioxarray`; it must now import it itself.

## Alternatives considered

- **Keep stacking, behind a config switch.**
  Leaves catena holding a reshape that belongs to the model, and keeps two code paths for every gridded feature.
- **Stop stacking but keep unstacking any MultiIndex on output.**
  Convenient for a model that stacks, but implicit, and wrong when the model dropped cells: the unstacked grid can lose whole rows and no longer match a store `create-store` built from a full run.
- **Provide a generic reshape step in the TOML spec.**
  An inline `[[node]]` expression such as `x.stack(pixel=("y", "x"))` already does this where a user wants it.
