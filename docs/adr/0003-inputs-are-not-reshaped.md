# 3. Inputs are not reshaped, and catena has no geospatial layer

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

Without stacking, the optional `geo` extra and the `catena.gridded` subpackage held two unrelated things:

- A geospatial layer of about 110 lines, the only code that used `rioxarray` or `pyproj`.
  When an input carried a CRS, it checked that every such input shared one CRS and grid, and added `latitude` and `longitude` inputs.
- The parallel subset I/O (shared Zarr stores, region writes, merging NetCDF parts) and the `catena gridded` commands.
  Since ADR 0002 this works over any dimension and needs only `xarray` and `zarr`, yet `catena gridded` refused to run without `rioxarray` and `pyproj`, and the core imported it lazily as though it needed the extra.

## Decision

1. **Inputs reach nodes with the dimensions their files have.** catena does not stack, transpose or rename them.
2. **Outputs are written as the nodes return them.** A result with a stacked (MultiIndex) dimension is refused with a catena error naming it, since no supported format can store one.
   catena does not unstack it.
3. **catena reads no meaning into a CRS.** The automatic grid check, the computed `latitude` and `longitude`, the `geo` extra and the `catena.gridded` subpackage are removed.
4. **Grid agreement is an opt-in input check.** A new `crs_equal` check compares the CRS each file stores, as text and with no dependencies; with `coords_equal` over the spatial coordinates it replaces the automatic check.
   Input checks now also run under `[subset]`: they are given the whole, unsliced files, so the reason for skipping them there did not hold.
5. **Latitude and longitude are downstream nodes.** A guide shows two node functions that compute them from an input's CRS with `rioxarray` and `pyproj`, and the test suite runs that code.
6. **Parallel subset I/O is core.** It lives in `catena.subsets`, and its commands are `catena subset create-store` and `catena subset merge`, needing only the `cli` extra.

## Consequences

- `[subset].dim` and `[blocking].dim` both name a dimension of the inputs, `y` for a grid; the finest block of a grid is one row.
- Contracts on gridded data declare the files' dimensions, such as `("time", "y", "x")`.
- A model that wants one spatial dimension stacks inside its own node and unstacks before returning, which is also where it can drop cells it does not need, such as open sea.
  Having dropped cells, it must reindex its results to the input grid: unstacking alone rebuilds only the rows and columns that kept a cell.
  viresco's models need this change.
- `stack_if_gridded`, `unstack_if_gridded` and `catena.gridded.spatial` are removed.
- Code that used `.rio` on a dataset relied on stacking having imported `rioxarray`; it must now import it itself.
- A pipeline whose gridded inputs disagree, and which declares no `crs_equal` or `coords_equal`, is no longer stopped: xarray's default inner join silently drops the cells that do not align.
  The docs say so where grids come up; making alignment exact during execution is a separate proposal.
- `crs_equal` compares stored text, so two equivalent CRSs written differently, by different tools say, fail it.
  The failure is loud and the check is optional; comparing with `pyproj` would fix it at the cost of a dependency.
- `latitude` and `longitude` are ordinary nodes, so the wiring check no longer needs an exemption for them, and `check_wiring` loses its `exempt` argument.
- viresco defines its own `latitude` and `longitude` nodes, and declares the grid checks it relies on.
- catena no longer depends on `rioxarray` or `pyproj`; they remain in the test dependency group for tests that write CRS-bearing files and for the guide.

## Alternatives considered

- **Keep stacking, behind a config switch.**
  Leaves catena holding a reshape that belongs to the model, and keeps two code paths for every gridded feature.
- **Stop stacking but keep unstacking any MultiIndex on output.**
  Convenient for a model that stacks, but implicit, and wrong when the model dropped cells: the unstacked grid can lose whole rows and no longer match a store `create-store` built from a full run.
- **Provide a generic reshape step in the TOML spec.**
  An inline `[[node]]` expression such as `x.stack(pixel=("y", "x"))` already does this where a user wants it.
- **Keep the geospatial layer as a single `catena/geo.py` behind lazy imports.**
  Small, but it keeps catena adding inputs nobody declared, triggered by metadata the user does not see, and keeps two dependencies for one convenience a downstream node provides in a few lines.
- **Keep the grid check automatic, without `rioxarray`.**
  Comparing coordinates needs to know which dimensions are spatial; inferring that from CF metadata is the guess the opt-in check avoids by having the user name them.
