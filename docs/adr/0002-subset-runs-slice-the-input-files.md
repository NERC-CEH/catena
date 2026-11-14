# 2. Subset runs slice the input files and write the final layout

- **Status:** proposed
- **Date:** 2026-11-14

## Context

Parallel runs over a large grid work by giving each `catena run` process a `[subset]`, a contiguous slice of one dimension.
NetCDF outputs go to one part file per process; Zarr outputs are region-written into a store that `catena gridded create-store` builds first.

Until now a subset was taken from the loaded arrays, after a gridded input had been stacked from `(y, x)` into `pixel`.
A gridded subset was therefore a range of pixels, and three things followed:

- Each process wrote the stacked `pixel` layout, with the `(y, x)` index flattened into plain coordinates, because a range of pixels need not be a rectangle.
- The shared Zarr store was 1-D in `pixel`, so `catena gridded merge` had to copy the whole output into a second, unstacked store: twice the disk, and a serial pass over everything the parallel runs had produced.
- The store builder was written for `pixel` alone, so a Zarr output could not be partitioned over any other dimension, though a NetCDF one could.

A failed Zarr subset run also went unnoticed.
The store carried no record of which regions had been written, and `merge` only verified completeness for NetCDF parts.

## Decision

1. **A subset slices the input files as they are opened**, before any grid is stacked.
   `dim` is a dimension of the files: `y` for a grid, `location` for a table of sites.
   Every file that has it must have the same length along it.
2. **A subset run writes the layout an ordinary run writes.**
   A band of whole rows unstacks to a rectangle, so there is no separate stacked output mode.
3. **The shared Zarr store is the finished output.**
   `create-store` runs the pipeline over one index of `dim`, takes each output's layout from the result, and resizes only `dim` to its full length.
   It works for any dimension.
4. **`merge` verifies a Zarr store instead of copying it.**
   A region write stores every chunk, including one holding only fill values, so a chunk missing from the store is a region no run wrote.
5. **The region is still given explicitly, and the coordinates are still compared by catena.**

## Consequences

- A gridded subset is a band of rows (or columns), not an arbitrary range of pixels, so the finest partition is one row.
- `[subset].dim` and `[blocking].dim` no longer mean the same thing for gridded data: a subset names a dimension of the files (`y`), a block names a dimension of the loaded arrays (`pixel`).
  They still compose, since blocking runs inside each subset.
- `dim = "pixel"` on a gridded input is now an error that lists the files' dimensions.
- `create-store` needs the partition dimension (`--dim`, or the config's `[subset]`), and its chunk size is in units of that dimension (`--chunk`, replacing `--pixel-chunk`).
  `[blocking].block_size` is only a sensible default when `[blocking]` partitions the same dimension.
- Storing all-fill chunks costs disk for regions that are entirely empty, such as open sea in a land model.
- `flatten_pixel_index`, `unstack_pixel`, the `stacked` argument of `get_outputs` and `io.subset_inputs` are removed.
- Misalignment between a subset and the store's chunks is reported before a run computes anything, where it was reported at the write.

## Alternatives considered

- **Keep pixel subsets and unstack on write when the range covers whole rows.**
  Less change, but the user would have to compute pixel ranges as multiples of the row length, and the stacked path would remain for ranges that are not.
- **`to_zarr(region="auto")`**, letting xarray find the region from the coordinates.
  Tried against xarray 2026.9.0 and rejected: when the partition dimension has no coordinate it writes to the start of the store without complaint, and it accepts a dataset whose other axes are a subset of the store's, writing a partial region.
- **Detect unwritten regions by scanning for NaN.**
  Reads the whole store, and cannot tell an unwritten region from one that is legitimately empty.
