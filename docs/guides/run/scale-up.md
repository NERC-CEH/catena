---
title: Scale up a pipeline
icon: lucide/gauge
---

# Scale up a pipeline

Four settings, roughly in order of the scale they address: caching, out-of-core dask, memory-bounded blocking, and parallel subset runs.
They compose, and none of them changes the node code.
To choose between them, see [Execution and scaling](../../concepts/execution.md).

## Caching results

A `[cache]` section persists node results to disk, and later runs read unchanged nodes back instead of recomputing them.

```toml
[cache]
path = ".catena_cache"   # default; resolved relative to the config file
```

This uses [Hamilton's caching](https://hamilton.apache.org/concepts/caching/): each node is keyed on its code and its inputs' fingerprints, so the cache invalidates itself when either changes.
catena fingerprints an `xarray.DataArray` by its values *and* metadata.

The payoff is in calibration loops.
In an `a → b → c → d` chain, changing `c`'s parameters leaves `a` and `b` cached, so only `c` and `d` recompute.

| Key | Description |
|-----|-------------|
| `path` | Cache directory (default `.catena_cache`, resolved against the config file). |
| `enabled` | Set `false` to keep the section but turn caching off. |
| `recompute` | `true` or a list of node names — force recompute (and re-cache) even on a hit. |
| `disable` | `true` or a list of node names — bypass the cache entirely for those nodes. |

`--cache`/`--no-cache` and `--cache-dir` override these from the command line.

## Out-of-core with dask

Nodes are plain xarray functions, so dask-backed inputs make them execute lazily and out-of-core with no code change.
When [driving catena from Python](drive-from-python.md#step-by-step), `.chunk()` the arrays from `load_inputs` before executing.
Combine with blocking to cap peak memory.

## Memory-bounded execution with `[blocking]`

`[blocking]` runs the whole DAG over one fixed-size block of a dimension at a time and concatenates the results.
Intermediate nodes are held in memory for one block at a time; the requested outputs of every block are held until they are written, so the outputs as a whole must still fit in memory.

```toml
[blocking]
block_size = 50      # indices of the partition dim processed at a time
dim = "y"            # a dim of the inputs, e.g. "location" if non-gridded
```

!!! warning "Outputs must vary over the partition dim"

    An output without the `dim` dimension, such as a grid-mean aggregate, cannot be concatenated, and catena raises an error.
    Drop such variables from `[outputs]` when blocking.

## Parallel subset runs

For parallelism, run independent `catena run` processes, each restricted to a contiguous slice of one dimension.
Give each process its slice on the command line, so that they all share one config:

```sh
catena run config.toml --subset y:0:50      # DIM:START:STOP, stop exclusive
catena run config.toml --subset y:50:100
```

From Python, pass `subset=SubsetSpec(0, 50, dim="y")` to `catena.run`.
A `[subset]` section in the config does the same for a single slice, and `--subset` replaces it:

```toml
[subset]
start = 0            # inclusive, zero-based
stop  = 50           # exclusive (Python slice convention)
dim   = "y"          # a dimension of the input files
```

The dimension is one the **input files** have, and every file that has it is sliced before anything else happens.
For gridded data that is a spatial dimension such as `y`, so a subset is a band of whole rows.
For non-gridded data it is whatever the files call it, such as `location` or `site`.
Every input that has the dimension must have the same length along it.

Each process reads only its slice and writes its outputs in the usual layout.
The processes share one output `path`, so catena keeps their writes apart, and `catena subset merge` finishes the job.

**NetCDF** — each process writes its own part (`weekly.nc` → `weekly_y0-50.nc`), with no setup needed:

```sh
parallel catena run config.toml --subset y:{} ::: 0:50 50:100 100:150
catena subset merge config.toml --dim y          # concatenate the parts into weekly.nc
```

`merge` refuses to run unless the parts cover the dimension end to end, so a run that died cannot leave a silent gap.
It finds the parts by the dimension in their names: pass `--dim`, or it reads the config's `[subset]`.
Pass `--out <path>` (single-output configs only) to choose a destination.

**Zarr** — all processes write regions of one shared store, created once up front:

```sh
catena subset create-store config.toml --dim y --chunk 50   # build the empty store(s)
parallel catena run config.toml --subset y:{} ::: 0:50 50:100 100:150
catena subset merge config.toml                    # check every region was written
```

The store is the finished output, in the layout an ordinary run writes, so nothing is copied afterwards.
`merge` checks it instead: every run writes all the chunks in its region, so a chunk missing from the store means a run failed or was never launched, and `merge` says how many.

!!! note "What `create-store` computes"

    `create-store` runs the pipeline over a single index of the dimension and takes each output's other axes, dims and dtype from the result, so the layout matches what the runs will write, derived axes included.

    The store therefore belongs to the config that created it.
    If a config change moves an output's time axis, `run` refuses to write into the stale store; re-create it with `--overwrite`.
    An output variable without the partition dimension, such as a total over it, cannot go in a shared store; write it to NetCDF.

!!! note "Chunk alignment for Zarr"

    Concurrent region writes are safe only when each subset's boundaries fall on the store's chunk boundaries along the partition dimension.
    `create-store --chunk` sets that chunk size; without it the whole dimension is one chunk, or `[blocking].block_size` if `[blocking]` partitions the same dimension.
    A misaligned subset is an error before the run computes anything, so keep subset ranges to multiples of the chunk size.
