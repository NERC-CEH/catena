---
title: Data formats
icon: lucide/database
---

# Data formats

catena picks a file's format from its extension.

## Supported formats

| Extension | Format | Loader | `[subset]` outputs |
|-----------|--------|--------|--------------------|
| `.nc`, `.netcdf` | NetCDF | `xarray` + `netcdf4` engine | One file per part |
| `.zarr` | Zarr store | `xarray` + `zarr` engine | Regions of a shared store |

Both are read as inputs and written as outputs, and hold any array: gridded data, multi-point data and single-site time series.

A scalar parameter belongs in the config itself, in a [module section](configuration.md#modules).

CSV and Parquet are not supported.
Convert a table to NetCDF first, and label its variables with their units if your nodes declare any:

```python
import pandas as pd

ds = pd.read_csv("daily.csv", index_col="time", parse_dates=True).to_xarray()
ds["temperature"].attrs["units"] = "degC"
ds.to_netcdf("daily.nc")
```

The table is defined in [`catena.formats.FORMATS`](modules/catena.formats.md).

## Spatial handling

Inputs keep the dimensions and coordinates they have in their files: catena does not stack, transpose, rename or reproject them, and reads no meaning into a CRS.
A gridded input arrives as `(time, y, x)`, or whatever its spatial dimensions are called, and a table of sites as `(time, site)`.

catena does not check that gridded inputs share a grid unless you ask it to.
Without a check, xarray aligns two arrays on mismatched coordinates by intersecting them, so a grid offset by rounding error gives a partial or empty result and no error.
Declare [`crs_equal` and `coords_equal`](configuration.md#validation) on the spatial coordinates to make a mismatch an error before anything runs.

A model that wants one spatial dimension can stack the grid itself, for example with `.stack(pixel=("y", "x"))`, and must unstack its results before they reach an output.
No supported format can store a stacked (MultiIndex) dimension, so catena refuses such an output and names the dimension.
A model that also drops cells after stacking, such as open sea, must restore the input's full grid, for example with `.unstack("pixel").reindex_like(elevation)`.
Unstacking alone rebuilds only the rows and columns that kept a cell, so the output would cover less than the input and would not fit a store `catena subset create-store` built.

For the latitude and longitude of each grid cell, see [Latitude and longitude](../guides/nodes/latitude-and-longitude.md).

## Temporal handling

catena finds the time dimension by its datetime-like coordinate (NumPy `datetime64` or a cftime index), so the dimension can have any name.
An input may have at most one such dimension.

File variables are named without a frequency suffix (`temperature`, not `temperature_daily`); catena appends the section's suffix to make the [node name](configuration.md#inputs), and infers nothing else from it.
Frequency is checked only where it is declared: by a `Freq` [contract](../guides/nodes/contracts.md), or across whole inputs by the [`time_equal` / `time_subset` checks](configuration.md#validation).

## Units metadata

Give input variables a CF-style `units` attribute so catena can validate them against your nodes' [contracts](../guides/nodes/contracts.md).
A missing or unparseable `units` attribute fails wherever a consumer declares a unit.

## Output provenance

A run from a config file stamps the config's full text and SHA-256 into each output's `attrs` (`catena_config`, `catena_config_sha256`), so every result records the pipeline that made it.
A run from an in-memory `ParsedConfig` has no text to stamp, so its outputs carry neither.
