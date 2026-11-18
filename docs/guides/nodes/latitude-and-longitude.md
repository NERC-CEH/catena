---
title: Latitude and longitude
icon: lucide/globe
---

# Latitude and longitude

catena passes inputs to your nodes as their files hold them, and adds no coordinates of its own.
A model that needs the latitude and longitude of each cell of a projected grid computes them in an ordinary node, from the CRS the input file carries.

## The module

The two nodes below take any gridded input that carries a CRS, here `elevation`, and reproject its grid to EPSG:4326 with [rioxarray](https://corteva.github.io/rioxarray/) and [pyproj](https://pyproj4.github.io/pyproj/), which you install yourself.
Rename the parameter to the input your pipeline has.

<!-- fmt:off -->
```python
--8<-- "docs/guides/nodes/latitude_longitude.py"
```
<!-- fmt:on -->

Each result is 2-D on the input's own spatial dimensions, in `(y, x)` order, so it lines up with every other array on that grid.
rioxarray must be able to tell which dimensions those are.
It recognises `x`/`y` and `longitude`/`latitude` by name, and any other names, such as `easting`/`northing`, only when their coordinates carry CF `axis` (`X`, `Y`) or `standard_name` (`projection_x_coordinate`, `projection_y_coordinate`) attributes.
If your files have neither, call `grid.rio.set_spatial_dims(x_dim=..., y_dim=...)` at the top of `_to_wgs84`; otherwise the nodes raise rioxarray's `MissingSpatialDimensionError`.
The declared `Unit` is a contract like any other: catena checks it against each consumer before the pipeline runs.
Under the CF unit registry `degrees_north` and `degrees_east` both satisfy a consumer that declares `Unit("degrees")`, so the contract cannot catch a swapped latitude and longitude.

## Wire it in

```toml
[inputs.static]
path = "static.nc"
vars = ["elevation"]
suffix = ""

[latitude_longitude]
_import_path = "latitude_longitude.py"
```

A node that takes a `latitude` or `longitude` parameter now receives the arrays, and an `[outputs.*]` section can write them.

## Check that the inputs share a grid

The nodes take their grid from one input.
If several inputs are meant to share it, say so in `[validation]`, so a mismatch is an error before anything runs:

```toml
[validation]
checks = [
  { check = "crs_equal", inputs = ["static", "climate"] },
  { check = "coords_equal", inputs = ["static", "climate"], coords = ["y", "x"] },
]
```

Name the gridded inputs rather than writing `["*"]`: `crs_equal` fails an input with no CRS, such as a table of parameters or sites.

See [Configuration › Validation](../../reference/configuration.md#validation).
