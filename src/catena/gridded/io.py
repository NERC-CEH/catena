"""Gridded (CRS) I/O: the optional geospatial layer.

Checks that gridded inputs share one grid, and computes 2-D `latitude`/`longitude`
by reprojecting it to EPSG:4326 (`compute_lat_lon`).

See `catena.gridded` for why every import of this module is lazy, and why importing
it is nonetheless cheap and dependency-free.
"""

import numpy as np
import xarray as xr

from ..errors import CatenaValueError


class MisalignedGridError(CatenaValueError):
    """Raised when two datasets do not share a common CRS and coordinates."""


def _ensure_rio() -> None:
    """Import rioxarray (registering the `.rio` accessor) or raise a clear error.

    The geospatial layer (grid checks + lat/lon reprojection) is optional;
    `rioxarray`/`pyproj` are only needed when CRS-bearing inputs are present.
    """
    try:
        import rioxarray as rioxarray  # registers the .rio accessor
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise ImportError(
            "geospatial (CRS-bearing) inputs require the optional 'geo' extra; "
            "install it with "
            '`pip install "ukceh-catena[geo] @ '
            'git+https://github.com/NERC-CEH/catena"`.'
        ) from exc


def has_crs(ds: xr.Dataset) -> bool:
    """Cheaply detect CRS metadata without importing rioxarray.

    Looks for the markers CF/rioxarray round-trips through NetCDF/Zarr: a `crs`
    global attribute, a `spatial_ref` grid-mapping coordinate, or a per-variable
    `grid_mapping` attribute. Used to decide whether to activate the (lazy)
    geospatial path, so non-gridded pipelines never touch rioxarray/pyproj.
    """
    if ds.attrs.get("crs") or "spatial_ref" in ds.coords:
        return True
    return any("grid_mapping" in da.attrs for da in ds.data_vars.values())


# ---------------------------------------------------------------------------
# Grid computation (CRS -> latitude/longitude)
# ---------------------------------------------------------------------------


def _check_common_grid(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    label1: str = "ds1",
    label2: str = "ds2",
    atol: float = 1e-6,
) -> None:
    """Raise MisalignedGridError if CRS and coordinates do not match."""
    _ensure_rio()
    if ds1.rio.crs != ds2.rio.crs:
        raise MisalignedGridError(
            f"Mismatched CRS! {label1}={ds1.rio.crs} ≠ {label2}={ds2.rio.crs}"
        )
    x1, y1 = ds1.rio.x_dim, ds1.rio.y_dim
    x2, y2 = ds2.rio.x_dim, ds2.rio.y_dim
    if not (x1 == x2 and y1 == y2):
        raise MisalignedGridError(
            f"Mismatched dimension names: {label1}=({x1}, {y1}) ≠ {label2}=({x2}, {y2})"
        )
    try:
        np.testing.assert_allclose(ds1[x1].values, ds2[x2].values, atol=atol)
        np.testing.assert_allclose(ds1[y1].values, ds2[y2].values, atol=atol)
    except AssertionError as e:
        raise MisalignedGridError(
            f"Mismatched coordinate values between {label1} and {label2}!"
        ) from e


def compute_lat_lon(
    spatial_datasets: dict[str, xr.Dataset],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute 2-D latitude and longitude DataArrays from CRS-bearing datasets.

    Both are in EPSG:4326, on the inputs' own spatial dimensions in `(y, x)` order
    (whatever `rioxarray` infers them to be called), and carry CF metadata:
    `units` of `degrees_north` / `degrees_east` and the matching `standard_name`.
    Under the CF unit registry either satisfies a declared `Unit("degrees")`.

    Raises `MisalignedGridError` unless every dataset shares one CRS and grid.
    """
    from pyproj import Transformer

    _ensure_rio()
    items = list(spatial_datasets.items())
    ref_name, ref_ds = items[0]
    for name, ds in items[1:]:
        _check_common_grid(ref_ds, ds, label1=ref_name, label2=name)

    x_dim, y_dim = ref_ds.rio.x_dim, ref_ds.rio.y_dim
    # indexing="ij": y varies along axis 0, x along axis 1.
    y_grid, x_grid = np.meshgrid(
        ref_ds[y_dim].values, ref_ds[x_dim].values, indexing="ij"
    )
    transformer = Transformer.from_crs(ref_ds.rio.crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

    coords = {y_dim: ref_ds[y_dim], x_dim: ref_ds[x_dim]}
    lat = xr.DataArray(
        lat_grid,
        dims=(y_dim, x_dim),
        coords=coords,
        name="latitude",
        attrs={"units": "degrees_north", "standard_name": "latitude"},
    )
    lon = xr.DataArray(
        lon_grid,
        dims=(y_dim, x_dim),
        coords=coords,
        name="longitude",
        attrs={"units": "degrees_east", "standard_name": "longitude"},
    )
    return lat, lon
