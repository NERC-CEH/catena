"""Latitude and longitude nodes for a gridded input that carries a CRS."""

from typing import Annotated

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import xarray_annotated as xra
from pyproj import Transformer
from xarray_annotated import Unit


def _to_wgs84(grid: xr.DataArray, axis: int, **attrs: str) -> xr.DataArray:
    """One EPSG:4326 coordinate (0 longitude, 1 latitude) for each cell of `grid`."""
    y_dim, x_dim = grid.rio.y_dim, grid.rio.x_dim
    y, x = np.meshgrid(grid[y_dim].values, grid[x_dim].values, indexing="ij")
    transformer = Transformer.from_crs(grid.rio.crs, "EPSG:4326", always_xy=True)
    values = transformer.transform(x, y)[axis]
    coords = {y_dim: grid[y_dim], x_dim: grid[x_dim]}
    return xr.DataArray(values, dims=(y_dim, x_dim), coords=coords, attrs=attrs)


@xra.check_annotations
def latitude(elevation: xr.DataArray) -> Annotated[xr.DataArray, Unit("degrees_north")]:
    """Latitude of each cell of the `elevation` grid."""
    return _to_wgs84(elevation, 1, units="degrees_north", standard_name="latitude")


@xra.check_annotations
def longitude(elevation: xr.DataArray) -> Annotated[xr.DataArray, Unit("degrees_east")]:
    """Longitude of each cell of the `elevation` grid."""
    return _to_wgs84(elevation, 0, units="degrees_east", standard_name="longitude")
