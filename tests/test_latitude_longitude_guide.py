"""The latitude/longitude nodes in `docs/guides/nodes/latitude-and-longitude.md`.

The guide includes `docs/guides/nodes/latitude_longitude.py` verbatim, so these
tests run the code a reader copies.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Unit

from catena.pipeline import dry_run, run

pytest.importorskip("rioxarray")
pytest.importorskip("pyproj")

GUIDE_MODULE = Path(__file__).parents[1] / "docs/guides/nodes/latitude_longitude.py"


@pytest.fixture
def bng_grid(tmp_path) -> Path:
    """A CRS-bearing elevation grid on British National Grid, starting at its true
    origin, 49°N 2°W."""
    ds = xr.Dataset(
        {
            "elevation": (
                ("northing", "easting"),
                np.arange(6.0).reshape(2, 3),
                {"units": "m"},
            )
        },
        coords={
            "easting": [400000.0, 401000.0, 402000.0],
            "northing": [-100000.0, -99000.0],
        },
    )
    # CF axis metadata is what lets rioxarray find the spatial dims when the file
    # is read back.
    ds.easting.attrs.update(axis="X", standard_name="projection_x_coordinate")
    ds.northing.attrs.update(axis="Y", standard_name="projection_y_coordinate")
    ds = ds.rio.set_spatial_dims(x_dim="easting", y_dim="northing")
    path = tmp_path / "static.nc"
    ds.rio.write_crs("EPSG:27700").to_netcdf(path)
    return path


def _config(tmp_path, grid: Path, extra: str = "") -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[inputs.static]
path = "{grid}"
vars = ["elevation"]
suffix = ""

[latitude_longitude]
_import_path = "{GUIDE_MODULE}"
{extra}
[outputs.coords]
path = "{tmp_path / "coords.nc"}"
suffix = ""
vars = ["latitude", "longitude"]
"""
    )
    return config


def test_nodes_lie_on_the_input_grid(tmp_path, bng_grid):
    report = run(_config(tmp_path, bng_grid))
    out = report.outputs["coords"]
    elevation = xr.open_dataset(bng_grid)["elevation"]
    for name, units in (("latitude", "degrees_north"), ("longitude", "degrees_east")):
        assert out[name].dims == ("northing", "easting")
        assert out[name].attrs["units"] == units
        xr.align(out[name], elevation, join="exact")
    # Within the ~100 m shift between the OSGB36 and WGS84 datums.
    origin = {"northing": 0, "easting": 0}
    assert float(out["latitude"][origin]) == pytest.approx(49.0, abs=0.01)
    assert float(out["longitude"][origin]) == pytest.approx(-2.0, abs=0.01)
    assert np.all(np.diff(out["latitude"].values, axis=0) > 0)  # north is +y
    assert np.all(np.diff(out["longitude"].values, axis=1) > 0)  # east is +x


def test_consumer_contract_is_checked_before_running(tmp_path, bng_grid):
    """A consumer declaring `Unit("degrees")` passes the whole-DAG contract check."""
    import sys
    import types

    @xra.check_annotations
    def scaled_latitude(
        latitude: Annotated[xr.DataArray, Unit("degrees")],
    ) -> Annotated[xr.DataArray, Unit("degrees")]:
        return latitude * 1

    module = types.ModuleType("latitude_consumer")
    scaled_latitude.__module__ = module.__name__
    module.scaled_latitude = scaled_latitude  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    try:
        extra = '\n[latitude_consumer]\n_import_path = "latitude_consumer"\n'
        with xra.enabled(True):
            report = dry_run(_config(tmp_path, bng_grid, extra))
        assert {s.name: s.status for s in report.stages}["contracts"] == "ok"
    finally:
        del sys.modules[module.__name__]
