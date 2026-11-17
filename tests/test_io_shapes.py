"""Inputs keep the dimensions they have in their files, and outputs are written as
the nodes return them.

Three layouts are checked: a CRS-bearing `(time, y, x)` grid, a `(time,)` single
point, and `(time, site)` multi-point data. A result with a stacked (MultiIndex)
dimension cannot be written, so `get_outputs` refuses it.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from catena.errors import CatenaValueError
from catena.io import get_outputs, load_inputs
from catena.specs import IOSpec

N_TIMES = 10
N_SITES = 5
N_Y, N_X = 2, 3
TIMES = pd.date_range("2020-01-01", periods=N_TIMES, freq="D")
RNG = np.random.default_rng(0)


def _write(ds: xr.Dataset, path) -> str:
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def grid_path(tmp_path):
    pytest.importorskip("rioxarray")
    ds = xr.Dataset(
        {"var_a": (["time", "y", "x"], RNG.random((N_TIMES, N_Y, N_X)))},
        coords={
            "time": TIMES,
            "y": np.linspace(51.0, 52.0, N_Y),
            "x": np.linspace(-1.0, 1.0, N_X),
        },
    )
    return _write(ds.rio.write_crs("EPSG:4326"), tmp_path / "grid.nc")


@pytest.fixture
def single_point_path(tmp_path):
    ds = xr.Dataset({"var_a": (["time"], RNG.random(N_TIMES))}, coords={"time": TIMES})
    return _write(ds, tmp_path / "point.nc")


@pytest.fixture
def multi_point_path(tmp_path):
    ds = xr.Dataset(
        {"var_a": (["time", "site"], RNG.random((N_TIMES, N_SITES)))},
        coords={"time": TIMES, "site": np.arange(N_SITES)},
    )
    return _write(ds, tmp_path / "sites.nc")


@pytest.mark.parametrize(
    ("fixture", "dims"),
    [
        ("grid_path", ("time", "y", "x")),
        ("single_point_path", ("time",)),
        ("multi_point_path", ("time", "site")),
    ],
)
def test_inputs_keep_their_file_dims(request, fixture, dims):
    path = request.getfixturevalue(fixture)
    inputs = load_inputs({"d": IOSpec(path=path, vars=["var_a"])})
    assert inputs["var_a_d"].dims == dims


def test_grid_lat_lon_are_2d_on_the_grid(grid_path):
    inputs = load_inputs({"d": IOSpec(path=grid_path, vars=["var_a"])})
    for name in ("latitude", "longitude"):
        assert inputs[name].dims == ("y", "x")
        for dim in ("y", "x"):
            assert inputs[name].indexes[dim].equals(inputs["var_a_d"].indexes[dim])


def test_grid_output_is_written_as_returned(grid_path):
    inputs = load_inputs({"d": IOSpec(path=grid_path, vars=["var_a"])})
    out = get_outputs(
        {"var_a_d": inputs["var_a_d"]}, {"d": IOSpec(path="out.nc", vars=["var_a"])}
    )
    assert out["d"]["var_a"].dims == ("time", "y", "x")


def test_stacked_output_is_refused(grid_path):
    inputs = load_inputs({"d": IOSpec(path=grid_path, vars=["var_a"])})
    stacked = inputs["var_a_d"].stack(pixel=("y", "x"))
    with pytest.raises(CatenaValueError, match=r"stacked dimension.*'pixel'.*unstack"):
        get_outputs({"var_a_d": stacked}, {"d": IOSpec(path="out.nc", vars=["var_a"])})
