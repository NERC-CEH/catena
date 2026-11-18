"""Tests for the input-Dataset compatibility suite (`catena.input_checks`)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from catena.config import CheckSpec
from catena.errors import CatenaValueError
from catena.input_checks import (
    CHECKS,
    Check,
    InputCheckError,
    coords_equal,
    crs_equal,
    run_input_checks,
    time_equal,
    time_subset,
)


def _ts_ds(times, name="v"):
    """A tiny time-indexed dataset."""
    return xr.Dataset(
        {name: ("time", np.arange(len(times), dtype=float))},
        coords={"time": pd.DatetimeIndex(times)},
    )


DAILY = pd.date_range("2020-01-01", periods=10, freq="D")


def _named(*datasets):
    """Name datasets `ds0`, `ds1`, … in the order given."""
    return {f"ds{i}": ds for i, ds in enumerate(datasets)}


# ---------------------------------------------------------------------------
# time_equal
# ---------------------------------------------------------------------------


class TestTimeEqual:
    def test_identical_passes(self):
        time_equal(_named(_ts_ds(DAILY), _ts_ds(DAILY, name="w")))

    def test_single_dataset_passes(self):
        time_equal(_named(_ts_ds(DAILY)))

    def test_differing_index_raises(self):
        with pytest.raises(ValueError, match="ds1 time index differs from ds0"):
            time_equal(_named(_ts_ds(DAILY), _ts_ds(DAILY[:5])))

    def test_missing_time_dim_raises(self):
        ds = xr.Dataset({"v": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
        with pytest.raises(ValueError, match="no time dimension"):
            time_equal(_named(_ts_ds(DAILY), ds))


# ---------------------------------------------------------------------------
# time_subset
# ---------------------------------------------------------------------------


class TestTimeSubset:
    def test_subset_passes(self):
        time_subset(_named(_ts_ds(DAILY), _ts_ds(DAILY[2:6])))

    def test_superfluous_timestamp_raises(self):
        extra = DAILY.append(pd.DatetimeIndex(["2021-01-01"]))
        with pytest.raises(ValueError, match=r"ds1 has 1 timestamp.* absent from ds0"):
            time_subset(_named(_ts_ds(DAILY), _ts_ds(extra)))

    def test_wrong_arity_raises(self):
        with pytest.raises(ValueError, match="exactly 2 datasets"):
            time_subset(_named(_ts_ds(DAILY)))

    def test_missing_time_dim_raises(self):
        ds = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})
        with pytest.raises(ValueError, match="no time dimension"):
            time_subset(_named(_ts_ds(DAILY), ds))


# ---------------------------------------------------------------------------
# coords_equal
# ---------------------------------------------------------------------------


class TestCoordsEqual:
    def test_matching_passes(self, daily_ds, static_ds):
        coords_equal(_named(daily_ds, static_ds), coords=["x", "y"])

    def test_within_atol_passes(self, daily_ds):
        shifted = daily_ds.assign_coords(x=daily_ds["x"] + 1e-9)
        coords_equal(_named(daily_ds, shifted), coords=["x"], atol=1e-6)

    def test_beyond_atol_raises(self, daily_ds):
        shifted = daily_ds.assign_coords(x=daily_ds["x"] + 1.0)
        with pytest.raises(ValueError, match="values differ"):
            coords_equal(_named(daily_ds, shifted), coords=["x"], atol=1e-6)

    def test_missing_coord_raises(self, daily_ds, static_ds):
        with pytest.raises(ValueError, match="missing"):
            coords_equal(_named(daily_ds, static_ds), coords=["nonexistent"])

    def test_shape_mismatch_raises(self, daily_ds):
        smaller = daily_ds.isel(x=slice(0, 1))
        with pytest.raises(ValueError, match="shape"):
            coords_equal(_named(daily_ds, smaller), coords=["x"])

    def test_datetime_coord_exact_match_passes(self):
        # Non-float coords take the exact-comparison branch.
        coords_equal(_named(_ts_ds(DAILY), _ts_ds(DAILY, name="w")), coords=["time"])

    def test_datetime_coord_mismatch_raises(self):
        shifted = _ts_ds(DAILY + pd.Timedelta(days=1))
        with pytest.raises(ValueError, match="values differ"):
            coords_equal(_named(_ts_ds(DAILY), shifted), coords=["time"])

    def test_missing_coord_in_later_dataset_raises(self, daily_ds):
        renamed = daily_ds.rename(x="col")
        with pytest.raises(ValueError, match="missing from ds1"):
            coords_equal(_named(daily_ds, renamed), coords=["x"])

    def test_single_dataset_passes(self, daily_ds):
        coords_equal(_named(daily_ds), coords=["x"])


# ---------------------------------------------------------------------------
# crs_equal
# ---------------------------------------------------------------------------


def _cf_grid(path, wkt: str | None, grid_mapping: str = "crs", **params):
    """Write a 2x2 grid whose variable names a CF grid mapping, read it back.

    `grid_mapping` is the variable's attribute, short (`"crs"`) or extended
    (`"crs: x y"`); the grid-mapping variable itself is always called `crs`.
    """
    from catena.formats import format_for

    attrs = {"grid_mapping_name": "transverse_mercator", **params}
    if wkt is not None:
        attrs["crs_wkt"] = wkt
    ds = xr.Dataset(
        {
            "v": (("y", "x"), np.zeros((2, 2)), {"grid_mapping": grid_mapping}),
            "crs": ((), 0, attrs),
        },
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    ds.to_netcdf(path)
    return format_for(str(path)).read(str(path))


class TestCrsEqual:
    def test_matching_passes(self, tmp_path):
        crs_equal(
            _named(
                _cf_grid(tmp_path / "a.nc", "WKT-A"),
                _cf_grid(tmp_path / "b.nc", "WKT-A"),
            )
        )

    def test_differing_raises(self, tmp_path):
        with pytest.raises(CatenaValueError, match="ds1 CRS differs from ds0"):
            crs_equal(
                _named(
                    _cf_grid(tmp_path / "a.nc", "WKT-A"),
                    _cf_grid(tmp_path / "b.nc", "WKT-B"),
                )
            )

    def test_parameters_compared_without_wkt(self, tmp_path):
        """A grid mapping given only by its CF parameters still defines a CRS."""

        def grid(name, easting):
            return _cf_grid(tmp_path / name, None, false_easting=easting)

        crs_equal(_named(grid("a.nc", 400000.0), grid("b.nc", 400000.0)))
        with pytest.raises(CatenaValueError, match="ds1 CRS differs from ds0"):
            crs_equal(_named(grid("a.nc", 400000.0), grid("c.nc", 0.0)))

    def test_extended_grid_mapping_form(self, tmp_path):
        """`grid_mapping = "crs: x y"` names the variable `crs`."""
        crs_equal(
            _named(
                _cf_grid(tmp_path / "a.nc", "WKT-A", grid_mapping="crs: x y"),
                _cf_grid(tmp_path / "b.nc", "WKT-A"),
            )
        )
        with pytest.raises(CatenaValueError, match="CRS differs"):
            crs_equal(
                _named(
                    _cf_grid(tmp_path / "a.nc", "WKT-A", grid_mapping="crs: x y"),
                    _cf_grid(tmp_path / "c.nc", "WKT-B", grid_mapping="crs: x y"),
                )
            )

    def test_global_crs_attribute(self):
        a = xr.Dataset({"v": ("x", [1.0])}, attrs={"crs": "EPSG:4326"})
        b = a.assign_attrs(crs="EPSG:27700")
        crs_equal(_named(a, a))
        with pytest.raises(CatenaValueError, match="CRS differs"):
            crs_equal(_named(a, b))

    def test_no_crs_raises(self):
        a = xr.Dataset({"v": ("x", [1.0])}, attrs={"crs": "EPSG:4326"})
        with pytest.raises(CatenaValueError, match="ds1 has no CRS"):
            crs_equal(_named(a, xr.Dataset({"v": ("x", [1.0])})))

    def test_rioxarray_written_crs(self, tmp_path):
        """The `spatial_ref` coordinate rioxarray writes is read the same way."""
        pytest.importorskip("rioxarray")
        from catena.formats import format_for

        def grid(name, epsg):
            path = tmp_path / name
            ds = xr.Dataset(
                {"v": (("y", "x"), np.zeros((2, 2)))},
                coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
            ).rio.write_crs(epsg)
            ds.to_netcdf(path)
            return format_for(str(path)).read(str(path))

        crs_equal(_named(grid("a.nc", "EPSG:27700"), grid("b.nc", "EPSG:27700")))
        with pytest.raises(CatenaValueError, match="CRS differs"):
            crs_equal(_named(grid("a.nc", "EPSG:27700"), grid("c.nc", "EPSG:4326")))


# ---------------------------------------------------------------------------
# Registry + run_input_checks
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_expected_checks_registered(self):
        assert set(CHECKS) == {
            "time_equal",
            "time_subset",
            "coords_equal",
            "crs_equal",
        }

    def test_arities(self):
        assert CHECKS["time_subset"].arity == 2
        assert CHECKS["time_equal"].arity == "variadic"


class TestRunInputChecks:
    def test_passing_checks_return_none(self):
        raw = {"a": _ts_ds(DAILY), "b": _ts_ds(DAILY, name="w")}
        run_input_checks(raw, [CheckSpec("time_equal", ["a", "b"])])

    def test_variadic_single_input_is_noop(self):
        raw = {"a": _ts_ds(DAILY)}
        run_input_checks(raw, [CheckSpec("time_equal", ["a"])])

    def test_failures_aggregate(self):
        raw = {
            "a": _ts_ds(DAILY),
            "b": _ts_ds(DAILY[:5]),
            "c": _ts_ds(DAILY[2:8]),
        }
        specs = [
            CheckSpec("time_equal", ["a", "b"]),
            CheckSpec("time_subset", ["b", "a"]),
        ]
        with pytest.raises(InputCheckError) as exc:
            run_input_checks(raw, specs)
        msg = str(exc.value)
        assert "2 input check(s) failed" in msg
        assert "time_equal" in msg
        assert "time_subset" in msg

    def test_failures_name_inputs_by_section_label(self):
        raw = {"daily": _ts_ds(DAILY), "short": _ts_ds(DAILY[:5])}
        with pytest.raises(InputCheckError) as exc:
            run_input_checks(raw, [CheckSpec("time_equal", ["daily", "short"])])
        assert "[inputs.short] time index differs from [inputs.daily]" in str(exc.value)

    def test_a_bug_in_a_check_is_not_reported_as_a_failed_check(self, monkeypatch):
        def broken(datasets):
            raise RuntimeError("bug")

        monkeypatch.setitem(CHECKS, "time_equal", Check(broken, "variadic"))
        raw = {"a": _ts_ds(DAILY), "b": _ts_ds(DAILY)}
        with pytest.raises(RuntimeError, match="bug"):
            run_input_checks(raw, [CheckSpec("time_equal", ["a", "b"])])

    def test_unknown_arity_backstop(self):
        # Direct call bypassing the parser: fixed-arity self-guard still fires.
        with pytest.raises(ValueError, match="exactly 2"):
            run_input_checks(
                {"a": _ts_ds(DAILY), "b": _ts_ds(DAILY), "c": _ts_ds(DAILY)},
                [CheckSpec("time_subset", ["a", "b", "c"])],
            )

    def test_kwargs_forwarded(self, daily_ds):
        shifted = daily_ds.assign_coords(x=daily_ds["x"] + 0.5)
        raw = {"a": daily_ds, "b": shifted}
        # atol wide enough to pass
        run_input_checks(
            raw, [CheckSpec("coords_equal", ["a", "b"], {"coords": ["x"], "atol": 1.0})]
        )
        # atol too tight to pass
        with pytest.raises(InputCheckError):
            run_input_checks(
                raw,
                [
                    CheckSpec(
                        "coords_equal", ["a", "b"], {"coords": ["x"], "atol": 1e-6}
                    )
                ],
            )


class TestRunInputChecksHook:
    """The cli/run.py `_run_input_checks` orchestration hook."""

    def test_no_checks_is_noop(self):
        from catena.config import ParsedConfig
        from catena.pipeline import _run_input_checks

        parsed = ParsedConfig(modules=[], driver_config={}, checks=[])
        assert _run_input_checks(parsed) == 0

    def test_subset_runs_checks_on_the_whole_files(self, tmp_path):
        """A `[subset]` slices the inputs for the run, not for the checks."""
        from catena.config import IOSpec, ParsedConfig, SubsetSpec
        from catena.pipeline import _run_input_checks

        specs = {}
        for label, y in (("a", [0.0, 1.0]), ("b", [0.0, 2.0])):
            path = tmp_path / f"{label}.nc"
            xr.Dataset({"v": ("y", [1.0, 2.0])}, coords={"y": y}).to_netcdf(path)
            specs[label] = IOSpec(path=str(path), vars=["v"])
        parsed = ParsedConfig(
            modules=[],
            driver_config={},
            input_specs=specs,
            # The first row agrees; only the whole files disagree.
            subset_spec=SubsetSpec(start=0, stop=1, dim="y"),
            checks=[CheckSpec("coords_equal", ["a", "b"], {"coords": ["y"]})],
        )
        with pytest.raises(InputCheckError, match="coords_equal"):
            _run_input_checks(parsed)
