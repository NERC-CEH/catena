"""Tests for the input-Dataset compatibility suite (`catena.input_checks`)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from catena.config import CheckSpec
from catena.input_checks import (
    CHECKS,
    Check,
    InputCheckError,
    coords_equal,
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
    """Name datasets ``ds0``, ``ds1``, … in the order given."""
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
# Registry + run_input_checks
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_expected_checks_registered(self):
        assert set(CHECKS) == {
            "time_equal",
            "time_subset",
            "coords_equal",
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

    def _parsed(self, subset):
        from catena.config import ParsedConfig, SubsetSpec

        return ParsedConfig(
            modules=[],
            driver_config={},
            input_specs={},
            subset_spec=SubsetSpec(start=0, stop=1, dim="pixel") if subset else None,
            checks=[CheckSpec("time_equal", ["a"])],
        )

    def test_no_checks_is_noop(self):
        from catena.config import ParsedConfig
        from catena.pipeline import _run_input_checks

        parsed = ParsedConfig(modules=[], driver_config={}, checks=[])
        assert _run_input_checks(parsed) == 0

    def test_subset_skips_with_warning(self):
        from catena.pipeline import _run_input_checks

        parsed = self._parsed(subset=True)
        with pytest.warns(UserWarning, match="input checks skipped under \\[subset\\]"):
            assert _run_input_checks(parsed) == 0
