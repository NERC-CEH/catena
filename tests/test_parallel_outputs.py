"""Tests for parallel subset output writing: parts, create-store, region writes."""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from catena.config import IOSpec, SubsetSpec
from catena.errors import CatenaFileExistsError, CatenaValueError
from catena.gridded.io import (
    _find_subset_parts,
    create_output_store,
    merge_subset_outputs,
    save_zarr_region,
    subset_suffix,
)
from catena.io import (
    assert_output_paths_writable,
    get_final_vars,
    get_outputs,
    load_inputs,
    save_outputs,
)

VAR = "mean_temperature"

# The session grid is 2x2, so `y` has two rows: one subset each.
ROWS = (SubsetSpec(0, 1, dim="y"), SubsetSpec(1, 2, dim="y"))


def _output_specs(path) -> dict[str, IOSpec]:
    return {"weekly": IOSpec(path=str(path), vars=[VAR])}


def _create_store(parsed, output_specs, **kwargs) -> list[str]:
    """`create_output_store` for a pipeline with its outputs swapped out.

    The store builder takes the whole `ParsedConfig` because it probes the DAG to
    learn each output's real layout; these tests vary only the output specs.
    """
    kwargs = {"dim": "y", "chunk": 1} | kwargs
    return create_output_store(replace(parsed, output_specs=output_specs), **kwargs)


def _execute(driver, config, spec: SubsetSpec | None, output_specs):
    inputs = load_inputs(config.input_specs, subset_spec=spec)
    final_vars: list[Any] = get_final_vars(output_specs)
    results = driver.execute(final_vars, inputs=inputs)  # type: ignore[reportArgumentType]
    return get_outputs(results, output_specs)


def _write_rows(driver, config, specs, rows=ROWS):
    for spec in rows:
        save_outputs(_execute(driver, config, spec, specs), specs, subset_spec=spec)


@pytest.fixture(scope="module")
def reference(pipeline_config, pipeline_driver):
    """The weekly output of a run over the whole grid."""
    specs = _output_specs("unused")
    return _execute(pipeline_driver, pipeline_config, None, specs)["weekly"].compute()


class TestSubsetSuffix:
    def test_subset_suffix(self):
        assert subset_suffix(SubsetSpec(0, 500, dim="y")) == "_y0-500"

    def test_subset_suffix_names_the_dim(self):
        spec = SubsetSpec(0, 10, dim="location")
        assert subset_suffix(spec) == "_location0-10"


class TestSubsetOverAGridDim:
    def test_a_subset_over_a_grid_dim_yields_whole_rows(self, pipeline_config):
        inputs = load_inputs(pipeline_config.input_specs, subset_spec=ROWS[0])
        full = load_inputs(pipeline_config.input_specs)
        # One of two rows of a 2x2 grid.
        for name in ("temperature_daily", "latitude"):
            assert dict(inputs[name].sizes)["y"] == 1
            assert dict(inputs[name].sizes)["x"] == 2
            assert inputs[name]["y"].values[0] == full[name]["y"].values[0]

    def test_a_dim_the_files_lack_is_refused(self, pipeline_config):
        with pytest.raises(CatenaValueError, match="dim 'pixel' is not a dimension"):
            load_inputs(
                pipeline_config.input_specs, subset_spec=SubsetSpec(0, 2, dim="pixel")
            )


# ---------------------------------------------------------------------------
# Part discovery: numeric ordering + completeness
# ---------------------------------------------------------------------------


class TestSubsetPartDiscovery:
    """`_find_subset_parts` orders parts numerically and proves they cover the grid.

    Exercised directly rather than through `merge_subset_outputs` because the
    interesting boundaries (where a lexicographic sort diverges from a numeric one)
    need far more pixels than the 2x2 session grid has.
    """

    def _touch(self, tmp_path, *names):
        for name in names:
            (tmp_path / name).touch()
        return tmp_path / "weekly.nc"

    def test_merge_sorts_parts_numerically(self, tmp_path):
        # Lexicographically, '_pixel1000-1500' sorts before '_pixel500-1000'.
        path = self._touch(
            tmp_path,
            "weekly_pixel500-1000.nc",
            "weekly_pixel1000-1500.nc",
            "weekly_pixel0-500.nc",
        )
        assert [p.name for p in _find_subset_parts(path, "pixel")] == [
            "weekly_pixel0-500.nc",
            "weekly_pixel500-1000.nc",
            "weekly_pixel1000-1500.nc",
        ]

    def test_merge_missing_part_raises(self, tmp_path):
        path = self._touch(tmp_path, "weekly_pixel0-500.nc", "weekly_pixel1000-1500.nc")
        with pytest.raises(ValueError, match="gap at pixel 500"):
            _find_subset_parts(path, "pixel")

    def test_overlapping_parts_raise(self, tmp_path):
        path = self._touch(
            tmp_path,
            "weekly_pixel0-500.nc",
            "weekly_pixel400-900.nc",
            "weekly_pixel900-1000.nc",
        )
        with pytest.raises(ValueError, match="overlap at pixel 500"):
            _find_subset_parts(path, "pixel")

    def test_parts_not_starting_at_zero_raise(self, tmp_path):
        path = self._touch(tmp_path, "weekly_pixel500-1000.nc")
        with pytest.raises(ValueError, match="gap at pixel 0"):
            _find_subset_parts(path, "pixel")

    def test_merge_ignores_non_part_files(self, tmp_path):
        # A stray '_partial' file matches the old glob but is not a subset part.
        path = self._touch(
            tmp_path, "weekly_pixel0-500.nc", "weekly_partial.nc", "weekly_pixelXX.nc"
        )
        assert [p.name for p in _find_subset_parts(path, "pixel")] == [
            "weekly_pixel0-500.nc"
        ]

    def test_no_parts_raises(self, tmp_path):
        path = self._touch(tmp_path, "weekly_partial.nc")
        with pytest.raises(FileNotFoundError, match="No subset parts"):
            _find_subset_parts(path, "pixel")


# ---------------------------------------------------------------------------
# NetCDF: unique suffixed files + merge
# ---------------------------------------------------------------------------


class TestNetcdfSubset:
    def test_subsets_write_distinct_files_and_merge(
        self, pipeline_config, pipeline_driver, reference, tmp_path
    ):
        out = tmp_path / "weekly.nc"
        specs = _output_specs(out)
        _write_rows(pipeline_driver, pipeline_config, specs)

        parts = sorted(tmp_path.glob("weekly_y*.nc"))
        assert [p.name for p in parts] == ["weekly_y0-1.nc", "weekly_y1-2.nc"]
        assert not out.exists()  # un-suffixed path untouched until merge

        assert merge_subset_outputs(specs, dim="y") == [str(out)]
        xr.testing.assert_allclose(xr.open_dataset(out), reference)

    def test_merge_refuses_to_silently_drop_a_failed_subset(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        """A subset run that never produced its part must not merge to a gap."""
        out = tmp_path / "weekly.nc"
        specs = _output_specs(out)
        _write_rows(pipeline_driver, pipeline_config, specs, rows=ROWS[1:])

        with pytest.raises(ValueError, match="gap at y 0"):
            merge_subset_outputs(specs, dim="y")

    def test_merge_needs_the_dim(self, tmp_path):
        with pytest.raises(CatenaValueError, match="--dim"):
            merge_subset_outputs(_output_specs(tmp_path / "weekly.nc"))

    def test_merge_out_override(self, pipeline_config, pipeline_driver, tmp_path):
        specs = _output_specs(tmp_path / "weekly.nc")
        _write_rows(pipeline_driver, pipeline_config, specs)
        dest = tmp_path / "custom.nc"
        assert merge_subset_outputs(specs, out=dest, dim="y") == [str(dest)]
        assert dest.exists()
        assert not (tmp_path / "weekly.nc").exists()

    def test_merge_out_rejected_for_multiple_outputs(self):
        specs = {
            "daily": IOSpec(path="a.nc", vars=[VAR]),
            "weekly": IOSpec(path="b.nc", vars=[VAR]),
        }
        with pytest.raises(ValueError, match="multiple"):
            merge_subset_outputs(specs, out="x.nc", dim="y")


# ---------------------------------------------------------------------------
# Zarr: shared pre-created store + region writes
# ---------------------------------------------------------------------------


class TestZarrSubset:
    def test_create_store_and_region_writes_reproduce_a_full_run(
        self, pipeline_config, pipeline_driver, reference, tmp_path
    ):
        store = tmp_path / "weekly.zarr"
        specs = _output_specs(store)

        assert _create_store(pipeline_config, specs) == [str(store)]

        # A freshly created store is all-NaN, in the layout a full run writes.
        empty = xr.open_zarr(store, consolidated=False)
        assert dict(empty.sizes) == dict(reference.sizes)
        assert empty[VAR].dims == reference[VAR].dims
        assert bool(np.isnan(empty[VAR].values).all())

        _write_rows(pipeline_driver, pipeline_config, specs)

        filled = xr.open_zarr(store, consolidated=False).compute()
        xr.testing.assert_allclose(filled, reference)
        # Nothing is left to merge; the store is checked and reported.
        assert merge_subset_outputs(specs) == [str(store)]

    def test_merge_reports_a_region_no_run_wrote(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        specs = _output_specs(tmp_path / "weekly.zarr")
        _create_store(pipeline_config, specs)
        _write_rows(pipeline_driver, pipeline_config, specs, rows=ROWS[:1])
        with pytest.raises(CatenaValueError, match=r"incomplete.*1 of 2 chunk"):
            merge_subset_outputs(specs)

    def test_a_region_of_only_fill_values_counts_as_written(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        """An all-NaN block (open sea, say) must not read as a failed run."""
        store = tmp_path / "weekly.zarr"
        specs = _output_specs(store)
        _create_store(pipeline_config, specs)
        for spec in ROWS:
            ds = _execute(pipeline_driver, pipeline_config, spec, specs)["weekly"]
            save_zarr_region(ds * np.nan, str(store), spec)
        assert merge_subset_outputs(specs) == [str(store)]

    def test_merge_out_rejected_for_a_store(self, tmp_path):
        with pytest.raises(CatenaValueError, match="in place"):
            merge_subset_outputs(_output_specs(tmp_path / "w.zarr"), out="x.zarr")

    def test_store_time_axis_comes_from_the_pipeline_not_the_label(
        self, pipeline_config, reference, tmp_path
    ):
        """The store's time axis is what the DAG produces, not what a label suggests.

        The output section here is labelled `monthly` — matching an input section
        that has a genuine (and different) monthly time axis — but it carries the
        *weekly* node. The store must take the weekly axis the pipeline actually
        produces.
        """
        store = tmp_path / "mislabelled.zarr"
        specs = {
            "monthly": IOSpec(
                path=str(store), vars={f"{VAR}_weekly": VAR}
            )  # node -> file var
        }
        _create_store(pipeline_config, specs)

        produced = reference.indexes["time"]
        monthly_input = load_inputs(pipeline_config.input_specs)[
            "dummy_variable_monthly"
        ].indexes["time"]
        got = xr.open_zarr(store, consolidated=False).indexes["time"]

        assert got.equals(produced)
        assert not got.equals(monthly_input)  # the label was a red herring

    def test_region_write_rejects_a_mismatched_time_axis(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        """A store whose time axis disagrees with the data is caught, not written.

        Region writes don't write coordinates, so without this guard the store would
        keep its own (wrong) timestamps and silently mislabel the data.
        """
        store = tmp_path / "weekly.zarr"
        specs = _output_specs(store)
        _create_store(pipeline_config, specs)

        ds = _execute(pipeline_driver, pipeline_config, ROWS[0], specs)["weekly"]
        shifted = ds.assign_coords(time=ds.indexes["time"] + pd.Timedelta(days=1))

        with pytest.raises(ValueError, match="'time' coordinate does not match"):
            save_zarr_region(shifted, str(store), ROWS[0])

    def test_region_write_rejects_rows_the_store_labels_differently(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        """Row 0's data must not be written where the store holds row 1."""
        store = tmp_path / "weekly.zarr"
        specs = _output_specs(store)
        _create_store(pipeline_config, specs)

        ds = _execute(pipeline_driver, pipeline_config, ROWS[0], specs)["weekly"]
        with pytest.raises(ValueError, match=r"'y' coordinate does not match.*1-2"):
            save_zarr_region(ds, str(store), ROWS[1])

    def test_misaligned_subset_is_refused_before_compute(
        self, pipeline_config, tmp_path
    ):
        specs = _output_specs(tmp_path / "weekly.zarr")
        _create_store(pipeline_config, specs, chunk=2)
        with pytest.raises(CatenaValueError, match="not aligned to the 'y' chunk"):
            assert_output_paths_writable(specs, SubsetSpec(1, 2, dim="y"))

    def test_subset_past_the_end_is_refused(self, pipeline_config, tmp_path):
        specs = _output_specs(tmp_path / "weekly.zarr")
        _create_store(pipeline_config, specs)
        with pytest.raises(CatenaValueError, match="past the end of 'y'"):
            assert_output_paths_writable(specs, SubsetSpec(2, 3, dim="y"))

    def test_multi_frequency_and_static_outputs(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        """create-store + region writes cover several frequencies, incl. static."""
        specs = {
            "daily": IOSpec(path=str(tmp_path / "daily.zarr"), vars=["temperature"]),
            "static": IOSpec(
                path=str(tmp_path / "static.zarr"), vars=["roughness"], suffix=""
            ),
        }
        assert len(_create_store(pipeline_config, specs)) == 2
        _write_rows(pipeline_driver, pipeline_config, specs)

        ref = _execute(pipeline_driver, pipeline_config, None, specs)
        for label in specs:
            got = xr.open_zarr(tmp_path / f"{label}.zarr", consolidated=False)
            xr.testing.assert_allclose(got.compute(), ref[label].compute())
        static = xr.open_zarr(tmp_path / "static.zarr", consolidated=False)
        assert "time" not in static.dims

    def test_create_store_refuses_to_clobber(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        """Re-creating an existing store needs --overwrite, to protect written data."""
        store = tmp_path / "weekly.zarr"
        specs = _output_specs(store)
        _create_store(pipeline_config, specs)
        _write_rows(pipeline_driver, pipeline_config, specs, rows=ROWS[:1])

        with pytest.raises(CatenaFileExistsError, match="overwrite"):
            _create_store(pipeline_config, specs)

        # overwrite=True recreates it as an empty (all-NaN) store.
        _create_store(pipeline_config, specs, overwrite=True)
        recreated = xr.open_zarr(store, consolidated=False).compute()
        assert bool(np.isnan(recreated[VAR].values).all())

    def test_create_store_needs_a_dim(self, pipeline_config, tmp_path):
        specs = _output_specs(tmp_path / "weekly.zarr")
        with pytest.raises(CatenaValueError, match="--dim"):
            create_output_store(replace(pipeline_config, output_specs=specs))

    def test_create_store_takes_the_dim_from_the_subset_section(
        self, pipeline_config, tmp_path
    ):
        store = tmp_path / "weekly.zarr"
        parsed = replace(
            pipeline_config, output_specs=_output_specs(store), subset_spec=ROWS[0]
        )
        create_output_store(parsed, chunk=1)
        assert xr.open_zarr(store, consolidated=False).sizes["y"] == 2

    def test_create_store_rejects_an_output_without_the_dim(self, tmp_path):
        """A store partitioned over `site` cannot hold a total over sites."""
        from catena.config import load_config

        data = tmp_path / "sites.nc"
        xr.Dataset({"t": (("site",), np.arange(4.0))}).to_netcdf(data)
        config = tmp_path / "config.toml"
        config.write_text(
            f"""\
[[node]]
name = "total"
inputs = ["t"]
expression = "t.sum('site')"

[inputs.sites]
path = "{data}"
suffix = ""
vars = ["t"]

[outputs.out]
path = "{tmp_path / "out.zarr"}"
suffix = ""
vars = ["total"]
"""
        )
        with pytest.raises(CatenaValueError, match="'total' has no 'site'"):
            create_output_store(load_config(config), dim="site")

    def test_subset_over_a_dim_inputs_disagree_on_is_refused(self, pipeline_config):
        """Daily, weekly and monthly inputs have different lengths of `time`."""
        with pytest.raises(CatenaValueError, match=r"disagree on the size.*'time'"):
            load_inputs(
                pipeline_config.input_specs, subset_spec=SubsetSpec(0, 2, dim="time")
            )


# ---------------------------------------------------------------------------
# CLI end-to-end: create-store -> subset runs -> merge
# ---------------------------------------------------------------------------


def _config_text(synthetic_data_dir, out_path):
    # The weekly time axis is *derived*: nothing on disk has it, and the output
    # section's label matches no input section. The Zarr store learns it by probing
    # the pipeline.
    return f"""\
[[resample]]
vars = ["temperature"]
from = "daily"
to = "weekly"
freq = "7D"

[[node]]
name = "{VAR}_weekly"
inputs = ["temperature_weekly"]
expression = "temperature_weekly"

[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["temperature"]

[outputs.weekly]
path = "{out_path}"
vars = ["{VAR}"]
"""


class TestCLIParallelWorkflow:
    def test_zarr_create_run_merge(self, tmp_path, synthetic_data_dir, reference):
        from typer.testing import CliRunner

        from catena.cli.app import app

        store = tmp_path / "weekly.zarr"
        runner = CliRunner()
        config = tmp_path / "config.toml"
        config.write_text(_config_text(synthetic_data_dir, store))

        r = runner.invoke(
            app, ["gridded", "create-store", str(config), "--dim", "y", "--chunk", "1"]
        )
        assert r.exit_code == 0, r.output
        assert store.exists()

        r = runner.invoke(app, ["run", str(config), "--subset", "y:0:1"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["gridded", "merge", str(config)])
        assert r.exit_code != 0  # half the store is still unwritten

        r = runner.invoke(app, ["run", str(config), "--subset", "y:1:2"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["gridded", "merge", str(config)])
        assert r.exit_code == 0, r.output

        filled = xr.open_zarr(store, consolidated=False).compute()
        xr.testing.assert_allclose(filled, reference)

    def test_chunk_defaults_to_a_block_size_over_the_same_dim(self, tmp_path):
        from typer.testing import CliRunner

        from catena.cli.app import app

        data = tmp_path / "sites.nc"
        xr.Dataset(
            {"t": (("site",), np.arange(6.0), {"units": "K"})},
            coords={"site": np.arange(6)},
        ).to_netcdf(data)
        store = tmp_path / "out.zarr"
        config = tmp_path / "config.toml"
        config.write_text(
            f"""\
[[node]]
name = "doubled"
inputs = ["t"]
expression = "t * 2"

[inputs.sites]
path = "{data}"
suffix = ""
vars = ["t"]

[outputs.out]
path = "{store}"
suffix = ""
vars = ["doubled"]

[blocking]
block_size = 2
dim = "site"
"""
        )
        runner = CliRunner()
        r = runner.invoke(
            app, ["gridded", "create-store", str(config), "--dim", "site"]
        )
        assert r.exit_code == 0, r.output
        for subset in ("site:0:2", "site:2:6"):
            r = runner.invoke(app, ["run", str(config), "--subset", subset])
            assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["gridded", "merge", str(config)])
        assert r.exit_code == 0, r.output
        np.testing.assert_array_equal(
            xr.open_zarr(store, consolidated=False)["doubled"].values,
            np.arange(6.0) * 2,
        )


# ---------------------------------------------------------------------------
# True concurrency: independent processes region-writing simultaneously
# ---------------------------------------------------------------------------


class TestConcurrentZarrWrites:
    """Mirror the real deployment: N OS processes writing disjoint regions at once.

    The sequential tests validate correctness but not concurrency safety. Here we
    launch genuinely independent `catena run` processes (as `parallel catena
    run` would) against one shared store and assert no writes are lost.
    """

    def test_independent_processes_no_lost_writes(
        self, tmp_path, synthetic_data_dir, reference
    ):
        import subprocess
        import sys

        from typer.testing import CliRunner

        from catena.cli.app import app

        store = tmp_path / "weekly.zarr"
        config = tmp_path / "config.toml"
        config.write_text(_config_text(synthetic_data_dir, store))

        # One process per row of the 2x2 grid, each writing its own chunk.
        r = CliRunner().invoke(
            app, ["gridded", "create-store", str(config), "--dim", "y", "--chunk", "1"]
        )
        assert r.exit_code == 0, r.output

        # Launch every subset at once; collect output after they are all running.
        subsets = ["y:0:1", "y:1:2"]
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from catena.cli import main; main()",
                    "run",
                    str(config),
                    "--subset",
                    subset,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for subset in subsets
        ]
        outputs = [p.communicate()[0] for p in procs]
        for subset, proc, out in zip(subsets, procs, outputs, strict=True):
            assert proc.returncode == 0, f"{subset} failed:\n{out}"

        filled = xr.open_zarr(store, consolidated=False).compute()
        xr.testing.assert_allclose(filled, reference)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSubsetErrors:
    def test_region_write_missing_store_raises(
        self, pipeline_config, pipeline_driver, tmp_path
    ):
        specs = _output_specs(tmp_path / "missing.zarr")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _write_rows(pipeline_driver, pipeline_config, specs, rows=ROWS[:1])

    def test_misaligned_subset_raises(self, pipeline_config, pipeline_driver, tmp_path):
        specs = _output_specs(tmp_path / "weekly.zarr")
        _create_store(pipeline_config, specs, chunk=2)
        with pytest.raises(ValueError, match="aligned"):
            _write_rows(pipeline_driver, pipeline_config, specs, rows=ROWS[1:])
