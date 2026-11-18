"""Tests for [subset]: running one slice of a dimension of the input files."""

import numpy as np
import pytest
import xarray as xr

from catena.build import build_driver
from catena.config import Config, IOSpec, SubsetSpec
from catena.errors import CatenaValueError
from catena.io import get_final_vars, load_inputs

_FINAL_VARS = get_final_vars({"weekly": IOSpec(path="", vars=["mean_temperature"])})

# The session grid is 2x2, so `y` has two rows.
_ROW_A = SubsetSpec(start=0, stop=1, dim="y")
_ROW_B = SubsetSpec(start=1, stop=2, dim="y")


# ---------------------------------------------------------------------------
# SubsetSpec config validation
# ---------------------------------------------------------------------------


class TestSubsetSpecValidation:
    def test_valid(self):
        spec = SubsetSpec.from_config({"start": 0, "stop": 500, "dim": "pixel"})
        assert spec.start == 0
        assert spec.stop == 500

    def test_dim_is_required(self):
        with pytest.raises(ValueError, match="'dim'"):
            SubsetSpec.from_config({"start": 0, "stop": 500})

    def test_dim_can_be_any_dimension(self):
        spec = SubsetSpec.from_config({"start": 0, "stop": 10, "dim": "location"})
        assert spec.dim == "location"

    def test_empty_dim_raises(self):
        with pytest.raises(ValueError, match="'dim'"):
            SubsetSpec.from_config({"start": 0, "stop": 10, "dim": ""})

    def test_missing_start_raises(self):
        with pytest.raises((ValueError, TypeError)):
            SubsetSpec.from_config({"stop": 500})

    def test_missing_stop_raises(self):
        with pytest.raises((ValueError, TypeError)):
            SubsetSpec.from_config({"start": 0})

    def test_negative_start_raises(self):
        with pytest.raises(ValueError, match="'start'"):
            SubsetSpec.from_config({"start": -1, "stop": 500})

    def test_negative_stop_raises(self):
        with pytest.raises(ValueError, match="'stop'"):
            SubsetSpec.from_config({"start": 0, "stop": -1})

    def test_stop_equal_to_start_raises(self):
        with pytest.raises(ValueError, match="'stop'"):
            SubsetSpec.from_config({"start": 5, "stop": 5})

    def test_stop_less_than_start_raises(self):
        with pytest.raises(ValueError, match="'stop'"):
            SubsetSpec.from_config({"start": 5, "stop": 3})

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="'start'"):
            SubsetSpec.from_config({"start": 0.5, "stop": 500})

    def test_parsed_from_toml(self):
        parsed = Config.loads(
            '[subset]\nstart = 0\nstop = 100\ndim = "pixel"\n'
        ).parse()
        assert parsed.subset_spec == SubsetSpec(start=0, stop=100, dim="pixel")

    def test_absent_section_gives_none(self):
        parsed = Config.loads("").parse()
        assert parsed.subset_spec is None


class TestMissingDim:
    def test_dim_on_no_input_raises_and_lists_dims(self, pipeline_config):
        spec = SubsetSpec(start=0, stop=2, dim="locaton")
        with pytest.raises(CatenaValueError, match=r"\[subset\] dim 'locaton'") as exc:
            load_inputs(pipeline_config.input_specs, subset_spec=spec)
        # The dimensions listed are the files' own, before stacking.
        assert "'y'" in str(exc.value)
        assert "pixel" not in str(exc.value)


# ---------------------------------------------------------------------------
# Correctness: subsets reconstruct the full inputs
# ---------------------------------------------------------------------------


class TestSubsetCorrectness:
    """Two complementary subsets concatenate back to the full inputs."""

    def test_rows_concatenate_to_the_full_inputs(
        self, pipeline_config, pipeline_inputs
    ):
        """Each subset holds its own rows of the grid."""
        inputs_a = load_inputs(pipeline_config.input_specs, subset_spec=_ROW_A)
        inputs_b = load_inputs(pipeline_config.input_specs, subset_spec=_ROW_B)

        for name, full_val in pipeline_inputs.items():
            combined = xr.concat([inputs_a[name], inputs_b[name]], dim="y")
            xr.testing.assert_identical(combined, full_val)

    def test_inputs_without_the_dim_are_unaffected(self, tmp_path):
        """A file with no `y` passes through whole."""
        from catena.io import subset_datasets

        series = xr.Dataset({"v": ("time", np.arange(3.0))})
        grid = xr.Dataset({"w": (("y", "x"), np.zeros((2, 2)))})
        out = subset_datasets({"series": series, "grid": grid}, _ROW_A)
        xr.testing.assert_identical(out["series"], series)
        assert out["grid"].sizes["y"] == 1


# ---------------------------------------------------------------------------
# Pipeline result: two subsets concatenate to the full result
# ---------------------------------------------------------------------------


class TestSubsetPipelineResult:
    """Running on subsets and concatenating reproduces the unsubsetted result."""

    def test_two_halves_match_full(self, pipeline_config, pipeline_inputs):
        dr = build_driver(
            pipeline_config.modules,
            pipeline_config.driver_config,
            node_specs=pipeline_config.node_specs,
        )
        reference = dr.execute(_FINAL_VARS, inputs=pipeline_inputs)  # type: ignore[reportArgumentType]

        inputs_a = load_inputs(pipeline_config.input_specs, subset_spec=_ROW_A)
        inputs_b = load_inputs(pipeline_config.input_specs, subset_spec=_ROW_B)
        result_a = dr.execute(_FINAL_VARS, inputs=inputs_a)  # type: ignore[reportArgumentType]
        result_b = dr.execute(_FINAL_VARS, inputs=inputs_b)  # type: ignore[reportArgumentType]

        for var in _FINAL_VARS:
            combined = xr.concat([result_a[var], result_b[var]], dim="y")
            xr.testing.assert_identical(combined, reference[var])


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


class TestCLISubset:
    """catena run with [subset] exits zero and writes only the subset's rows."""

    @pytest.fixture
    def subset_config_toml(self, tmp_path, synthetic_data_dir):
        out_path = tmp_path / "outputs.nc"
        # Load only what the pipeline consumes (temperature_daily), so a real
        # `catena run` stays free of the unused-input WiringWarning. Broad
        # multi-section configs are exercised by the parser tests instead.
        content = f"""\
[[node]]
name = "mean_temperature_weekly"
inputs = ["temperature_daily"]
expression = "temperature_daily.resample(time='7D').mean()"

[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["temperature"]

[outputs.weekly]
path = "{out_path}"
vars = ["mean_temperature"]

[subset]
start = 0
stop = 1
dim = "y"
"""
        p = tmp_path / "config.toml"
        p.write_text(content)
        return p, out_path

    def _subset_path(self, out_path):
        """The auto-suffixed NetCDF path a [subset] run writes to."""
        from catena.subsets import subset_suffix

        return out_path.with_name(
            f"{out_path.stem}{subset_suffix(_ROW_A)}{out_path.suffix}"
        )

    def test_exits_zero(self, subset_config_toml):
        from typer.testing import CliRunner

        from catena.cli.app import app

        config_path, _ = subset_config_toml
        result = CliRunner().invoke(app, ["run", str(config_path)])
        assert result.exit_code == 0, result.output

    def test_output_is_one_row_of_the_grid(self, subset_config_toml):
        """A subset run writes a uniquely-suffixed file in the usual grid layout."""
        from typer.testing import CliRunner

        from catena.cli.app import app

        config_path, out_path = subset_config_toml
        CliRunner().invoke(app, ["run", str(config_path)])

        # The un-suffixed path is never written; the suffixed one holds one row.
        assert not out_path.exists()
        ds = xr.open_dataset(self._subset_path(out_path))
        assert dict(ds.sizes) | {"time": 0} == {"time": 0, "y": 1, "x": 2}

    def test_output_values_match_full_run(self, subset_config_toml, tmp_path):
        from typer.testing import CliRunner

        from catena import run
        from catena.cli.app import app
        from catena.config import load_config

        config_path, out_path = subset_config_toml
        CliRunner().invoke(app, ["run", str(config_path)])
        subset_ds = xr.open_dataset(self._subset_path(out_path))

        parsed = load_config(config_path)
        parsed.subset_spec = None
        run(parsed)
        xr.testing.assert_allclose(
            subset_ds, xr.open_dataset(out_path).isel(y=slice(0, 1))
        )


class TestSubsetFromString:
    def test_valid(self):
        assert SubsetSpec.from_string("pixel:0:500") == SubsetSpec(0, 500, dim="pixel")

    @pytest.mark.parametrize("text", ["0:500", "pixel:0", "pixel:a:5", "pixel:-1:5"])
    def test_malformed_raises(self, text):
        with pytest.raises(CatenaValueError, match="DIM:START:STOP"):
            SubsetSpec.from_string(text)

    def test_bounds_are_validated_as_in_the_config(self):
        with pytest.raises(CatenaValueError, match="must be greater than"):
            SubsetSpec.from_string("pixel:5:5")


class TestSubsetOverride:
    """One shared config, with each run's subset given on the command line."""

    @pytest.fixture
    def shared_config(self, tmp_path, synthetic_data_dir):
        out_path = tmp_path / "outputs.nc"
        config = tmp_path / "config.toml"
        config.write_text(
            f"""\
[[node]]
name = "mean_temperature_weekly"
inputs = ["temperature_daily"]
expression = "temperature_daily.resample(time='7D').mean()"

[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["temperature"]

[outputs.weekly]
path = "{out_path}"
vars = ["mean_temperature"]
"""
        )
        return config, out_path

    def test_cli_runs_merge_to_the_full_result(self, shared_config, tmp_path):
        from typer.testing import CliRunner

        from catena import run
        from catena.cli.app import app

        config, out_path = shared_config
        runner = CliRunner()
        for subset in ("y:0:1", "y:1:2"):
            result = runner.invoke(app, ["run", str(config), "--subset", subset])
            assert result.exit_code == 0, result.output
        assert not out_path.exists()
        result = runner.invoke(app, ["subset", "merge", str(config), "--dim", "y"])
        assert result.exit_code == 0, result.output
        merged = xr.open_dataset(out_path).load()
        out_path.unlink()

        run(config)
        xr.testing.assert_allclose(merged, xr.open_dataset(out_path))

    def test_run_argument_replaces_the_config_section(self, shared_config):
        from catena import run

        config, _ = shared_config
        with open(config, "a") as f:
            f.write('\n[subset]\nstart = 0\nstop = 1\ndim = "y"\n')
        report = run(config, subset=_ROW_B)
        (written,) = report.written
        assert written.path.name == "outputs_y1-2.nc"

    def test_dry_run_takes_the_same_argument(self, shared_config):
        from catena import dry_run

        config, _ = shared_config
        with pytest.raises(CatenaValueError, match="dim 'locaton'"):
            dry_run(config, subset=SubsetSpec(0, 2, dim="locaton"))

    def test_malformed_cli_subset_is_a_usage_error(self, shared_config):
        from typer.testing import CliRunner

        from catena.cli.app import app

        config, _ = shared_config
        result = CliRunner().invoke(app, ["run", str(config), "--subset", "0:2"])
        assert result.exit_code != 0
        assert "DIM:START:STOP" in result.output

    def test_merge_takes_the_dim_when_the_config_has_no_subset(self, tmp_path):
        """A non-pixel partition given by --subset must be named to `merge` too."""
        from typer.testing import CliRunner

        from catena.cli.app import app

        data = tmp_path / "sites.nc"
        xr.Dataset(
            {"t": (("site",), np.arange(4.0), {"units": "K"})},
            coords={"site": np.arange(4)},
        ).to_netcdf(data)
        out_path = tmp_path / "out.nc"
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
path = "{out_path}"
suffix = ""
vars = ["doubled"]
"""
        )
        runner = CliRunner()
        for subset in ("site:0:2", "site:2:4"):
            result = runner.invoke(app, ["run", str(config), "--subset", subset])
            assert result.exit_code == 0, result.output
        result = runner.invoke(app, ["subset", "merge", str(config), "--dim", "site"])
        assert result.exit_code == 0, result.output
        np.testing.assert_array_equal(
            xr.open_dataset(out_path)["doubled"].values, np.arange(4.0) * 2
        )
