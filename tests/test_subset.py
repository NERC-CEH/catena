"""Tests for the [subset] spatial pixel slicing feature."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from catena.build import build_driver
from catena.config import Config, IOSpec, SubsetSpec
from catena.errors import CatenaValueError
from catena.io import get_final_vars, load_inputs

_FINAL_VARS = get_final_vars({"weekly": IOSpec(path="", vars=["mean_temperature"])})


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
        assert "pixel" in str(exc.value)


# ---------------------------------------------------------------------------
# Correctness: subsets reconstruct the full inputs
# ---------------------------------------------------------------------------


class TestSubsetCorrectness:
    """Two complementary subsets concatenate back to the full inputs."""

    def test_pixel_coords_correct(self, pipeline_config, pipeline_inputs):
        """Each subset contains the right pixel coordinates."""
        spec_a = SubsetSpec(start=0, stop=2, dim="pixel")
        spec_b = SubsetSpec(start=2, stop=4, dim="pixel")

        inputs_a = load_inputs(pipeline_config.input_specs, subset_spec=spec_a)
        inputs_b = load_inputs(pipeline_config.input_specs, subset_spec=spec_b)

        for name, full_val in pipeline_inputs.items():
            if not (isinstance(full_val, xr.DataArray) and "pixel" in full_val.dims):
                continue
            combined = xr.concat([inputs_a[name], inputs_b[name]], dim="pixel")
            xr.testing.assert_identical(combined, full_val)

    def test_no_pixel_inputs_unaffected(self, pipeline_config, pipeline_inputs):
        """Inputs without the partition dim pass through unchanged."""
        spec = SubsetSpec(start=0, stop=2, dim="pixel")
        inputs_sub = load_inputs(pipeline_config.input_specs, subset_spec=spec)

        for name, full_val in pipeline_inputs.items():
            if "pixel" in full_val.dims:
                continue
            xr.testing.assert_identical(inputs_sub[name], full_val)


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

        inputs_a = load_inputs(
            pipeline_config.input_specs, subset_spec=SubsetSpec(0, 2, dim="pixel")
        )
        inputs_b = load_inputs(
            pipeline_config.input_specs, subset_spec=SubsetSpec(2, 4, dim="pixel")
        )
        result_a = dr.execute(_FINAL_VARS, inputs=inputs_a)  # type: ignore[reportArgumentType]
        result_b = dr.execute(_FINAL_VARS, inputs=inputs_b)  # type: ignore[reportArgumentType]

        for var in _FINAL_VARS:
            combined = xr.concat([result_a[var], result_b[var]], dim="pixel")
            xr.testing.assert_identical(combined, reference[var])


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


class TestCLISubset:
    """catena run with [subset] exits zero and writes only the subset pixels."""

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
stop = 2
dim = "pixel"
"""
        p = tmp_path / "config.toml"
        p.write_text(content)
        return p, out_path

    def _subset_path(self, out_path):
        """The auto-suffixed NetCDF path a [subset] run writes to."""
        from catena.gridded.io import subset_suffix

        spec = SubsetSpec(0, 2, dim="pixel")
        return out_path.with_name(
            f"{out_path.stem}{subset_suffix(spec)}{out_path.suffix}"
        )

    def test_exits_zero(self, subset_config_toml):
        from typer.testing import CliRunner

        from catena.cli.app import app

        config_path, _ = subset_config_toml
        result = CliRunner().invoke(app, ["run", str(config_path)])
        assert result.exit_code == 0, result.output

    def test_output_is_stacked_subset(self, subset_config_toml):
        """A subset run writes a uniquely-suffixed, stacked (pixel) file."""
        from typer.testing import CliRunner

        from catena.cli.app import app

        config_path, out_path = subset_config_toml
        CliRunner().invoke(app, ["run", str(config_path)])

        # The un-suffixed path is never written; the suffixed one holds 2 pixels.
        assert not out_path.exists()
        ds = xr.open_dataset(self._subset_path(out_path))
        assert ds.sizes["pixel"] == 2

    def test_output_values_match_full_run(self, subset_config_toml):
        from typer.testing import CliRunner

        from catena.cli.app import app
        from catena.config import load_config
        from catena.io import get_outputs

        config_path, out_path = subset_config_toml
        CliRunner().invoke(app, ["run", str(config_path)])
        subset_ds = xr.open_dataset(self._subset_path(out_path))

        parsed = load_config(config_path)
        parsed.subset_spec = None
        dr = build_driver(
            parsed.modules, parsed.driver_config, node_specs=parsed.node_specs
        )
        full_inputs = load_inputs(parsed.input_specs)
        final_vars: list[Any] = get_final_vars(parsed.output_specs)
        full_results = dr.execute(final_vars, inputs=full_inputs)  # type: ignore[reportArgumentType]
        ref_ds = get_outputs(full_results, parsed.output_specs, stacked=True)["weekly"]

        for var in subset_ds.data_vars:
            np.testing.assert_allclose(
                subset_ds[var].transpose("time", "pixel").values,
                ref_ds[var].transpose("time", "pixel").isel(pixel=slice(0, 2)).values,
                equal_nan=True,
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
        for subset in ("pixel:0:2", "pixel:2:4"):
            result = runner.invoke(app, ["run", str(config), "--subset", subset])
            assert result.exit_code == 0, result.output
        assert not out_path.exists()
        result = runner.invoke(app, ["gridded", "merge", str(config)])
        assert result.exit_code == 0, result.output
        merged = xr.open_dataset(out_path).load()
        out_path.unlink()

        run(config)
        xr.testing.assert_allclose(merged, xr.open_dataset(out_path))

    def test_run_argument_replaces_the_config_section(self, shared_config):
        from catena import run

        config, _ = shared_config
        with open(config, "a") as f:
            f.write('\n[subset]\nstart = 0\nstop = 1\ndim = "pixel"\n')
        report = run(config, subset=SubsetSpec(2, 4, dim="pixel"))
        (written,) = report.written
        assert written.path.name == "outputs_pixel2-4.nc"

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
        result = runner.invoke(app, ["gridded", "merge", str(config), "--dim", "site"])
        assert result.exit_code == 0, result.output
        np.testing.assert_array_equal(
            xr.open_dataset(out_path)["doubled"].values, np.arange(4.0) * 2
        )
