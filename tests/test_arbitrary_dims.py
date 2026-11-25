"""Phase 2: non-temporal, non-geospatial pipelines with arbitrary dims/labels.

These exercise the generalised I/O layer: input section labels need not be known
frequencies, variables may carry arbitrary dimensions (no `time`, no CRS), and
the frequency-suffix naming is opt-out.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from catena.build import build_driver, build_driver_from_config
from catena.config import IOSpec, SubsetSpec, load_config
from catena.errors import CatenaValueError
from catena.io import effective_suffix, get_final_vars, get_outputs, load_inputs


def _write_scene(path) -> None:
    """A (location, band) reflectance cube — no time, no pixel, no CRS."""
    ds = xr.Dataset(
        {"reflectance": (("location", "band"), np.linspace(0, 1, 12).reshape(4, 3))},
        coords={"location": np.arange(4), "band": ["red", "nir", "swir"]},
    )
    ds.to_netcdf(path, engine="netcdf4")


def test_effective_suffix_rules():
    # The default suffix is `_<label>` for every section label...
    assert effective_suffix("daily", IOSpec(path="", vars=[])) == "_daily"
    assert effective_suffix("static", IOSpec(path="", vars=[])) == "_static"
    assert effective_suffix("scene", IOSpec(path="", vars=[])) == "_scene"
    # ...and bare names for the flat [inputs]/[outputs] form
    assert effective_suffix("inputs", IOSpec(path="", vars=[])) == ""
    assert effective_suffix("outputs", IOSpec(path="", vars=[])) == ""
    assert effective_suffix("inputs", IOSpec(path="", vars=[], suffix="_x")) == "_x"
    # explicit override wins, including bare names on any label
    assert effective_suffix("daily", IOSpec(path="", vars=[], suffix="")) == ""
    assert effective_suffix("x", IOSpec(path="", vars=[], suffix="_y")) == "_y"


class TestArbitraryDimLoading:
    def test_arbitrary_label_and_dims(self, tmp_path):
        _write_scene(tmp_path / "scene.nc")
        specs = {"scene": IOSpec(path=str(tmp_path / "scene.nc"), vars=["reflectance"])}
        inputs = load_inputs(specs)
        # arbitrary label -> suffix; arbitrary dims preserved; no time/pixel/grid
        da = inputs["reflectance_scene"]
        assert da.dims == ("location", "band")

    def test_bare_suffix_for_non_static_label(self, tmp_path):
        ds = xr.Dataset({"threshold": ((), 0.5)})
        ds.to_netcdf(tmp_path / "params.nc", engine="netcdf4")
        specs = {
            "params": IOSpec(
                path=str(tmp_path / "params.nc"), vars=["threshold"], suffix=""
            )
        }
        inputs = load_inputs(specs)
        assert "threshold" in inputs  # bare, not "threshold_params"


class TestNonTemporalPipeline:
    def test_end_to_end(self, tmp_path):
        _write_scene(tmp_path / "scene.nc")
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[[node]]\n"
            'name = "mean_reflectance_scene"\n'
            'inputs = ["reflectance_scene"]\n'
            "expression = \"reflectance_scene.mean('band')\"\n\n"
            "[inputs.scene]\n"
            f'path = "{tmp_path / "scene.nc"}"\n'
            'vars = ["reflectance"]\n\n'
            "[outputs.scene]\n"
            f'path = "{tmp_path / "out.nc"}"\n'
            'vars = ["mean_reflectance"]\n'
        )
        parsed = load_config(cfg)
        inputs = load_inputs(parsed.input_specs)
        dr = build_driver(
            parsed.modules, parsed.driver_config, node_specs=parsed.node_specs
        )
        final_vars = get_final_vars(parsed.output_specs)
        assert final_vars == ["mean_reflectance_scene"]
        results = dr.execute(final_vars, inputs=inputs)  # type: ignore[reportArgumentType]
        out = get_outputs(results, parsed.output_specs)["scene"]
        assert out["mean_reflectance"].dims == ("location",)
        assert out.sizes["location"] == 4


class TestBlockingArbitraryDim:
    def test_blocked_matches_unblocked_over_location(self, tmp_path):
        from catena.blocking import execute_blocked
        from catena.config import BlockingSpec

        _write_scene(tmp_path / "scene.nc")
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[[node]]\n"
            'name = "scaled_scene"\n'
            'inputs = ["reflectance_scene"]\n'
            'expression = "reflectance_scene * 2.0"\n\n'
            "[inputs.scene]\n"
            f'path = "{tmp_path / "scene.nc"}"\n'
            'vars = ["reflectance"]\n'
        )
        parsed = load_config(cfg)
        inputs = load_inputs(parsed.input_specs)
        dr = build_driver(
            parsed.modules, parsed.driver_config, node_specs=parsed.node_specs
        )
        final = ["scaled_scene"]

        ref = dr.execute(final, inputs=inputs)  # type: ignore[reportArgumentType]
        spec = BlockingSpec(block_size=2, dim="location")
        blocked = execute_blocked(dr, inputs, final, spec)
        xr.testing.assert_identical(blocked["scaled_scene"], ref["scaled_scene"])


class TestSubsetArbitraryDim:
    """[subset] partitions any dimension of the input files."""

    def _config(self, tmp_path, subset: str = "") -> Path:
        _write_scene(tmp_path / "scene.nc")
        cfg = tmp_path / f"config{'_subset' if subset else ''}.toml"
        cfg.write_text(
            "[[node]]\n"
            'name = "scaled_scene"\n'
            'inputs = ["reflectance_scene"]\n'
            'expression = "reflectance_scene * 2.0"\n\n'
            "[inputs.scene]\n"
            f'path = "{tmp_path / "scene.nc"}"\n'
            'vars = ["reflectance"]\n\n'
            "[outputs.scene]\n"
            f'path = "{tmp_path / "out.nc"}"\n'
            'vars = ["scaled"]\n' + subset
        )
        return cfg

    def test_subset_over_non_pixel_dim(self, tmp_path):
        """Subset parts over `location` merge back to the full run, exactly."""
        from catena.io import save_outputs
        from catena.subsets import merge_subset_outputs

        # Reference: the whole thing in one go.
        parsed = load_config(self._config(tmp_path))
        dr = build_driver(
            parsed.modules, parsed.driver_config, node_specs=parsed.node_specs
        )
        final = get_final_vars(parsed.output_specs)
        reference = get_outputs(
            dr.execute(final, inputs=load_inputs(parsed.input_specs)),  # type: ignore[reportArgumentType]
            parsed.output_specs,
        )["scene"]

        # Two independent "processes", each over half the 4 locations.
        for start, stop in ((0, 2), (2, 4)):
            spec = SubsetSpec(start=start, stop=stop, dim="location")
            inputs = load_inputs(parsed.input_specs, subset_spec=spec)
            assert inputs["reflectance_scene"].sizes["location"] == 2
            results = dr.execute(final, inputs=inputs)  # type: ignore[reportArgumentType]
            save_outputs(
                get_outputs(results, parsed.output_specs),
                parsed.output_specs,
                subset_spec=spec,
            )

        parts = sorted(p.name for p in tmp_path.glob("out_location*.nc"))
        assert parts == ["out_location0-2.nc", "out_location2-4.nc"]

        merge_subset_outputs(parsed.output_specs, dim="location")
        merged = xr.open_dataset(tmp_path / "out.nc")
        xr.testing.assert_allclose(
            merged["scaled"].transpose(*reference["scaled"].dims),
            reference["scaled"],
        )

    def test_zarr_store_partitioned_over_non_pixel_dim(self, tmp_path):
        """A shared Zarr store works over `location` as it does over a grid row."""
        from catena import run
        from catena.subsets import create_output_store, merge_subset_outputs

        cfg = self._config(
            tmp_path, subset='\n[subset]\nstart = 0\nstop = 2\ndim = "location"\n'
        )
        parsed = load_config(cfg)
        store = tmp_path / "out.zarr"
        zarr_specs = {"scene": IOSpec(path=str(store), vars=["scaled"])}
        parsed = replace(parsed, output_specs=zarr_specs, subset_spec=None)

        create_output_store(parsed, dim="location", chunk=2)
        for start, stop in ((0, 2), (2, 4)):
            run(parsed, subset=SubsetSpec(start=start, stop=stop, dim="location"))
        merge_subset_outputs(zarr_specs)

        reference = tmp_path / "reference.zarr"
        run(
            replace(
                parsed,
                output_specs={
                    "scene": replace(zarr_specs["scene"], path=str(reference))
                },
            )
        )
        xr.testing.assert_allclose(
            xr.open_zarr(store, consolidated=False).compute(),
            xr.open_zarr(reference, consolidated=False).compute(),
        )


class TestDimensionCreatedByANode:
    """A node can add a dimension no input has: here `levels` builds `level`.

    `profile` combines it with an input on `location`, so it carries both, and
    blocking and subset runs partition `location` while `level` is left whole.
    """

    @staticmethod
    def _config(tmp_path, outputs: str, extra: str = "") -> Path:
        xr.Dataset(
            {"field": ("location", np.arange(1.0, 5.0))},
            coords={"location": np.arange(4)},
        ).to_netcdf(tmp_path / "field.nc", engine="netcdf4")
        (tmp_path / "nodes.py").write_text(
            "import numpy as np\n"
            "import xarray as xr\n\n\n"
            "def levels(*, n_levels: int = 3) -> xr.DataArray:\n"
            "    depth = np.arange(n_levels) + 0.5\n"
            '    return xr.DataArray(depth, dims="level", coords={"level": depth})\n\n\n'
            "def profile(field: xr.DataArray, levels: xr.DataArray) -> xr.DataArray:\n"
            "    return field * levels\n"
        )
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[nodes]\n_import_path = "nodes.py"\n\n'
            f'[inputs]\npath = "{tmp_path / "field.nc"}"\nvars = ["field"]\n\n'
            f"[outputs]\n{outputs}\n{extra}"
        )
        return cfg

    def _netcdf_config(self, tmp_path, extra: str = "", vars_: str = '"profile"'):
        outputs = f'path = "{tmp_path / "out.nc"}"\nvars = [{vars_}]\n'
        return self._config(tmp_path, outputs, extra)

    def test_blocked_matches_unblocked(self, tmp_path):
        from catena.blocking import execute_blocked
        from catena.config import BlockingSpec

        parsed = load_config(self._netcdf_config(tmp_path))
        inputs = load_inputs(parsed.input_specs)
        dr = build_driver_from_config(parsed)
        ref = dr.execute(["profile"], inputs=inputs)  # type: ignore[reportArgumentType]
        blocked = execute_blocked(
            dr, inputs, ["profile"], BlockingSpec(block_size=3, dim="location")
        )
        assert ref["profile"].sizes == {"location": 4, "level": 3}
        xr.testing.assert_identical(blocked["profile"], ref["profile"])

    def test_zarr_subsets_merge_to_the_full_run(self, tmp_path):
        from catena import run
        from catena.subsets import create_output_store, merge_subset_outputs

        store = tmp_path / "out.zarr"
        parsed = load_config(
            self._config(tmp_path, f'path = "{store}"\nvars = ["profile"]\n')
        )
        create_output_store(parsed, dim="location", chunk=2)
        template = xr.open_zarr(store, consolidated=False)
        assert template["profile"].sizes == {"location": 4, "level": 3}

        for start, stop in ((0, 2), (2, 4)):
            run(parsed, subset=SubsetSpec(start=start, stop=stop, dim="location"))
        merge_subset_outputs(parsed.output_specs)

        reference = tmp_path / "reference.zarr"
        run(
            replace(
                parsed,
                output_specs={
                    "outputs": replace(
                        parsed.output_specs["outputs"], path=str(reference)
                    )
                },
            )
        )
        xr.testing.assert_identical(
            xr.open_zarr(store, consolidated=False).compute(),
            xr.open_zarr(reference, consolidated=False).compute(),
        )

    def test_blocking_refuses_the_dimension_node_as_an_output(self, tmp_path):
        from catena import run

        cfg = self._netcdf_config(
            tmp_path,
            extra='\n[blocking]\nblock_size = 2\ndim = "location"\n',
            vars_='"profile", "levels"',
        )
        with pytest.raises(CatenaValueError, match="recombine 'levels'"):
            run(cfg)

    def test_zarr_store_refuses_the_dimension_node_as_an_output(self, tmp_path):
        from catena.subsets import create_output_store

        parsed = load_config(
            self._config(
                tmp_path,
                f'path = "{tmp_path / "out.zarr"}"\nvars = ["profile", "levels"]\n',
            )
        )
        with pytest.raises(CatenaValueError, match="'levels' has no 'location'"):
            create_output_store(parsed, dim="location", chunk=2)
