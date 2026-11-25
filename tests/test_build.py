"""Tests for catena.build — build_driver wiring."""

import numpy as np
import pytest
import xarray as xr
from hamilton import driver

from catena.build import build_driver
from catena.config import Config, NodeSpec


class TestBuildDriverReturnType:
    def test_empty_module_list(self):
        assert isinstance(build_driver([], {}), driver.Driver)

    def test_resample_preset_builds(self):
        parsed = Config(
            {
                "resample": [
                    {"vars": ["x"], "from": "daily", "to": "weekly", "freq": "7D"}
                ]
            }
        ).parse()
        assert isinstance(
            build_driver(
                parsed.modules, parsed.driver_config, node_specs=parsed.node_specs
            ),
            driver.Driver,
        )

    def test_node_module(self):
        specs = [
            NodeSpec(
                name="y",
                inputs=["x"],
                expression="x * 2",
                import_path=None,
                function=None,
            )
        ]
        assert isinstance(build_driver(["node"], {}, node_specs=specs), driver.Driver)

    def test_allow_module_overrides_flag(self):
        specs = [
            NodeSpec(
                name="y",
                inputs=["x"],
                expression="x * 2",
                import_path=None,
                function=None,
            )
        ]
        dr = build_driver(["node"], {}, allow_module_overrides=True, node_specs=specs)
        assert isinstance(dr, driver.Driver)

    def test_importable_custom_module(self):
        """A dotted import path to a real module loads like any built-in."""
        dr = build_driver(["catena.transforms"], {})
        assert isinstance(dr, driver.Driver)


class TestBuildDriverErrors:
    def test_non_importable_custom_module_raises(self):
        with pytest.raises(ValueError, match="Could not import"):
            build_driver(["does_not_exist_pkg.module"], {})

    def test_node_module_without_specs_raises(self):
        with pytest.raises(ValueError, match="no node_specs"):
            build_driver(["node"], {})


class TestBuildDriverDoesNotMutateConfig:
    """build_driver must not write Hamilton's power-user flag into the caller's dict."""

    def test_build_driver_does_not_mutate_config(self):
        config = {"threshold": 0.5}
        pristine = dict(config)
        build_driver([], config)
        assert config == pristine


class TestBuildDriverDAGStructure:
    """Verify that the built driver exposes expected DAG nodes."""

    def test_node_driver_exposes_generated_node(self):
        specs = [
            NodeSpec(
                name="my_var",
                inputs=["a", "b"],
                expression="a + b",
                import_path=None,
                function=None,
            )
        ]
        dr = build_driver(["node"], {}, node_specs=specs)
        available = {v.name for v in dr.list_available_variables()}
        assert "my_var" in available


class TestBuildDriverFromConfig:
    def test_resolves_a_relative_module_against_the_config(self, tmp_path):
        """`base` is passed through, which a hand-written `build_driver` call omits."""
        from catena import build_driver_from_config
        from catena.config import load_config

        (tmp_path / "nodes.py").write_text("def one() -> int:\n    return 1\n")
        config = tmp_path / "config.toml"
        config.write_text('[model]\n_import_path = "nodes.py"\n')
        dr = build_driver_from_config(load_config(config))
        assert dr.execute(["one"]) == {"one": 1}


class TestExactAlignment:
    """Nodes run under `arithmetic_join="exact"`, so a coordinate mismatch raises."""

    @pytest.fixture
    def dr(self):
        spec = NodeSpec(
            name="total",
            inputs=["a", "b"],
            expression="a + b",
            import_path=None,
            function=None,
        )
        return build_driver(["node"], {}, node_specs=[spec])

    @staticmethod
    def _arrays(offset: float):
        a = xr.DataArray(np.ones(3), dims="y", coords={"y": [0.0, 1.0, 2.0]})
        return {"a": a, "b": a.assign_coords(y=a.y + offset)}

    def test_mismatched_coordinates_raise(self, dr):
        with pytest.raises(xr.AlignmentError, match="'y'"):
            dr.execute(["total"], inputs=self._arrays(1e-9))

    def test_matching_coordinates_run(self, dr):
        result = dr.execute(["total"], inputs=self._arrays(0.0))
        assert result["total"].sizes == {"y": 3}

    def test_broadcasting_a_new_dimension_is_unaffected(self, dr):
        inputs = self._arrays(0.0)
        inputs["b"] = xr.DataArray(np.ones(2), dims="time")
        assert dr.execute(["total"], inputs=inputs)["total"].sizes == {
            "y": 3,
            "time": 2,
        }

    def test_option_is_restored_after_a_failed_run(self, dr):
        before = xr.get_options()["arithmetic_join"]
        with pytest.raises(xr.AlignmentError):
            dr.execute(["total"], inputs=self._arrays(1e-9))
        assert xr.get_options()["arithmetic_join"] == before
