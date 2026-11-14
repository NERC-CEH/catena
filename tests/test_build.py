"""Tests for catena.build — build_driver wiring."""

import pytest
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
