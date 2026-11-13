"""Tests for the general fan-out [[node]] mechanism and node contract decls."""

import pytest
import xarray as xr
from xarray_annotated import Declared, Dims, Dtype, declarations_from_signature

from catena.build import build_driver
from catena.config import Config, NodeSpec, expand_node_entries
from catena.errors import CatenaValueError
from catena.nodegen import make_node_module


def _node(name, inputs, expression, **kw):
    return NodeSpec(
        name=name,
        inputs=list(inputs),
        expression=expression,
        import_path=None,
        function=None,
        **kw,
    )


class TestForEachExpansion:
    def test_substitutes_var_in_all_string_fields(self):
        entries = expand_node_entries(
            [
                {
                    "for_each": ["a", "b"],
                    "name": "{var}_out",
                    "inputs": ["{var}_in"],
                    "expression": "{var}_in * 2",
                }
            ]
        )
        assert [e["name"] for e in entries] == ["a_out", "b_out"]
        assert entries[0]["inputs"] == ["a_in"]
        assert entries[0]["expression"] == "a_in * 2"
        assert "for_each" not in entries[0]

    def test_empty_for_each_is_rejected(self):
        with pytest.raises(CatenaValueError, match=r"for_each = \[\]"):
            expand_node_entries(
                [{"name": "x", "inputs": [], "expression": "1", "for_each": []}]
            )

    def test_entry_without_for_each_passes_through(self):
        entries = expand_node_entries([{"name": "x", "inputs": [], "expression": "1"}])
        assert len(entries) == 1
        assert entries[0]["name"] == "x"

    def test_config_fan_out_generates_and_runs(self):
        parsed = Config(
            {
                "node": [
                    {
                        "for_each": ["a", "b"],
                        "name": "{var}_doubled",
                        "inputs": ["{var}"],
                        "expression": "{var} * 2",
                    }
                ]
            }
        ).parse()
        assert {s.name for s in parsed.node_specs} == {
            "a_doubled",
            "b_doubled",
        }
        dr = build_driver(
            parsed.modules, parsed.driver_config, node_specs=parsed.node_specs
        )
        out = dr.execute(["a_doubled"], inputs={"a": xr.DataArray([1.0, 2.0])})
        assert list(out["a_doubled"].values) == [2.0, 4.0]

    def test_fan_out_name_collision_raises(self):

        config = Config(
            {
                "node": [
                    {
                        "for_each": ["a", "a"],
                        "name": "{var}_x",
                        "inputs": ["{var}"],
                        "expression": "{var}",
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="Duplicate node name"):
            config.parse()


class TestSchemaOnNode:
    def test_schema_markers_annotated_and_decorated(self):
        # annotate() builds the schema return contract and check_annotations is applied,
        # so both markers are readable off the built node's signature.
        mod = make_node_module(
            [_node("f", ["a"], "a", dims=["time", "x"], dtype="float64")]
        )
        _, out = declarations_from_signature(mod.f)
        # annotate() emits markers in a fixed order (unit, dims, dtype, coords).
        assert out == Declared(dims=Dims("time", "x"), dtype=Dtype("float64"))

    def test_declared_dims_readable_from_generated_signature(self):
        mod = make_node_module([_node("f", ["a"], "a", dims=["time", "x"])])
        _, out = declarations_from_signature(mod.f)
        assert out == Declared(dims=Dims("time", "x"))

    def test_config_parses_and_validates_dims_dtype(self):
        spec = (
            Config(
                {
                    "node": [
                        {
                            "name": "f",
                            "inputs": ["a"],
                            "expression": "a",
                            "dims": ["time", "x"],
                            "dtype": "float64",
                        }
                    ]
                }
            )
            .parse()
            .node_specs[0]
        )
        assert spec.dims == ["time", "x"]
        assert spec.dtype == "float64"

    def test_invalid_dtype_raises(self):

        config = Config(
            {
                "node": [
                    {"name": "f", "inputs": ["a"], "expression": "a", "dtype": "nope"}
                ]
            }
        )
        with pytest.raises(ValueError, match="invalid dtype"):
            config.parse()


class TestPerVariableContracts:
    """A contract key given as a table keyed by variable, under fan-out."""

    def _expand(self, **contract):
        return expand_node_entries(
            [
                {
                    "for_each": ["a", "b"],
                    "name": "{var}_out",
                    "inputs": ["{var}"],
                    "expression": "{var}",
                    **contract,
                }
            ]
        )

    def test_each_node_gets_its_own_value(self):
        a, b = self._expand(units={"a": "K", "b": "mm"})
        assert (a["units"], b["units"]) == ("K", "mm")

    def test_a_plain_value_applies_to_every_node(self):
        a, b = self._expand(dims=["time"])
        assert a["dims"] == b["dims"] == ["time"]

    def test_a_variable_left_out_declares_nothing(self):
        a, b = self._expand(units={"a": "K"})
        assert a["units"] == "K"
        assert "units" not in b

    def test_an_unknown_variable_is_rejected(self):
        with pytest.raises(CatenaValueError, match=r"units names \['c'\]"):
            self._expand(units={"a": "K", "c": "mm"})

    def test_a_table_without_fan_out_is_rejected(self):
        with pytest.raises(CatenaValueError, match="needs 'for_each'"):
            expand_node_entries(
                [{"name": "x", "inputs": [], "expression": "1", "units": {"x": "K"}}]
            )

    def test_per_variable_units_reach_the_generated_nodes(self):
        specs = (
            Config(
                {
                    "node": [
                        {
                            "for_each": ["a", "b"],
                            "name": "{var}_out",
                            "inputs": ["{var}"],
                            "expression": "{var}",
                            "units": {"a": "K", "b": "mm"},
                        }
                    ]
                }
            )
            .parse()
            .node_specs
        )
        assert {s.name: s.units for s in specs} == {"a_out": "K", "b_out": "mm"}


class TestUnknownNodeKeys:
    @pytest.mark.parametrize("key", ["passthrough", "unitz"])
    def test_unknown_key_is_rejected(self, key):
        config = Config(
            {"node": [{"name": "f", "inputs": ["a"], "expression": "a", key: True}]}
        )
        with pytest.raises(CatenaValueError, match=f"unrecognised key.*{key}"):
            config.parse()
