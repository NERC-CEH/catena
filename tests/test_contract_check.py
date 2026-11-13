"""Tests for the build-time (static) DAG unit-consistency check."""

import sys
import types
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray_annotated as xra
from hamilton import driver
from hamilton.function_modifiers import extract_fields
from hamilton.settings import ENABLE_POWER_USER_MODE
from xarray_annotated.schema import Coords, Dims, Dtype

from catena.build import build_driver
from catena.config import NodeSpec, ResampleSpec
from catena.contract_check import (
    check_dag_contracts,
    check_input_contracts,
)
from catena.errors import CatenaValueError

_DAILY = xr.DataArray(
    [1.0, 2.0, 3.0],
    dims="time",
    coords={"time": pd.date_range("2025-01-01", periods=3, freq="D")},
    attrs={"units": "Pa"},
)


def _da():
    return xr.DataArray([1.0])


@pytest.fixture
def register():
    """Build Hamilton-scannable modules from functions and clean up afterwards.

    Hamilton only picks up a module's functions when they live in ``sys.modules``
    and their ``__module__`` matches the module name (mirrors ``node.py``).
    """
    names: list[str] = []

    def _make(name: str, *funcs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for fn in funcs:
            fn.__module__ = name
            setattr(mod, fn.__name__, fn)
        sys.modules[name] = mod
        names.append(name)
        return mod

    yield _make

    for name in names:
        sys.modules.pop(name, None)


def _build(*mods) -> driver.Driver:
    return (
        driver.Builder()
        .with_modules(*mods)
        .with_config({ENABLE_POWER_USER_MODE: True})
        .build()
    )


def _producer(unit: str):
    """A node producing ``gpp_weekly`` with the given declared output unit."""

    class Out(TypedDict):
        gpp_weekly: Annotated[xr.DataArray, xra.Unit(unit)]

    @extract_fields()
    @xra.check_annotations
    def producer() -> Out:  # type: ignore[valid-type]
        return {"gpp_weekly": _da()}

    return producer


def _consumer(unit: str, name: str = "consumer", in_name: str = "gpp_weekly"):
    """A node consuming ``in_name`` with the given declared input unit.

    Built via ``exec`` so the consumed parameter (= the upstream node name) and
    the function name are both dynamic. The output node is named ``f"{name}_out"``
    so multiple consumers can coexist in one graph without a name collision.
    """
    src = (
        "from typing import Annotated, TypedDict\n"
        "import xarray as xr\n"
        "from hamilton.function_modifiers import extract_fields\n"
        "import xarray_annotated as xra\n"
        f"class _Out(TypedDict):\n"
        f"    {name}_out: Annotated[xr.DataArray, xra.Unit('t ha-1')]\n"
        "@extract_fields()\n"
        "@xra.check_annotations\n"
        f"def {name}({in_name}: Annotated[xr.DataArray, xra.Unit({unit!r})]) -> _Out:\n"
        f"    return {{{name + '_out'!r}: {in_name}}}\n"
    )
    ns: dict = {}
    exec(src, ns)
    return ns[name]


def _bare_producer(unit: str, name: str = "flux"):
    """A *single-output* node producing ``name`` via a bare ``Annotated`` return.

    Unlike :func:`_producer` (a ``TypedDict`` + ``extract_fields`` multi-output
    node), this exercises the static check's other producer shape: a node whose
    own name *is* the produced variable and whose unit is the bare return
    annotation. Future model components may use either shape.
    """
    src = (
        "from typing import Annotated\n"
        "import xarray as xr\n"
        "import xarray_annotated as xra\n"
        "@xra.check_annotations\n"
        f"def {name}() -> Annotated[xr.DataArray, xra.Unit({unit!r})]:\n"
        "    return xr.DataArray([1.0])\n"
    )
    ns: dict = {}
    exec(src, ns)
    return ns[name]


def _plain_consumer(name: str = "plain_cons", in_name: str = "gpp_weekly"):
    """A consumer of ``in_name`` that declares **no** unit on the input.

    Used to pin the opt-in contract: an un-annotated consumer of a typed
    producer contributes no declaration, so the edge is not checked.
    """
    src = (
        "from typing import Annotated, TypedDict\n"
        "import xarray as xr\n"
        "from hamilton.function_modifiers import extract_fields\n"
        "import xarray_annotated as xra\n"
        f"class _Out(TypedDict):\n"
        f"    {name}_out: Annotated[xr.DataArray, xra.Unit('t ha-1')]\n"
        "@extract_fields()\n"
        "@xra.check_annotations\n"
        f"def {name}({in_name}: xr.DataArray) -> _Out:\n"
        f"    return {{{name + '_out'!r}: {in_name}}}\n"
    )
    ns: dict = {}
    exec(src, ns)
    return ns[name]


# ---------------------------------------------------------------------------
# Dimensional incompatibility (always a finding)
# ---------------------------------------------------------------------------


class TestDimensionalMismatch:
    def _dr(self, register):
        prod = register("uc_prod", _producer("g m-2 d-1"))
        cons = register("uc_cons", _consumer("kg"))
        return _build(prod, cons)

    def test_strict_raises(self, register):
        dr = self._dr(register)
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="dimensionally incompatible"),
        ):
            check_dag_contracts(dr)

    def test_off_is_silent(self, register):
        dr = self._dr(register)
        with xra.enabled(False):
            check_dag_contracts(dr)

    def test_message_names_node_and_units(self, register):
        dr = self._dr(register)
        with xra.enabled(True), pytest.raises(ValueError, match="gpp_weekly") as exc:
            check_dag_contracts(dr)
        msg = str(exc.value)
        assert "'g m-2 d-1'" in msg
        assert "'kg'" in msg


# ---------------------------------------------------------------------------
# Compatible but unequal units (nothing converts them between nodes)
# ---------------------------------------------------------------------------


class TestExactMatch:
    def _dr(self, register):
        prod = register("ue_prod", _producer("Pa"))
        cons = register("ue_cons", _consumer("hPa"))
        return _build(prod, cons)

    def test_compatible_flagged(self, register):
        dr = self._dr(register)
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="different units"),
        ):
            check_dag_contracts(dr)

    def test_compatible_fails_the_build(self, register):
        register("ue2_prod", _producer("m"))
        register("ue2_cons", _consumer("km"))
        with (
            xra.enabled(True),
            pytest.raises(CatenaValueError, match="different units"),
        ):
            build_driver(["ue2_prod", "ue2_cons"], {})

    def test_different_spellings_of_one_unit_pass(self, register):
        prod = register("ue3_prod", _producer("Pa"))
        cons = register("ue3_cons", _consumer("pascal"))
        with xra.enabled(True):
            check_dag_contracts(_build(prod, cons))


# ---------------------------------------------------------------------------
# Shared external input consumed with conflicting units (no producer)
# ---------------------------------------------------------------------------


class TestSharedInputConflict:
    def test_conflicting_consumers_flagged(self, register):
        a = register("us_a", _consumer("Pa", name="consumer_a"))
        b = register("us_b", _consumer("kg", name="consumer_b"))
        dr = _build(a, b)
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="dimensionally incompatible"),
        ):
            check_dag_contracts(dr)


# ---------------------------------------------------------------------------
# Consistent declarations pass (synthetic + real models)
# ---------------------------------------------------------------------------


class TestBareReturnProducer:
    """A single-output node (bare ``Annotated`` return) is a checkable producer.

    The existing producer tests all use ``TypedDict`` + ``extract_fields``; this
    covers the other node shape so the static check stays robust to model
    components that emit a single declared output.
    """

    def _dr(self, register, prod_unit, cons_unit):
        prod = register("br_prod", _bare_producer(prod_unit, name="flux"))
        cons = register("br_cons", _consumer(cons_unit, in_name="flux"))
        return _build(prod, cons)

    def test_incompatible_raises(self, register):
        dr = self._dr(register, "g m-2 d-1", "kg")
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="dimensionally incompatible"),
        ):
            check_dag_contracts(dr)

    def test_equal_units_pass(self, register):
        dr = self._dr(register, "g m-2 d-1", "g m-2 d-1")
        with (
            xra.enabled(True),
        ):
            check_dag_contracts(dr)

    def test_exact_string_mismatch_flagged(self, register):
        dr = self._dr(register, "Pa", "hPa")
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="different units"),
        ):
            check_dag_contracts(dr)


class TestOptInContract:
    """Unit checking is opt-in per edge: an edge is only compared when *both*
    sides declare a unit. An un-annotated consumer (or producer) is skipped, so
    partially-annotated future models never trigger false positives."""

    def test_unannotated_consumer_of_typed_producer_not_checked(self, register):
        prod = register("oc_prod", _producer("Pa"))
        cons = register("oc_cons", _plain_consumer())
        dr = _build(prod, cons)
        with (
            xra.enabled(True),
        ):
            check_dag_contracts(dr)


class TestSwitch:
    """The checks follow xarray-annotated's on/off switch, block over variable."""

    def _dr(self, register):
        prod = register("sw_prod", _producer("g m-2 d-1"))
        cons = register("sw_cons", _consumer("kg"))
        return _build(prod, cons)

    def test_variable_off_skips_both_checks(self, register, monkeypatch):
        monkeypatch.setenv("XARRAY_ANNOTATED_ENABLED", "false")
        dr = self._dr(register)
        check_dag_contracts(dr)
        check_input_contracts(dr, {"gpp_weekly": xr.DataArray([1.0])})

    def test_variable_unset_checks(self, register, monkeypatch):
        monkeypatch.delenv("XARRAY_ANNOTATED_ENABLED")
        with pytest.raises(CatenaValueError, match="incompatible"):
            check_dag_contracts(self._dr(register))

    def test_block_off_beats_variable_on(self, register, monkeypatch):
        monkeypatch.setenv("XARRAY_ANNOTATED_ENABLED", "true")
        dr = self._dr(register)
        with xra.enabled(False):
            check_dag_contracts(dr)
            check_input_contracts(dr, {"gpp_weekly": xr.DataArray([1.0])})

    def test_block_on_beats_variable_off(self, register, monkeypatch):
        monkeypatch.setenv("XARRAY_ANNOTATED_ENABLED", "false")
        dr = self._dr(register)
        with xra.enabled(True), pytest.raises(CatenaValueError, match="incompatible"):
            check_dag_contracts(dr)
        with (
            xra.enabled(True),
            pytest.raises(CatenaValueError, match="no 'units' attribute"),
        ):
            check_input_contracts(dr, {"gpp_weekly": xr.DataArray([1.0])})


class TestConsistent:
    def test_matching_units_pass(self, register):
        prod = register("uk_prod", _producer("Pa"))
        cons = register("uk_cons", _consumer("Pa"))
        dr = _build(prod, cons)
        with (
            xra.enabled(True),
        ):
            check_dag_contracts(dr)

    def test_producer_consumer_same_unit_clean(self, register):
        register("clean_prod", _producer("g m-2 d-1"))
        register("clean_cons", _consumer("g m-2 d-1"))
        with xra.enabled(True):
            dr = build_driver(["clean_prod", "clean_cons"], {})
            check_dag_contracts(dr)


# ---------------------------------------------------------------------------
# build_driver invokes the check (gated by the switch)
# ---------------------------------------------------------------------------


class TestBuildDriverIntegration:
    def test_build_driver_runs_check_when_enabled(self, register):
        register("ub_prod", _producer("g m-2 d-1"))
        register("ub_cons", _consumer("kg"))
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="dimensionally incompatible"),
        ):
            build_driver(["ub_prod", "ub_cons"], {})

    def test_build_driver_skips_when_disabled(self, register):
        register("ub2_prod", _producer("g m-2 d-1"))
        register("ub2_cons", _consumer("kg"))
        with xra.enabled(False):
            build_driver(["ub2_prod", "ub2_cons"], {})


# ---------------------------------------------------------------------------
# Resample: a resampled var is checked by what its [[resample]] declares
# ---------------------------------------------------------------------------


def _resample_node_specs(*specs):
    """Build the node specs each ResampleSpec generates (as config does)."""
    from catena.config import resample_node_specs

    return [node for spec in specs for node in resample_node_specs(spec)]


def _resample(vars_, source, target, freq, **contract):
    return ResampleSpec(
        vars=vars_, source=source, target=target, freq=freq, contract=contract
    )


class TestResampleDeclarations:
    """A producer emits ``gpp_weekly`` ('g m-2 d-1'), resampled to ``gpp_monthly``."""

    def _build(self, register, consumer_unit, **contract):
        register("rs_prod", _producer("g m-2 d-1"))  # produces gpp_weekly
        register("rs_cons", _consumer(consumer_unit, in_name="gpp_monthly"))
        specs = _resample_node_specs(
            _resample(["gpp"], "weekly", "monthly", "1ME", **contract)
        )
        return build_driver(["rs_prod", "node", "rs_cons"], {}, node_specs=specs)

    def test_declared_unit_is_checked_against_the_consumer(self, register):
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"'gpp_monthly'.*output of gpp_monthly"),
        ):
            self._build(register, "kg", units="g m-2 d-1")

    def test_edge_into_the_resample_is_left_to_runtime(self, register):
        # Like a [[node]], a resample declares its output only.
        with xra.enabled(True):
            self._build(register, "kg", units="kg")

    def test_loosely_typed_producer_wires_into_a_declaring_resample(self, register):
        def gpp_daily() -> xr.DataArray | None: ...

        register("lt_prod", gpp_daily)
        specs = _resample_node_specs(
            _resample(["gpp"], "daily", "weekly", "7D", units="K")
        )
        with xra.enabled(True):
            build_driver(["lt_prod", "node"], {}, node_specs=specs)

    def test_consistent_declarations_pass(self, register):
        with xra.enabled(True):
            self._build(register, "g m-2 d-1", units="g m-2 d-1")

    def test_undeclared_unit_leaves_the_edges_unchecked(self, register):
        with xra.enabled(True):
            self._build(register, "kg")

    def test_per_variable_units_are_checked_per_variable(self, register):
        register(
            "pv_prods", _bare_producer("K", name="t"), _bare_producer("mm", name="p")
        )
        register("pv_cons", _consumer("mm", name="pv", in_name="t_weekly"))
        specs = _resample_node_specs(
            _resample(["t", "p"], "", "weekly", "7D", units={"t": "K", "p": "mm"})
        )
        with xra.enabled(True), pytest.raises(ValueError, match="'t_weekly'"):
            build_driver(["pv_prods", "node", "pv_cons"], {}, node_specs=specs)

    def test_declared_dims_are_checked(self, register):
        register("rd_prod", _schema_producer("arr_daily", Dims("time", "x")))
        register(
            "rd_cons",
            _named_schema_consumer("rd", Dims("time", "y"), in_name="arr_weekly"),
        )
        specs = _resample_node_specs(
            _resample(["arr"], "daily", "weekly", "7D", dims=["time", "x"])
        )
        with xra.enabled(True), pytest.raises(ValueError, match="'arr_weekly'"):
            build_driver(["rd_prod", "node", "rd_cons"], {}, node_specs=specs)


# ---------------------------------------------------------------------------
# Node: a declared `units=` makes the node a checkable producer
# ---------------------------------------------------------------------------


class TestNodePropagation:
    def _build(self, register, consumer_unit):
        register("dv_cons", _consumer(consumer_unit, in_name="flux"))
        specs = [
            NodeSpec(
                name="flux",
                inputs=["a", "b"],
                expression="a + b",
                import_path=None,
                function=None,
                units="g m-2 d-1",
            )
        ]
        return build_driver(["node", "dv_cons"], {}, node_specs=specs)

    def test_incompatible_consumer_of_node_var_raises(self, register):
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match="flux"),
        ):
            self._build(register, "kg")

    def test_compatible_consumer_of_node_var_passes(self, register):
        with xra.enabled(True):
            self._build(register, "g m-2 d-1")


# ---------------------------------------------------------------------------
# Runtime: a failing input names the node it was declared on
# ---------------------------------------------------------------------------


class TestRuntimeQualname:
    def test_missing_units_error_includes_qualname(self):
        @xra.check_annotations
        def my_model_node(vpd: Annotated[xr.DataArray, xra.Unit("Pa")]) -> xr.DataArray:
            return vpd

        with (
            xra.enabled(True),
            pytest.raises(
                xra.units.UnitError, match=r"\[.*my_model_node\].*no 'units'"
            ),
        ):
            my_model_node(vpd=xr.DataArray([1.0]))

    def test_unparseable_units_error_includes_qualname(self):
        @xra.check_annotations
        def another_node(vpd: Annotated[xr.DataArray, xra.Unit("Pa")]) -> xr.DataArray:
            return vpd

        da = xr.DataArray([1.0], attrs={"units": "fraction"})
        with (
            xra.enabled(True),
            pytest.raises(
                xra.units.UnitError, match=r"\[.*another_node\].*unparseable"
            ),
        ):
            another_node(vpd=da)


# ---------------------------------------------------------------------------
# Schema facets (dims / dtype / coords): the DAG check is facet-generic
# ---------------------------------------------------------------------------


def _schema_producer(name: str, *markers):
    """A single-output node producing ``name`` with the given schema markers."""

    def prod() -> Annotated[(xr.DataArray, *markers)]:  # type: ignore[valid-type]
        return _da()

    prod.__name__ = name
    return prod


def _schema_consumer(*markers, in_name: str = "arr"):
    """A node consuming ``in_name`` declaring the given schema markers on it."""
    src = (
        "from typing import Annotated\n"
        "import xarray as xr\n"
        f"def cons({in_name}: Annotated[(xr.DataArray, *_markers)]) -> xr.DataArray:\n"
        f"    return {in_name}\n"
    )
    ns: dict = {"_markers": markers}
    exec(src, ns)
    return ns["cons"]


def _named_schema_consumer(fn_name: str, *markers, in_name: str = "arr"):
    """A named node consuming ``in_name``, declaring the given schema markers on it."""
    src = (
        "from typing import Annotated\n"
        "import xarray as xr\n"
        f"def {fn_name}({in_name}: Annotated[(xr.DataArray, *_markers)]) -> xr.DataArray:\n"
        f"    return {in_name}\n"
    )
    ns: dict = {"_markers": markers}
    exec(src, ns)
    return ns[fn_name]


class TestAllPairsEdgeCheck:
    """Every pair of declarations is compared, not just each against the first.

    Three of the four edge predicates are non-transitive: a *loose* declaration is
    compatible with everything, so it cannot serve as a base for a star-wise check.
    ``Dims("x","y")`` is compatible with both ``Dims("x","y", ordered=True)`` and
    ``Dims("y","x", ordered=True)`` — yet those two provably conflict. With no
    producer to anchor on, only an all-pairs comparison finds it.
    """

    def test_conflicting_consumers_caught_via_a_loose_declaration(self, register):
        mod = register(
            "ap_cons",
            # Deliberately loose first: it is compatible with each of the other two,
            # so a check anchored on it alone sees no problem.
            _named_schema_consumer("c1_loose", Dims("time", "x")),
            _named_schema_consumer("c2_time_x", Dims("time", "x", ordered=True)),
            _named_schema_consumer("c3_x_time", Dims("x", "time", ordered=True)),
        )
        dr = _build(mod)
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"dims: .* are incompatible") as exc,
        ):
            check_dag_contracts(dr)
        # The conflicting *pair* is named, not just the mismatch with the base.
        message = str(exc.value)
        assert "c2_time_x" in message
        assert "c3_x_time" in message

    def test_mutually_compatible_consumers_still_pass(self, register):
        mod = register(
            "ap_ok",
            _named_schema_consumer("d1", Dims("time", "x")),
            _named_schema_consumer("d2", Dims("x", "time")),
            _named_schema_consumer("d3", Dims("time", "x", ordered=True)),
        )
        dr = _build(mod)
        with xra.enabled(True):
            check_dag_contracts(dr)


class TestSchemaDagCheck:
    """The build-time edge check compares dims and dtype markers, always raising
    ``ValueError`` on a provable mismatch (a pipeline-definition error)."""

    def _dr(self, register, prod_markers, cons_markers):
        prod = register("scd_prod", _schema_producer("arr", *prod_markers))
        cons = register("scd_cons", _schema_consumer(*cons_markers))
        return _build(prod, cons)

    def test_dim_sets_differ_raises(self, register):
        dr = self._dr(register, [Dims("time", "x")], [Dims("time", "y")])
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"dims: .* are incompatible"),
        ):
            check_dag_contracts(dr)

    def test_dim_order_differ_raises_when_both_ordered(self, register):
        dr = self._dr(
            register,
            [Dims("time", "x", ordered=True)],
            [Dims("x", "time", ordered=True)],
        )
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"dims: .* are incompatible"),
        ):
            check_dag_contracts(dr)

    def test_same_dim_set_any_order_passes(self, register):
        dr = self._dr(register, [Dims("time", "x")], [Dims("x", "time")])
        with xra.enabled(True):
            check_dag_contracts(dr)

    def test_dtype_kinds_differ_raises(self, register):
        dr = self._dr(register, [Dtype("float64")], [Dtype("int32")])
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"dtype: .* are incompatible"),
        ):
            check_dag_contracts(dr)

    def test_different_widths_raise(self, register):
        dr = self._dr(register, [Dtype("float64")], [Dtype("float32")])
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"dtype: .* are incompatible"),
        ):
            check_dag_contracts(dr)

    def test_kind_matches_any_width(self, register):
        dr = self._dr(register, [Dtype("float")], [Dtype("float32")])
        with xra.enabled(True):
            check_dag_contracts(dr)

    def test_coords_are_not_checked_at_edges(self, register):
        # Coords are lower bounds, so mismatched coord declarations never raise.
        dr = self._dr(register, [Coords("time")], [Coords("time", "lat")])
        with xra.enabled(True):
            check_dag_contracts(dr)

    def test_disabled_skips_schema(self, register):
        dr = self._dr(register, [Dims("time", "x")], [Dims("time", "y")])
        with xra.enabled(False):
            check_dag_contracts(dr)


class TestSchemaInputCheck:
    """``check_input_contracts`` validates a loaded array's dims/dtype/coords
    against its consumer's declaration, from file metadata alone."""

    def _dr(self, register, *markers):
        cons = register("sci_cons", _schema_consumer(*markers))
        return _build(cons)

    def test_dim_mismatch_raises(self, register):
        dr = self._dr(register, Dims("time", "x"))
        bad = xr.DataArray(np.zeros((2, 3)), dims=("time", "y"))
        with xra.enabled(True), pytest.raises(CatenaValueError, match="dims"):
            check_input_contracts(dr, {"arr": bad})

    def test_dim_match_passes(self, register):
        dr = self._dr(register, Dims("time", "x"))
        good = xr.DataArray(np.zeros((2, 3)), dims=("time", "x"))
        with xra.enabled(True):
            check_input_contracts(dr, {"arr": good})

    def test_dtype_kind_mismatch_raises(self, register):
        dr = self._dr(register, Dtype("float64"))
        bad = xr.DataArray(np.zeros((3,), dtype="int64"), dims=("x",))
        with xra.enabled(True), pytest.raises(CatenaValueError, match="dtype"):
            check_input_contracts(dr, {"arr": bad})

    def test_every_failing_input_is_reported_at_once(self, register):
        consumers = register(
            "sci_many",
            _named_schema_consumer("e1", Dims("x"), in_name="first"),
            _named_schema_consumer("e2", Dtype("float64"), in_name="second"),
        )
        dr = _build(consumers)
        inputs = {
            "first": xr.DataArray([1.0], dims="y"),
            "second": xr.DataArray(np.zeros(1, dtype="int64"), dims="x"),
        }
        with xra.enabled(True), pytest.raises(CatenaValueError) as exc:
            check_input_contracts(dr, inputs)
        message = str(exc.value)
        assert "'first' dims mismatch" in message
        assert "'second' dtype mismatch" in message

    def test_every_failing_facet_of_one_input_is_reported(self, register):
        def consumer(
            x: Annotated[xr.DataArray, xra.Unit("Pa"), Dims("x")],
        ) -> xr.DataArray:
            return x

        dr = _build(register("sci_two_facets", consumer))
        wrong = xr.DataArray([1.0], dims="y", attrs={"units": "K"})
        with xra.enabled(True), pytest.raises(CatenaValueError) as exc:
            check_input_contracts(dr, {"x": wrong})
        message = str(exc.value)
        assert "input 'x'" in message
        assert "'x' dims mismatch" in message

    def test_none_passes_where_the_hint_allows_it(self, register):
        def consumer(
            x: Annotated[xr.DataArray, xra.Unit("Pa")] | None = None,
        ) -> xr.DataArray:
            return xr.DataArray(0.0) if x is None else x

        dr = _build(register("sci_optional", consumer))
        with xra.enabled(True):
            check_input_contracts(dr, {"x": None})

    def test_none_raises_if_another_consumer_forbids_it(self, register):
        def allows(
            x: Annotated[xr.DataArray, xra.Unit("Pa")] | None = None,
        ) -> xr.DataArray:
            return xr.DataArray(0.0) if x is None else x

        def forbids(x: Annotated[xr.DataArray, xra.Unit("Pa")]) -> xr.DataArray:
            return x

        dr = _build(register("sci_mixed_optional", allows, forbids))
        with (
            xra.enabled(True),
            pytest.raises(CatenaValueError, match="declared a DataArray, got None"),
        ):
            check_input_contracts(dr, {"x": None})

    def test_malformed_marker_on_an_edge_is_a_catena_error(self, register):
        def first(x: Annotated[xr.DataArray, xra.Unit("not_a_unit")]) -> xr.DataArray:
            return x

        def second(x: Annotated[xr.DataArray, xra.Unit("Pa")]) -> xr.DataArray:
            return x

        dr = _build(register("sci_malformed", first, second))
        with (
            xra.enabled(True),
            pytest.raises(CatenaValueError, match="malformed contract declaration"),
        ):
            check_dag_contracts(dr)

    @pytest.mark.parametrize(
        "marker",
        [xra.Unit("not_a_unit"), xra.Dtype("not_a_dtype"), xra.Freq("not_a_freq")],
        ids=["unit", "dtype", "freq"],
    )
    @pytest.mark.parametrize(
        "check",
        [check_dag_contracts, lambda dr: check_input_contracts(dr, {"x": _DAILY})],
        ids=["edge", "input"],
    )
    def test_malformed_marker_on_a_lone_declaration(self, register, marker, check):
        def consumer(x: Annotated[xr.DataArray, marker]) -> xr.DataArray:
            return x

        dr = _build(register(f"sci_lone_{type(marker).__name__}", consumer))
        with (
            xra.enabled(True),
            pytest.raises(CatenaValueError, match="malformed contract declaration"),
        ):
            check(dr)

    def test_dimensionality_failure_names_the_input_once(self, register):
        def consumer(x: Annotated[xr.DataArray, xra.Unit("kg")]) -> xr.DataArray:
            return x

        dr = _build(register("sci_named_once", consumer))
        with xra.enabled(True), pytest.raises(CatenaValueError) as exc:
            check_input_contracts(dr, {"x": _DAILY})
        assert str(exc.value).count("'x'") == 1


def _percent():
    """A fresh node reading ``x`` and producing its output, both declared in ``%``."""

    @xra.check_annotations
    def percent(
        x: Annotated[xr.DataArray, xra.Unit("%")],
    ) -> Annotated[xr.DataArray, xra.Unit("%")]:
        return x

    return percent


def _input_in(unit: str) -> xr.DataArray:
    return _DAILY.assign_attrs(units=unit)


class TestParameterizedNodes:
    """Nodes generated by Hamilton's ``@parameterize_sources`` and ``@parameterize``.

    Neither the parameter nor the function name is a node, so declarations are read
    off the generated nodes and keyed by the nodes they bind to.
    """

    def test_mismatch_on_a_parameterised_edge_is_caught(self, register):
        from hamilton.function_modifiers import parameterize_sources

        @parameterize_sources(mass_nee={"flux": "nee"})
        @xra.check_annotations
        def to_mass(flux: Annotated[xr.DataArray, xra.Unit("kg")]) -> xr.DataArray:
            return flux

        prod = register("ps_prod", _bare_producer("umol m-2 s-1", name="nee"))
        cons = register("ps_cons", to_mass)
        dr = _build(prod, cons)
        with (
            xra.enabled(True),
            pytest.raises(ValueError, match=r"'nee'.*input of mass_nee \(to_mass\)"),
        ):
            check_dag_contracts(dr)

    def test_shared_parameter_names_do_not_collide(self, register):
        from hamilton.function_modifiers import parameterize_sources

        @parameterize_sources(mass_a={"flux": "a"})
        @xra.check_annotations
        def as_mass(
            flux: Annotated[xr.DataArray, xra.Unit("g m-2 d-1")],
        ) -> xr.DataArray:
            return flux

        @parameterize_sources(molar_b={"flux": "b"})
        @xra.check_annotations
        def as_molar(
            flux: Annotated[xr.DataArray, xra.Unit("umol m-2 s-1")],
        ) -> xr.DataArray:
            return flux

        prods = register(
            "ps_prods",
            _bare_producer("g m-2 d-1", name="a"),
            _bare_producer("umol m-2 s-1", name="b"),
        )
        cons = register("ps_cons2", as_mass, as_molar)
        dr = _build(prods, cons)
        with xra.enabled(True):
            check_dag_contracts(dr)

    def _parameterize(self, register, *others):
        from hamilton.function_modifiers import parameterize, source

        masked = parameterize(masked_daily={"x": source("x_daily")})(_percent())
        return _build(register("pz_cons", masked), *others)

    def test_parameterize_edge_is_caught(self, register):
        prod = register("pz_prod", _bare_producer("m3 m-3", name="x_daily"))
        dr = self._parameterize(register, prod)
        with xra.enabled(True), pytest.raises(CatenaValueError, match="'x_daily'"):
            check_dag_contracts(dr)

    def test_parameterize_output_is_read_off_the_node(self, register):
        prod = register("pz_prod2", _bare_producer("%", name="x_daily"))
        down = register(
            "pz_down", _consumer("m3 m-3", name="down", in_name="masked_daily")
        )
        dr = self._parameterize(register, prod, down)
        with (
            xra.enabled(True),
            pytest.raises(
                CatenaValueError, match=r"output of masked_daily \(percent\)"
            ),
        ):
            check_dag_contracts(dr)

    def test_input_check_follows_the_bound_source(self, register):
        dr = self._parameterize(register)
        with xra.enabled(True), pytest.raises(CatenaValueError, match="'x_daily'"):
            check_input_contracts(dr, {"x_daily": _input_in("K")})
        with xra.enabled(True):
            check_input_contracts(dr, {"x_daily": _input_in("%")})


class TestSubdagNodes:
    """Nodes generated by Hamilton's ``@subdag``.

    Inside a sub-DAG every node is namespaced, and ``source()`` adds an identity
    node carrying the inner parameter's hint, so the edge into the sub-DAG is
    checked against its real source.
    """

    def _dr(self, register, *others, typed_wrapper: bool = False):
        from hamilton.function_modifiers import source, subdag

        inner = register("sd_inner", _percent())
        out_hint = (
            Annotated[xr.DataArray, xra.Unit("%")] if typed_wrapper else xr.DataArray
        )

        def masked_daily(percent: xr.DataArray) -> xr.DataArray:
            return percent

        masked_daily.__annotations__["return"] = out_hint
        wrapper = subdag(inner, inputs={"x": source("x_daily")})(masked_daily)
        return _build(register("sd_outer", wrapper), *others)

    def test_edge_into_the_subdag_is_caught(self, register):
        prod = register("sd_prod", _bare_producer("m3 m-3", name="x_daily"))
        with xra.enabled(True), pytest.raises(CatenaValueError, match="'x_daily'"):
            check_dag_contracts(self._dr(register, prod))

    def test_typed_wrapper_output_is_checked(self, register):
        prod = register("sd_prod2", _bare_producer("%", name="x_daily"))
        down = register(
            "sd_down", _consumer("m3 m-3", name="down", in_name="masked_daily")
        )
        dr = self._dr(register, prod, down, typed_wrapper=True)
        with xra.enabled(True), pytest.raises(CatenaValueError, match="'masked_daily'"):
            check_dag_contracts(dr)

    def test_input_check_follows_the_bound_source(self, register):
        dr = self._dr(register)
        with xra.enabled(True), pytest.raises(CatenaValueError, match="'x_daily'"):
            check_input_contracts(dr, {"x_daily": _input_in("K")})

    def test_unrelated_node_sharing_an_inner_name_is_not_compared(self, register):
        prods = register(
            "sd_prods",
            _bare_producer("%", name="x_daily"),
            _bare_producer("K", name="x"),
        )
        down = register("sd_down2", _consumer("K", name="down2", in_name="x"))
        with xra.enabled(True):
            check_dag_contracts(self._dr(register, prods, down))
