"""Tests for `catena.scaffold`: generating a config, and listing what it loads."""

import importlib.util
import shutil
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path
from textwrap import dedent

import pytest
import xarray_annotated as xra
from xarray_annotated import Freq, Unit

import catena
from catena.config import Config
from catena.errors import CatenaFileExistsError, CatenaValueError
from catena.importing import ENTRY_POINT_GROUP, clear_registry_cache
from catena.scaffold import (
    ANY_OTHER,
    InputChange,
    generate_config,
    input_requirements,
    variant_inputs,
)

RECIPES = Path(__file__).parent.parent / "recipes"

HEADER = """\
from typing import Annotated
import xarray as xr
from xarray_annotated import Freq, Unit
"""


def _module(tmp_path: Path, name: str, body: str) -> str:
    """Write a node module beside the config and return its file name."""
    (tmp_path / f"{name}.py").write_text(HEADER + dedent(body))
    return f"{name}.py"


@pytest.fixture
def two_models(tmp_path):
    """A daily model feeding a weekly one, with settings on both."""
    _module(
        tmp_path,
        "first",
        """
        def wet(
            rain: Annotated[xr.DataArray, Unit("mm d-1"), Freq("D")],
            *,
            threshold: float = 1.0,
            label: str | None = None,
        ) -> Annotated[xr.DataArray, Unit("mm d-1"), Freq("D")]:
            return rain > threshold
        """,
    )
    _module(
        tmp_path,
        "second",
        """
        def wet_weeks(
            wet: Annotated[xr.DataArray, Unit("mm d-1"), Freq("D")],
            soil: Annotated[xr.DataArray, Unit("m")],
            *,
            threshold: float = 1.0,
            window: int = 7,
        ) -> Annotated[xr.DataArray, Freq("7D")]:
            return wet.resample(time="7D").mean() * soil
        """,
    )
    return tmp_path


class TestGenerateConfig:
    def test_inputs_are_what_nothing_produces(self, two_models):
        data = generate_config(["first.py", "second.py"], base=two_models).data
        assert data["inputs"] == {"path": "inputs.nc", "vars": ["rain", "soil"]}

    def test_group_splits_sections_by_label(self, two_models):
        seen = {}

        def group(name, declared):
            seen[name] = declared
            freqs = {d.freq for d in declared if d.freq is not None}
            return "static" if not freqs else str(freqs.pop().freq)

        data = generate_config(
            ["first.py", "second.py"], base=two_models, group=group
        ).data
        assert data["inputs"] == {
            "D": {"path": "inputs/D.nc", "suffix": "", "vars": ["rain"]},
            "static": {"path": "inputs/static.nc", "suffix": "", "vars": ["soil"]},
        }
        assert data["outputs"] == {
            "7D": {"path": "outputs/7D.nc", "suffix": "", "vars": ["wet_weeks"]}
        }
        assert [d.unit for d in seen["rain"]] == [Unit("mm d-1")]
        assert [d.freq for d in seen["wet_weeks"]] == [Freq("7D")]

    def test_an_undeclared_output_is_grouped_with_no_declarations(self, tmp_path):
        _module(
            tmp_path, "a", "def y(x: xr.DataArray) -> xr.DataArray:\n    return x\n"
        )
        seen = {}

        def group(name, declared):
            seen[name] = declared
            return "all"

        generate_config(["a.py"], base=tmp_path, group=group)
        assert seen == {"x": (), "y": ()}

    def test_outputs_are_what_nothing_consumes(self, two_models):
        data = generate_config(["first.py", "second.py"], base=two_models).data
        assert data["outputs"] == {"path": "outputs.nc", "vars": ["wet_weeks"]}

    def test_a_shared_setting_is_written_once(self, two_models):
        data = generate_config(["first.py", "second.py"], base=two_models).data
        assert data["first"] == {"_import_path": "first.py", "threshold": 1.0}
        assert data["second"] == {"_import_path": "second.py", "window": 7}

    def test_a_none_default_is_left_out(self, two_models):
        data = generate_config(["first.py"], base=two_models).data
        assert "label" not in data["first"]

    def test_the_generated_config_parses(self, two_models):
        generated = generate_config(["first.py", "second.py"], base=two_models)
        Config(generated.data, base=two_models).parse()
        assert generated.notes == []

    def test_different_defaults_are_noted(self, tmp_path):
        _module(tmp_path, "a", "def x(*, k: int = 1) -> int:\n    return k\n")
        _module(tmp_path, "b", "def y(*, k: int = 2) -> int:\n    return k\n")
        generated = generate_config(["a.py", "b.py"], base=tmp_path)
        assert generated.data["a"]["k"] == 1
        assert "k" not in generated.data["b"]
        assert any("'k'" in note and "[b]" in note for note in generated.notes)

    def test_a_required_setting_is_noted(self, tmp_path):
        _module(
            tmp_path,
            "a",
            """
            def x(v: xr.DataArray, *, k: int) -> xr.DataArray:
                return v * k
            """,
        )
        generated = generate_config(["a.py"], base=tmp_path)
        assert generated.data["inputs"]["vars"] == ["v"]
        assert any("'k'" in note for note in generated.notes)

    def test_an_optional_input_is_not_loaded(self, tmp_path):
        _module(
            tmp_path,
            "a",
            """
            def x(v: xr.DataArray, w: xr.DataArray | None = None) -> xr.DataArray:
                return v
            """,
        )
        data = generate_config(["a.py"], base=tmp_path).data
        assert data["inputs"]["vars"] == ["v"]

    def test_a_contract_conflict_is_noted(self, tmp_path):
        _module(
            tmp_path,
            "a",
            """
            def x(v: xr.DataArray) -> Annotated[xr.DataArray, Unit("m")]:
                return v

            def y(x: Annotated[xr.DataArray, Unit("s")]) -> xr.DataArray:
                return x
            """,
        )
        with xra.enabled(True):
            generated = generate_config(["a.py"], base=tmp_path)
        assert any("contract conflict" in note for note in generated.notes)

    def test_two_modules_defining_one_node_are_refused(self, tmp_path):
        body = "def x(v: xr.DataArray) -> xr.DataArray:\n    return v\n"
        _module(tmp_path, "a", body)
        _module(tmp_path, "b", body)
        with pytest.raises(CatenaValueError, match="cannot be built into one DAG"):
            generate_config(["a.py", "b.py"], base=tmp_path)

    def test_two_entries_for_one_section_are_refused(self, tmp_path):
        _module(tmp_path, "a", "")
        (tmp_path / "sub").mkdir()
        _module(tmp_path / "sub", "a", "")
        with pytest.raises(CatenaValueError, match=r"\[a\]"):
            generate_config(["a.py", "sub/a.py"], base=tmp_path)

    def test_a_module_named_like_a_builtin_section_is_refused(self, tmp_path):
        _module(tmp_path, "inputs", "")
        with pytest.raises(CatenaValueError, match="parses itself"):
            generate_config(["inputs.py"], base=tmp_path)

    def test_a_malformed_name_is_refused(self):
        with pytest.raises(CatenaValueError, match="neither"):
            generate_config(["not a module"])

    def test_a_dotted_module_carries_its_import_path(self):
        data = generate_config(["catena.transforms"]).data
        assert data["transforms"]["_import_path"] == "catena.transforms"


VARIANTS = """
from hamilton.function_modifiers import config


def model(extra: xr.DataArray, *, method: str = "plain") -> xr.DataArray:
    return extra


@config.when(method="plain")
def extra__plain() -> xr.DataArray:
    return xr.DataArray(0.0)


@config.when(method="corrected")
def extra__corrected(correction: xr.DataArray) -> xr.DataArray:
    return correction


@config.when_in(method=["corrected", "other"])
def report(extra: xr.DataArray) -> xr.DataArray:
    return extra
"""

COUPLING = """
from hamilton.function_modifiers import config


def growth(vwc: xr.DataArray) -> xr.DataArray:
    return vwc


@config.when(vwc_from="splash")
def vwc__splash(soil_moisture: xr.DataArray) -> xr.DataArray:
    return soil_moisture
"""


class TestVariantInputs:
    """The issue #64 cases: inputs only a non-default variant needs."""

    def test_each_named_value_and_any_other_is_listed(self, tmp_path):
        changes = variant_inputs([_module(tmp_path, "m", VARIANTS)], base=tmp_path)
        assert changes == {
            "method": {
                "corrected": InputChange(added=("correction",)),
                "plain": InputChange(),
                "other": InputChange(added=("extra",)),
                ANY_OTHER: InputChange(added=("extra",)),
            }
        }

    def test_a_setting_only_a_decorator_reads_adds_and_removes(self, tmp_path):
        changes = variant_inputs([_module(tmp_path, "c", COUPLING)], base=tmp_path)
        assert changes["vwc_from"] == {
            "splash": InputChange(added=("soil_moisture",), removed=("vwc",)),
            ANY_OTHER: InputChange(),
        }

    def test_when_not_and_when_not_in_values_are_read(self, tmp_path):
        name = _module(
            tmp_path,
            "n",
            """
            from hamilton.function_modifiers import config


            @config.when_not(mode="a")
            def x__not_a(p: xr.DataArray) -> xr.DataArray:
                return p


            @config.when_not_in(mode=["b", "c"])
            def y(q: xr.DataArray) -> xr.DataArray:
                return q
            """,
        )
        assert list(variant_inputs([name], base=tmp_path)["mode"]) == [
            "a",
            "b",
            "c",
            ANY_OTHER,
        ]

    def test_a_custom_resolver_alone_has_unknown_values(self, tmp_path):
        name = _module(
            tmp_path,
            "u",
            """
            from hamilton.function_modifiers import config


            @config(lambda cfg: cfg.get("flag") == 1, config_used=["flag"])
            def x(p: xr.DataArray) -> xr.DataArray:
                return p
            """,
        )
        assert variant_inputs([name], base=tmp_path) == {"flag": {}}
        notes = generate_config([name], base=tmp_path).notes
        assert any("'flag' selects variants" in note for note in notes)

    def test_a_value_that_does_not_build_carries_the_error(self, tmp_path):
        name = _module(
            tmp_path,
            "e",
            """
            from hamilton.function_modifiers import config


            @config.when(m="a")
            def x__a() -> xr.DataArray:
                return xr.DataArray(0.0)


            @config.when_in(m=["a", "b"])
            def x__b() -> xr.DataArray:
                return xr.DataArray(1.0)
            """,
        )
        change = variant_inputs([name], base=tmp_path)["m"]["a"]
        assert change.error is not None
        assert change
        tomllib.loads(generate_config([name], base=tmp_path).dumps())

    def test_a_module_without_config_decorators_has_none(self, two_models):
        assert variant_inputs(["first.py", "second.py"], base=two_models) == {}


class TestVariantNotes:
    def test_values_that_change_alike_share_a_note(self, tmp_path):
        notes = generate_config([_module(tmp_path, "m", VARIANTS)], base=tmp_path).notes
        assert notes == [
            "[m] method = 'corrected' also needs 'correction'.",
            "[m] method = 'other' or any other value also needs 'extra'.",
        ]

    def test_a_decorator_only_setting_names_its_section(self, tmp_path):
        generated = generate_config([_module(tmp_path, "c", COUPLING)], base=tmp_path)
        assert generated.notes == [
            "[c] vwc_from = 'splash' also needs 'soil_moisture'; no longer needs 'vwc'."
        ]
        assert generated.data["inputs"]["vars"] == ["vwc"]


class TestRegisteredModules:
    @pytest.fixture
    def registered(self, monkeypatch):
        entry = EntryPoint(
            name="resampling", value="catena.transforms", group=ENTRY_POINT_GROUP
        )
        monkeypatch.setattr("catena.importing.entry_points", lambda group=None: [entry])
        clear_registry_cache()
        yield
        clear_registry_cache()

    def test_a_registered_module_is_a_bare_section(self, registered):
        data = generate_config(["resampling"]).data
        assert "_import_path" not in data["resampling"]

    def test_a_path_named_like_a_registered_module_keeps_its_import_path(
        self, registered, tmp_path
    ):
        _module(tmp_path, "resampling", "")
        data = generate_config(["resampling.py"], base=tmp_path).data
        assert data["resampling"] == {"_import_path": "resampling.py"}


class TestWriting:
    def test_notes_head_the_file(self, tmp_path):
        _module(tmp_path, "a", "def x(*, k: int) -> int:\n    return k\n")
        text = generate_config(["a.py"], base=tmp_path).dumps()
        assert text.startswith("# TODO: 'k' is a required setting")
        assert tomllib.loads(text)["a"] == {"_import_path": "a.py"}

    def test_dump_creates_parents_and_refuses_to_overwrite(self, two_models):
        generated = generate_config(["first.py"], base=two_models)
        path = two_models / "sub" / "config.toml"
        generated.dump(path)
        with pytest.raises(CatenaFileExistsError):
            generated.dump(path)
        generated.dump(path, overwrite_ok=True)


def test_the_flux_recipe_runs_once_its_paths_are_set(tmp_path):
    """A generated config is the recipe's own, but for its file paths."""
    shutil.copy(RECIPES / "flux_pipeline" / "nodes.py", tmp_path)
    spec = importlib.util.spec_from_file_location(
        "flux_make_data", RECIPES / "flux_pipeline" / "make_data.py"
    )
    assert spec is not None
    assert spec.loader is not None
    make_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_data)
    flux, satellite = make_data.write_inputs(tmp_path / "data")

    def by_file(name, declared):
        return "satellite" if name == "sat_gpp" else "flux"

    generated = generate_config(["nodes.py"], base=tmp_path, group=by_file)
    assert generated.notes == []
    generated.data["inputs"]["satellite"]["path"] = str(satellite)
    generated.data["inputs"]["flux"]["path"] = str(flux)
    generated.dump(tmp_path / "config.toml")
    with xra.enabled(True):
        report = catena.dry_run(tmp_path / "config.toml")
    assert [stage for stage in report.stages if stage.status == "failed"] == []


class TestInputRequirements:
    def test_lists_each_variable_with_its_declarations(self, two_models):
        config = Config(
            {
                "first": {"_import_path": "first.py"},
                "second": {"_import_path": "second.py"},
                "inputs": {
                    "daily": {"path": "d.nc", "vars": {"rain": "precip"}},
                    "static": {"path": "s.nc", "suffix": "", "vars": ["soil"]},
                },
            },
            base=two_models,
        )
        requirements = input_requirements(config.parse())
        rain = requirements["daily"]["precip"]
        assert rain.node == "rain"
        assert [d.unit for d in rain.declared] == [Unit("mm d-1")]
        assert [d.freq for d in rain.declared] == [Freq("D")]
        assert rain.consumers == ("wet",)
        assert requirements["static"]["soil"].node == "soil"

    def test_an_unconsumed_variable_declares_nothing(self, two_models):
        config = Config(
            {
                "first": {"_import_path": "first.py"},
                "inputs": {"path": "d.nc", "vars": ["rain", "spare"]},
            },
            base=two_models,
        )
        requirement = input_requirements(config.parse())["inputs"]["spare"]
        assert requirement.declared == ()
        assert requirement.consumers == ()

    def test_a_section_without_vars_is_refused(self, two_models):
        config = Config(
            {"first": {"_import_path": "first.py"}, "inputs": {"path": "d.nc"}},
            base=two_models,
        )
        with pytest.raises(CatenaValueError, match="omits 'vars'"):
            input_requirements(config.parse())
