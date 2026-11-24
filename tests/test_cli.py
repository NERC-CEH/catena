"""Tests for the catena CLI: argument handling, exit codes, and what it prints.

The CLI is a shim over `catena.pipeline` and `catena.graph`, so the pipeline
behaviour itself is tested in `test_pipeline.py` and `test_graph.py`. What is
left here is the part only the command can get wrong.
"""

import shutil
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from catena._version import __version__
from catena.cli import main
from catena.cli.app import app
from catena.errors import (
    CatenaError,
    CatenaFileExistsError,
    CatenaFileNotFoundError,
    CatenaPermissionError,
    CatenaValueError,
)
from catena.input_checks import InputCheckError

runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionFlag:
    @pytest.mark.parametrize("flag", ["--version", "-v"])
    def test_exits_zero(self, flag):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0

    @pytest.mark.parametrize("flag", ["--version", "-v"])
    def test_shows_version_string(self, flag):
        result = runner.invoke(app, [flag])
        assert __version__ in result.output


def test_help_keeps_bracketed_section_names():
    result = runner.invoke(app, ["graph", "--help"], terminal_width=200)
    assert "[outputs.*] sections" in result.output


class TestMissingTyperExtra:
    """Without the extra, `catena` must explain itself, not traceback.

    The import boundary itself is checked in `test_layout.py`.
    """

    def test_entry_point_hints_at_the_extra(self, monkeypatch, capsys):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "typer" or name.startswith("typer."):
                raise ModuleNotFoundError("No module named 'typer'", name="typer")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "catena.cli.app", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        assert (
            '"ukceh-catena[cli] @ git+https://github.com/NERC-CEH/catena"'
            in capsys.readouterr().err
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_exits_zero(self, config_toml):
        result = runner.invoke(app, ["run", str(config_toml)])
        assert result.exit_code == 0, result.output

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["run", str(tmp_path / "nonexistent.toml")])
        assert result.exit_code != 0

    def test_run_without_outputs_prints_notice(self, tmp_path, synthetic_data_dir):
        # Previously exited 0 with no output and no message, which looked like a
        # successful run that had produced files somewhere.
        cfg = tmp_path / "no_outputs.toml"
        cfg.write_text(
            f"""\
[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["temperature"]
"""
        )
        result = runner.invoke(app, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "nothing to execute" in result.output

    @pytest.fixture
    def writing_config(self, tmp_path, synthetic_data_dir):
        """A config that actually writes a file, so the run has something to report."""
        cfg = tmp_path / "writes.toml"
        cfg.write_text(
            f"""\
[[node]]
name = "warmth_daily"
inputs = ["temperature_daily"]
expression = "temperature_daily * 2"

[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["temperature"]

[outputs.daily]
path = "{tmp_path / "out.nc"}"
vars = ["warmth"]
"""
        )
        return cfg

    def test_reports_what_it_wrote(self, writing_config):
        """A run that writes files says so: silence looked the same as doing nothing."""
        result = runner.invoke(app, ["run", str(writing_config)])
        assert result.exit_code == 0, result.output
        assert "inputs loaded:" in result.output
        assert "wrote" in result.output
        assert "1 variable(s)" in result.output
        assert "Run completed in" in result.output

    def test_written_paths_are_relative_to_the_working_directory(
        self, writing_config, monkeypatch
    ):
        """Output paths resolve against the config's directory, so they arrive absolute."""
        monkeypatch.chdir(writing_config.parent)
        result = runner.invoke(app, ["run", str(writing_config)])
        assert result.exit_code == 0, result.output
        assert "wrote out.nc" in result.output

    def test_failing_pipeline_exits_non_zero(self, tmp_path, synthetic_data_dir):
        """A hard failure from the library must reach the shell as an exit code."""
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f"""\
[inputs.daily]
path = "{synthetic_data_dir / "daily.nc"}"
vars = ["temperature"]

[outputs.daily]
path = "{blocker / "out.nc"}"
vars = ["temperature"]
"""
        )
        result = runner.invoke(app, ["run", str(cfg)])
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


class TestGraphCommand:
    """The command's own job: where the file goes and in which format."""

    @pytest.mark.skipif(not shutil.which("dot"), reason="graphviz not installed")
    def test_generates_dot_file(self, config_toml, tmp_path):
        out = tmp_path / "pipeline"
        result = runner.invoke(app, ["graph", str(config_toml), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.with_suffix(".dot").exists()
        assert "mean_temperature_weekly" in out.with_suffix(".dot").read_text()

    @pytest.mark.skipif(not shutil.which("dot"), reason="graphviz not installed")
    def test_all_nodes_draws_beyond_the_outputs(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text(
            '[[node]]\nname = "kept"\ninputs = ["x"]\nexpression = "x"\n'
            '[[node]]\nname = "unused"\ninputs = ["x"]\nexpression = "x"\n'
            '[outputs.products]\npath = "out.nc"\nvars = {kept = "kept"}\n'
        )
        out = tmp_path / "pipeline"
        args = ["graph", str(config), "--output", str(out)]
        assert runner.invoke(app, args).exit_code == 0
        assert "unused" not in out.with_suffix(".dot").read_text()
        result = runner.invoke(app, [*args, "--all-nodes"])
        assert result.exit_code == 0, result.output
        assert "unused" in out.with_suffix(".dot").read_text()

    def test_view_options_reach_build_graph(self, tmp_path, monkeypatch):
        calls = []

        def fake_build_graph(config, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(source="digraph {}")

        monkeypatch.setattr("catena.cli.graph.build_graph", fake_build_graph)
        config = tmp_path / "config.toml"
        config.write_text("")
        args = ["graph", str(config), "--output", str(tmp_path / "pipeline")]
        result = runner.invoke(
            app,
            [
                *args,
                "--upstream-of",
                "a",
                "--upstream-of",
                "b",
                "--between",
                "c",
                "d",
                "--hide-inputs",
                "--deduplicate-inputs",
            ],
        )
        assert result.exit_code == 0, result.output
        assert runner.invoke(app, args).exit_code == 0
        given, default = calls
        assert given["upstream_of"] == ["a", "b"]
        assert given["between"] == ("c", "d")
        assert given["hide_inputs"]
        assert given["deduplicate_inputs"]
        assert default["upstream_of"] == ()
        assert default["downstream_of"] == ()
        assert default["between"] is None
        assert not default["hide_inputs"]
        assert not default["deduplicate_inputs"]

    def test_missing_config_fails(self, tmp_path):
        result = runner.invoke(app, ["graph", str(tmp_path / "no.toml")])
        assert result.exit_code != 0


class TestGraphRendering:
    """--svg/--png/--pdf render through the graphviz API, and failures are surfaced."""

    @pytest.mark.skipif(not shutil.which("dot"), reason="graphviz not installed")
    def test_png_is_written(self, config_toml, tmp_path):
        out = tmp_path / "pipeline"
        result = runner.invoke(
            app, ["graph", str(config_toml), "--output", str(out), "--png"]
        )
        assert result.exit_code == 0, result.output
        png = out.with_suffix(".png")
        assert png.exists()
        assert png.read_bytes().startswith(b"\x89PNG")

    @pytest.mark.skipif(not shutil.which("dot"), reason="graphviz not installed")
    def test_svg_is_written(self, config_toml, tmp_path):
        out = tmp_path / "pipeline"
        result = runner.invoke(
            app, ["graph", str(config_toml), "--output", str(out), "--svg"]
        )
        assert result.exit_code == 0, result.output
        assert "<svg" in out.with_suffix(".svg").read_text()

    def test_graph_png_reports_missing_dot(self, config_toml, tmp_path, monkeypatch):
        # Previously subprocess.run(["dot", ...]) ran without check=True, so a
        # missing binary produced no output, no error and a zero exit code.
        import graphviz

        def boom(*args, **kwargs):
            raise graphviz.ExecutableNotFound(["dot"])

        monkeypatch.setattr(graphviz.Digraph, "pipe", boom)

        out = tmp_path / "pipeline"
        result = runner.invoke(
            app, ["graph", str(config_toml), "--output", str(out), "--png"]
        )
        assert result.exit_code != 0
        assert "graphviz" in result.output.lower()
        assert not out.with_suffix(".png").exists()


class TestModuleBesideTheConfig:
    """A `.py` file named by an `_import_path` resolves against the config.

    The CLI used to append the working directory to `sys.path` for this. It no
    longer touches `sys.path` at all, so the config has to be self-contained --
    and it is, from any working directory.
    """

    @pytest.fixture
    def local_pipeline(self, tmp_path):
        (tmp_path / "local_nodes.py").write_text(
            "def doubled(scalar: float) -> float:\n    return 2 * scalar\n"
        )
        (tmp_path / "config.toml").write_text(
            '[local]\n_import_path = "local_nodes.py"\n'
        )
        return tmp_path / "config.toml"

    def test_resolves_from_the_config_directory(self, local_pipeline, monkeypatch):
        monkeypatch.chdir(local_pipeline.parent)
        result = runner.invoke(app, ["graph", "config.toml", "--output", "graph"])
        assert result.exit_code == 0, result.output

    def test_resolves_from_an_unrelated_directory(
        self, local_pipeline, tmp_path_factory
    ):
        """The case the old cwd-on-sys.path behaviour could not handle."""
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        result = runner.invoke(
            app,
            ["graph", str(local_pipeline), "--output", str(elsewhere / "graph")],
        )
        assert result.exit_code == 0, result.output


class TestErrorRendering:
    """`main` prints a CatenaError alone; anything else keeps its traceback."""

    def test_catena_error_prints_message_without_traceback(self, monkeypatch, capsys):
        def _raise():
            raise CatenaValueError("something the user can fix")

        monkeypatch.setattr("catena.cli.app.app", _raise)
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: something the user can fix" in captured.err
        assert "Traceback" not in captured.err

    def test_other_exceptions_propagate(self, monkeypatch):
        """A bug must keep its frames, so `main` must not swallow it."""

        def _raise():
            raise TypeError("a bug, not a bad input")

        monkeypatch.setattr("catena.cli.app.app", _raise)
        with pytest.raises(TypeError, match="a bug"):
            main()

    def test_input_check_error_is_a_catena_error(self):
        assert issubclass(InputCheckError, CatenaError)


class TestErrorTypes:
    """Concrete errors keep the stdlib type a library caller would catch."""

    @pytest.mark.parametrize(
        ("error", "stdlib"),
        [
            (CatenaValueError, ValueError),
            (CatenaFileNotFoundError, FileNotFoundError),
            (CatenaPermissionError, PermissionError),
            (CatenaFileExistsError, FileExistsError),
        ],
    )
    def test_inherits_stdlib_type(self, error, stdlib):
        assert issubclass(error, stdlib)
        assert issubclass(error, CatenaError)

    def test_base_is_not_a_value_error(self):
        """`except ValueError` must not catch a missing-file error by accident."""
        assert not issubclass(CatenaError, ValueError)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class TestInitCommand:
    NODES = (
        "import xarray as xr\n\ndef y(x: xr.DataArray) -> xr.DataArray:\n    return x\n"
    )

    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch):
        (tmp_path / "nodes.py").write_text(self.NODES)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_writes_the_config(self, workdir):
        result = runner.invoke(app, ["init", "-m", "nodes.py"])
        assert result.exit_code == 0, result.output
        assert 'vars = [\n    "x",\n]' in (workdir / "config.toml").read_text()

    def test_a_py_path_is_rewritten_relative_to_the_config(self, workdir):
        result = runner.invoke(app, ["init", "-m", "nodes.py", "-o", "sub/c.toml"])
        assert result.exit_code == 0, result.output
        assert '_import_path = "../nodes.py"' in (workdir / "sub/c.toml").read_text()

    def test_refuses_to_overwrite_without_force(self, workdir):
        (workdir / "config.toml").write_text("")
        result = runner.invoke(app, ["init", "-m", "nodes.py"])
        assert isinstance(result.exception, CatenaFileExistsError)
        result = runner.invoke(app, ["init", "-m", "nodes.py", "--force"])
        assert result.exit_code == 0, result.output

    def test_prints_the_notes(self, workdir):
        (workdir / "nodes.py").write_text("def y(*, k: int) -> int:\n    return k\n")
        result = runner.invoke(app, ["init", "-m", "nodes.py"])
        assert "TODO: 'k' is a required setting" in result.output

    def test_interactive_toggles_and_adds_paths(self, workdir, monkeypatch):
        monkeypatch.setattr(
            "catena.cli.init.discover_registered_modules",
            lambda: {"transforms": None, "hidden": None},
        )
        result = runner.invoke(app, ["init"], input="\n9\n1 nodes.py\n1\n\n")
        assert result.exit_code == 0, result.output
        assert "[ ] 1: transforms" in result.output
        assert "hidden" in result.output
        assert "Choose at least one module." in result.output
        assert "No module number 9." in result.output
        assert "[nodes]" in (workdir / "config.toml").read_text()
        assert "[transforms]" not in (workdir / "config.toml").read_text()

    def test_factory_excludes_and_postprocesses(self, workdir, monkeypatch):
        from catena.cli.init import make_init_app

        monkeypatch.setattr(
            "catena.cli.init.discover_registered_modules",
            lambda: {"transforms": None, "hidden": None},
        )

        def postprocess(data):
            data["validation"] = {"checks": []}

        custom = make_init_app(
            exclude={"hidden"},
            group=lambda name, declared: "everything",
            postprocess=postprocess,
        )
        result = runner.invoke(custom, input="nodes.py\n\n")
        assert result.exit_code == 0, result.output
        assert "hidden" not in result.output
        text = (workdir / "config.toml").read_text()
        assert "[validation]" in text
        assert "[inputs.everything]" in text
