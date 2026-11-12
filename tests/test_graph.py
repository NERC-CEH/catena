"""Tests for `catena.build_graph` and the styling it applies.

The heavy post-processing (`relabel_with_units`, `color_edges_by_frequency`,
`cluster_nodes_by_frequency`) pattern-matches Hamilton's rendered Graphviz body,
so most of these are canaries: if Hamilton changes how it renders, the patterns
stop matching and every feature **silently degrades to a no-op**. Testing them
against `build_graph` rather than through the CLI keeps them pointed at the
library, where the behaviour lives.
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import xarray as xr

from catena import build_graph
from catena.build import build_driver
from catena.config import load_config
from catena.graph import (
    GraphStyle,
    _import_function,
    assign_freq_colors,
    cluster_nodes_by_frequency,
    color_edges_by_frequency,
    compose_styles,
    infer_frequencies,
    link_style,
    make_style_function,
    relabel_with_units,
    separate_config_inputs,
)
from catena.graph_style import (
    DEFAULT_PALETTE,
    FREQ_COLOR_CYCLE,
    GraphvizSpec,
    load_graphviz_spec,
)
from catena.pipeline import prepare_config

requires_dot = pytest.mark.skipif(
    not shutil.which("dot"), reason="graphviz not installed"
)


class TestBuildGraph:
    """`build_graph` returns the styled Digraph; nothing is written to disk."""

    @requires_dot
    def test_returns_a_digraph_without_writing(self, config_toml, tmp_path):
        digraph = build_graph(config_toml)
        assert digraph.source
        assert not list(tmp_path.glob("*.dot"))

    @requires_dot
    def test_accepts_a_parsed_config(self, config_toml):
        """A caller may parse, adjust the spec, then visualise it in memory."""
        digraph = build_graph(load_config(config_toml))
        assert "mean_temperature_weekly" in digraph.source

    @requires_dot
    def test_style_file_pins_a_frequency_colour(self, config_toml, tmp_path):
        style = tmp_path / "style.toml"
        # The config's node declares freq = "7D"; pinning it overrides the cycle.
        style.write_text('[palette]\n"7D" = "#123456"\n')
        assert "#123456" in build_graph(config_toml, style=style).source


class TestGraphvizBodySurgery:
    """Canaries for the post-processing that rewrites Hamilton's rendered body.

    These assert the surgery actually applied, so a rendering change upstream
    fails loudly here instead of quietly producing a plain graph.
    """

    @pytest.fixture
    def dot_source(self, config_toml):
        return build_graph(config_toml).source

    @requires_dot
    def test_dot_contains_declared_units(self, dot_source):
        # The declared unit replaced the DataArray type ...
        assert "<i>degC</i>" in dot_source
        # ... and no node with a declared unit still shows the type it replaced.
        for line in dot_source.splitlines():
            if line.strip().startswith("mean_temperature_weekly "):
                assert "DataArray" not in line

    @requires_dot
    def test_dot_contains_freq_clusters(self, dot_source):
        # One cluster per distinct declared frequency; the config declares 7D.
        assert "subgraph cluster_7D {" in dot_source
        assert 'label="7D"' in dot_source

    @requires_dot
    def test_dot_edges_coloured(self, dot_source):
        edges = [ln for ln in dot_source.splitlines() if " -> " in ln]
        assert edges, "no edges rendered at all"
        assert any("color=" in ln for ln in edges)


class TestFreqColorAssignment:
    """Colours are assigned to declared frequencies from a cycle, not a fixed table."""

    def test_distinct_freqs_get_distinct_colours(self):
        colors = assign_freq_colors(
            {"a": "7D", "b": "1ME", "c": "7D"}, dict(DEFAULT_PALETTE)
        )
        assert set(colors) == {"7D", "1ME"}
        assert colors["7D"] != colors["1ME"]

    def test_colours_come_from_the_cycle_in_first_seen_order(self):
        colors = assign_freq_colors({"a": "1ME", "b": "7D"}, dict(DEFAULT_PALETTE))
        assert colors["1ME"] == FREQ_COLOR_CYCLE[0]
        assert colors["7D"] == FREQ_COLOR_CYCLE[1]

    def test_palette_entry_pins_a_frequency(self):
        colors = assign_freq_colors({"a": "7D"}, {**DEFAULT_PALETTE, "7D": "#123456"})
        assert colors["7D"] == "#123456"

    def test_no_frequencies_no_colours(self):
        assert assign_freq_colors({}, dict(DEFAULT_PALETTE)) == {}


class TestCustomStyleFunction:
    def _mock_node(self, tags=None, type_=None, name=""):
        node = MagicMock()
        node.tags = tags or {}
        node.type = type_ or object
        node.name = name
        return node

    def _style(
        self,
        output_vars: "set[str] | frozenset[str]" = frozenset(),
        freq_map: "dict[str, str] | None" = None,
    ):
        freq_map = freq_map if freq_map is not None else {}
        colors = assign_freq_colors(freq_map, dict(DEFAULT_PALETTE))
        return (
            make_style_function(GraphvizSpec(), set(output_vars), freq_map, colors),
            colors,
        )

    def test_declared_freq_node_coloured_and_labelled(self):
        # The fill comes from the node's *declared* frequency, not its name.
        node = self._mock_node(type_=xr.DataArray, name="gpp_smoothed")
        style_fn, colors = self._style(freq_map={"gpp_smoothed": "7D"})
        style, _, label = style_fn(node=node, node_class="default")
        assert style["fillcolor"] == colors["7D"]
        assert label == "7D"

    def test_output_node_gets_highlight_border(self):
        node = self._mock_node(type_=xr.DataArray, name="gpp_monthly")
        style_fn, colors = self._style({"gpp_monthly"}, freq_map={"gpp_monthly": "1ME"})
        style, _, label = style_fn(node=node, node_class="default")
        assert style["color"] == DEFAULT_PALETTE["output"]
        assert "penwidth" in style
        # frequency fill is retained alongside the output border
        assert style["fillcolor"] == colors["1ME"]
        assert label == "output"

    def test_node_without_a_declared_freq_has_empty_style(self):
        # A name suffix alone means nothing now — only declarations count.
        node = self._mock_node(type_=xr.DataArray, name="gpp_daily")
        style_fn, _ = self._style()
        style, _, label = style_fn(node=node, node_class="default")
        assert style == {}
        assert label is None


class TestGraphPostProcessing:
    def test_relabel_replaces_type_with_unit(self):
        digraph = SimpleNamespace(
            body=[
                "\tgpp_weekly [label=<<b>gpp_weekly</b><br /><br /><i>DataArray</i>>]\n",
                "\tlatitude [label=<<b>latitude</b><br /><br /><i>DataArray</i>>]\n",
            ]
        )
        relabel_with_units(digraph, {"gpp_weekly": "g m-2 d-1"})  # type: ignore[arg-type]
        assert "<i>g m-2 d-1</i>" in digraph.body[0]
        # nodes without a declared unit keep their original type
        assert "<i>DataArray</i>" in digraph.body[1]

    def test_relabel_input_table_rows(self):
        row = "<tr><td>temperature_daily</td><td>DataArray</td></tr>"
        other = "<tr><td>latitude</td><td>DataArray</td></tr>"
        digraph = SimpleNamespace(
            body=[f'\t_inputs [label=<<table border="0">{row}{other}</table>>]\n']
        )
        relabel_with_units(digraph, {"temperature_daily": "degC"})  # type: ignore[arg-type]
        assert "<td>temperature_daily</td><td>degC</td>" in digraph.body[0]
        # rows for inputs without a declared unit are untouched
        assert "<td>latitude</td><td>DataArray</td>" in digraph.body[0]

    def test_relabel_input_row_of_a_widened_type(self):
        """An untyped second consumer makes Hamilton render an input as ``Any``."""
        row = "<tr><td>precipitation</td><td>Any</td></tr>"
        other = "<tr><td>scale</td><td>Any</td></tr>"
        digraph = SimpleNamespace(
            body=[f'\t_inputs [label=<<table border="0">{row}{other}</table>>]\n']
        )
        relabel_with_units(digraph, {"precipitation": "mm"})  # type: ignore[arg-type]
        assert "<td>precipitation</td><td>mm</td>" in digraph.body[0]
        assert "<td>scale</td><td>Any</td>" in digraph.body[0]

    @requires_dot
    def test_input_with_an_untyped_consumer_shows_its_unit(self, tmp_path):
        (tmp_path / "typed.py").write_text(
            "from typing import Annotated\n\n"
            "import xarray as xr\n"
            "import xarray_annotated as xra\n\n\n"
            "@xra.check_annotations\n"
            "def wet(precipitation: Annotated[xr.DataArray, xra.Unit('mm')]) -> xr.DataArray:\n"
            "    return precipitation\n"
        )
        config = tmp_path / "config.toml"
        config.write_text(
            """\
[typed]
_import_path = "typed.py"

[[node]]
name = "doubled"
inputs = ["precipitation"]
expression = "precipitation * 2"
"""
        )
        source = build_graph(config).source
        assert "<td>precipitation</td><td>mm</td>" in source

    def test_color_edges_by_source_frequency(self):
        digraph = SimpleNamespace(
            body=[
                "\ttemperature_weekly -> gpp_weekly\n",
                "\tlatitude -> gpp_weekly\n",
            ]
        )
        color_edges_by_frequency(
            digraph,  # type: ignore[arg-type]
            {"temperature_weekly": "7D"},
            {"7D": "#8da0cb"},
        )
        assert 'color="#8da0cb"' in digraph.body[0]
        # edges from unknown-frequency sources are untouched
        assert "color=" not in digraph.body[1]

    def test_infer_frequencies_by_neighbour_consensus(self):
        # mymodel: weekly in, weekly out -> weekly.
        edges = [("temperature_weekly", "mymodel"), ("mymodel", "gpp_weekly")]
        freq = infer_frequencies(
            edges, {"temperature_weekly": "weekly", "gpp_weekly": "weekly"}
        )
        assert freq["mymodel"] == "weekly"

    def test_infer_frequencies_conflict_stays_unresolved(self):
        # a node bridging daily and monthly has no consensus -> not assigned.
        edges = [("temperature_daily", "bridge"), ("bridge", "soc_monthly")]
        freq = infer_frequencies(
            edges, {"temperature_daily": "daily", "soc_monthly": "monthly"}
        )
        assert "bridge" not in freq

    def test_cluster_groups_nodes_by_frequency(self):
        digraph = SimpleNamespace(
            body=[
                "\tgpp_weekly [label=<<b>gpp_weekly</b>>]\n",
                "\ttemperature_daily [label=<<b>temperature_daily</b>>]\n",
                "\tsurface_type [label=<<b>surface_type</b>>]\n",  # ungrouped
                "\t_gpp_weekly_inputs [label=<<table></table>>]\n",  # joins 7D
                "\ttemperature_daily -> gpp_weekly\n",
                "\t_gpp_weekly_inputs -> gpp_weekly\n",
            ]
        )
        cluster_nodes_by_frequency(
            digraph,  # type: ignore[arg-type]
            {"gpp_weekly": "7D", "temperature_daily": "1D"},
            {"7D": "#8da0cb", "1D": "#fc8d62", "1ME": "#a6d854"},
        )
        source = "".join(digraph.body)
        assert "subgraph cluster_7D {" in source
        assert "subgraph cluster_1D {" in source
        # a frequency with no members emits no empty cluster
        assert "cluster_1ME" not in source
        # the input table joins the cluster of the node it feeds
        weekly = source.split("cluster_7D {", 1)[1].split("}", 1)[0]
        assert "_gpp_weekly_inputs" in weekly
        assert "gpp_weekly [label" in weekly
        # ungrouped nodes stay outside any cluster
        plant_idx = source.index("surface_type [label")
        assert plant_idx > source.index("}")  # after the last cluster brace
        # every node is declared before any edge (clustering pitfall guard)
        assert source.index("gpp_weekly [label") < source.index(" -> ")

    def test_cluster_id_is_sanitised_and_label_quoted(self):
        # An anchored offset alias contains a hyphen, which Graphviz will not accept
        # in a bare cluster id.
        digraph = SimpleNamespace(body=["\tgpp [label=<<b>gpp</b>>]\n"])
        cluster_nodes_by_frequency(
            digraph,  # type: ignore[arg-type]
            {"gpp": "W-SUN"},
            {"W-SUN": "#8da0cb"},
        )
        source = "".join(digraph.body)
        assert "subgraph cluster_W_SUN {" in source
        assert 'label="W-SUN"' in source


class TestConfigInputs:
    """A node's config inputs get a table of their own, apart from its data."""

    @pytest.fixture
    def config_source(self, tmp_path) -> str:
        (tmp_path / "scaling.py").write_text(
            "import xarray as xr\n\n\n"
            "def scaled(\n"
            "    precipitation: xr.DataArray, *, factor: float = 2.0, mode: str = 'a'\n"
            ") -> xr.DataArray:\n"
            "    return precipitation * factor\n\n\n"
            "def flagged(scaled: xr.DataArray, *, strict: bool = False) -> xr.DataArray:\n"
            "    return scaled\n"
        )
        config = tmp_path / "config.toml"
        config.write_text('[scaling]\n_import_path = "scaling.py"\nfactor = 3.0\n')
        return build_graph(config).source

    @staticmethod
    def _line(source: str, node_id: str) -> str:
        return next(
            line
            for line in source.splitlines()
            if line.strip().startswith(f"{node_id} [")
        )

    @requires_dot
    def test_config_leaves_the_input_table(self, config_source):
        inputs = self._line(config_source, "_scaled_inputs")
        assert "<td>precipitation</td>" in inputs
        assert "factor" not in inputs
        assert "mode" not in inputs

    @requires_dot
    def test_config_table_shows_values(self, config_source):
        config = self._line(config_source, "_scaled_config")
        assert "shape=note" in config
        # set in the config, else the signature default
        assert "<td>factor</td><td>3.0</td>" in config
        assert "<td>mode</td><td>&quot;a&quot;</td>" in config
        assert "_scaled_config -> scaled" in config_source

    @requires_dot
    def test_input_table_of_only_config_is_removed(self, config_source):
        assert "_flagged_inputs" not in config_source
        assert "<td>strict</td><td>false</td>" in self._line(
            config_source, "_flagged_config"
        )

    @requires_dot
    def test_floating_config_box_is_dropped_once_tabled(self, config_source):
        assert not any(
            line.strip().startswith("factor [") for line in config_source.splitlines()
        )

    @requires_dot
    def test_catena_own_hamilton_setting_is_not_drawn(self, config_source):
        assert "hamilton.enable_power_user_mode" not in config_source

    @requires_dot
    def test_legend_has_a_config_entry(self, config_source):
        legend = config_source.split("cluster__legend", 1)[1]
        assert "config [" in legend

    def test_no_config_inputs_leaves_the_body_alone(self):
        body = [
            '\t_f_inputs [label=<<table border="0"><tr><td>x</td><td>1</td></tr></table>>]\n'
        ]
        digraph = SimpleNamespace(body=list(body))
        separate_config_inputs(digraph, {})  # type: ignore[arg-type]
        assert digraph.body == body


_VIEWS_MODULE = """\
from typing import Annotated

import xarray as xr
from xarray_annotated import Freq, Unit

DailyK = Annotated[xr.DataArray, Unit("K"), Freq("D")]
DailyC = Annotated[xr.DataArray, Unit("degC"), Freq("D")]
Rain = Annotated[xr.DataArray, Unit("mm"), Freq("D")]


def daily_a(temperature: DailyK, precipitation: Rain, *, offset: float = 0.0) -> DailyC:
    return temperature - 273.15 + offset


def daily_b(temperature: DailyK, precipitation: Rain, *, offset: float = 0.0) -> DailyC:
    return temperature - 273.15 + offset


def combine(daily_a: DailyC) -> Annotated[xr.DataArray, Unit("degC")]:
    return daily_a


def weekly(
    combine: Annotated[xr.DataArray, Unit("degC")],
) -> Annotated[xr.DataArray, Unit("degC"), Freq("7D")]:
    return combine.resample(time="7D").mean()


def unrelated(precipitation: Rain) -> Rain:
    return precipitation
"""


def _styled(dr, view: str, *args, outputs=(), **kwargs):
    """Draw ``view`` with catena's styling."""
    style = GraphStyle.from_driver(dr, outputs=outputs)
    return style.apply(getattr(dr, view)(*args, **style.hamilton_kwargs(), **kwargs))


def _node_line(source: str, node_id: str) -> str:
    return next(
        line for line in source.splitlines() if line.strip().startswith(f"{node_id} [")
    )


def _cluster(source: str, freq: str) -> str:
    """Return the body of the cluster for ``freq``, or ``""`` if there is none."""
    _, found, rest = source.partition(f"subgraph cluster_{freq} {{")
    return rest.split("\t}", 1)[0] if found else ""


@pytest.fixture(scope="module")
def views_driver(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("views")
    (tmp_path / "views.py").write_text(_VIEWS_MODULE)
    config = tmp_path / "config.toml"
    config.write_text('[views]\n_import_path = "views.py"\n')
    parsed, _ = prepare_config(config)
    return build_driver(
        modules=parsed.modules,
        config=parsed.driver_config,
        node_specs=parsed.node_specs,
        base=parsed.base,
        registered=parsed.registered_modules,
    )


@pytest.fixture(scope="module")
def upstream(views_driver) -> str:
    return _styled(
        views_driver, "display_upstream_of", "weekly", outputs=["weekly"]
    ).source


@pytest.fixture(scope="module")
def path(views_driver) -> str:
    return _styled(views_driver, "visualize_path_between", "daily_a", "weekly").source


@requires_dot
class TestHamiltonViews:
    """catena's styling applied to Hamilton's other display methods.

    Hamilton's views share one renderer, so the body surgery should hold for a
    pruned or path view as it does for the whole DAG.
    """

    def test_upstream_view_draws_only_what_the_node_needs(self, upstream):
        assert "weekly [" in upstream
        assert "daily_a [" in upstream
        assert "daily_b [label" not in upstream
        assert "unrelated" not in upstream

    @pytest.mark.xfail(strict=True, reason="config tables cover the whole DAG")
    def test_upstream_view_omits_config_of_undrawn_nodes(self, upstream):
        # The dangling edge would make Graphviz draw a bare `daily_b` node.
        assert "daily_b" not in upstream

    def test_upstream_view_keeps_units(self, upstream):
        assert "<i>degC</i>" in _node_line(upstream, "daily_a")
        inputs = _node_line(upstream, "_daily_a_inputs")
        assert "<td>temperature</td><td>K</td>" in inputs
        assert "<td>precipitation</td><td>mm</td>" in inputs

    def test_upstream_view_separates_config(self, upstream):
        assert "offset" not in _node_line(upstream, "_daily_a_inputs")
        assert "<td>offset</td><td>0.0</td>" in _node_line(upstream, "_daily_a_config")

    def test_upstream_view_clusters_and_outlines_outputs(self, upstream):
        assert "daily_a [" in _cluster(upstream, "D")
        assert "weekly [" in _cluster(upstream, "7D")
        assert DEFAULT_PALETTE["output"] in _node_line(upstream, "weekly")

    def test_upstream_view_renders(self, views_driver):
        digraph = _styled(views_driver, "display_upstream_of", "weekly")
        assert digraph.pipe(format="svg")

    def test_path_view_keeps_hamiltons_path_colour(self, path):
        edge = next(ln for ln in path.splitlines() if "daily_a -> combine" in ln)
        # catena's frequency colour is there, but Hamilton's red comes last and wins
        assert edge.rstrip().endswith("color=red]")
        assert "color=red" in _node_line(path, "combine")

    def test_path_view_renders(self, views_driver):
        digraph = _styled(views_driver, "visualize_path_between", "daily_a", "weekly")
        assert digraph.pipe(format="svg")

    def test_deduplicated_inputs_keep_each_nodes_config(self, views_driver):
        source = _styled(
            views_driver, "display_all_functions", deduplicate_inputs=True
        ).source
        tables = [
            ln for ln in source.splitlines() if "_inputs [" in ln and "<table" in ln
        ]
        shared = [ln for ln in tables if "<td>temperature</td>" in ln]
        assert len(shared) == 1, "Hamilton should draw one shared input table"
        assert "offset" not in shared[0]
        assert "<td>offset</td>" in _node_line(source, "_daily_a_config")
        assert "<td>offset</td>" in _node_line(source, "_daily_b_config")
        assert "_daily_a_config -> daily_a" in source
        assert "_daily_b_config -> daily_b" in source

    def test_cluster_does_not_depend_on_the_view(self, views_driver):
        # `combine` bridges D and 7D, so in the whole DAG it joins neither.
        whole = _styled(views_driver, "display_all_functions").source
        assert "combine [" not in _cluster(whole, "D")
        pruned = _styled(views_driver, "display_upstream_of", "combine").source
        assert "combine [" not in _cluster(pruned, "D")

    @pytest.mark.xfail(strict=True, reason="config tables ignore hide_inputs")
    def test_hide_inputs_hides_config_tables(self, views_driver):
        source = _styled(views_driver, "display_all_functions", hide_inputs=True).source
        assert "_inputs [" not in source
        assert "_config [" not in source

    @pytest.mark.xfail(
        strict=True, reason="Hamilton keeps the last output's style for the legend"
    )
    def test_output_legend_swatch_has_no_frequency_fill(self, views_driver):
        source = _styled(
            views_driver, "display_all_functions", outputs=["daily_a", "weekly"]
        ).source
        legend = source.split("cluster__legend", 1)[1]
        swatch = next(
            ln for ln in legend.splitlines() if ln.strip().startswith("output [")
        )
        assert FREQ_COLOR_CYCLE[0] not in swatch
        assert FREQ_COLOR_CYCLE[1] not in swatch


class TestComposeStyles:
    @staticmethod
    def _fn(attrs, legend=None):
        return lambda *, node, node_class: (dict(attrs), node_class, legend)

    def test_later_attributes_win(self):
        composed = compose_styles(
            self._fn({"fillcolor": "red", "color": "blue"}),
            self._fn({"fillcolor": "green"}),
        )
        style, base, legend = composed(node=None, node_class="function")
        assert style == {"fillcolor": "green", "color": "blue"}
        assert (base, legend) == ("function", None)

    def test_last_named_legend_wins(self):
        composed = compose_styles(
            self._fn({}, legend="7D"), self._fn({}, legend="output"), self._fn({})
        )
        _, _, legend = composed(node=None, node_class="function")
        assert legend == "output"


_LINKS_MODULE = '''\
from typing import TypedDict

import xarray as xr
from hamilton.function_modifiers import config, extract_fields


class Parts(TypedDict):
    left: xr.DataArray
    right: xr.DataArray


@extract_fields({"left": xr.DataArray, "right": xr.DataArray})
def split(whole: xr.DataArray) -> Parts:
    """Split the whole in two.

    Longer description that the tooltip leaves out.
    """
    return {"left": whole, "right": whole}


@config.when(method="equilibrate")
def start__equilibrate(left: xr.DataArray) -> xr.DataArray:
    """Spin up to equilibrium."""
    return left


def undocumented(right: xr.DataArray) -> xr.DataArray:
    return right
'''


def _url(fn) -> str:
    return f"api.html#{fn.__qualname__}"


@pytest.fixture(scope="module")
def links_driver(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("links")
    (tmp_path / "links.py").write_text(_LINKS_MODULE)
    config = tmp_path / "config.toml"
    config.write_text('[links]\n_import_path = "links.py"\nmethod = "equilibrate"\n')
    parsed, _ = prepare_config(config)
    return build_driver(
        modules=parsed.modules,
        config=parsed.driver_config,
        base=parsed.base,
        registered=parsed.registered_modules,
    )


@pytest.fixture(scope="module")
def links_source(links_driver) -> str:
    style = GraphStyle.from_driver(links_driver, link=_url)
    return style.apply(
        links_driver.display_all_functions(**style.hamilton_kwargs())
    ).source


@requires_dot
class TestLinks:
    def test_function_node_links_to_its_function(self, links_source):
        assert 'URL="api.html#split"' in _node_line(links_source, "split")

    def test_tooltip_is_the_first_docstring_paragraph(self, links_source):
        line = _node_line(links_source, "split")
        assert 'tooltip="Split the whole in two."' in line

    def test_field_nodes_are_not_linked(self, links_source):
        assert "URL=" not in _node_line(links_source, "left")
        assert "URL=" not in _node_line(links_source, "right")

    def test_config_when_node_links_by_qualname(self, links_source):
        assert 'URL="api.html#start__equilibrate"' in _node_line(links_source, "start")

    def test_undocumented_function_gets_no_tooltip(self, links_source):
        line = _node_line(links_source, "undocumented")
        assert "URL=" in line
        assert "tooltip=" not in line

    def test_inputs_are_not_linked(self, links_source):
        assert "URL=" not in _node_line(links_source, "_split_inputs")

    def test_link_returning_none_draws_no_link(self, links_driver):
        style = GraphStyle.from_driver(
            links_driver, link=lambda fn: None if fn.__name__ == "split" else "x"
        )
        source = links_driver.display_all_functions(**style.hamilton_kwargs()).source
        assert "URL=" not in _node_line(source, "split")
        assert "URL=" in _node_line(source, "undocumented")

    def test_link_style_composes_with_hamilton_directly(self, links_driver):
        links = link_style(links_driver, _url, tooltip=False)
        source = links_driver.display_upstream_of(
            "start", custom_style_function=links
        ).source
        line = _node_line(source, "start")
        assert "URL=" in line
        assert "tooltip=" not in line

    def test_style_file_names_a_link_function(self, tmp_path, monkeypatch):
        (tmp_path / "docs_links.py").write_text(
            "def api_url(fn):\n    return 'docs/' + fn.__name__\n"
        )
        (tmp_path / "links.py").write_text(_LINKS_MODULE)
        monkeypatch.syspath_prepend(str(tmp_path))
        style = tmp_path / "style.toml"
        style.write_text('link_function = "docs_links:api_url"\nlink_tooltip = false\n')
        config = tmp_path / "config.toml"
        config.write_text(
            '[links]\n_import_path = "links.py"\nmethod = "equilibrate"\n'
        )
        source = build_graph(config, style=style).source
        line = _node_line(source, "split")
        assert 'URL="docs/split"' in line
        assert "tooltip=" not in line

    def test_style_function_is_composed_over_the_builtin(self, tmp_path, monkeypatch):
        (tmp_path / "my_style.py").write_text(
            "def style(*, node, node_class):\n"
            "    return {'fontcolor': 'navy'}, node_class, None\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        style = tmp_path / "style.toml"
        style.write_text('style_function = "my_style:style"\n')
        (tmp_path / "views.py").write_text(_VIEWS_MODULE)
        config = tmp_path / "config.toml"
        config.write_text('[views]\n_import_path = "views.py"\n')
        source = build_graph(config, style=style).source
        line = _node_line(source, "weekly")
        assert "fontcolor=navy" in line
        # the built-in frequency fill is still there underneath
        assert FREQ_COLOR_CYCLE[1] in line or FREQ_COLOR_CYCLE[0] in line


class TestGraphvizSpec:
    def test_none_returns_defaults(self):
        spec = load_graphviz_spec(None)
        assert spec.palette == DEFAULT_PALETTE
        assert spec.style_function is None
        assert spec.show_legend is True
        assert spec.cluster_by_frequency is True

    def test_cluster_by_frequency_can_be_disabled(self, tmp_path):
        f = tmp_path / "style.toml"
        f.write_text("cluster_by_frequency = false\n")
        spec = load_graphviz_spec(f)
        assert spec.cluster_by_frequency is False

    def test_partial_palette_is_deep_merged(self, tmp_path):
        f = tmp_path / "style.toml"
        f.write_text('[palette]\n"7D" = "#000000"\n')
        spec = load_graphviz_spec(f)
        assert spec.palette["7D"] == "#000000"
        # untouched categories fall back to the defaults
        assert spec.palette["output"] == DEFAULT_PALETTE["output"]

    def test_graph_attr_collected_into_kwargs(self, tmp_path):
        f = tmp_path / "style.toml"
        f.write_text('[graph_attr]\nrankdir = "TB"\n')
        spec = load_graphviz_spec(f)
        assert spec.graphviz_kwargs == {"graph_attr": {"rankdir": "TB"}}

    def test_unknown_key_raises(self, tmp_path):
        f = tmp_path / "style.toml"
        f.write_text("bogus = 1\n")
        with pytest.raises(ValueError, match="Unknown key"):
            load_graphviz_spec(f)


class TestImportStyleFunction:
    def test_imports_module_function(self):
        import os.path

        assert _import_function("os.path:join", "style_function") is os.path.join

    def test_rejects_malformed_path(self):
        with pytest.raises(ValueError, match="module:function"):
            _import_function("not_a_reference", "style_function")


class TestStrayGraphvizSection:
    def test_science_config_rejects_graphviz_section(self, tmp_path):
        # Styling belongs in a `graph --style` file, not the science config. It used
        # to be swallowed silently, which masks typos; now it is an error.
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[graphviz]\nshow_legend = true\n"
            '[[node]]\nname = "y"\ninputs = ["x"]\nexpression = "x * 2"\n'
        )
        with pytest.raises(ValueError, match="_import_path"):
            load_config(cfg)
