"""Build a styled Graphviz visualisation of a configured pipeline.

`build_graph` returns a ``graphviz.Digraph``, which renders inline in a notebook.
Write it with ``digraph.source`` or render it with ``digraph.pipe(format=...)``.

`GraphStyle` applies the same styling to any of Hamilton's ``Driver`` display
methods, such as ``visualize_path_between`` or ``display_downstream_of``: pass
`GraphStyle.hamilton_kwargs` to the method, then `GraphStyle.apply` to the
digraph it returns.

**Everything shown is read off the built DAG's contracts.**
Node fills and edge colours come from declared `xarray_annotated.temporal.Freq`
contracts, and node labels carry each node's declared unit in place of the
near-universal ``DataArray`` type text. A node with no declared frequency is
left uncoloured.

Each node's data inputs and config inputs are drawn as separate tables. A config
input is one the driver config sets, or a keyword-only parameter of the node's
function; its table shows the value in use, from the config or else the default.

Styling defaults live in `catena.graph_style`; a ``--style`` TOML file is
deep-merged over them, and may name a ``style_function`` composed over the
built-in one, and a ``link_function`` that links nodes to their documentation.
"""

import html
import importlib
import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hamilton import graph_types
from hamilton.driver import Driver
from hamilton.settings import ENABLE_POWER_USER_MODE

from .build import build_driver
from .contract_check import _collect_declarations
from .errors import CatenaValueError
from .graph_style import FREQ_COLOR_CYCLE, GraphvizSpec, load_graphviz_spec
from .io import get_final_vars
from .pipeline import ConfigSource, prepare_config

if TYPE_CHECKING:
    # graphviz ships in the optional `viz` extra, and `catena.build_graph` is
    # re-exported from the package root, so importing it here would make
    # `import catena` fail for anyone who did not install that extra.
    import graphviz

__all__ = ["GraphStyle", "build_graph", "compose_styles", "link_style"]

StyleFunction = Callable[..., tuple[dict, str | None, str | None]]


def make_style_function(
    spec: GraphvizSpec,
    output_vars: set[str],
    freq_map: dict[str, str],
    freq_colors: dict[str, str],
) -> StyleFunction:
    """Build the default ``custom_style_function`` for ``display_all_functions``.

    Nodes are filled by their **declared frequency** (``freq_map``, read off the DAG's
    contracts — not guessed from name suffixes), and nodes corresponding to requested
    outputs additionally get a thick coloured border so the pipeline's end products
    stand out.  Units are *not* handled here (the style hook cannot set a node's
    label); they are injected later by :func:`relabel_with_units`.
    """
    palette = spec.palette

    def custom_style(
        *, node: graph_types.HamiltonNode, node_class: str
    ) -> tuple[dict, str | None, str | None]:
        freq = freq_map.get(node.name)

        style: dict = {}
        if freq is not None:
            style["fillcolor"] = freq_colors[freq]

        if node.name in output_vars:
            style["color"] = palette["output"]
            style["penwidth"] = "2.5"
            return style, node_class, "output"

        return style, node_class, freq

    return custom_style


def compose_styles(*style_functions: StyleFunction) -> StyleFunction:
    r"""Combine Hamilton ``custom_style_function``\ s into one.

    Hamilton's display methods take a single style function. Each node's style
    dicts are merged left to right, so a later function overrides an earlier
    one's attributes. The legend entry comes from the last function that names
    one.
    """

    def composed(
        *, node: graph_types.HamiltonNode, node_class: str
    ) -> tuple[dict, str | None, str | None]:
        style: dict = {}
        base: str | None = node_class
        legend: str | None = None
        for function in style_functions:
            attrs, fn_base, fn_legend = function(node=node, node_class=node_class)
            style.update(attrs)
            if fn_legend is not None:
                base, legend = fn_base, fn_legend
        return style, base, legend

    return composed


LinkFunction = Callable[[Callable], str | None]


def _own_functions(dr: Driver) -> dict[str, Callable]:
    """Return ``node -> function`` for the nodes a function itself defines.

    A field node (one per ``@extract_fields`` output) shares its originating
    function with the node it depends on, so it is left out; external inputs,
    which Hamilton gives the consuming function as their originating function,
    are left out too.
    """
    own: dict[str, Callable] = {}
    for n in dr.graph.nodes.values():
        if n.user_defined or not n.originating_functions:
            continue
        fn = n.originating_functions[0]
        if any(
            not dep.user_defined
            and dep.originating_functions
            and dep.originating_functions[0] is fn
            for dep in n.dependencies
        ):
            continue
        own[n.name] = fn
    return own


def _summary(fn: Callable) -> str | None:
    """Return the first paragraph of ``fn``'s docstring, on one line."""
    doc = inspect.getdoc(fn)
    if not doc:
        return None
    return " ".join(doc.split("\n\n", 1)[0].split())


def _link_style(
    own: Mapping[str, Callable], link: LinkFunction, *, tooltip: bool
) -> StyleFunction:
    def style(
        *, node: graph_types.HamiltonNode, node_class: str
    ) -> tuple[dict, str | None, str | None]:
        fn = own.get(node.name)
        url = link(fn) if fn is not None else None
        if fn is None or url is None:
            return {}, node_class, None
        attrs = {"URL": url}
        if tooltip and (summary := _summary(fn)) is not None:
            attrs["tooltip"] = summary
        return attrs, node_class, None

    return style


def link_style(
    dr: Driver, link: LinkFunction, *, tooltip: bool = True
) -> StyleFunction:
    """Build a Hamilton ``custom_style_function`` that links nodes to their docs.

    Each node a function defines gets ``URL`` set to ``link(function)``, which
    Graphviz renders as a link in SVG output. Field nodes from
    ``@extract_fields`` share their parent's function and get no link, nor do
    inputs and config.

    Parameters
    ----------
    dr
        The driver whose DAG will be drawn.
    link
        Receives a node's originating function and returns its URL, or ``None``
        for no link. ``function.__module__`` and ``function.__qualname__`` give
        the dotted path that most API documentation tools anchor on.
    tooltip
        Also set ``tooltip`` to the first paragraph of the function's docstring.

    Examples
    --------
    Compose it with catena's own styling::

        style = GraphStyle.from_driver(dr)
        links = link_style(dr, lambda fn: f"api.html#{fn.__module__}.{fn.__qualname__}")
        digraph = dr.display_all_functions(
            **{
                **style.hamilton_kwargs(),
                "custom_style_function": compose_styles(style.node_style, links),
            }
        )
    """
    return _link_style(_own_functions(dr), link, tooltip=tooltip)


def _import_function(path: str, key: str) -> Callable:
    """Import a ``"module:function"`` reference given under ``key``."""
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise CatenaValueError(
            f"{key} must be of the form 'module:function', got {path!r}."
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _node_maps(dr: Driver) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(name -> unit, name -> frequency)`` maps for the DAG's nodes.

    Both are read from the declarations the contract checks use. A node's
    *produced* declaration takes precedence over a *consumed* one (an input
    parameter's), so pure input nodes are covered too.

    The frequency map comes from **declared** `xarray_annotated.temporal.Freq`
    contracts (``"7D"``, ``"1ME"``), not from name suffixes. Section labels are inert
    (`catena.io.load_inputs`), so a pipeline whose resample targets are called
    ``raw``/``smoothed`` is grouped exactly as well as one using ``daily``/``weekly``.
    """
    produced, consumed, _ = _collect_declarations(dr)
    units: dict[str, str] = {}
    freqs: dict[str, str] = {}
    for name, consumers in consumed.items():
        for decl, _ in consumers:
            if decl.unit is not None:
                units.setdefault(name, decl.unit.unit)
            if decl.freq is not None:
                freqs.setdefault(name, decl.freq.freq)
    for name, (decl, _) in produced.items():
        if decl.unit is not None:
            units[name] = decl.unit.unit
        if decl.freq is not None:
            freqs[name] = decl.freq.freq
    return units, freqs


def assign_freq_colors(
    freq_map: dict[str, str], palette: dict[str, str]
) -> dict[str, str]:
    """Map each distinct declared frequency to a colour.

    Frequencies are arbitrary strings, not a fixed vocabulary, so colours are
    drawn from a qualitative **cycle** in first-seen order rather than looked up in a
    fixed ``daily``/``weekly``/``monthly`` table. A ``[palette]`` entry naming a
    frequency (``"7D" = "#ff7f00"``) pins that one, for users who want stable colours
    across renders.
    """
    colors: dict[str, str] = {}
    taken = 0
    for freq in dict.fromkeys(freq_map.values()):  # distinct, first-seen order
        if freq in palette:
            colors[freq] = palette[freq]
        else:
            colors[freq] = FREQ_COLOR_CYCLE[taken % len(FREQ_COLOR_CYCLE)]
            taken += 1
    return colors


# A single ``<td>name</td><td>type</td>`` cell pair inside an input table. The type is
# not always ``DataArray``: an untyped second consumer widens it to ``Any``.
_INPUT_ROW_RE = re.compile(r"<td>(?P<name>[^<]+)</td><td>[^<]*</td>")


def _config_inputs(
    dr: Driver, driver_config: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Return ``node -> {config input -> displayed value}`` for the DAG's nodes.

    A config input is an external input that is a keyword-only parameter of the
    consuming function, or that the driver config sets. Its value is the
    config's, else the signature default, else ``required``.
    """
    result: dict[str, dict[str, str]] = {}
    for n in dr.graph.nodes.values():
        if n.user_defined or not n.originating_functions:
            continue
        params = inspect.signature(n.originating_functions[0]).parameters
        found: dict[str, str] = {}
        for name in n.input_types:
            dep = dr.graph.nodes.get(name)
            if dep is None or not dep.user_defined:
                continue
            param = params.get(name)
            if name in driver_config:
                found[name] = _format_config_value(driver_config[name])
            elif param is None or param.kind is not param.KEYWORD_ONLY:
                continue
            elif param.default is not param.empty:
                found[name] = _format_config_value(param.default)
            else:
                found[name] = "required"
        if found:
            result[n.name] = found
    return result


def _format_config_value(value: Any) -> str:
    """Render a config value as it would be written in TOML."""
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


# Hamilton's own style for config nodes, plus the margin its input tables use.
_CONFIG_STYLE = "fontname=Helvetica margin=0.15 shape=note style=filled"
_LEGEND_CONFIG = "\t\tconfig [fontname=Helvetica shape=note style=filled]\n"
_TABLE_ROW_RE = re.compile(r"<tr><td>(?P<name>[^<]+)</td><td>[^<]*</td></tr>")


def _node_id(line: str) -> str:
    return line.split("[label=", 1)[0].strip().strip('"')


def _config_table(target: str, keys: dict[str, str]) -> list[str]:
    """Return the DOT lines for ``target``'s config table and its edge."""
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td></tr>"
        for key, value in keys.items()
    )
    name = f"_{target}_config"
    return [
        f'\t{name} [label=<<table border="0">{rows}</table>> {_CONFIG_STYLE}]\n',
        f"\t{name} -> {target}\n",
    ]


def separate_config_inputs(
    digraph: "graphviz.Digraph", config_inputs: dict[str, dict[str, str]]
) -> None:
    """Move each node's config inputs into a table of their own, in-place.

    Hamilton lists every external input of a node in one ``_<fn>_inputs``
    table, so data and config read alike. Here the config rows leave that
    table for a ``_<fn>_config`` table in Hamilton's config style, which shows
    each value rather than its type. Hamilton's free-floating box for a config
    key is dropped, as is the one for the Hamilton setting `catena.build_driver`
    adds, and an input table left empty is removed along with its edge.

    Only nodes in ``digraph`` get a config table, so a pruned view stays pruned.
    If Hamilton drew no input tables at all (``hide_inputs=True``), no config
    tables are drawn either, and the config boxes are left as Hamilton drew them.
    """
    drawn = {
        _node_id(line)
        for line in digraph.body
        if " -> " not in line and "[label=" in line
    }
    if any(node_id.endswith("_inputs") for node_id in drawn):
        hidden = {key for keys in config_inputs.values() for key in keys}
        config_inputs = {t: k for t, k in config_inputs.items() if t in drawn}
    else:
        hidden, config_inputs = set(), {}
    hidden.add(ENABLE_POWER_USER_MODE)
    shown = {key for keys in config_inputs.values() for key in keys}

    # Strip the config rows first, so the edges of emptied tables are known
    # whichever order Hamilton wrote the definitions and edges in.
    tables: dict[int, str] = {}
    emptied: set[str] = set()
    for i, line in enumerate(digraph.body):
        if " -> " in line or "[label=<<table" not in line:
            continue
        node_id = _node_id(line)
        target = node_id.removeprefix("_").removesuffix("_inputs")
        if not node_id.endswith("_inputs") or target not in config_inputs:
            continue
        tables[i] = _TABLE_ROW_RE.sub(
            lambda m, keys=config_inputs[target]: (
                "" if m.group("name") in keys else m.group(0)
            ),
            line,
        )
        if "<tr>" not in tables[i]:
            emptied.add(node_id)

    body: list[str] = []
    pending = dict(config_inputs)
    in_legend = False
    legend_at: int | None = None
    legend_has_config = False
    for i, line in enumerate(digraph.body):
        if "subgraph cluster__legend" in line:
            in_legend = True
            legend_at = len(body)
        if in_legend:
            legend_has_config |= line.strip().startswith("config [")
            in_legend = line.strip() != "}"
            body.append(line)
        elif " -> " in line:
            if line.split(" -> ", 1)[0].strip().strip('"') not in emptied:
                body.append(line)
        elif "[label=" in line and _node_id(line) in hidden and "shape=note" in line:
            continue
        elif i in tables:
            target = _node_id(line).removeprefix("_").removesuffix("_inputs")
            if _node_id(line) not in emptied:
                body.append(tables[i])
            body.extend(_config_table(target, pending.pop(target)))
        else:
            body.append(line)

    # A node whose external inputs are all config has no input table.
    for target, keys in pending.items():
        body.extend(_config_table(target, keys))

    if shown and legend_at is not None and not legend_has_config:
        # Hamilton orders the legend config first, after the graph attributes.
        body.insert(legend_at + 2, _LEGEND_CONFIG)
    digraph.body[:] = body


_LEGEND_ATTR_RE = re.compile(r'(?:URL|tooltip)="(?:[^"\\]|\\.)*" ?')
_FILL_RE = re.compile(r'fillcolor=(?:"[^"]*"|\S+) ?')


def clean_legend(digraph: "graphviz.Digraph") -> None:
    """Strip per-node attributes from Hamilton's legend, in place.

    Hamilton draws a legend entry with the style of the last node that named
    it, so the ``output`` swatch would carry that node's frequency fill and any
    entry its link. Links and tooltips are dropped from every entry, and the
    ``output`` swatch keeps only its outline.
    """
    in_legend = False
    for i, line in enumerate(digraph.body):
        if "subgraph cluster__legend" in line:
            in_legend = True
        if not in_legend:
            continue
        if line.strip() == "}":
            break
        line = _LEGEND_ATTR_RE.sub("", line)
        if line.strip().startswith("output ["):
            line = _FILL_RE.sub("", line)
        digraph.body[i] = line.replace(" ]", "]")


def relabel_with_units(digraph: "graphviz.Digraph", unit_map: dict[str, str]) -> None:
    """Rewrite node labels in-place so the unit replaces the ``DataArray`` type.

    Hamilton renders every function node's label as ``name`` over the type string
    ``DataArray`` (it strips the ``Annotated`` wrapper), and groups inputs into
    tables of ``name``/type rows.  Since virtually every node is a ``DataArray``,
    that type text is noise; here we swap it for the declared unit in both places.
    Nodes with no declared unit keep their original (informative) type.
    """

    def _row(match: "re.Match[str]") -> str:
        unit = unit_map.get(match.group("name"))
        if unit is None:
            return match.group(0)
        return f"<td>{match.group('name')}</td><td>{html.escape(unit)}</td>"

    for i, line in enumerate(digraph.body):
        if "[label=<" not in line:
            continue
        name = line.split("[label=", 1)[0].strip().strip('"')
        unit = unit_map.get(name)
        if unit:
            line = line.replace("<i>DataArray</i>", f"<i>{html.escape(unit)}</i>", 1)
        if "<table" in line:
            line = _INPUT_ROW_RE.sub(_row, line)
        digraph.body[i] = line


def color_edges_by_frequency(
    digraph: "graphviz.Digraph", freq_map: dict[str, str], freq_colors: dict[str, str]
) -> None:
    """Colour each edge in-place by the declared frequency of its source node."""
    for i, line in enumerate(digraph.body):
        if " -> " not in line:
            continue
        src = line.split(" -> ", 1)[0].strip().strip('"')
        freq = freq_map.get(src)
        if freq is None:
            continue
        color = freq_colors[freq]
        if "[" in line:
            digraph.body[i] = line.replace("[", f'[color="{color}" ', 1)
        else:
            digraph.body[i] = line.rstrip("\n") + f' [color="{color}"]\n'


class GraphStyle:
    """catena's styling for a driver's DAG, applicable to any Hamilton view.

    Build it once from a driver, pass `hamilton_kwargs` to any of Hamilton's
    display methods, then `apply` it to the digraph that method returns::

        style = GraphStyle.from_driver(dr, outputs=["gpp_weekly"])
        digraph = dr.visualize_path_between(
            "nee_raw", "gpp_weekly", **style.hamilton_kwargs()
        )
        style.apply(digraph)

    Everything shown is read off the whole DAG, so a node is styled the same
    in every view that draws it.
    """

    def __init__(
        self,
        *,
        spec: GraphvizSpec,
        outputs: frozenset[str],
        units: dict[str, str],
        freqs: dict[str, str],
        freq_colors: dict[str, str],
        config_inputs: dict[str, dict[str, str]],
        own_functions: dict[str, Callable],
        link: LinkFunction | None,
    ) -> None:
        self.spec = spec
        self.outputs = outputs
        self._units = units
        self._freqs = freqs
        self._freq_colors = freq_colors
        self._config_inputs = config_inputs
        self._own_functions = own_functions
        self._link = link

    @classmethod
    def from_driver(
        cls,
        dr: Driver,
        *,
        outputs: Iterable[str] = (),
        spec: GraphvizSpec | None = None,
        link: LinkFunction | None = None,
    ) -> "GraphStyle":
        """Read the styling off ``dr``'s DAG.

        Parameters
        ----------
        dr
            The driver whose DAG will be drawn.
        outputs
            Node names to outline as the pipeline's end products.
        spec
            Styling options; the defaults when omitted. See `catena.graph_style`.
        link
            Link each node to its documentation; see `link_style`. Overrides the
            spec's ``link_function``.
        """
        spec = spec if spec is not None else GraphvizSpec()
        if link is None and spec.link_function is not None:
            link = _import_function(spec.link_function, "link_function")
        units, freqs = _node_maps(dr)
        return cls(
            spec=spec,
            outputs=frozenset(outputs),
            units=units,
            freqs=freqs,
            freq_colors=assign_freq_colors(freqs, spec.palette),
            config_inputs=_config_inputs(dr, dr.graph.config),
            own_functions=_own_functions(dr),
            link=link,
        )

    @property
    def node_style(self) -> StyleFunction:
        """The Hamilton ``custom_style_function`` for this DAG.

        catena's frequency fill and output outline, then the spec's
        ``style_function`` and the links, each composed over the last.
        """
        styles = [
            make_style_function(
                self.spec, set(self.outputs), self._freqs, self._freq_colors
            )
        ]
        if self.spec.style_function is not None:
            styles.append(_import_function(self.spec.style_function, "style_function"))
        if self._link is not None:
            styles.append(
                _link_style(
                    self._own_functions, self._link, tooltip=self.spec.link_tooltip
                )
            )
        return compose_styles(*styles)

    def hamilton_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for any of Hamilton's ``Driver`` display methods."""
        return {
            "custom_style_function": self.node_style,
            "graphviz_kwargs": self.spec.graphviz_kwargs,
            "show_legend": self.spec.show_legend,
        }

    def apply(self, digraph: "graphviz.Digraph") -> "graphviz.Digraph":
        """Style a digraph Hamilton drew from this DAG, in place.

        Separates config inputs into their own tables, puts each declared unit
        in place of the ``DataArray`` type, colours each edge by its source
        node's declared frequency, and strips per-node attributes from the
        legend.

        Returns
        -------
        graphviz.Digraph
            ``digraph`` itself, for chaining.
        """
        separate_config_inputs(digraph, self._config_inputs)
        relabel_with_units(digraph, self._units)
        color_edges_by_frequency(digraph, self._freqs, self._freq_colors)
        clean_legend(digraph)
        return digraph


def build_graph(
    config: ConfigSource,
    *,
    style: str | Path | None = None,
    allow_overrides: bool = False,
    link: LinkFunction | None = None,
    outputs_only: bool = True,
) -> "graphviz.Digraph":
    """Build a styled visualisation of the DAG a config describes.

    Each node shows its declared unit, requested outputs are highlighted, and
    nodes and edges are coloured by declared temporal frequency.

    By default only what the ``[outputs.*]`` sections need is drawn; a config
    with no outputs draws the whole DAG.

    Parameters
    ----------
    config
        Path to a TOML config file, or a `ParsedConfig` to visualise directly.
    style
        Optional TOML file overriding the default styling. See
        `catena.graph_style`.
    allow_overrides
        Allow a later module to override a node defined by an earlier one.
    link
        Link each node to its documentation; see `link_style`.
    outputs_only
        Draw only the nodes the requested outputs depend on. ``False`` draws
        every node the config's modules define.

    Returns
    -------
    graphviz.Digraph
        The styled graph. Write it with ``digraph.source`` (DOT) or render it
        with ``digraph.pipe(format=...)``.

    Attention
    ---------
    This requires the ``viz`` extra:
    ``pip install "ukceh-catena[viz] @ git+https://github.com/NERC-CEH/catena"``.
    """
    parsed, _ = prepare_config(config)

    dr = build_driver(
        modules=parsed.modules,
        config=parsed.driver_config,
        node_specs=parsed.node_specs,
        allow_module_overrides=allow_overrides,
        base=parsed.base,
        registered=parsed.registered_modules,
    )
    outputs = get_final_vars(parsed.output_specs)
    graph_style = GraphStyle.from_driver(
        dr,
        outputs=outputs,
        spec=load_graphviz_spec(Path(style) if style is not None else None),
        link=link,
    )

    if outputs_only and outputs:
        missing = [repr(name) for name in outputs if name not in dr.graph.nodes]
        if missing:
            raise CatenaValueError(
                f"Cannot draw the outputs: no node is called {', '.join(missing)}. "
                "Check the [outputs.*] sections, or draw every node instead."
            )
        digraph = dr.display_upstream_of(*outputs, **graph_style.hamilton_kwargs())
    else:
        digraph = dr.display_all_functions(**graph_style.hamilton_kwargs())
    if digraph is None:
        raise CatenaValueError(
            "Failed to build the graph visualisation. Is graphviz installed? "
            "Install the visualisation extra with "
            '`pip install "ukceh-catena[viz] @ '
            'git+https://github.com/NERC-CEH/catena"`.'
        )
    return graph_style.apply(digraph)
