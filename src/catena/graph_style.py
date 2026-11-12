"""Styling configuration for the ``catena graph`` DAG visualisation.

The canonical default styling lives in :class:`GraphvizSpec`, a typed dataclass
that is the single source of truth.  Users may supply an optional TOML file via
``catena graph --style <file>`` whose keys are deep-merged over these defaults,
so one style can be reused across many science configs without duplicating it.

Recognised top-level keys in a ``--style`` file::

    style_function = "my_pkg.styling:my_style"   # composed over the built-in style
    link_function = "my_pkg.docs:api_url"        # node function -> URL, or None
    link_tooltip = true                          # docstring summary on hover
    show_legend = true

    [palette]                                    # pin a colour to a frequency
    "7D" = "#ff7f00"                             # ... or override `output`

    [graph_attr]                                 # graphviz graph attributes
    rankdir = "TB"

    [node_attr]                                  # graphviz node attributes
    fontname = "Courier"

    [edge_attr]                                  # graphviz edge attributes
    color = "#444444"
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CatenaValueError

# ColorBrewer "Set2": a colour-blind-friendly qualitative palette. These work both as
# node fill colours (with black text) and as edge colours. Frequencies are arbitrary
# offset aliases ("7D", "1ME", "W-SUN"), not a fixed vocabulary, so colours are
# assigned from this cycle in first-seen order rather than from a fixed table.
FREQ_COLOR_CYCLE: tuple[str, ...] = (
    "#fc8d62",
    "#8da0cb",
    "#a6d854",
    "#66c2a5",
    "#ffd92f",
    "#e5c494",
)

#: Colours for the fixed, non-frequency categories. A ``[palette]`` entry may also
#: name a *frequency* (``"7D"``) to pin it against `FREQ_COLOR_CYCLE`.
DEFAULT_PALETTE: dict[str, str] = {
    "output": "#e7298a",  # border highlight on requested output nodes
}

# Top-level keys accepted in a ``--style`` TOML file.
_GRAPHVIZ_ATTR_KEYS = ("graph_attr", "node_attr", "edge_attr")
_KNOWN_KEYS = frozenset(
    {
        "palette",
        "style_function",
        "link_function",
        "link_tooltip",
        "show_legend",
        *_GRAPHVIZ_ATTR_KEYS,
    }
)


@dataclass(frozen=True)
class GraphvizSpec:
    """Resolved styling for a single ``catena graph`` invocation.

    Attributes
    ----------
    palette
        Category → hex-colour map (see :data:`DEFAULT_PALETTE`).
    style_function
        Optional ``"module:function"`` import path to a Hamilton
        ``custom_style_function``, composed over the built-in one.
    link_function
        Optional ``"module:function"`` import path to a function that takes a
        node's originating function and returns its documentation URL, or
        ``None``. See `catena.graph.link_style`.
    link_tooltip
        Whether a linked node also shows its docstring summary on hover.
    show_legend
        Whether to draw the legend.
    graphviz_kwargs
        Kwargs forwarded to ``Driver.display_all_functions`` (``graph_attr``,
        ``node_attr``, ``edge_attr`` nested dicts).
    """

    palette: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PALETTE))
    style_function: str | None = None
    link_function: str | None = None
    link_tooltip: bool = True
    show_legend: bool = True
    graphviz_kwargs: dict[str, Any] = field(default_factory=dict)


def load_graphviz_spec(path: Path | None) -> GraphvizSpec:
    """Load a styling spec from ``path``, falling back to the built-in defaults.

    When ``path`` is ``None`` the dataclass defaults are returned unchanged.
    Otherwise the TOML file is parsed and deep-merged over the defaults: the
    ``palette`` is updated key-by-key (so a file need only override the colours
    it cares about), and the graphviz attribute tables are collected into
    ``graphviz_kwargs``.  Unknown top-level keys raise ``ValueError``.
    """
    if path is None:
        return GraphvizSpec()

    with Path(path).open("rb") as file:
        data = tomllib.load(file)

    unknown = set(data) - _KNOWN_KEYS
    if unknown:
        raise CatenaValueError(
            f"Unknown key(s) in graphviz style file {path}: {sorted(unknown)}. "
            f"Recognised keys: {sorted(_KNOWN_KEYS)}."
        )

    palette = {**DEFAULT_PALETTE, **data.get("palette", {})}
    graphviz_kwargs = {k: data[k] for k in _GRAPHVIZ_ATTR_KEYS if k in data}

    return GraphvizSpec(
        palette=palette,
        style_function=data.get("style_function"),
        link_function=data.get("link_function"),
        link_tooltip=data.get("link_tooltip", True),
        show_legend=data.get("show_legend", True),
        graphviz_kwargs=graphviz_kwargs,
    )
