---
title: Visualise the graph
icon: lucide/git-fork
---

# Visualise the graph

`catena graph` renders the DAG without running it, which is the quickest way to check that a config wired up the way you meant.

```sh
catena graph config.toml --pdf
```

This writes `pipeline.pdf` (and the `.dot` source).
Pass `--svg` for SVG, `--png` for PNG, and `-o/--output` to change the base filename.
`catena graph` needs the `viz` extra **and** the [Graphviz system binary](../install.md#graphviz).

## What the graph shows

The graphs on this page are drawn from a small pipeline: two daily inputs, three nodes from a module, and a `[[resample]]` to the weekly output.

=== "config.toml"

    ```toml
    --8<-- "docs/guides/validate/graph_example/config.toml"
    ```

=== "nodes.py"

    <!-- fmt:off -->
    ```python
    --8<-- "docs/guides/validate/graph_example/nodes.py"
    ```
    <!-- fmt:on -->

```python exec="true" session="visualise"
import sys

sys.path.insert(0, "docs/_scripts")
from catena_blocks import graph_example

print(graph_example())
```

- It draws what the `[outputs.*]` sections need; pass `--all-nodes` to draw every node the config's modules define.
  A config with no outputs draws every node.
- Each node shows its declared unit, and requested outputs are outlined.
- Nodes, and the edges leaving them, are coloured by their declared `freq` contract, never by section label; a node with no declared frequency is left uncoloured.
- A node's data inputs and its config are drawn as separate tables, and the config table shows each value in use: the config's, or else the function's default.
- In SVG output, hovering over a node shows the first paragraph of its docstring and its declared contract (unit, dims, dtype, coords, freq).
  A `[[node]]` or `[[resample]]` shows its expression, or the function it calls.

## Drawing part of the graph

A large DAG is easier to read a piece at a time.
Each of these draws part of it in place of what the outputs need, and they cannot be combined with each other or with `--all-nodes`:

| Option | Draws |
|---|---|
| `--upstream-of NODE` | `NODE` and everything it depends on |
| `--downstream-of NODE` | `NODE` and everything that depends on it |
| `--between START END` | the paths from `START` to `END` |

Repeat `--upstream-of` or `--downstream-of` to name several nodes.

Two more options thin out the input tables, in any view:

- `--hide-inputs` leaves out the tables of data inputs and config.
- `--deduplicate-inputs` draws each data input once, shared by every node that uses it; each node keeps its own config table.

Here only what `aridity_index_daily` needs is drawn, without the input tables:

```python exec="true" session="visualise"
print(graph_example("--upstream-of", "aridity_index_daily", "--hide-inputs"))
```

## What to look for

Structural mistakes that a config file hides are obvious in the graph:

- With `--all-nodes`, a node with no downstream consumer: nothing you asked for uses it.
- Two chains that should have met but did not, usually because a name differs by a suffix.
- A node in the wrong colour: a `freq` contract says something you did not intend.

`catena graph` builds the driver, so it also runs the contract check: if the config draws, its declared edges agree.

## Customising the styling

Pass a style file with `-s`/`--style` to override colours, layout, the legend, or the style function, and reuse it across pipelines:

```sh
catena graph config.toml --style recipes/graphviz.toml --pdf
```

The commented [`recipes/graphviz.toml`](https://github.com/NERC-CEH/catena/blob/main/recipes/graphviz.toml) template lists every key.

This style file pins the colours of the two frequencies, lays the graph out top to bottom, and names a style function:

=== "style.toml"

    ```toml
    --8<-- "docs/guides/validate/graph_example/style.toml"
    ```

=== "styling.py"

    <!-- fmt:off -->
    ```python
    --8<-- "docs/guides/validate/graph_example/styling.py"
    ```
    <!-- fmt:on -->

```python exec="true" session="visualise"
print(graph_example("--style", "style.toml"))
```

A style function receives each Hamilton node, including the tags set on its function with Hamilton's [`@tag`](https://hamilton.apache.org/reference/decorators/tag/) decorator, so `by_stage` dashes `precipitation_checked`, tagged `stage="qc"` in `nodes.py`.
It is composed over catena's own style function, and the attributes it returns win, so it keeps `rounded,filled` to keep the node's shape and frequency fill.
`PYTHONPATH=.` makes `styling.py` importable; a style function in an installed package needs nothing extra.

## Linking nodes to API documentation

A style file can name a `link_function`: it receives each node's Python function and returns a URL, or `None` for no link.

```toml
link_function = "my_package.docs:api_url"
```

```python
def api_url(fn):
    return f"../reference/api.html#{fn.__module__}.{fn.__qualname__}"
```

In SVG output each linked node is clickable.
Field nodes from `@extract_fields` share their function with the node that defines them, so only that node is linked.
Nodes from `[[node]]` and `[[resample]]` entries have no documentation page and are never linked.
Set `tooltips = false` in the style file to drop the hover text.

### In a documentation page

An SVG shown through `<img>`, or Markdown's `![](graph.svg)`, has no working links.
[dotdoc](https://jmarshrossney.github.io/dotdoc/)'s [`graph_html`](https://jmarshrossney.github.io/dotdoc/api/#dotdoc.html.graph_html) renders the digraph as an HTML fragment that inlines the SVG instead:

```python
from dotdoc import graph_html

html = graph_html(catena.build_graph("config.toml", link=api_url))
```

Write `html` into the page source, or return `mo.Html(html)` from a marimo notebook that is exported to the page.
Zensical and Material for MkDocs preview a link's target on hover, but only for HTML links, so `graph_html` lays an invisible link with a `data-preview` attribute over each linked node's name.
See its documentation for the other options, such as zoom controls and dark-scheme colours.
The [recipes](../../recipes/flux-pipeline.md#the-dag) are built this way, linking each node to the documentation of `nodes.py` at the end of the page.

## From Python

`catena.build_graph` returns the same styled `graphviz.Digraph`, which renders inline in a notebook:

```python
import catena

catena.build_graph("config.toml")
catena.build_graph("config.toml", outputs_only=False)  # every node
catena.build_graph("config.toml", link=api_url)  # link to the docs
catena.build_graph("config.toml", downstream_of=["temperature_daily"])
catena.build_graph("config.toml", between=("temperature_daily", "aridity_index_daily"))
catena.build_graph("config.toml", hide_inputs=True)
```

### Styling Hamilton's other views

`build_graph` covers the views above.
For anything else Hamilton's `Driver` can draw, or to pass it options `build_graph` does not, [`GraphStyle`](../../reference/modules/catena.graph.md#catena.graph.GraphStyle) applies catena's styling to the digraph directly.
Pass its `hamilton_kwargs()` to the method, then `apply` it to the digraph the method returns:

```python
from catena import build_driver_from_config, load_config
from catena.graph import GraphStyle

parsed = load_config("config.toml")
dr = build_driver_from_config(parsed)
style = GraphStyle.from_driver(dr, outputs=["aridity_index_daily"])
digraph = dr.visualize_path_between(
    "temperature_daily", "aridity_index_daily", **style.hamilton_kwargs()
)
style.apply(digraph)
```

Frequencies and units are read off the whole DAG, so a node looks the same in every view.

### Adding your own styling

Hamilton's display methods take a single `custom_style_function`.
dotdoc's [`compose_styles`](https://jmarshrossney.github.io/dotdoc/api/#dotdoc.hamilton.compose_styles) merges several into one, later attributes winning, so your own style function can sit on top of catena's:

```python
from dotdoc.hamilton import compose_styles, link_style


def mine(*, node, node_class):
    return {"fontcolor": "navy"}, node_class, None


kwargs = style.hamilton_kwargs()
kwargs["custom_style_function"] = compose_styles(
    style.node_style, mine, link_style(dr, api_url)
)
style.apply(dr.display_all_functions(**kwargs))
```

To leave out catena's styling entirely, call Hamilton's method without `GraphStyle`.
