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
It draws what the `[outputs.*]` sections need; pass `--all-nodes` to draw every node the config's modules define.
A config with no outputs draws every node.
Each node shows its declared unit, and requested outputs are highlighted.
Nodes are coloured and clustered by their declared `freq` contract, never by section label; a node with no declared frequency joins its neighbours' cluster when they all agree.
A node's data inputs and its config are drawn as separate tables, and the config table shows each value in use: the config's, or else the function's default.
[Pipeline 101](../../recipes/pipeline-101.md#the-graph-catena-builds) shows both.

Pass `--png` for PNG, and `-o/--output` to change the base filename.
`catena graph` needs the `viz` extra **and** the [Graphviz system binary](../install.md#graphviz).

## What to look for

Structural mistakes that a config file hides are obvious in the graph:

- With `--all-nodes`, a node with no downstream consumer: nothing you asked for uses it.
- Two chains that should have met but did not, usually because a name differs by a suffix.
- A node in the wrong cluster: a `freq` contract says something you did not intend.

`catena graph` builds the driver, so it also runs the contract check: if the config draws, its declared edges agree.

## Customising the styling

Pass a style file with `-s`/`--style` to override colours, layout, the legend, or the style function, and reuse it across pipelines:

```sh
catena graph config.toml --style recipes/graphviz.toml --pdf
```

The commented [`recipes/graphviz.toml`](https://github.com/NERC-CEH/catena/blob/main/recipes/graphviz.toml) template lists every key.

## Linking nodes to API documentation

A style file can name a `link_function`: it receives each node's Python function and returns a URL, or `None` for no link.

```toml
link_function = "my_package.docs:api_url"
```

```python
def api_url(fn):
    return f"../reference/api.html#{fn.__module__}.{fn.__qualname__}"
```

In SVG output each linked node is clickable, and hovering shows the first paragraph of its docstring (set `link_tooltip = false` to turn that off).
Field nodes from `@extract_fields` share their function with the node that defines them, so only that node is linked.

## From Python

`catena.build_graph` returns the same styled `graphviz.Digraph`, which renders inline in a notebook:

```python
import catena

catena.build_graph("config.toml")
catena.build_graph("config.toml", outputs_only=False)  # every node
catena.build_graph("config.toml", link=api_url)        # link to the docs
```

### Styling Hamilton's other views

`build_graph` draws one of two views.
Hamilton's `Driver` has more, such as `visualize_path_between` and `display_downstream_of`, and [`GraphStyle`](../../reference/modules/catena.graph.md#catena.graph.GraphStyle) applies catena's styling to any of them.
Pass its `hamilton_kwargs()` to the method, then `apply` it to the digraph the method returns:

```python
from catena import build_driver, load_config
from catena.graph import GraphStyle

parsed = load_config("config.toml")
dr = build_driver(
    modules=parsed.modules,
    config=parsed.driver_config,
    node_specs=parsed.node_specs,
)
style = GraphStyle.from_driver(dr, outputs=["aridity_index_daily"])
digraph = dr.visualize_path_between(
    "temperature_daily", "aridity_index_daily", **style.hamilton_kwargs()
)
style.apply(digraph)
```

Frequencies and units are read off the whole DAG, so a node looks the same in every view.

### Adding your own styling

Hamilton's display methods take a single `custom_style_function`.
[`compose_styles`](../../reference/modules/catena.graph.md#catena.graph.compose_styles) merges several into one, later attributes winning, so your own style function can sit on top of catena's:

```python
from catena.graph import compose_styles, link_style

def mine(*, node, node_class):
    return {"fontcolor": "navy"}, node_class, None

kwargs = style.hamilton_kwargs()
kwargs["custom_style_function"] = compose_styles(
    style.node_style, mine, link_style(dr, api_url)
)
style.apply(dr.display_all_functions(**kwargs))
```

To leave out catena's styling entirely, call Hamilton's method without `GraphStyle`.
