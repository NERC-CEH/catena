"""Start a pipeline: write a config from a set of modules, and list what it loads.

`generate_config` writes a starting config for a set of modules. Name each
module by the section an installed package registers it under, or by an
`_import_path` (a dotted module name or a `.py` file):

```python
from catena.scaffold import generate_config

generated = generate_config(["diagnostics", "nodes.py"])
generated.dump("config.toml")
```

The modules are built into one DAG, and the config gets:

- one module section each, holding the keyword-only settings their node
  functions declare, at their defaults;
- an `[inputs]` section listing every data input the modules need and none of
  them produces, under its node name;
- an `[outputs]` section listing the nodes no other node consumes.

How variables are split between files depends on the data, so by default every
input goes in one section and every output in another, at the placeholder paths
`inputs.nc` and `outputs.nc`. Pass `group` to split them into labelled sections
instead. It is called with each node's name and the contracts declared for it,
and returns the section label to put the node in; the sections' paths are then
`inputs/<label>.nc` and `outputs/<label>.nc`. Grouping by declared frequency:

```python
def by_freq(name, declared):
    freqs = {d.freq for d in declared if d.freq is not None}
    return "static" if not freqs else str(freqs.pop().freq)

generate_config(["diagnostics"], group=by_freq)
```

What the generator cannot write is listed in `GeneratedConfig.notes`, and at
the top of the file `GeneratedConfig.dumps` writes: settings with no default,
settings two modules give different defaults, contract conflicts between the
modules, and the inputs a setting brings in or drops when it selects another
`@config.when*` variant.

`variant_inputs` lists those variant inputs for each setting a `@config.when*`
decorator reads, value by value:

```python
from catena.scaffold import ANY_OTHER, variant_inputs

changes = variant_inputs(["pmodel"])
changes["method_kphio"]["sandoval"].added    # ('aridity_index',)
changes["method_kphio"][ANY_OTHER].added     # ()
```

The values are the ones the `config.when`, `when_not`, `when_in` and
`when_not_in` decorators name, plus `ANY_OTHER` for every value none of them
names, which all select the same variants. Each value is built with every other
setting at its default, so a variant that a combination of two settings selects
is not listed.

`input_requirements` lists, for each input section of a parsed config, the
variables it loads and the contracts their consumers declare for them, without
reading any data. It is what a tool that writes input files, such as a synthetic
data generator, needs to know.
"""

import inspect
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w
import xarray as xr
from hamilton import driver, graph_types
from hamilton.function_modifiers import config as hamilton_config
from hamilton.function_modifiers.base import NodeResolver
from hamilton.graph_utils import find_functions
from xarray_annotated import Declared, declaration_from_hint

from .build import build_driver, build_driver_from_config
from .config import Config
from .contract_check import _collect_declarations, dag_contract_conflicts
from .errors import CatenaFileExistsError, CatenaValueError
from .importing import (
    import_user_module,
    is_file_form,
    is_valid_module_path,
    registered_module,
)
from .io import var_mapping
from .specs import RECOGNISED_SECTIONS, ParsedConfig, io_section_name
from .wiring_check import terminal_nodes

__all__ = [
    "ANY_OTHER",
    "GeneratedConfig",
    "Group",
    "InputChange",
    "InputRequirement",
    "generate_config",
    "input_requirements",
    "variant_inputs",
]


class _AnyOther:
    """The type of `ANY_OTHER`."""

    def __repr__(self) -> str:
        return "<any other>"


ANY_OTHER = _AnyOther()
"""Stands for every value of a setting that no `@config.when*` decorator names."""

Group = Callable[[str, tuple[Declared, ...]], str]
"""Choose a node's section label from its name and the contracts declared for it.

For an input, the declarations are its consumers'; for an output, its producer's.
Either may be empty.
"""


@dataclass
class GeneratedConfig:
    """A config written by `generate_config`, with what it could not fill in.

    Attributes
    ----------
    data
        The config, as the dict `catena.config.Config` takes. Edit it in place
        before writing, for instance to add `[validation]` checks.
    notes
        One line per thing to finish by hand before the config runs.
    """

    data: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def dumps(self) -> str:
        """Return the config as TOML, with `notes` as a comment block at the top."""
        header = "".join(
            "# TODO: " + note.replace("\n", "\n#       ") + "\n" for note in self.notes
        )
        body = tomli_w.dumps(self.data)
        return f"{header}\n{body}" if header else body

    def dump(self, path: str | Path, overwrite_ok: bool = False) -> None:
        """Write the config to `path`, refusing to replace a file unless asked.

        Missing parent directories are created.
        """
        path = Path(path)
        if path.exists() and not overwrite_ok:
            raise CatenaFileExistsError(
                f"There is already a file at {path}. Pass `overwrite_ok=True` to "
                f"replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps())


@dataclass(frozen=True)
class InputRequirement:
    """One variable an input section loads, and what its consumers declare for it.

    Attributes
    ----------
    node
        The node name the variable is bound to.
    declared
        One declaration per consumer that declares one, in the order Hamilton
        lists the consumers. Empty when no consumer declares a contract.
    consumers
        The node declaring each entry of `declared`.
    """

    node: str
    declared: tuple[Declared, ...] = ()
    consumers: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputChange:
    """What one value of a setting changes about the inputs a config must load.

    Attributes
    ----------
    added
        Required inputs the value brings in, compared with the defaults.
    removed
        Inputs the defaults need and the value does not.
    error
        Why the modules do not build with this value, if they do not; `added`
        and `removed` are then empty.
    """

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    error: str | None = None

    def __bool__(self) -> bool:
        """Return True if the value changes anything."""
        return bool(self.added or self.removed or self.error)


def variant_inputs(
    modules: Sequence[str], *, base: str | Path | None = None
) -> dict[str, dict[Any, InputChange]]:
    """List, for each setting a `@config.when*` reads, what each value changes.

    Parameters
    ----------
    modules
        As for `generate_config`.
    base
        As for `generate_config`.

    Returns
    -------
    dict[str, dict[Any, InputChange]]
        For each setting, sorted by name, each value a decorator names and then
        `ANY_OTHER`, mapped to how the inputs differ from those the defaults
        need. A setting only custom `config(resolver, config_used=[...])`
        decorators read, whose values cannot be known, maps to an empty dict.

    Raises
    ------
    CatenaValueError
        As for `generate_config`, if the modules do not build at their defaults.
    """
    modules_ = _Modules.load(modules, base)
    return _variant_changes(modules_, modules_.build(modules_.config))


def _variant_changes(
    modules_: "_Modules", dr: driver.Driver
) -> dict[str, dict[Any, InputChange]]:
    """Return `variant_inputs` for loaded modules, given their default build."""
    default = _unbound(dr)
    changes: dict[str, dict[Any, InputChange]] = {}
    for setting, (_, values) in sorted(_selectors(modules_.imported).items()):
        changes[setting] = {}
        if not values:
            continue
        for value in (*values, ANY_OTHER):
            try:
                unbound = _unbound(modules_.build(modules_.config | {setting: value}))
            except CatenaValueError as exc:
                changes[setting][value] = InputChange(error=str(exc))
                continue
            changes[setting][value] = InputChange(
                added=tuple(sorted(unbound - default)),
                removed=tuple(sorted(default - unbound)),
            )
    return changes


def generate_config(
    modules: Sequence[str],
    *,
    base: str | Path | None = None,
    group: Group | None = None,
) -> GeneratedConfig:
    """Write a starting config for `modules`.

    Parameters
    ----------
    modules
        Each entry is the section name an installed package registers a module
        under, or else an `_import_path`: a dotted module name, or a path to a
        `.py` file. A registered name is written as a bare section; anything
        else as a section named after the module, carrying its `_import_path`.
    base
        The directory the config will be written to. A relative `.py` path
        resolves against it, as it will when the config is loaded.
    group
        Splits inputs and outputs into labelled sections; see `Group`. Without
        it, the config has one `[inputs]` and one `[outputs]` section.

    Returns
    -------
    GeneratedConfig
        The config, and the notes on what is left to fill in by hand.

    Raises
    ------
    CatenaValueError
        If a module cannot be found or imported, if two entries name the same
        section, or if the modules cannot be built into one DAG (two of them
        defining the same node, for instance).
    """
    modules_ = _Modules.load(modules, base)
    base, sections, defaults = modules_.base, modules_.sections, modules_.defaults
    clashes = modules_.clashes
    dr = modules_.build(modules_.config)

    notes = [
        f"setting {key!r} defaults to {value!r} in [{owner}] and to {other!r} in "
        f"[{elsewhere}]; one value serves both, so choose one."
        for key, (owner, value, elsewhere, other) in clashes.items()
    ]

    produced, consumed = _collect_declarations(dr)
    required = _required(dr)
    data_inputs: dict[str, tuple[Declared, ...]] = {}
    for node in graph_types.HamiltonGraph.from_graph(dr.graph).nodes:
        if not node.is_external_input or node.name not in required:
            continue
        if _is_data(node.type):
            data_inputs[node.name] = tuple(d for d, _ in consumed.get(node.name, ()))
        elif node.name not in defaults:
            notes.append(
                f"{node.name!r} is a required setting with no default; set it in "
                f"the section of the module that uses it."
            )

    outputs: dict[str, tuple[Declared, ...]] = {}
    types_ = {v.name: v.type for v in dr.list_available_variables()}
    for name in terminal_nodes(dr):
        if name in produced:
            outputs[name] = (produced[name][0],)
        elif _is_data(types_[name]):
            outputs[name] = ()

    notes.extend(f"contract conflict: {fault}" for fault in dag_contract_conflicts(dr))
    owners = {key: owner for key, (owner, _) in _selectors(modules_.imported).items()}
    owners |= {key: owner for key, (owner, _) in defaults.items()}
    notes.extend(_variant_notes(_variant_changes(modules_, dr), owners))

    data: dict[str, Any] = {}
    for section, (path, registered) in sections.items():
        settings = {
            key: value for key, (owner, value) in defaults.items() if owner == section
        }
        data[section] = ({} if registered else {"_import_path": path}) | settings
    if data_inputs:
        data["inputs"] = _io_tables("inputs", data_inputs, group)
    if outputs:
        data["outputs"] = _io_tables("outputs", outputs, group)

    generated = GeneratedConfig(data=data, notes=notes)
    _verify(generated, base, dr)
    return generated


def input_requirements(
    parsed: ParsedConfig, dr: driver.Driver | None = None
) -> dict[str, dict[str, InputRequirement]]:
    """List what each input section of `parsed` loads, without reading any data.

    Parameters
    ----------
    parsed
        A parsed config.
    dr
        The driver `parsed` builds, if already built. Otherwise it is built here,
        without the contract check.

    Returns
    -------
    dict[str, dict[str, InputRequirement]]
        For each input section label, the file variables it loads, each with the
        node it is bound to and the contracts its consumers declare.

    Raises
    ------
    CatenaValueError
        If an input section omits `vars`: it then loads whatever the file holds,
        which cannot be known before the file exists.
    """
    if dr is None:
        dr = build_driver_from_config(parsed, check_contracts=False)
    _, consumed = _collect_declarations(dr)
    requirements: dict[str, dict[str, InputRequirement]] = {}
    for label, spec in parsed.input_specs.items():
        if spec.vars is None:
            raise CatenaValueError(
                f"{io_section_name('inputs', label)} omits 'vars', so what it "
                f"loads depends on a file that may not exist yet. List the "
                f"variables to load."
            )
        section: dict[str, InputRequirement] = {}
        for node, file_var in var_mapping(label, spec).items():
            declarations = consumed.get(node, [])
            section[file_var] = InputRequirement(
                node=node,
                declared=tuple(decl for decl, _ in declarations),
                consumers=tuple(who for _, who in declarations),
            )
        requirements[label] = section
    return requirements


@dataclass(frozen=True)
class _Modules:
    """The requested modules, imported, with their settings at their defaults."""

    base: Path | None
    sections: dict[str, tuple[str, bool]]
    imported: dict[str, types.ModuleType]
    defaults: dict[str, tuple[str, Any]]
    clashes: dict[str, tuple[str, Any, str, Any]]

    @classmethod
    def load(cls, modules: Sequence[str], base: str | Path | None) -> "_Modules":
        """Import `modules` and collect their settings."""
        base_ = None if base is None else Path(base).resolve()
        sections = _sections(modules)
        imported = {
            section: import_user_module(path, base_)
            for section, (path, _) in sections.items()
        }
        defaults, clashes = _settings(imported)
        return cls(base_, sections, imported, defaults, clashes)

    @property
    def config(self) -> dict[str, Any]:
        """Return each setting at its default, as the driver config."""
        return {key: value for key, (_, value) in self.defaults.items()}

    def build(self, config: dict[str, Any]) -> driver.Driver:
        """Build the modules into one driver, without the contract check."""
        try:
            return build_driver(
                [path for path, _ in self.sections.values()],
                config,
                base=self.base,
                check_contracts=False,
            )
        except CatenaValueError:
            raise
        except Exception as exc:  # Hamilton raises bare exceptions for a bad graph.
            raise CatenaValueError(
                f"The modules {list(self.sections)} cannot be built into one DAG: {exc}"
            ) from exc


def _selectors(
    imported: dict[str, types.ModuleType],
) -> dict[str, tuple[str, list[Any]]]:
    """Map each setting a `@config` decorator reads to the values decorators name.

    A setting is listed if any decorator reads it, by its public
    `optional_config`, with the first section whose module does. Its values
    come from the `config.when*` resolvers, whose key-value pairs Hamilton keeps
    only in a closure: `resolves` in `ConfigResolver.when*` closes over the one
    mapping it was called with.
    """
    found: dict[str, tuple[str, list[Any]]] = {}
    for section, module in imported.items():
        for _, func in find_functions(module):
            for resolver in getattr(func, NodeResolver.get_lifecycle_name(), []):
                if not isinstance(resolver, hamilton_config):
                    continue
                for key in resolver.optional_config() or {}:
                    found.setdefault(key, (section, []))
                for key, named in _named_values(resolver).items():
                    values = found[key][1]
                    values.extend(v for v in named if v not in values)
    return found


def _named_values(resolver: hamilton_config) -> dict[str, list[Any]]:
    """Return the values a `config.when*` decorator names, by setting."""
    resolves = getattr(resolver.does_resolve, "resolves", None)
    closure = getattr(resolves, "__closure__", None)
    if resolves is None or not closure:
        return {}
    if not isinstance(pairs := closure[0].cell_contents, dict):
        return {}
    if resolves.__qualname__.split(".")[1] in ("when_in", "when_not_in"):
        return {key: list(group) for key, group in pairs.items()}
    return {key: [value] for key, value in pairs.items()}


def _unbound(dr: driver.Driver) -> set[str]:
    """Return the inputs some node requires and the driver config does not set."""
    externals = {
        node.name
        for node in graph_types.HamiltonGraph.from_graph(dr.graph).nodes
        if node.is_external_input
    }
    return (_required(dr) & externals) - set(dr.graph.config)


def _variant_notes(
    changes: dict[str, dict[Any, InputChange]], owners: dict[str, str]
) -> list[str]:
    """Word what each setting's values change, merging values that change alike."""
    notes = []
    for setting, by_value in changes.items():
        where = f"[{owners[setting]}] "
        if not by_value:
            notes.append(
                f"{where}{setting!r} selects variants with values catena cannot "
                f"list; check the module for the inputs each one needs."
            )
            continue
        alike: dict[InputChange, list[Any]] = {}
        for value, change in by_value.items():
            if change:
                alike.setdefault(change, []).append(value)
        for change, values in alike.items():
            named = " or ".join(
                "any other value" if v is ANY_OTHER else repr(v) for v in values
            )
            parts = []
            if change.error is not None:
                parts.append(f"does not build: {change.error}")
            if change.added:
                parts.append(f"also needs {_names(change.added)}")
            if change.removed:
                parts.append(f"no longer needs {_names(change.removed)}")
            effect = "; ".join(parts)
            notes.append(f"{where}{setting} = {named} {effect}.")
    return notes


def _names(names: tuple[str, ...]) -> str:
    """Join input names for a note."""
    return ", ".join(repr(name) for name in names)


def _sections(modules: Sequence[str]) -> dict[str, tuple[str, bool]]:
    """Map each requested module's section to its import path and how it is named.

    The flag is True for a registered module, which is written as a bare section.
    """
    sections: dict[str, tuple[str, bool]] = {}
    for module in modules:
        if (found := registered_module(module)) is not None:
            section, path, registered = module, found.import_path, True
        elif is_file_form(module) or is_valid_module_path(module):
            section, path, registered = _section_name(module), module, False
        else:
            raise CatenaValueError(
                f"{module!r} is neither a registered module, a dotted module path, "
                f"nor a path to a .py file."
            )
        if section in sections:
            raise CatenaValueError(
                f"{module!r} and {sections[section][0]!r} would both be written as "
                f"[{section}]."
            )
        sections[section] = (path, registered)
    return sections


def _section_name(import_path: str) -> str:
    """Name a section after the last component of its module's import path."""
    if is_file_form(import_path):
        name = Path(import_path).stem
    else:
        name = import_path.rsplit(".", 1)[-1]
    if name in RECOGNISED_SECTIONS:
        raise CatenaValueError(
            f"{import_path!r} would be written as [{name}], a section catena parses "
            f"itself. Rename the module."
        )
    return name


def _settings(
    imported: dict[str, types.ModuleType],
) -> tuple[dict[str, tuple[str, Any]], dict[str, tuple[str, Any, str, Any]]]:
    """Collect each module's keyword-only settings that have a default.

    Returns `(defaults, clashes)`. `defaults` maps each setting to the first
    section declaring it and its default there; a setting with a `None` default
    is left out, as TOML has no null. `clashes` maps a setting two sections give
    different defaults to both sections and both defaults.
    """
    defaults: dict[str, tuple[str, Any]] = {}
    clashes: dict[str, tuple[str, Any, str, Any]] = {}
    for section, module in imported.items():
        for _, func in find_functions(module):
            for param in inspect.signature(func).parameters.values():
                if (
                    param.kind is not inspect.Parameter.KEYWORD_ONLY
                    or param.default is inspect.Parameter.empty
                    or param.default is None
                ):
                    continue
                if param.name not in defaults:
                    defaults[param.name] = (section, param.default)
                    continue
                owner, value = defaults[param.name]
                if owner != section and value != param.default:
                    clashes.setdefault(
                        param.name, (owner, value, section, param.default)
                    )
    return defaults, clashes


def _required(dr: driver.Driver) -> set[str]:
    """Return the names some node requires, rather than takes optionally."""
    return {
        dep
        for node in graph_types.HamiltonGraph.from_graph(dr.graph).nodes
        for dep in node.required_dependencies
    }


def _is_data(hint: Any) -> bool:
    """Report whether a node of type `hint` holds data, rather than a setting."""
    if declaration_from_hint(hint) is not None:
        return True
    if typing.get_origin(hint) is typing.Annotated:
        hint = typing.get_args(hint)[0]
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return any(_is_data(arg) for arg in typing.get_args(hint))
    return isinstance(hint, type) and issubclass(hint, xr.DataArray)


def _io_tables(
    kind: str, nodes: dict[str, tuple[Declared, ...]], group: Group | None
) -> dict[str, Any]:
    """Return the `[inputs]` or `[outputs]` table holding `nodes`."""
    if group is None:
        return {"path": f"{kind}.nc", "vars": sorted(nodes)}
    groups: dict[str, list[str]] = {}
    for name, declared in sorted(nodes.items()):
        groups.setdefault(group(name, declared), []).append(name)
    return {
        label: {"path": f"{kind}/{label}.nc", "suffix": "", "vars": names}
        for label, names in sorted(groups.items())
    }


def _verify(generated: GeneratedConfig, base: Path | None, dr: driver.Driver) -> None:
    """Parse the generated config, and note any required input it leaves unbound.

    Required settings with no default are already noted, so only an input missed
    for another reason is added. That would be a bug here, but is reported
    rather than raised so that the config can still be written and inspected.
    """
    parsed = Config(generated.data, base=base).parse()
    bound = set(parsed.driver_config)
    for label, spec in parsed.input_specs.items():
        bound |= set(var_mapping(label, spec))
    noted = " ".join(generated.notes)
    externals = {
        node.name
        for node in graph_types.HamiltonGraph.from_graph(dr.graph).nodes
        if node.is_external_input
    }
    for name in sorted((_required(dr) & externals) - bound):
        if repr(name) not in noted:
            generated.notes.append(f"{name!r} is required but bound by nothing.")
