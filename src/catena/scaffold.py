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
- one `[inputs.<label>]` section per declared frequency, listing every data input
  the modules need and none of them produces, under its node name;
- one `[outputs.<label>]` section per declared frequency, listing the nodes no
  other node consumes.

Section labels come from the declared `Freq` (`freq_D`, `freq_1ME`), with
`no_freq` for inputs that declare none; a config with a single input or output
group uses the unlabelled `[inputs]` or `[outputs]` form. Every path follows one
rule, to be replaced with your own: `inputs/<label>.nc` and `outputs/<label>.nc`,
or `inputs.nc` and `outputs.nc` for the unlabelled form.

What the generator cannot write is listed in `GeneratedConfig.notes`, and at
the top of the file `GeneratedConfig.dumps` writes: settings with no default,
settings two modules give different defaults, and contract conflicts between the
modules.

`input_requirements` lists, for each input section of a parsed config, the
variables it loads and the contracts their consumers declare for them, without
reading any data. It is what a tool that writes input files, such as a synthetic
data generator, needs to know.
"""

import inspect
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w
import xarray as xr
from hamilton import driver, graph_types
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
    "GeneratedConfig",
    "InputRequirement",
    "generate_config",
    "input_requirements",
]

# Label of the input group whose consumers declare no frequency.
_NO_FREQ = "no_freq"


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
        header = "".join(f"# TODO: {note}\n" for note in self.notes)
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


def generate_config(
    modules: Sequence[str], *, base: str | Path | None = None
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
    base = None if base is None else Path(base).resolve()
    sections = _sections(modules)
    import_paths = [path for path, _ in sections.values()]
    imported = {
        section: import_user_module(path, base)
        for section, (path, _) in sections.items()
    }

    defaults, clashes = _settings(imported)
    try:
        dr = build_driver(
            import_paths,
            {key: value for key, (_, value) in defaults.items()},
            base=base,
            check_contracts=False,
        )
    except CatenaValueError:
        raise
    except Exception as exc:  # Hamilton raises bare exceptions for a bad graph.
        raise CatenaValueError(
            f"The modules {list(sections)} cannot be built into one DAG: {exc}"
        ) from exc

    notes = [
        f"setting {key!r} defaults to {value!r} in [{owner}] and to {other!r} in "
        f"[{elsewhere}]; one value serves both, so choose one."
        for key, (owner, value, elsewhere, other) in clashes.items()
    ]

    produced, consumed = _collect_declarations(dr)
    required = _required(dr)
    data_inputs: dict[str, str] = {}
    for node in graph_types.HamiltonGraph.from_graph(dr.graph).nodes:
        if not node.is_external_input or node.name not in required:
            continue
        if _is_data(node.type):
            data_inputs[node.name] = _freq_label(
                next(
                    (d for d, _ in consumed.get(node.name, ()) if d.freq is not None),
                    None,
                )
            )
        elif node.name not in defaults:
            notes.append(
                f"{node.name!r} is a required setting with no default; set it in "
                f"the section of the module that uses it."
            )

    outputs: dict[str, str] = {}
    types_ = {v.name: v.type for v in dr.list_available_variables()}
    for name in terminal_nodes(dr):
        if _is_data(types_[name]) or name in produced:
            declared = produced.get(name, (None, ""))[0]
            outputs[name] = _freq_label(declared)

    notes.extend(f"contract conflict: {fault}" for fault in dag_contract_conflicts(dr))

    data: dict[str, Any] = {}
    for section, (path, registered) in sections.items():
        settings = {
            key: value for key, (owner, value) in defaults.items() if owner == section
        }
        data[section] = ({} if registered else {"_import_path": path}) | settings
    if data_inputs:
        data["inputs"] = _io_tables("inputs", data_inputs)
    if outputs:
        data["outputs"] = _io_tables("outputs", outputs)

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


def _freq_label(declared: Declared | None) -> str:
    """Return the section label for nodes declared with `declared`'s frequency."""
    if declared is None or declared.freq is None:
        return _NO_FREQ
    return f"freq_{declared.freq.freq}"


def _io_tables(kind: str, labels: dict[str, str]) -> dict[str, Any]:
    """Return the `[inputs]` or `[outputs]` table grouping `labels`' nodes."""
    groups: dict[str, list[str]] = {}
    for name, label in sorted(labels.items()):
        groups.setdefault(label, []).append(name)
    if len(groups) == 1:
        (names,) = groups.values()
        return {"path": f"{kind}.nc", "vars": names}
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
