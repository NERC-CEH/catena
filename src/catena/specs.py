"""The parsed configuration data model: one dataclass per config section.

`catena.config` builds these from a TOML file; `run`, `dry_run` and
`build_graph` accept the resulting `ParsedConfig` in place of a path.

Each spec validates itself in `from_config`, so a malformed section fails at parse
time with a message naming the section, never later inside a DAG node.
"""

import keyword
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from .errors import CatenaValueError

_VALID_AGGFUNCS: frozenset[str] = frozenset(
    {"mean", "sum", "max", "min", "first", "last"}
)

#: Node names and node inputs a config may not use. Each one names a helper bound
#: in every generated node module's namespace, so a node called ``xr`` would shadow
#: that helper for every later node's expression. `catena.nodegen` binds them and
#: `NodeSpec.from_config` rejects them at parse time; the two are kept in step by
#: ``test_nodegen.py::test_reserved_names_match_generated_namespace``.
RESERVED_NODE_NAMES: frozenset[str] = frozenset(
    {"xr", "xra", "Any", "__import_module", "__transforms"}
)

#: Top-level config sections and keys catena parses itself. A section named here is
#: never treated as a user module, so a package registering one of these names would
#: register something no config could ever reach.
RECOGNISED_SECTIONS: frozenset[str] = frozenset(
    {
        "inputs",
        "outputs",
        "validation",
        "node",
        "resample",
        "cache",
        "blocking",
        "subset",
    }
)

#: Keys declaring a generated node's output contract. Under fan-out, each may be a
#: table keyed by variable instead of one value for every variable.
CONTRACT_KEYS: tuple[str, ...] = (
    "units",
    "dims",
    "dtype",
    "coords",
    "freq",
    "overwrite_units",
)

_NODE_KEYS: frozenset[str] = frozenset(
    {"name", "inputs", "expression", "_import_path", "function", "for_each"}
) | frozenset(CONTRACT_KEYS)

# `freq` is the resample's own target, and a resample keeps its input's units label,
# so `overwrite_units` would relabel values it never converted.
_RESAMPLE_CONTRACT_KEYS: tuple[str, ...] = ("units", "dims", "dtype", "coords")
_RESAMPLE_OWN_KEYS: frozenset[str] = frozenset(
    {"vars", "from", "to", "freq", "aggfunc"}
)
_RESAMPLE_KEYS: frozenset[str] = _RESAMPLE_OWN_KEYS | frozenset(_RESAMPLE_CONTRACT_KEYS)


def _assert_known_keys(entry: dict, allowed: frozenset[str], where: str) -> None:
    """Reject keys a config entry does not recognise."""
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise CatenaValueError(
            f"{where} has unrecognised key(s) {unknown}. Recognised keys: "
            f"{sorted(allowed)}."
        )


def _assert_valid_per_var_tables(
    entry: dict, keys: Iterable[str], names: list[str], where: str
) -> None:
    """Reject a per-variable contract table naming a variable that is not fanned out.

    A table value is ``{var = value}``. Only fan-out can resolve one, so a table on
    an entry with no variables is rejected too.
    """
    for key in keys:
        value = entry.get(key)
        if not isinstance(value, dict):
            continue
        if not names:
            raise CatenaValueError(
                f"{where}: {key} is a table keyed by variable, which needs "
                f"'for_each' to say what the variables are."
            )
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise CatenaValueError(
                f"{where}: {key} names {unknown}, which are not among the variables "
                f"{names}."
            )


# ---------------------------------------------------------------------------
# Spec dataclasses (the parsed data model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisteredModule:
    """A config section resolved to a module by an installed package.

    Produced when a section carries no ``_import_path`` and its name matches one an
    installed package declares in the ``catena.modules`` entry-point group (see
    `catena.importing`). `run` and `dry_run` report these, so a section whose code
    comes from somewhere the config does not name is still visible.
    """

    #: The config section that named it.
    section: str
    #: The dotted module path the package registered.
    import_path: str
    #: The distribution that registered it.
    distribution: str


@dataclass
class ResampleSpec:
    """Specification for a single [[resample]] entry.

    ``source`` and ``target`` (TOML ``from`` / ``to``) are **node-name suffixes**, not
    frequencies: ``from = "daily"`` reads ``{var}_daily``, ``to = "weekly"`` writes
    ``{var}_weekly``, and nothing is inferred from either (see `catena.io.load_inputs`
    on inert labels). ``from = ""`` reads the bare name ``{var}``; ``to`` must not be
    empty. ``freq`` alone says what happens to the time axis, and is each output's
    declared frequency.

    ``contract`` holds the other declared output facets (``units``, ``dims``,
    ``dtype``, ``coords``), each one value for every variable or a table keyed by
    variable. A facet left undeclared is not checked.

    Built into node specs by `catena.config.resample_node_specs`.
    """

    vars: list[str]
    source: str
    target: str
    freq: str
    aggfunc: str = "mean"
    contract: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the spec, however it was built."""
        if not isinstance(self.vars, list) or not all(
            isinstance(v, str) for v in self.vars
        ):
            raise CatenaValueError(
                f"[[resample]] 'vars' must be a list of names, got {self.vars!r}."
            )
        where = f"[[resample]] of {self.vars}"
        if "overwrite_units" in self.contract:
            raise CatenaValueError(
                f"{where}: overwrite_units cannot be set, as a resample keeps its "
                f"input's units label and does not convert the values."
            )
        _assert_known_keys(
            self.contract, frozenset(_RESAMPLE_CONTRACT_KEYS), f"{where} contract"
        )
        if self.aggfunc not in _VALID_AGGFUNCS:
            raise CatenaValueError(
                f"{where}: Unsupported aggfunc '{self.aggfunc}'. "
                f"Supported: {sorted(_VALID_AGGFUNCS)}"
            )
        if self.target == "":
            raise CatenaValueError(
                f"{where} has to = '', which would write each result over its own "
                f"source name. Give 'to' a suffix, e.g. to = 'weekly' writes "
                f"'{{var}}_weekly'."
            )

        from xarray_annotated.temporal import Freq, assert_valid_freq

        try:
            assert_valid_freq(Freq(self.freq), f"[[resample]] to '{self.target}' freq")
        except ValueError as exc:
            raise CatenaValueError(str(exc)) from exc
        if Freq(self.freq).any_day:
            raise CatenaValueError(
                f"[[resample]] to '{self.target}' freq must be a pandas offset "
                f"alias, as {self.freq!r} says how often to sample but not where to "
                f"stamp it. Use one such as '1ME' or 'YE'."
            )
        _assert_valid_per_var_tables(
            self.contract, _RESAMPLE_CONTRACT_KEYS, self.vars, where
        )
        for var in self.vars:
            _assert_valid_contract(self.contract_for(var), f"{where}, '{var}'")

    def contract_for(self, var: str) -> dict[str, Any]:
        """Return the contract declared for one variable, tables resolved."""
        out: dict[str, Any] = {}
        for key, value in self.contract.items():
            if isinstance(value, dict):
                if var not in value:
                    continue
                value = value[var]
            out[key] = value
        return out

    @classmethod
    def from_config(cls, entry: dict) -> Self:
        """Construct and validate from a raw [[resample]] TOML entry."""
        # overwrite_units passes through to the contract for its own error.
        _assert_known_keys(
            entry,
            _RESAMPLE_KEYS | {"overwrite_units"},
            f"[[resample]] of {entry.get('vars')}",
        )
        missing = [key for key in ("vars", "from", "to", "freq") if key not in entry]
        if missing:
            raise CatenaValueError(
                f"[[resample]] entry is missing required key(s) {missing}. Every "
                f"entry needs 'vars', 'from' and 'to' (the node-name suffixes to read "
                f"from and write to) and 'freq' (the target pandas offset alias, e.g. "
                f"'7D', '1ME', 'W-SUN')."
            )
        return cls(
            vars=entry["vars"],
            source=entry["from"],
            target=entry["to"],
            freq=entry["freq"],
            aggfunc=entry.get("aggfunc", "mean"),
            contract={k: v for k, v in entry.items() if k not in _RESAMPLE_OWN_KEYS},
        )


def _assert_node_identifier(value: Any, field_: str, node_name: Any) -> None:
    """Reject a node name / input that is unsafe to interpolate into node source.

    `catena.nodegen` builds each node's ``def`` line by string formatting, so a
    name that is not a plain Python identifier fails as an opaque ``SyntaxError``
    deep in module generation (or, worse, injects statements). The generated
    module's own namespace names are reserved too: a node called ``xr`` would
    shadow the helper for every later node's expression.
    """
    where = f"[[node]] '{node_name}' {field_}"
    if not isinstance(value, str) or not value.isidentifier():
        raise CatenaValueError(
            f"{where}: {value!r} is not a valid Python identifier, and node names "
            f"and inputs become identifiers in the generated node module."
        )
    if keyword.iskeyword(value):
        raise CatenaValueError(f"{where}: {value!r} is a Python keyword.")
    if value in RESERVED_NODE_NAMES:
        raise CatenaValueError(
            f"{where}: {value!r} is reserved — it names a helper bound in every "
            f"generated node module ({sorted(RESERVED_NODE_NAMES)}). Choose "
            f"another name."
        )


def _assert_valid_contract(entry: dict, where: str) -> None:
    """Reject a generated node's malformed output contract at parse time."""
    from xarray_annotated import (
        annotate,
        assert_valid_declaration,
        declaration_from_hint,
    )

    try:
        declared = declaration_from_hint(
            annotate(
                unit=entry.get("units"),
                dims=entry.get("dims"),
                dtype=entry.get("dtype"),
                coords=entry.get("coords"),
                freq=entry.get("freq"),
            )
        )
        if declared is not None:
            assert_valid_declaration(declared, where)
    except TypeError as exc:
        raise CatenaValueError(f"{where}: {exc}") from exc
    except ValueError as exc:
        raise CatenaValueError(str(exc)) from exc


def _assert_valid_overwrite_units(entry: dict, node_name: Any) -> None:
    """Reject an ``overwrite_units`` the node cannot apply."""
    if "overwrite_units" not in entry:
        return
    value = entry["overwrite_units"]
    if not isinstance(value, bool):
        raise CatenaValueError(
            f"node '{node_name}': overwrite_units must be true or false, not {value!r}."
        )
    if value and entry.get("units") is None:
        raise CatenaValueError(
            f"node '{node_name}': overwrite_units = true needs a declared 'units' "
            f"to overwrite with."
        )


@dataclass
class NodeSpec:
    """Specification for a single (already fan-out-expanded) [[node]] entry.

    ``units`` / ``dims`` / ``dtype`` / ``coords`` / ``freq`` declare the node's output
    contract: validated at runtime, and read by the build-time check. An output with no
    ``units`` label is stamped with the declared unit; one that keeps its input's label
    still has to match, unless ``overwrite_units`` replaces it with the declared unit.
    A node declaring none of them is not checked.
    """

    name: str
    inputs: list[str]
    expression: str | None
    import_path: str | None
    function: str | None
    units: str | None = None
    dims: list[str] | None = None
    dtype: str | None = None
    coords: list[str] | None = None
    freq: str | None = None
    overwrite_units: bool = False

    @classmethod
    def from_config(cls, entry: dict) -> Self:
        """Construct and validate from a raw (expanded) [[node]] TOML entry."""
        name = entry.get("name")
        _assert_known_keys(entry, _NODE_KEYS - {"for_each"}, f"[[node]] '{name}'")
        _assert_node_identifier(name, "name", name)
        for inp in entry.get("inputs", []):
            _assert_node_identifier(inp, "inputs", name)
        has_expression = "expression" in entry
        has_import_path = "_import_path" in entry
        has_function = "function" in entry
        if has_expression and (has_import_path or has_function):
            raise CatenaValueError(
                f"Node entry for '{name}' must specify either "
                "'expression' or ('_import_path' + 'function'), not both."
            )
        if not has_expression and not (has_import_path or has_function):
            raise CatenaValueError(
                f"Node entry for '{name}' must specify either "
                "'expression' or ('_import_path' + 'function')."
            )
        if has_import_path != has_function:
            missing = "function" if has_import_path else "_import_path"
            present = "_import_path" if has_import_path else "function"
            raise CatenaValueError(
                f"Node entry for '{name}' specifies '{present}' but is missing "
                f"'{missing}'. A function node needs both keys: "
                f"'_import_path' (the module) and 'function' (the name within it)."
            )
        _assert_valid_contract(entry, f"node '{name}'")
        _assert_valid_overwrite_units(entry, name)
        return cls(
            name=entry["name"],
            inputs=entry["inputs"],
            expression=entry.get("expression"),
            import_path=entry.get("_import_path"),
            function=entry.get("function"),
            units=entry.get("units"),
            dims=entry.get("dims"),
            dtype=entry.get("dtype"),
            coords=entry.get("coords"),
            freq=entry.get("freq"),
            overwrite_units=entry.get("overwrite_units", False),
        )


@dataclass
class CacheSpec:
    """Specification for the [cache] section.

    ``recompute`` and ``disable`` follow Hamilton's caching API: each is either
    a boolean (apply to all nodes) or a list of node names.
    """

    path: str = ".catena_cache"
    recompute: bool | list[str] = field(default_factory=list)
    disable: bool | list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, entry: dict) -> Self:
        """Construct and validate from a raw [cache] TOML entry."""

        def _coerce(key: str) -> bool | list[str]:
            val = entry.get(key, [])
            if isinstance(val, bool):
                return val
            if isinstance(val, list) and all(isinstance(v, str) for v in val):
                return val
            raise CatenaValueError(
                f"[cache] '{key}' must be a boolean or a list of node names, "
                f"got {val!r}."
            )

        return cls(
            path=entry.get("path", ".catena_cache"),
            recompute=_coerce("recompute"),
            disable=_coerce("disable"),
        )


def assert_dim_name(value: Any, where: str) -> str:
    """Validate a required dimension-name key (``[blocking].dim``, ``[subset].dim``)."""
    if not isinstance(value, str) or not value:
        raise CatenaValueError(f"{where} must be a non-empty string, got {value!r}.")
    return value


@dataclass
class BlockingSpec:
    """Specification for the [blocking] section.

    Controls how a partition dimension (``dim``) is split into fixed-size
    sequential blocks to bound peak memory usage. ``dim`` may be any dimension:
    ``pixel`` for gridded data, or ``location``, say, for non-gridded pipelines.
    """

    block_size: int
    dim: str

    @classmethod
    def from_config(cls, entry: dict) -> Self:
        """Construct and validate from a raw [blocking] TOML entry."""
        block_size = entry.get("block_size")
        if not isinstance(block_size, int) or block_size < 1:
            raise CatenaValueError(
                "[blocking] 'block_size' must be a positive integer, "
                f"got {block_size!r}."
            )
        dim = assert_dim_name(entry.get("dim"), "[blocking] 'dim'")
        return cls(block_size=block_size, dim=dim)


@dataclass
class SubsetSpec:
    """Specification for the [subset] section.

    Selects a contiguous slice ``[start, stop)`` of one dimension (``dim``) so that
    independent ``catena run`` processes can each handle a different chunk of the same
    input files. ``stop`` is exclusive (Python slice convention).

    ``dim`` mirrors `BlockingSpec.dim`: the two mechanisms partition the same way
    and differ only in *who* runs the parts (one process sequentially vs. many
    processes concurrently), so a non-gridded pipeline can subset over ``location``
    or ``site`` just as it can block over it. The one place ``pixel`` is still
    special is the gridded Zarr store (`catena.gridded.io.create_output_store`),
    whose layout *is* the pixel grid; it rejects any other ``dim``.
    """

    start: int
    stop: int
    dim: str

    @classmethod
    def from_config(cls, entry: dict) -> Self:
        """Construct and validate from a raw [subset] TOML entry."""

        def _index(key: str) -> int:
            value = entry.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CatenaValueError(
                    f"[subset] {key!r} must be a non-negative integer, got {value!r}."
                )
            return value

        start = _index("start")
        stop = _index("stop")
        if stop <= start:
            raise CatenaValueError(
                f"[subset] 'stop' ({stop}) must be greater than 'start' ({start})."
            )
        dim = assert_dim_name(entry.get("dim"), "[subset] 'dim'")
        return cls(start=start, stop=stop, dim=dim)


@dataclass
class IOSpec:
    """I/O specification for a single input or output section.

    ``vars`` maps this section's file variables to Hamilton node names, in one of
    three forms:

    - a **list** ``["temperature", ...]`` — the node name is derived from the file
      variable and the section's suffix (``{var}{suffix}``), the convenient default;
    - a **mapping** ``{node_name: file_var}`` — an explicit, suffix-free alias, e.g.
      ``{temperature_daily = "t2m"}`` (input: read file var ``t2m`` as node
      ``temperature_daily``) or ``{gpp_daily = "gpp"}`` (output: write node
      ``gpp_daily`` to file var ``gpp``). Use this to decouple file naming from DAG
      naming, or to alias a variable without renaming the file;
    - **omitted** (``None``) — *inputs only*: bind **every** variable in the file,
      through the suffix. An empty list is rejected:
      binding nothing is never what a section is for.

    ``suffix`` controls the list/load-everything forms' node names. When ``None``
    (the default) the effective suffix is derived from the section label
    (``_<label>``). Set ``suffix = ""`` on any section for bare names, or
    ``suffix = "_x"`` to choose an explicit suffix. It is ignored for the mapping
    form (which is already explicit). See ``catena.io.effective_suffix``.
    """

    path: str
    vars: list[str] | dict[str, str] | None = None
    suffix: str | None = None


def _validate_vars(label: str, vars_: Any) -> list[str] | dict[str, str]:
    """Validate a section's ``vars`` is a list[str] or a dict[str, str]."""
    if isinstance(vars_, dict):
        bad = [
            (k, v)
            for k, v in vars_.items()
            if not isinstance(k, str) or not isinstance(v, str)
        ]
        if bad:
            raise CatenaValueError(
                f"[{label}] 'vars' mapping must be {{node_name = file_var}} with "
                f"string keys and values, got offending entries {bad!r}."
            )
        return dict(vars_)
    if isinstance(vars_, list) and all(isinstance(v, str) for v in vars_):
        return list(vars_)
    raise CatenaValueError(
        f"[{label}] 'vars' must be a list of names or a {{node_name = file_var}} "
        f"mapping, got {vars_!r}."
    )


@dataclass
class CheckSpec:
    """One entry of ``[validation].checks``: a named input-compatibility check.

    ``check`` is a key in `catena.input_checks.CHECKS`; ``inputs`` are the resolved
    ``[inputs.*]`` section labels to pass (``["*"]`` already expanded at parse
    time); ``kwargs`` are the remaining inline-table keys forwarded verbatim.
    """

    check: str
    inputs: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedConfig:
    """Parsed pipeline configuration, ready to pass to build_driver."""

    modules: list[str]
    driver_config: dict[str, Any]
    node_specs: list["NodeSpec"] = field(default_factory=list)
    input_specs: dict[str, "IOSpec"] = field(default_factory=dict)
    output_specs: dict[str, "IOSpec"] = field(default_factory=dict)
    cache_spec: "CacheSpec | None" = None
    blocking_spec: "BlockingSpec | None" = None
    subset_spec: "SubsetSpec | None" = None
    checks: list["CheckSpec"] = field(default_factory=list)
    #: The directory relative paths in the config resolve against, normally the
    #: one holding the config file. ``None`` for a config with no file, which can
    #: then only use absolute paths and dotted ``_import_path`` names.
    base: Path | None = None
    #: Sections whose module came from an installed package rather than an
    #: ``_import_path``, in config order.
    registered_modules: list["RegisteredModule"] = field(default_factory=list)
