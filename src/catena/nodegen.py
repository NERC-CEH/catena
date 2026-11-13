"""Generate Hamilton-compatible node modules from config specs."""

import hashlib
import linecache
import sys
import types
from dataclasses import astuple
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

import xarray as xr
import xarray_annotated as xra

from .errors import CatenaValueError
from .importing import import_user_module
from .specs import NodeSpec


def _node_namespace(base: Path | None = None) -> dict[str, Any]:
    """Build the namespace every generated node's body is ``exec``'d in."""
    return {
        "xr": xr,
        "xra": xra,
        "Any": Any,
        # A `[[node]]` naming '_import_path' + 'function' resolves it the same way a
        # config section does, so a relative .py path is relative to the config.
        "__import_module": partial(import_user_module, base=base),
        # Available to node expressions (e.g. the [[resample]] preset desugars to
        # ``__transforms.resample(...)``).
        "__transforms": import_module("catena.transforms"),
    }


def _module_name(node_specs: list[NodeSpec], base: Path | None) -> str:
    """Return a stable module name, keyed on the specs it is generated from.

    Hamilton requires the generated module to live in ``sys.modules`` (it resolves a
    node's originating function through it), so the registration cannot simply be
    dropped. A random per-build name would therefore leak one entry *per build* —
    unbounded in a long-lived process such as a calibration loop or a test session.

    Keying the name on the specs' content, and on the directory a relative
    ``_import_path`` resolves against, means rebuilding the same config reuses one
    entry, while two different configs still get distinct modules.
    """
    payload = repr((str(base), [astuple(spec) for spec in node_specs])).encode()
    return f"catena_node_generated_{hashlib.sha256(payload).hexdigest()[:12]}"


def make_node_module(
    node_specs: list[NodeSpec], base: Path | None = None
) -> types.ModuleType:
    """Generate a Hamilton-compatible module with one function per node spec.

    The node's *body* is built by ``exec`` (a ``[[node]]`` expression is arbitrary
    user code), but its declared output contract is attached programmatically:
    ``xarray_annotated.annotate`` builds the ``Annotated`` return hint from the raw
    spec values, then the decorators are applied as ordinary calls — no
    annotation/decorator source is generated. A node declaring ``units`` has its
    output stamped with them by ``xarray_annotated.stamp_output_units`` (which
    replaces a present label only under ``overwrite_units``) before
    ``xarray_annotated.check_annotations`` checks it.

    The module is registered in ``sys.modules`` under a content-derived name (see
    `_module_name`), which Hamilton requires and which keeps repeated builds of the
    same config from accumulating entries.
    """
    mod = types.ModuleType(_module_name(node_specs, base))
    ns: dict = _node_namespace(base)
    for spec in node_specs:
        source = _build_fn_code(spec)
        filename = f"<catena node {spec.name} in {mod.__name__}>"
        # Hamilton's result cache versions a node by hashing its source, which it
        # reads with `inspect.getsource`. An `exec`'d function has no file to read
        # from, so without this the node hashes as unversioned and a changed
        # `expression` is served stale from the cache. Registering the source under
        # the name it was compiled with is what `inspect` looks up.
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        try:
            code = compile(source, filename, "exec")
        except SyntaxError as exc:
            raise CatenaValueError(
                f"[[node]] '{spec.name}' expression {spec.expression!r} is not "
                f"valid Python: {exc.msg}"
            ) from exc
        exec(code, ns)
        fn = _decorate(ns[spec.name], spec)
        fn.__module__ = mod.__name__
        setattr(mod, spec.name, fn)
    sys.modules[mod.__name__] = mod
    return mod


def _decorate(fn: Any, spec: NodeSpec) -> Any:
    """Attach the node's declared output contract to the bare ``exec``'d function."""
    # A declared unit/dims/dtype/coords/freq makes the node a typed producer: validated
    # at runtime and read by the build-time check. Without any, the node is unchecked.
    fn.__annotations__["return"] = xra.annotate(
        unit=spec.units,
        dims=spec.dims or None,
        dtype=spec.dtype,
        coords=spec.coords or None,
        freq=spec.freq,
    )
    if spec.inputs_share_output and (spec.units is not None or spec.dims):
        kept = xra.annotate(unit=spec.units, dims=spec.dims or None)
        for inp in spec.inputs:
            fn.__annotations__[inp] = kept
    if spec.units is not None:
        fn = xra.stamp_output_units(overwrite=spec.overwrite_units)(fn)
    return xra.check_annotations(fn)


def _build_fn_code(spec: NodeSpec) -> str:
    """Return source for the bare node function (params + body, no decorators)."""
    params = ", ".join(f"{inp}: Any" for inp in spec.inputs)
    if spec.expression is not None:
        body = f"    return {spec.expression}"
    else:
        kwargs = ", ".join(f"{inp}={inp}" for inp in spec.inputs)
        body = (
            f"    _fn = getattr(__import_module({spec.import_path!r}), "
            f"{spec.function!r})\n"
            f"    return _fn({kwargs})"
        )
    return f"def {spec.name}({params}):\n{body}\n"
