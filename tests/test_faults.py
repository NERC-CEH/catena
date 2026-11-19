"""The set of fault kinds is public API, so each one is documented and produced."""

import pathlib
import re
import typing

import pytest

import catena
from catena import faults
from catena.faults import Fault, FaultKind

KINDS = typing.get_args(FaultKind.__value__)


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_is_documented(kind):
    assert re.search(rf"^\| `\"{kind}\"` \|", faults.__doc__ or "", re.MULTILINE)


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_is_produced(kind):
    """A kind named nowhere but its own module is dead public API."""
    root = pathlib.Path(catena.__file__).parent
    sources = (p.read_text() for p in root.rglob("*.py") if p.name != "faults.py")
    assert any(f'"{kind}"' in text for text in sources)


def test_str_is_the_message():
    assert str(Fault("plan_invalid", "cannot plan")) == "cannot plan"
