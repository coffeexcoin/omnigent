"""Tests for policy wire-schema validation helpers."""

from __future__ import annotations

import pytest

from omnigent.policies.schema import validate_actor_context


def test_validate_actor_context_accepts_exact_nonempty_shape() -> None:
    """A canonical actor mapping is copied into the typed wire shape."""
    source = {"run_as": "alice@example.com"}

    actor = validate_actor_context(source)

    assert actor == source
    assert actor is not source


@pytest.mark.parametrize(
    "value",
    [
        "alice@example.com",
        {},
        {"run_as": ""},
        {"run_as": "   "},
        {"run_as": None},
        {"run_as": 7},
        {"run_as": "a" * 321},
        {"run_as": "alice@example.com", "role": "admin"},
    ],
)
def test_validate_actor_context_rejects_malformed_values(value: object) -> None:
    """Malformed or widened actor payloads fail closed at trust boundaries."""
    with pytest.raises(ValueError, match="actor"):
        validate_actor_context(value)


def test_validate_actor_context_allows_missing_actor() -> None:
    """Actor context remains optional for non-forwarded/direct call paths."""
    assert validate_actor_context(None) is None
