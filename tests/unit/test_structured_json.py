"""Tests for the single-object JSON recovery utility (real-adapter-repair §3)."""

from __future__ import annotations

import pytest

from egrag.structured_json import StructuredOutputError, recover_json_object


@pytest.mark.unit
def test_strict_valid_json_is_not_recovered() -> None:
    r = recover_json_object('{"answer": "x", "citations": []}')
    assert r.data == {"answer": "x", "citations": []}
    assert r.recovered is False


@pytest.mark.unit
def test_trailing_whitespace_is_strict() -> None:
    r = recover_json_object('  {"a": 1}\n\n  ')
    assert r.data == {"a": 1}
    assert r.recovered is False  # strip + whole-string parse still works


@pytest.mark.unit
def test_json_then_prose_suffix_is_recovered() -> None:
    r = recover_json_object('{"answer": "Polish", "citations": []}, where "Polish" is the answer.')
    assert r.data == {"answer": "Polish", "citations": []}
    assert r.recovered is True


@pytest.mark.unit
def test_prose_prefix_then_one_object_is_recovered() -> None:
    r = recover_json_object('Here is the result: {"answer": "yes", "citations": ["c1"]}')
    assert r.data["answer"] == "yes"
    assert r.recovered is True


@pytest.mark.unit
def test_braces_inside_strings_are_handled() -> None:
    r = recover_json_object('prefix {"answer": "a }{ b", "note": "{not json}"} suffix')
    assert r.data == {"answer": "a }{ b", "note": "{not json}"}
    assert r.recovered is True


@pytest.mark.unit
def test_two_competing_objects_are_rejected() -> None:
    with pytest.raises(StructuredOutputError, match="competing"):
        recover_json_object('{"answer": "a"} {"answer": "b"}')


@pytest.mark.unit
def test_one_object_plus_invalid_brace_group_is_not_competing() -> None:
    # A stray {non-json} prose group is not a competing *valid* object.
    r = recover_json_object('{"answer": "a", "citations": []} and then {not json here}')
    assert r.data["answer"] == "a"
    assert r.recovered is True


@pytest.mark.unit
def test_truncated_object_is_rejected() -> None:
    with pytest.raises(StructuredOutputError, match=r"truncated|no valid"):
        recover_json_object('{"answer": "a", "citations": [')


@pytest.mark.unit
def test_top_level_array_is_rejected() -> None:
    with pytest.raises(StructuredOutputError, match="not an object"):
        recover_json_object('[{"answer": "a"}, {}]')


@pytest.mark.unit
def test_scalar_is_rejected() -> None:
    with pytest.raises(StructuredOutputError, match="not an object"):
        recover_json_object("42")


@pytest.mark.unit
def test_empty_is_rejected() -> None:
    with pytest.raises(StructuredOutputError, match="empty"):
        recover_json_object("   \n  ")


@pytest.mark.unit
def test_pure_prose_is_rejected() -> None:
    with pytest.raises(StructuredOutputError, match="no valid"):
        recover_json_object("I cannot answer this question.")
