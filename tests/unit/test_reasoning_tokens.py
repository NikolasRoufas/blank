"""Token-counter and budget tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from egrag.reasoning import CharacterTokenCounter, WhitespaceTokenCounter
from egrag.reasoning.models import TokenBudget


@pytest.mark.unit
def test_character_counter_is_conservative_and_deterministic() -> None:
    counter = CharacterTokenCounter(chars_per_token=4.0)
    assert counter.count("") == 0
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2  # ceil(5/4)
    assert counter.count("abcde") == counter.count("abcde")


@pytest.mark.unit
def test_whitespace_counter() -> None:
    assert WhitespaceTokenCounter().count("one two three") == 3
    assert WhitespaceTokenCounter().count("") == 0


@pytest.mark.unit
def test_character_counter_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        CharacterTokenCounter(chars_per_token=0.0)


@pytest.mark.unit
def test_token_budget_reserves_output() -> None:
    budget = TokenBudget(total=100, reserved_output=30)
    assert budget.available == 70
    with pytest.raises(ValidationError):
        TokenBudget(total=10, reserved_output=20)  # reserve exceeds total
