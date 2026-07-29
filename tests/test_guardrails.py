import pytest

from src.guardrails import DISCLAIMER, add_disclaimer, check_emergency

EMERGENCY_EXAMPLES = [
    "I've been having chest pain for the last hour",
    "My friend is unconscious and not breathing",
    "I think I'm having a heart attack",
    "I took too many pills, is that an overdose?",
    "I want to kill myself",
    "I've been having thoughts of suicide",
    "She is bleeding badly, severe bleeding from a cut",
    "I think this is anaphylaxis, my throat is closing up",
]

NON_EMERGENCY_EXAMPLES = [
    "What are common symptoms of the flu?",
    "How does ibuprofen work?",
    "What foods are good for managing diabetes?",
    "Can you explain what cholesterol is?",
    "What is a normal resting heart rate?",
]


@pytest.mark.parametrize("text", EMERGENCY_EXAMPLES)
def test_check_emergency_detects_red_flag_language(text):
    assert check_emergency(text) is True


@pytest.mark.parametrize("text", NON_EMERGENCY_EXAMPLES)
def test_check_emergency_does_not_flag_routine_questions(text):
    assert check_emergency(text) is False


def test_check_emergency_is_case_insensitive():
    assert check_emergency("CHEST PAIN and can't breathe") is True


def test_add_disclaimer_appends_disclaimer_text():
    result = add_disclaimer("Ibuprofen is an NSAID.")

    assert result.startswith("Ibuprofen is an NSAID.")
    assert DISCLAIMER in result
