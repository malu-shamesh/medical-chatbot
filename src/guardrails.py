"""Safety guardrails for the medical chatbot.

A medical chatbot carries real safety and liability weight that a generic RAG
assistant doesn't: a wrong or delayed answer to someone describing a medical
emergency, or a confident-sounding diagnosis/prescription from an LLM, can cause
real harm. These guardrails run before (and after) the RAG/LLM call rather than
being left to prompt wording alone, since prompt instructions can be missed or
worked around but a keyword check on the input cannot.

This module implements three layers:
1. Emergency detection: bypass the LLM entirely and return a fixed response
   directing the user to emergency services, for inputs matching red-flag
   symptoms or crisis language.
2. Scope guard: a system prompt addendum instructing the model to describe
   general, retrieved information only - never to diagnose, prescribe, or give
   dosing advice - and to defer to a licensed professional for anything specific
   to the user's situation.
3. Disclaimer: appended to every non-emergency response, so the informational
   (not diagnostic) nature of the tool is never left implicit.
"""

import re

EMERGENCY_PATTERNS = [
    r"\bsuicid\w*\b",
    r"\bkill (myself|me)\b",
    r"\bself[\s-]?harm\b",
    r"\bcan'?t breathe\b",
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    r"\bsevere (bleeding|blood loss)\b",
    r"\bunconscious\b",
    r"\bnot breathing\b",
    r"\boverdose\b",
    r"\banaphyla\w*\b",
    r"\bpoison\w*\b",
]
_EMERGENCY_REGEX = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please call your local "
    "emergency number (911 in the US) or go to the nearest emergency room right "
    "away. If you are in crisis or thinking about suicide, you can also call or "
    "text 988 (Suicide & Crisis Lifeline, US) to talk to someone now.\n\n"
    "This chatbot is not equipped to help with emergencies and cannot provide "
    "the immediate care you may need."
)

DISCLAIMER = (
    "\n\n---\n"
    "This information is for general educational purposes only and is not a "
    "substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider about your specific situation."
)

SCOPE_GUARD_INSTRUCTIONS = (
    "You are a medical information assistant. Rules you must follow:\n"
    "- Only describe general medical information found in the retrieved context.\n"
    "- Never diagnose a condition, prescribe or recommend a specific medication, "
    "dosage, or treatment plan for the user's personal situation.\n"
    "- If the question requires a diagnosis or personalized treatment decision, "
    "say that a licensed healthcare provider should be consulted, and explain "
    "what general information is available instead.\n"
    "- If the retrieved context does not contain the answer, say so plainly "
    "rather than guessing."
)


def check_emergency(text: str) -> bool:
    """Return True if text matches emergency/crisis language requiring an immediate,
    non-LLM response rather than a RAG-generated answer."""
    return bool(_EMERGENCY_REGEX.search(text))


def add_disclaimer(answer: str) -> str:
    return answer + DISCLAIMER
