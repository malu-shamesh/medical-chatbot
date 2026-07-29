"""Image-based question answering via a vision-capable LLM.

For medical images (e.g. a photo of a rash or a scan), the model describes
general visual characteristics and possible information categories only - it
never states a diagnosis. This is enforced by the same scope-guard instructions
used for text queries, plus an explicit reminder in the image prompt, since a
vision model asked to describe a medical photo is especially prone to producing
diagnosis-shaped language ("this looks like...") unless told not to.
"""

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.guardrails import EMERGENCY_RESPONSE, SCOPE_GUARD_INSTRUCTIONS, add_disclaimer, check_emergency

DEFAULT_VISION_MODEL = "gpt-4o-mini"

IMAGE_SYSTEM_INSTRUCTIONS = (
    SCOPE_GUARD_INSTRUCTIONS
    + "\nYou are looking at a medical image the user has shared. Describe only "
    "general, visible characteristics (e.g. color, shape, size, location) and "
    "what general categories of condition such characteristics can be "
    "associated with in medical literature. Do not state or imply a diagnosis "
    "for this specific image."
)


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def ask_about_image(image_path: Path, question: str, model: str = DEFAULT_VISION_MODEL) -> dict:
    """Return {'answer': str, 'is_emergency': bool} for a question about a medical image."""
    if check_emergency(question):
        return {"answer": EMERGENCY_RESPONSE, "is_emergency": True}

    llm = ChatOpenAI(model=model, temperature=0.2)
    image_data = _encode_image(image_path)

    message = HumanMessage(content=[
        {"type": "text", "text": f"{IMAGE_SYSTEM_INSTRUCTIONS}\n\nQuestion: {question}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ])

    response = llm.invoke([message])
    return {"answer": add_disclaimer(response.content), "is_emergency": False}
