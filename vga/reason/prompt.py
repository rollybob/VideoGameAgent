"""
The controller contract handed to the VLM: system prompt + the `act` tool schema.

We force a single `act` tool call (strict schema) so the model returns a
guaranteed-valid structured action instead of free text we would have to parse.
The button enum IS the abstract controller from core/contract.py - keep them in
sync (base.BUTTON_CHOICES is the source of truth).
"""

from __future__ import annotations

from .base import BUTTON_CHOICES

SYSTEM_PROMPT = """\
You are VGA, an agent that plays video games from the screen alone, the way a
human does. You are given one screenshot of a Game Boy Advance game and must
choose the single best controller input to make progress right now.

How to play well:
- Read what is on screen. Menus, dialogue, tutorials, and prompts tell you what
  to do - follow them like a new player would.
- Advance dialogue and confirm menus with A. Cancel or back out with B.
- Move with the D-pad (up/down/left/right). Open the menu with start.
- If nothing needs doing this instant (an animation is playing, the screen is
  mid-transition), choose "wait".
- Pick exactly ONE input. You will see the result on the next screenshot and can
  react then. Do not try to plan a long sequence in one step.

You see pixels only - there is no hidden game state available to you. Base every
decision on what is visible in the image. Always respond by calling the `act`
tool. Keep your reason to one short sentence."""


def act_tool() -> dict:
    """The strict `act` tool. button is the abstract controller; repeats allows a
    short mash (e.g. tapping through a stretch of dialogue)."""
    return {
        "name": "act",
        "description": "Issue one controller input for the current frame.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One short sentence: why this input now.",
                },
                "button": {
                    "type": "string",
                    "enum": BUTTON_CHOICES,
                    "description": "The controller input to issue (or 'wait' to do nothing).",
                },
                "repeats": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "How many times to tap the button (1 unless mashing through dialogue).",
                },
            },
            "required": ["reason", "button", "repeats"],
            "additionalProperties": False,
        },
    }


def user_text(goal: str, step: int, last_action: str) -> str:
    """The per-frame text that accompanies the screenshot."""
    lines = [f"Step {step}. This is the current screen."]
    if goal:
        lines.append(f"Your objective: {goal}")
    if last_action:
        lines.append(f"Your previous input was: {last_action}")
    lines.append("Choose the single best controller input now via the act tool.")
    return "\n".join(lines)
