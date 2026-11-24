import json
from pathlib import Path

from .schemas import response_schema

SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.md").read_text(encoding="utf-8")


def system_prompt(capabilities):
    schema = response_schema([motion["name"] for motion in capabilities["motions"]],
                             capabilities["lighting"] is not None)
    return (
        f"{SYSTEM_PROMPT}\n"
        "Use emotion_expression with one autonomous motion for ordinary spoken replies, including factual answers. "
        "For a lighting request with an autonomous gesture, use emotion_expression with one lighting and one motion action. "
        "Use light_control for lighting alone. Use motion_control for a commanded motion and combined_control "
        "when the user explicitly commands both motion and lighting. "
        "Use conversation or no_action with an empty actions array only when no motion is appropriate. "
        "Use stop only with a sole stop action. Use at most one action of each type. "
        "Respect allow_explicit for commanded actions and allow_autonomous for gestures chosen by LUMO. "
        "Report simulated capabilities honestly. If there is no matching safe capability, explain briefly. "
        "Brightness is 0–100; relative adjustments are percentage points.\n\n"
        f"Runtime capabilities and current hardware state (JSON):\n{json.dumps(capabilities)}\n\n"
        f"Behavior Event schema (JSON):\n{json.dumps(schema)}"
    )
