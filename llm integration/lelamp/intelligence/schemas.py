"""Strict JSON wire protocol for model decisions."""

import json
import math
from dataclasses import dataclass

INTENTS = ("motion_control", "light_control", "emotion_expression", "conversation",
           "combined_control", "stop", "no_action")
EMOTIONS = ("neutral", "happy", "excited", "sad", "frustrated", "tired",
            "confused", "curious", "surprised", "unknown")
BRIGHTNESS_LIMITS = {"set_brightness": (0, 100), "increase_brightness": (1, 100),
                     "decrease_brightness": (1, 100)}
LIGHT_OPERATIONS = ("turn_on", "turn_off", *BRIGHTNESS_LIMITS, "reading_mode", "night_mode")
ACTION_SETS = {
    "motion_control": ({"motion"},),
    "light_control": ({"lighting"},),
    "emotion_expression": (set(), {"motion"}, {"lighting"}, {"motion", "lighting"}),
    "conversation": (set(),),
    "combined_control": ({"motion", "lighting"},),
    "stop": ({"stop"},),
    "no_action": (set(),),
}
MAX_SPEECH = 1000


class InvalidEvent(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    type: str
    name: str | None = None
    operation: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class BehaviorEvent:
    intent: str
    user_emotion: str
    speech: str
    actions: tuple[Action, ...]

    @classmethod
    def parse(cls, raw):
        try:
            data = json.loads(raw, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant)
        except (TypeError, ValueError, RecursionError):
            raise InvalidEvent("Expected one valid JSON object with unique fields") from None
        _require_fields(data, {"intent", "user_emotion", "speech", "actions"})
        intent, emotion, speech, actions = (data[key] for key in ("intent", "user_emotion", "speech", "actions"))
        if intent not in INTENTS or emotion not in EMOTIONS:
            raise InvalidEvent("Invalid intent or emotion")
        if not isinstance(speech, str) or len(speech) > MAX_SPEECH:
            raise InvalidEvent("Invalid speech")
        try:
            speech.encode("utf-8")
        except UnicodeEncodeError:
            raise InvalidEvent("Invalid speech encoding") from None
        if not isinstance(actions, list) or len(actions) > 2:
            raise InvalidEvent("Expected an array of up to two actions")
        parsed = tuple(_parse_action(action) for action in actions)
        kinds = {action.type for action in parsed}
        if len(kinds) != len(parsed) or kinds not in ACTION_SETS[intent]:
            raise InvalidEvent("Conflicting actions or intent/action mismatch")
        return cls(intent, emotion, speech, parsed)


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidEvent("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise InvalidEvent("Non-finite JSON number")


def _require_fields(data, fields):
    if not isinstance(data, dict) or set(data) != fields:
        raise InvalidEvent("Missing or unexpected fields")


def _parse_action(data):
    if not isinstance(data, dict):
        raise InvalidEvent("Expected an action object")
    kind = data.get("type")
    if kind == "motion":
        _require_fields(data, {"type", "name"})
        if not isinstance(data["name"], str) or not data["name"]:
            raise InvalidEvent("Invalid motion name")
        return Action(kind, name=data["name"])
    if kind == "stop":
        _require_fields(data, {"type"})
        return Action(kind)
    if kind != "lighting":
        raise InvalidEvent("Unknown action type")
    operation = data.get("operation")
    if not isinstance(operation, str) or operation not in LIGHT_OPERATIONS:
        raise InvalidEvent("Unknown lighting operation")
    limits = BRIGHTNESS_LIMITS.get(operation)
    _require_fields(data, {"type", "operation", "value"} if limits else {"type", "operation"})
    value = data.get("value")
    if limits and (type(value) not in (int, float) or not limits[0] <= value <= limits[1] or not math.isfinite(value)):
        raise InvalidEvent("Brightness value outside operation range")
    return Action(kind, operation=operation, value=value)


def _object(**fields):
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def response_schema(motion_names, lighting):
    """Prompt schema. Local validation additionally checks intent/action consistency."""
    motions = [_object(type={"const": "motion"}, name={"enum": motion_names})] if motion_names else []
    lights = []
    if lighting:
        for operation in LIGHT_OPERATIONS:
            fields = {"type": {"const": "lighting"}, "operation": {"const": operation}}
            if operation in BRIGHTNESS_LIMITS:
                low, high = BRIGHTNESS_LIMITS[operation]
                fields["value"] = {"type": "number", "minimum": low, "maximum": high}
            lights.append(_object(**fields))
    return _object(intent={"enum": INTENTS}, user_emotion={"enum": EMOTIONS},
                   speech={"type": "string", "maxLength": MAX_SPEECH},
                   actions={"type": "array", "maxItems": 2,
                            "items": {"oneOf": [_object(type={"const": "stop"}), *motions, *lights]}})
