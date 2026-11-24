"""Model-facing registry and per-lamp authorization for recorded gestures."""

import hashlib
import json
from dataclasses import asdict, dataclass

from ..demo_recordings import AI_MOTION_DIR, VOICE_RECORDINGS, available_recordings, recording_trajectory
from ..intelligence.schemas import LIGHT_OPERATIONS

MOTION_DESCRIPTIONS = {
    "nod": "Nod to acknowledge, agree or say thanks.",
    "headshake": "Shake the head.",
    "curious": "Express curiosity or interest.",
    "excited": "Express excitement or celebrate good news.",
    "happy_wiggle": "Greet playfully or express happiness.",
    "sad": "Express sadness or gentle empathy.",
    "scanning": "Play a preset look-around gesture without sensing or tracking anyone.",
    "shock": "Express surprise.",
    "shy": "Express shyness.",
    "wake_up": "Play a wake-up gesture on request.",
    "idle": "Play the preset idle recording once on request.",
    "turn_left": "Turn left using a locally authored and calibrated recording.",
    "turn_right": "Turn right using a locally authored and calibrated recording.",
    "raise_arm": "Raise the lamp arm using a locally authored and calibrated recording.",
    "lower_head": "Lower the lamp head using a locally authored and calibrated recording.",
}
EXPLICIT_ONLY = {"headshake", "scanning", "wake_up", "idle", "turn_left", "turn_right",
                 "raise_arm", "lower_head"}


def profile_fingerprint(profile):
    data = {key: value for key, value in profile.data.items() if key != "ai_motion"}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def recording_fingerprint(recording):
    return hashlib.sha256((AI_MOTION_DIR / recording).read_bytes()).hexdigest()


@dataclass(frozen=True)
class MotionCapability:
    name: str
    description: str
    recording: str
    enabled: bool
    hardware_verified: bool
    allow_autonomous: bool
    allow_explicit: bool = True
    recording_sha256: str | None = None


class CapabilityRegistry:
    def __init__(self, *, simulated, motion=False, profile=None, scale=0.25,
                 lighting=None, tts_mode=None, recording_cache=None):
        if simulated and motion:
            raise ValueError("Physical motion cannot use a simulated capability registry")
        self.simulated = simulated
        self.lighting = lighting
        self.tts_mode = tts_mode
        recordings = set(available_recordings(AI_MOTION_DIR))
        approvals = self.motion_approvals(profile) if motion else {}
        self.motions = {}
        for name, description in MOTION_DESCRIPTIONS.items():
            recording = VOICE_RECORDINGS.get(name, name)
            filename = recording + ".csv"
            exists = recording in recordings
            approval = approvals.get(name)
            autonomous = name not in EXPLICIT_ONLY
            if approval:
                digest, max_scale, allow_autonomous = approval
                autonomous = autonomous and allow_autonomous
                if not exists:
                    raise ValueError(f"Recording for {name} differs from the locally verified file")
                if scale > max_scale:
                    raise ValueError(f"Requested scale exceeds the locally verified scale for {name}")
                recording_trajectory(profile, profile.neutral, recording, scale, expected_sha256=digest,
                                     recordings_dir=AI_MOTION_DIR, cache=recording_cache)
            self.motions[name] = MotionCapability(
                name, description, filename, exists and (simulated or bool(approval)),
                bool(approval), autonomous, recording_sha256=approval[0] if approval else None)
        if motion and not approvals:
            raise ValueError("Profile requires at least one locally verified ai_motion action")

    @staticmethod
    def motion_approvals(profile):
        if profile is None:
            raise ValueError("Physical motion requires a MotionProfile")
        config = profile.data.get("ai_motion", {})
        if not isinstance(config, dict) or set(config) != {"profile_sha256", "actions"}:
            raise ValueError("Profile requires ai_motion with profile_sha256 and actions")
        if config["profile_sha256"] != profile_fingerprint(profile):
            raise ValueError("Local motion authorization does not match the current profile")
        actions = config["actions"]
        if not isinstance(actions, dict):
            raise ValueError("ai_motion.actions must be an object")
        approvals = {}
        for name, entry in actions.items():
            required = {"recording_sha256", "max_scale"}
            if (name not in MOTION_DESCRIPTIONS or not isinstance(entry, dict)
                    or not required <= set(entry) <= required | {"allow_autonomous"}):
                raise ValueError("Invalid locally verified action entry")
            limit = entry["max_scale"]
            digest = entry["recording_sha256"]
            autonomous = entry.get("allow_autonomous", False)
            if type(limit) not in (int, float) or not 0 < limit <= 1 or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Invalid recording hash or verified scale")
            if type(autonomous) is not bool:
                raise ValueError("allow_autonomous must be a boolean")
            approvals[name] = digest, limit, autonomous
        return approvals

    def prompt_data(self, *, motion_ready=True, lighting_ready=True):
        motions = []
        for capability in self.motions.values():
            if capability.enabled and motion_ready:
                info = asdict(capability)
                del info["recording_sha256"]
                motions.append(info)
        lighting = None
        if self.lighting is not None and lighting_ready:
            lighting = {"simulated": self.lighting.simulated, "operations": LIGHT_OPERATIONS,
                        **self.lighting.state()}
        return {"motion_simulated": self.simulated, "motions": motions,
                "lighting": lighting, "tts": self.tts_mode, "stop": True}
