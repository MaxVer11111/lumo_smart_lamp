"""Retarget recorded joint deltas without changing the recording's time scale."""

import bisect
import csv
import hashlib
import io
import math
from collections import OrderedDict
from pathlib import Path

from .behavior.performance import mark
from .demo_motion import JOINTS

RECORDINGS_DIR = Path(__file__).parent / "recordings"
AI_MOTION_DIR = Path(__file__).resolve().with_name("motion")
VOICE_RECORDINGS = {name: name for name in (
    "nod", "headshake", "curious", "excited", "happy_wiggle", "sad",
    "scanning", "shock", "shy", "wake_up", "idle",
)}
VOICE_RECORDINGS["greet"] = "happy_wiggle"


def available_recordings(recordings_dir=None):
    folder = RECORDINGS_DIR if recordings_dir is None else recordings_dir
    return sorted(p.stem for p in folder.glob("*.csv"))


class RecordingCache:
    """Store immutable, validated poses for one runtime's authorized recordings."""

    def __init__(self):
        self._entries = OrderedDict()

    def get(self, key):
        if key in self._entries:
            poses, info = self._entries[key]
            self._entries.move_to_end(key)
            return [dict(zip(JOINTS, pose)) for pose in poses], info.copy()
        return None

    def store(self, key, frames, info):
        if len(frames) <= 8192:
            self._entries[key] = tuple(tuple(frame[joint] for joint in JOINTS) for frame in frames), info.copy()
            if len(self._entries) > 16:
                self._entries.popitem(last=False)


def adapt_recording(profile, name, scale=0.25, fps=30, *, expected_sha256=None, recordings_dir=None,
                    cache=None):
    """Use one uniform scale for positions AND velocities, preserving coordination.

    The requested scale is a ceiling. Further shrink it to retain 20% of the
    available positional margin and 10% of the configured speed budget.
    """
    folder = RECORDINGS_DIR if recordings_dir is None else recordings_dir
    if name not in available_recordings(folder):
        raise ValueError(f"Unknown recording: {name}")
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Recording scale must be > 0 and <= 1")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    raw = (folder / f"{name}.csv").read_bytes()
    if expected_sha256 is None:
        return _adapt_csv(raw, profile, name, scale, fps)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Recording differs from the locally verified file")
    if cache is None:
        return _adapt_csv(raw, profile, name, scale, fps)

    from .behavior.capabilities import profile_fingerprint

    effective_limits = tuple((profile.neutral[j], tuple(profile.limits[j]), profile.speeds[j]) for j in JOINTS)
    key = (name, digest, profile_fingerprint(profile), profile.data["calibration_sha256"],
           effective_limits, scale, fps)
    if hit := cache.get(key):
        return hit
    frames, info = _adapt_csv(raw, profile, name, scale, fps)
    cache.store(key, frames, info)
    return frames, info


def _adapt_csv(raw, profile, name, scale, fps):
    with io.StringIO(raw.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        if set(reader.fieldnames or ()) != {"timestamp", *JOINTS}:
            raise ValueError("Recording must contain timestamp and exactly five joint columns")
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if len(rows) < 2 or any(not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("Recording requires at least two finite frames")
    times = [row["timestamp"] - rows[0]["timestamp"] for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Recording timestamps must be strictly increasing")
    deltas = [{j: row[j] - rows[0][j] for j in JOINTS} for row in rows]
    actual_scale = scale
    for j in JOINTS:
        low, high = profile.limits[j]
        neutral = profile.neutral[j]
        for row in deltas:
            delta = row[j]
            if delta > 0:
                actual_scale = min(actual_scale, 0.8 * (high - neutral) / delta)
            elif delta < 0:
                actual_scale = min(actual_scale, 0.8 * (neutral - low) / -delta)
        for index in range(1, len(rows)):
            velocity = abs(deltas[index][j] - deltas[index - 1][j]) / (times[index] - times[index - 1])
            if velocity:
                actual_scale = min(actual_scale, 0.9 * profile.speeds[j] / velocity)
    if actual_scale <= 0 or not math.isfinite(actual_scale):
        raise ValueError("No motion margin around neutral; adjust the validated neutral pose")
    count = math.ceil(times[-1] * fps)
    frames = []
    for index in range(count + 1):
        t = min(index / fps, times[-1])
        right = min(bisect.bisect_right(times, t), len(times) - 1)
        left = max(0, right - 1)
        weight = (t - times[left]) / (times[right] - times[left])
        pose = {j: profile.neutral[j] + actual_scale * (
            deltas[left][j] + weight * (deltas[right][j] - deltas[left][j])) for j in JOINTS}
        profile.validate(pose)
        frames.append(pose)
    return frames, {"recording": name, "requested_scale": scale, "actual_scale": actual_scale,
                    "source_seconds": times[-1], "adapted_seconds": count / fps}


def recording_trajectory(profile, start, name, scale=0.25, fps=30, *, expected_sha256=None,
                         recordings_dir=None, cache=None):
    mark("adaptation_started")
    frames, info = adapt_recording(profile, name, scale, fps, expected_sha256=expected_sha256,
                                   recordings_dir=recordings_dir, cache=cache)
    mark("adaptation_finished")
    intro = profile.trajectory(start, "", fps, targets=[profile.neutral])
    outro = profile.trajectory(frames[-1], "", fps, targets=[profile.neutral])
    return intro + frames[1:] + outro[1:], info
