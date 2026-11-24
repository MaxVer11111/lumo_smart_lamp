"""Validated demo trajectories; positions use calibrated RANGE_M100_100 units."""

import csv
import hashlib
import json
import math
import threading
import time
from pathlib import Path

JOINTS = tuple(f"{name}.pos" for name in (
    "base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"
))
ACTIONS = ("nod", "headshake", "greet")


def calibration_digest(calibration):
    from dataclasses import asdict
    data = {name: asdict(value) for name, value in calibration.items()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class MotionProfile:
    def __init__(self, data):
        self.data = data
        self.neutral = data["neutral"]
        self.limits = data["limits"]
        self.speeds = data["max_speed"]
        self.amplitudes = data["amplitudes"]
        if data.get("units") != "range_m100_100":
            raise ValueError("Profile units must be range_m100_100")
        if not isinstance(data.get("lamp_id"), str) or not data["lamp_id"]:
            raise ValueError("Profile requires lamp_id")
        if not isinstance(data.get("calibration_sha256"), str):
            raise ValueError("Profile requires calibration_sha256")
        for values in (self.neutral, self.limits, self.speeds):
            if set(values) != set(JOINTS):
                raise ValueError("Profile must specify all five joints")
        for joint in JOINTS:
            low, high = self.limits[joint]
            speed = self.speeds[joint]
            if not all(math.isfinite(v) for v in (low, high, speed)):
                raise ValueError("Limits and speeds must be finite")
            if not -100 <= low < high <= 100 or speed <= 0:
                raise ValueError("Invalid limits or speed")
        if set(self.amplitudes) != set(ACTIONS):
            raise ValueError("Specify nod, headshake and greet amplitudes")
        if not all(math.isfinite(v) and v > 0 for v in self.amplitudes.values()):
            raise ValueError("Amplitudes must be positive and finite")
        self.validate(self.neutral)
        for name in ACTIONS:
            self.keyframes(name)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def validate(self, pose):
        if set(pose) != set(JOINTS):
            raise ValueError("Pose must contain exactly five joints")
        for joint, value in pose.items():
            low, high = self.limits[joint]
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Position outside validated limits: {joint}={value}")

    def keyframes(self, name):
        if name not in ACTIONS:
            raise ValueError(f"Unknown action: {name}")
        joint = "wrist_pitch.pos" if name == "nod" else "base_yaw.pos"
        frames = [dict(self.neutral)]
        for sign in (1, -1, 1, -1, 0):
            pose = dict(self.neutral)
            pose[joint] += sign * self.amplitudes[name]
            self.validate(pose)
            frames.append(pose)
        return frames

    def trajectory(self, start, name, fps=30, targets=None):
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.validate(start)
        frames = [dict(start)]
        for target in self.keyframes(name) if targets is None else targets:
            self.validate(target)
            # Smoothstep's peak derivative is 1.5; enforce per-joint speed.
            duration = max(0.5, *(1.5 * abs(target[j] - start[j]) / self.speeds[j] for j in JOINTS))
            count = math.ceil(duration * fps)
            for index in range(1, count + 1):
                t = index / count
                weight = t * t * (3 - 2 * t)
                pose = {j: start[j] + (target[j] - start[j]) * weight for j in JOINTS}
                self.validate(pose)
                frames.append(pose)
            start = target
        return frames


def export_csv(path, frames, fps=30):
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["timestamp", *JOINTS])
        writer.writeheader()
        for index, pose in enumerate(frames):
            writer.writerow({"timestamp": index / fps, **pose})


class DemoPlayer:
    """Synchronous single-writer player. stop() may be called from another thread."""

    def __init__(self, robot, profile, fps=30):
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.robot, self.profile, self.fps = robot, profile, fps
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def stop(self):
        # Holding torque is retained; never command a return-to-neutral on stop.
        self._cancel.set()

    def play(self, name, targets=None, *, source="generated", scale=0.25):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("An action is already playing")
        try:
            self._cancel.clear()
            if not self.robot.calibration or not self.robot.is_calibrated:
                raise RuntimeError("Missing or mismatched calibration")
            if self.robot.id != self.profile.data["lamp_id"]:
                raise ValueError("Profile belongs to a different lamp")
            if calibration_digest(self.robot.calibration) != self.profile.data["calibration_sha256"]:
                raise ValueError("Calibration changed; validate the motion profile again")
            observation = self.robot.get_observation()
            start = {joint: observation[joint] for joint in JOINTS}
            if source == "recording":
                from .demo_recordings import recording_trajectory
                frames, _ = recording_trajectory(self.profile, start, name, scale, self.fps)
            elif source == "generated":
                frames = self.profile.trajectory(start, name, self.fps, targets=targets)
            else:
                raise ValueError(f"Unknown motion source: {source}")
            for pose in frames:
                if self._cancel.is_set():
                    return False
                tick = time.monotonic()
                self.robot.send_action(pose)
                if self._cancel.wait(max(0, 1 / self.fps - (time.monotonic() - tick))):
                    return False
            return True
        finally:
            self._lock.release()
