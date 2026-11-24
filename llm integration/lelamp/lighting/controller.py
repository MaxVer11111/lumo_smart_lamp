"""Logical brightness controller with an optional verified hardware output."""

import math
import threading


class LightingController:
    def __init__(self, *, write_output=None, default=65, reading=85, night=12, fade_seconds=0.3):
        self._levels = {"default": default, "reading": reading, "night": night}
        if not all(math.isfinite(value) and 0 < value <= 100 for value in self._levels.values()):
            raise ValueError("Lighting mode levels must be in (0, 100]")
        self._fade_seconds = self._duration(fade_seconds)
        self._state_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._write = write_output
        self._brightness = default
        self._on = False
        if self._write is not None:
            self._write(0)

    @property
    def simulated(self):
        return self._write is None

    def state(self):
        with self._state_lock:
            return {"brightness": self._brightness, "on": self._on}

    def get_brightness(self):
        return self.state()["brightness"]

    def _apply(self, brightness, on):
        if self._write is not None:
            self._write(brightness if on else 0)
        with self._state_lock:
            self._brightness, self._on = brightness, on

    def turn_off(self):
        with self._output_lock:
            self._apply(self.get_brightness(), False)

    def turn_on(self):
        with self._output_lock:
            self._apply(self.get_brightness() or self._levels["default"], True)

    @staticmethod
    def _validate_value(value):
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("Brightness must be in [0, 100]")

    @staticmethod
    def _duration(seconds):
        if not math.isfinite(seconds) or not 0 <= seconds <= 30:
            raise ValueError("Lighting fade duration must be in [0, 30]")
        return seconds

    def set_brightness(self, value):
        self._validate_value(value)
        with self._output_lock:
            self._apply(value, value > 0)

    def _adjust(self, delta):
        if not math.isfinite(delta) or not 1 <= abs(delta) <= 100:
            raise ValueError("Relative adjustment must be in [1, 100]")
        with self._output_lock:
            value = min(100, max(0, self.get_brightness() + delta))
            self._apply(value, value > 0)

    def increase_brightness(self, value):
        if value < 1:
            raise ValueError("Expected positive brightness adjustment")
        self._adjust(value)

    def decrease_brightness(self, value):
        if value < 1:
            raise ValueError("Expected positive brightness adjustment")
        self._adjust(-value)

    def fade_to(self, value, duration=None, *, cancel=None):
        """Local-only fade from the current output level with a cooperative stop."""
        self._validate_value(value)
        seconds = self._fade_seconds if duration is None else self._duration(duration)
        steps = max(1, math.ceil(seconds * 20))
        token = threading.Event() if cancel is None else cancel
        with self._output_lock:
            state = self.state()
            initial = state["brightness"] if state["on"] else 0
            for step in range(1, steps + 1):
                if token.wait(seconds / steps):
                    return False
                level = value if step == steps else initial + (value - initial) * step / steps
                self._apply(level, level > 0)
        return True

    def set_mode(self, mode):
        self.set_brightness(self._levels[mode])

    def execute(self, operation, value):
        operations = {
            "turn_on": self.turn_on, "turn_off": self.turn_off,
            "set_brightness": self.set_brightness, "increase_brightness": self.increase_brightness,
            "decrease_brightness": self.decrease_brightness,
        }
        modes = {"reading_mode": "reading", "night_mode": "night"}
        if operation in modes:
            self.set_mode(modes[operation])
        elif value is None:
            operations[operation]()
        else:
            operations[operation](value)
