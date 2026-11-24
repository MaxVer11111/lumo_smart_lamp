"""Bounded PCM capture with optional local WebRTC endpoint detection."""

import io
import math
import os
import time
import wave
from array import array
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event

RATE = 16000


class CaptureError(RuntimeError):
    pass


class NoSpeech(CaptureError):
    pass


def pcm_wav(pcm):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return output.getvalue()


@dataclass(frozen=True)
class VADSettings:
    frame_ms: int = 20
    silence_ms: int = 800
    min_speech_ms: int = 200
    max_speech_seconds: float = 15
    pre_roll_ms: int = 300
    wait_timeout: float = 10
    mode: int = 1

    def __post_init__(self):
        if not (self.frame_ms in (10, 20, 30) and self.mode in range(4)
                and 3 * self.frame_ms <= self.silence_ms <= 5000
                and 3 * self.frame_ms <= self.min_speech_ms <= 5000
                and 0 <= self.pre_roll_ms <= 2000 and 0 < self.wait_timeout <= 60
                and self.min_speech_ms + self.silence_ms <= self.max_speech_seconds * 1000
                and self.max_speech_seconds + self.pre_roll_ms / 1000 <= 30):
            raise ValueError("Invalid VAD timing or aggressiveness settings")

    @classmethod
    def from_env(cls):
        if os.getenv("VAD_PROVIDER", "webrtc").strip().lower() != "webrtc":
            raise ValueError("VAD_PROVIDER must be webrtc")
        return cls(int(os.getenv("VAD_FRAME_MS", "20")), int(os.getenv("VAD_SILENCE_MS", "800")),
                   int(os.getenv("VAD_MIN_SPEECH_MS", "200")),
                   float(os.getenv("VAD_MAX_SPEECH_SECONDS", "15")),
                   int(os.getenv("VAD_PRE_ROLL_MS", "300")), float(os.getenv("VAD_WAIT_TIMEOUT", "10")),
                   int(os.getenv("VAD_MODE", "1")))

    def detector(self):
        try:
            import webrtcvad
        except ImportError:
            raise ValueError("WebRTC VAD requires uv run --extra vad") from None
        vad = webrtcvad.Vad(self.mode)
        return lambda frame: vad.is_speech(frame, RATE)


class MicrophoneCapture:
    def __init__(self, vad=None, *, reuse=False):
        self.vad = vad
        self.reuse = reuse
        self.frame_ms = vad.frame_ms if vad else 20
        self._stream = None
        self._device = None
        self._mailbox = None

    def _callback(self, data, frames, clock, status):
        mailbox = self._mailbox
        if mailbox is None:
            return
        queue, lost = mailbox
        if status.input_overflow:
            lost.set()
            return
        # Estimate the end of the captured frame on the same monotonic clock as the trace.
        elapsed = clock.currentTime - clock.inputBufferAdcTime if clock.inputBufferAdcTime else 0
        end = time.monotonic() - max(0, elapsed - frames / RATE)
        try:
            queue.put_nowait((bytes(data), end))
        except Full:
            lost.set()

    def record(self, device, duration, ticket, *, consume=None, check=None):
        import sounddevice as sd

        if not 0 < duration <= 30:
            raise ValueError("Recording duration must be between 0 and 30 seconds")
        trace = ticket.trace
        trace.mark("recording_started")
        queue, lost = Queue(maxsize=1000 // self.frame_ms), Event()
        error = True
        try:
            detect = self.vad.detector() if self.vad else None
            if self._stream is None or self._device != device:
                self.close()
                self._stream = sd.RawInputStream(
                    samplerate=RATE, channels=1, dtype="int16", blocksize=RATE * self.frame_ms // 1000,
                    callback=self._callback, device=device)
                self._device = device
            self._mailbox = queue, lost
            self._stream.start()

            peak = 0
            loudest_rms = 0.0

            def next_frame():
                nonlocal peak, loudest_rms
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    if ticket.cancel.is_set():
                        raise CaptureError("Recording cancelled")
                    if check:
                        check()
                    if lost.is_set():
                        raise CaptureError("Microphone audio was lost")
                    try:
                        data, end = queue.get(timeout=.05)
                    except Empty:
                        continue
                    if len(data) != RATE * self.frame_ms // 500:
                        raise CaptureError("Unexpected microphone frame size")
                    if detect:
                        samples = array("h", data)
                        peak = max(peak, max(map(abs, samples)))
                        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
                        loudest_rms = max(loudest_rms, rms)
                    return data, end
                raise CaptureError("Microphone stopped delivering audio")

            captured = []
            deliver = consume or captured.append
            if detect:
                try:
                    self._utterance(next_frame, deliver, detect, trace)
                except NoSpeech as exc:
                    device_name = "default" if device is None else device
                    raise NoSpeech(
                        f"{exc}; input={device_name}, VAD_MODE={self.vad.mode}, "
                        f"PCM16 peak={peak}/32768, loudest frame RMS={loudest_rms:.0f}/32768"
                    ) from None
            else:
                remaining = round(duration * RATE) * 2
                while remaining:
                    data, _ = next_frame()
                    block = data[:remaining]
                    deliver(block)
                    remaining -= len(block)
            if lost.is_set() or ticket.cancel.is_set():
                raise CaptureError("Recording cancelled or microphone audio was lost")
            error = False
            return pcm_wav(b"".join(captured)) if consume is None else None
        finally:
            trace.mark("recording_ended")
            self._mailbox = None
            if self._stream is not None:
                try:
                    self._stream.abort()
                except Exception:
                    self.close()
                    raise
                if error or not self.reuse:
                    self.close()

    def _utterance(self, next_frame, deliver, detect, trace):
        cfg = self.vad
        onset_window = 5
        recent_voice = deque(maxlen=3)
        pre_roll = deque(maxlen=max(onset_window, math.ceil(cfg.pre_roll_ms / cfg.frame_ms)))
        wait_frames = math.ceil(cfg.wait_timeout * 1000 / cfg.frame_ms)
        min_voiced_frames = math.ceil(cfg.min_speech_ms / cfg.frame_ms)
        max_frames = int(cfg.max_speech_seconds * 1000 / cfg.frame_ms)

        # Hold a tentative utterance locally until it contains enough voiced audio.
        # Discard a short candidate that ends before confirmation and keep waiting.
        pending = []
        total_voiced = best_candidate = 0
        for frame_number in range(1, wait_frames + 1):
            data, end = next_frame()
            pre_roll.append(data)
            speaking = detect(data)
            total_voiced += int(speaking)
            if not pending:
                if not speaking:
                    continue
                recent_voice.append((frame_number, end))
                first_frame, first_end = recent_voice[0]
                if len(recent_voice) < recent_voice.maxlen or frame_number - first_frame >= onset_window:
                    continue
                pending.extend(pre_roll)
                first_voice = first_end - cfg.frame_ms / 1000
                voiced_frames = len(recent_voice)
                silence_ms = 0
            else:
                pending.append(data)
                voiced_frames += int(speaking)
                silence_ms = 0 if speaking else silence_ms + cfg.frame_ms

            best_candidate = max(best_candidate, voiced_frames)
            if voiced_frames >= min_voiced_frames:
                break
            if silence_ms >= cfg.silence_ms or frame_number - first_frame + 1 >= max_frames:
                trace.increment("vad_ignored_short_onset")
                pending.clear()
                recent_voice.clear()
        else:
            trace.increment("vad_wait_voiced_ms", total_voiced * cfg.frame_ms)
            trace.increment("vad_best_candidate_ms", best_candidate * cfg.frame_ms)
            raise NoSpeech(
                f"No speech detected before the microphone timeout "
                f"(best candidate {best_candidate * cfg.frame_ms} ms voiced; "
                f"minimum {cfg.min_speech_ms} ms)"
            )

        trace.mark("speech_started", first_voice)
        for frame in pending:
            deliver(frame)
        last_voice = end
        for _ in range(frame_number - first_frame + 1, max_frames):
            data, end = next_frame()
            deliver(data)
            if detect(data):
                last_voice = end
                silence_ms = 0
            else:
                silence_ms += cfg.frame_ms
                if silence_ms >= cfg.silence_ms:
                    trace.mark("speech_ended", last_voice)
                    trace.mark("endpoint_detected")
                    return
        raise CaptureError("Speech exceeded the recording limit before a complete endpoint")

    def close(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
