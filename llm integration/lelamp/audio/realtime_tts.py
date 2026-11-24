"""Qwen-TTS-Realtime synthesis with bounded, cancellable PCM playback."""

import asyncio
import base64
import logging
import os
import time
from queue import Empty, Full, Queue
from threading import Event

from .realtime import (RealtimeCancelled, RealtimeError, cancellable, connection, endpoint, receive,
                       send, timeout_from_env, wait_for_event)
from .tts import SAMPLE_RATE
from ..behavior.performance import current_trace

logger = logging.getLogger(__name__)
BLOCK_MS = 20
BLOCK_FRAMES = SAMPLE_RATE * BLOCK_MS // 1000
BLOCK_BYTES = BLOCK_FRAMES * 2
SILENCE = bytes(BLOCK_BYTES)


class _Playback:
    """PortAudio consumes complete blocks without waiting for the network."""

    def __init__(self, sd, cancel, initial_blocks, capacity):
        self.sd, self.cancel = sd, cancel
        self.blocks = Queue(capacity)
        self.initial = initial_blocks
        self.eof, self.drained = Event(), Event()
        self.first_output = None
        self.rebuffering = False
        self.underruns = 0
        self.device_underflows = 0
        self.callback_failed = False

    def callback(self, outdata, frames, clock, status):
        if self.cancel.is_set():
            raise self.sd.CallbackAbort
        if frames != BLOCK_FRAMES:
            self.callback_failed = True
            raise self.sd.CallbackAbort
        if status.output_underflow:
            self.device_underflows += 1
        if self.rebuffering and self.blocks.qsize() < self.initial and not self.eof.is_set():
            outdata[:] = SILENCE
            return
        self.rebuffering = False
        try:
            block = self.blocks.get_nowait()
        except Empty:
            if self.eof.is_set():
                raise self.sd.CallbackStop
            self.underruns += 1
            self.rebuffering = True
            outdata[:] = SILENCE
            return
        outdata[:] = block
        if self.first_output is None:
            self.first_output = time.monotonic()


class RealtimeTTS:
    mode = "speaker"

    def __init__(self, url, key, voice, device, *, initial_ms=100, max_ms=2000,
                 connect_timeout=10, timeout=30, reuse=False):
        if not key or not voice:
            raise ValueError("Set TTS_STREAMING_VOICE and TTS_API_KEY or DASHSCOPE_API_KEY")
        if not BLOCK_MS <= initial_ms <= max_ms <= 10000:
            raise ValueError("Invalid TTS initial or maximum buffer length")
        self.url, self.key, self.voice, self.device = url, key, voice, device
        self.capacity = max_ms // BLOCK_MS
        self.initial_blocks = min(self.capacity, (initial_ms + BLOCK_MS - 1) // BLOCK_MS)
        self.connect_timeout, self.timeout, self.reuse = connect_timeout, timeout, reuse
        self._stream = self._playback = None

    @classmethod
    def from_env(cls, *, reuse=False):
        if os.getenv("TTS_STREAMING_PROVIDER", "qwen_realtime").strip().lower() != "qwen_realtime":
            raise ValueError("TTS_STREAMING_PROVIDER must be qwen_realtime")
        device = os.getenv("TTS_DEVICE", "").strip()
        return cls(
            endpoint(os.getenv("TTS_STREAMING_URL", "").strip(),
                     os.getenv("TTS_BASE_URL", "") or os.getenv("ASR_BASE_URL", ""),
                     os.getenv("TTS_STREAMING_MODEL", "qwen3-tts-flash-realtime")),
            os.getenv("TTS_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip(),
            os.getenv("TTS_STREAMING_VOICE", "").strip(),
            int(device) if device else None,
            initial_ms=int(os.getenv("TTS_INITIAL_BUFFER_MS", "100")),
            max_ms=int(os.getenv("TTS_MAX_BUFFER_MS", "2000")),
            connect_timeout=timeout_from_env("TTS_CONNECT_TIMEOUT", 10),
            timeout=timeout_from_env("TTS_STREAMING_TIMEOUT", 30),
            reuse=reuse,
        )

    def speak(self, text, cancel, interaction_id):
        if cancel.is_set():
            return False
        import sounddevice as sd

        playback = _Playback(sd, cancel, self.initial_blocks, self.capacity)
        trace = current_trace.get()
        self._playback = playback
        completed = False
        try:
            asyncio.run(cancellable(self._stream_speech(text, playback, trace), cancel.is_set))
            completed = True
            return True
        except RealtimeCancelled:
            return False
        except RealtimeError as exc:
            logger.warning("interaction=%s streaming TTS failed: %s", interaction_id, exc)
            raise
        except TimeoutError:
            logger.warning("interaction=%s streaming TTS timed out", interaction_id)
            raise RealtimeError("Streaming TTS timed out; reply is available in the terminal") from None
        finally:
            try:
                if self._stream is not None and not completed:
                    self._stream.abort()
            finally:
                try:
                    if not completed or not self.reuse:
                        self.close()
                finally:
                    self._playback = None
                    if trace and playback.first_output is not None:
                        trace.mark("tts_first_output", playback.first_output)
                    if trace and playback.underruns:
                        trace.increment("tts_rebuffers", playback.underruns)
                    if trace and playback.device_underflows:
                        trace.increment("tts_device_underruns", playback.device_underflows)
                    if playback.underruns or playback.device_underflows:
                        logger.info("interaction=%s tts_rebuffers=%s device_underruns=%s",
                                    interaction_id, playback.underruns, playback.device_underflows)

    def _start(self, playback):
        if playback.cancel.is_set():
            raise RealtimeCancelled("Streaming playback cancelled")
        if self._stream is None:
            self._stream = playback.sd.RawOutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=self.device,
                blocksize=BLOCK_FRAMES, latency="low", callback=self._callback, finished_callback=self._finished)
        self._stream.start()

    def _callback(self, *args):
        self._playback.callback(*args)

    def _finished(self):
        self._playback.drained.set()

    async def _stream_speech(self, text, playback, trace):
        started = False

        def start_if_ready(force=False):
            nonlocal started
            if not started and (force or playback.blocks.qsize() >= playback.initial):
                self._start(playback)
                started = True

        async def enqueue(block):
            async with asyncio.timeout(self.timeout):
                while True:
                    if playback.callback_failed or playback.drained.is_set():
                        raise RealtimeError("Speaker stopped before receiving all streaming audio")
                    try:
                        playback.blocks.put_nowait(block)
                    except Full:
                        await asyncio.sleep(.005)
                    else:
                        start_if_ready()
                        return

        if trace:
            trace.mark("tts_requested")
        async with connection(self.url, self.key, self.connect_timeout) as ws:
            async with asyncio.timeout(self.timeout):
                await send(ws, "session.update", session={
                    "voice": self.voice, "mode": "commit", "response_format": "pcm",
                    "sample_rate": SAMPLE_RATE})
                await wait_for_event(ws, "session.updated")
                await send(ws, "input_text_buffer.append", text=text)
                await send(ws, "input_text_buffer.commit")
            tail = b""
            pcm_received = False
            while True:
                async with asyncio.timeout(self.timeout):
                    event = await receive(ws)
                kind = event.get("type")
                if kind == "response.audio.delta":
                    try:
                        pcm = base64.b64decode(event["delta"], validate=True)
                    except (KeyError, ValueError, TypeError):
                        raise RealtimeError("DashScope returned malformed PCM data") from None
                    if len(pcm) % 2:
                        raise RealtimeError("DashScope returned a partial PCM16 sample")
                    if not pcm:
                        continue
                    if not pcm_received and trace:
                        trace.mark("tts_first_pcm")
                    pcm_received = True
                    data = tail + pcm
                    end = len(data) // BLOCK_BYTES * BLOCK_BYTES
                    for offset in range(0, end, BLOCK_BYTES):
                        await enqueue(data[offset:offset + BLOCK_BYTES])
                    tail = data[end:]
                elif kind == "response.done":
                    response = event.get("response")
                    if not pcm_received or not isinstance(response, dict) or response.get("status") != "completed":
                        raise RealtimeError("DashScope did not complete speech synthesis")
                    break
                elif kind == "session.finished":
                    raise RealtimeError("DashScope ended before completing speech synthesis")

            if tail:
                await enqueue(tail.ljust(BLOCK_BYTES, b"\x00"))
            playback.eof.set()
            start_if_ready(force=True)
            async with asyncio.timeout(self.timeout):
                await send(ws, "session.finish")
                await wait_for_event(ws, "session.finished")

        # Drain is bounded by the queue and the audio device, independent of total speech length.
        async with asyncio.timeout(self.timeout + self.capacity * BLOCK_MS / 1000):
            while not playback.drained.is_set():
                await asyncio.sleep(.01)
        if playback.callback_failed or playback.first_output is None:
            raise RealtimeError("Speaker failed to play streaming audio")
        # PortAudio requires stop() after a callback-completed stream before restarting it.
        self._stream.stop()
        if trace:
            trace.mark("tts_playback_completed")

    def close(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
