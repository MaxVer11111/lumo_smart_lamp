"""One isolated Qwen-ASR-Realtime session per captured utterance."""

import asyncio
import base64
import logging
import os
from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event, Thread

from .capture import RATE
from .realtime import (RealtimeError, cancellable, connection, endpoint, receive, send,
                       timeout_from_env, wait_for_event)
from ..ai_config import redact_secrets
from ..demo_speech import ASRServiceError

logger = logging.getLogger(__name__)
TRANSCRIPTION = "conversation.item.input_audio_transcription."


class RealtimeASR:
    def __init__(self, url, key, *, language="en", chunk_ms=40, connect_timeout=10, final_timeout=5):
        if not key:
            raise ValueError("Realtime ASR requires DASHSCOPE_API_KEY")
        if not 10 <= chunk_ms <= 200:
            raise ValueError("ASR_CHUNK_MS must be between 10 and 200")
        self.url, self.key, self.language = url, key, language
        self.chunk_bytes = chunk_ms * RATE * 2 // 1000
        self.connect_timeout, self.final_timeout = connect_timeout, final_timeout

    @classmethod
    def from_env(cls):
        return cls(
            endpoint(os.getenv("ASR_REALTIME_URL", "").strip(), os.getenv("ASR_BASE_URL", ""),
                     os.getenv("ASR_REALTIME_MODEL", "qwen3-asr-flash-realtime")),
            os.getenv("DASHSCOPE_API_KEY", "").strip(),
            language=os.getenv("ASR_LANGUAGE", "en").strip(),
            chunk_ms=int(os.getenv("ASR_CHUNK_MS", "40")),
            connect_timeout=timeout_from_env("ASR_CONNECT_TIMEOUT", 10),
            final_timeout=timeout_from_env("ASR_FINAL_TIMEOUT", 5),
        )

    @contextmanager
    def session(self, ticket, frame_ms):
        capture = _Utterance(self, ticket, frame_ms)
        capture.worker.start()
        try:
            yield capture
        finally:
            capture.cancel.set()
            capture.worker.join()


class _Utterance:
    def __init__(self, provider, ticket, frame_ms):
        self.provider, self.ticket = provider, ticket
        self.frames = Queue(maxsize=12000 // frame_ms)
        self.sealed, self.done, self.cancel = Event(), Event(), Event()
        self.commit_requested = self.finish_requested = False
        self.error = self.transcript = None
        self.worker = Thread(target=self._run, name="lumo-asr")

    def _cancelled(self):
        return self.cancel.is_set() or self.ticket.cancel.is_set()

    def check(self):
        if self._cancelled():
            raise ASRServiceError("Realtime ASR cancelled")
        if self.done.is_set() and self.error:
            raise ASRServiceError(self.error)

    def append(self, pcm):
        self.check()
        try:
            self.frames.put_nowait(pcm)
        except Full:
            raise ASRServiceError("Realtime ASR could not keep up with microphone capture") from None

    def final(self):
        self.check()
        self.sealed.set()
        if not self.done.wait(self.provider.final_timeout):
            raise ASRServiceError("Realtime ASR final transcription timed out")
        self.check()
        return self.transcript

    def _run(self):
        try:
            self.transcript = asyncio.run(cancellable(self._recognize(), self._cancelled))
        except (RealtimeError, TimeoutError) as exc:
            self.error = str(exc) or "Realtime ASR session timed out"
        except Exception:
            self.error = "Realtime ASR session failed"
        finally:
            self.done.set()

    async def _recognize(self):
        provider = self.provider
        self.ticket.trace.mark("asr_requested")
        async with connection(provider.url, provider.key, provider.connect_timeout, retry=True) as ws:
            session = {"modalities": ["text"], "input_audio_format": "pcm",
                       "sample_rate": RATE, "turn_detection": None}
            if provider.language:
                session["input_audio_transcription"] = {"language": provider.language}
            async with asyncio.timeout(provider.connect_timeout):
                await send(ws, "session.update", session=session)
                await wait_for_event(ws, "session.updated")

            sender = asyncio.create_task(self._upload(ws))
            receiver = asyncio.create_task(self._read(ws))
            try:
                done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    sender.result()
                    return await receiver
                transcript = receiver.result()
                await sender
                return transcript
            finally:
                for task in (sender, receiver):
                    task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _upload(self, ws):
        chunk = bytearray()
        size = self.provider.chunk_bytes
        while True:
            try:
                chunk.extend(self.frames.get_nowait())
            except Empty:
                if self.sealed.is_set() and self.frames.empty():
                    break
                await asyncio.sleep(.01)
                continue
            while len(chunk) >= size:
                await send(ws, "input_audio_buffer.append", audio=base64.b64encode(chunk[:size]).decode("ascii"))
                del chunk[:size]
        if chunk:
            await send(ws, "input_audio_buffer.append", audio=base64.b64encode(chunk).decode("ascii"))
        self.commit_requested = True
        self.ticket.trace.mark("asr_commit_requested")
        await send(ws, "input_audio_buffer.commit")
        self.finish_requested = True
        await send(ws, "session.finish")

    async def _read(self, ws):
        item_id = transcript = None
        while True:
            event = await receive(ws)
            kind = event.get("type")
            if kind == TRANSCRIPTION + "text":
                if event.get("text") or event.get("stash"):
                    self.ticket.trace.mark("asr_partial")
                    if logger.isEnabledFor(logging.DEBUG):
                        preview = str(event.get("text", "")) + str(event.get("stash", ""))
                        logger.debug("interaction=%s partial=%r", self.ticket.id, redact_secrets(preview))
            elif kind in ("input_audio_buffer.committed", TRANSCRIPTION + "completed"):
                if not self.commit_requested:
                    raise RealtimeError("Realtime ASR returned a commit or final result before capture was submitted")
                # The WebSocket carries exactly one client commit. Some responses
                # do not identify the item, so compare IDs whenever present.
                received_id = event.get("item_id")
                if received_id is not None and not isinstance(received_id, str):
                    raise RealtimeError("Realtime ASR returned an invalid item ID")
                if received_id and item_id and received_id != item_id:
                    raise RealtimeError("Realtime ASR returned conflicting item IDs for one capture")
                item_id = received_id or item_id

                if kind == "input_audio_buffer.committed":
                    self.ticket.trace.mark("asr_committed")
                    if not received_id:
                        self.ticket.trace.increment("asr_commit_without_item_id")
                    continue

                final = event.get("transcript")
                if not isinstance(final, str) or not final.strip():
                    raise RealtimeError("Realtime ASR returned an empty or invalid final transcript")
                if transcript is not None and transcript != final:
                    raise RealtimeError("Realtime ASR returned conflicting final transcripts for one capture")
                transcript = final
                self.ticket.trace.mark("asr_final")
            elif kind == TRANSCRIPTION + "failed":
                raise RealtimeError("DashScope could not recognize the utterance")
            elif kind == "session.finished":
                if not self.finish_requested or transcript is None:
                    raise RealtimeError("Realtime ASR finished without a final transcription for the submitted capture")
                self.ticket.trace.mark("asr_session_finished")
                return transcript
