import asyncio
import base64
import json
import tempfile
import threading
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lelamp.ai_runtime import AIRuntime
from lelamp.audio.capture import CaptureError, MicrophoneCapture, NoSpeech, VADSettings, pcm_wav
from lelamp.audio.realtime_asr import RealtimeASR, TRANSCRIPTION
from lelamp.audio.realtime_tts import BLOCK_BYTES, BLOCK_FRAMES, RealtimeTTS
from lelamp.behavior.capabilities import CapabilityRegistry
from lelamp.behavior.dispatcher import CommandDispatcher
from lelamp.behavior.manager import BehaviorManager, Interaction
from lelamp.behavior.motion import MockMotion, RecordedMotion
from lelamp.behavior.performance import Benchmark, Trace, current_trace, summarize_groups
from lelamp.demo_motion import JOINTS, DemoPlayer
from lelamp.demo_recordings import AI_MOTION_DIR, RecordingCache
from test_demo import Calibration
from test_llm_integration import raw, verified_profile


class VADTests(unittest.TestCase):
    def frames(self, values, cfg=None, *, delivered=None):
        capture = MicrophoneCapture(cfg or VADSettings())
        trace = Trace()
        if delivered is None:
            delivered = []
        frames = iter(((value, index), trace.started + (index + 1) * .02)
                      for index, value in enumerate(values))
        capture._utterance(lambda: next(frames), delivered.append, lambda frame: bool(frame[0]), trace)
        return trace.snapshot(), delivered

    def test_preroll_and_mid_sentence_pause_preserve_both_clauses(self):
        values = [0] * 20 + [1] * 12 + [0] * 30 + [1] * 7 + [0] * 40
        trace, data = self.frames(values)
        self.assertEqual([frame[1] for frame in data], list(range(8, len(values))))
        self.assertEqual(trace["events_ms"]["speech_started"], 400)
        self.assertEqual(trace["events_ms"]["speech_ended"], 1380)

    def test_fragmented_onset_can_confirm_speech_without_three_consecutive_frames(self):
        values = [0] * 10 + [1, 0] * 10 + [0] * 40
        trace, data = self.frames(values)
        self.assertEqual(trace["events_ms"]["speech_started"], 200)
        self.assertEqual(trace["events_ms"]["speech_ended"], 580)
        self.assertEqual(data[10], (1, 10))

    def test_short_startup_noise_does_not_end_recording_or_reach_asr(self):
        # The reported 80 ms trigger is followed by silence and deliberate user speech.
        values = [0] * 4 + [1] * 4 + [0] * 65 + [1] * 14 + [0] * 40
        for silence_ms in (400, 800):
            with self.subTest(silence_ms=silence_ms):
                trace, data = self.frames(values, VADSettings(silence_ms=silence_ms))
                self.assertEqual(data[0][1], 61)
                self.assertEqual(trace["events_ms"]["speech_started"], 1460)
                self.assertEqual(trace["counts"]["vad_ignored_short_onset"], 1)

    def test_noise_only_uses_original_wait_budget_and_uploads_nothing(self):
        # Two discarded triggers must not keep extending the wait deadline.
        values = ([1] * 4 + [0] * 46) * 2
        delivered = []
        with self.assertRaisesRegex(NoSpeech, "best candidate 80 ms voiced; minimum 200 ms"):
            self.frames(values, VADSettings(wait_timeout=2), delivered=delivered)
        self.assertEqual(delivered, [])

    def test_unfinished_long_speech_fails_closed(self):
        with self.assertRaises(CaptureError):
            self.frames([1] * 60, VADSettings(max_speech_seconds=1.2))

    def test_timeout_reports_device_signal_level_and_vad_threshold(self):
        stream = Mock()
        callback = None

        def open_stream(**kwargs):
            nonlocal callback
            callback = kwargs["callback"]
            return stream

        def feed_quiet_frames():
            for _ in range(5):
                callback(b"\x02\x00" * 320, 320, SimpleNamespace(inputBufferAdcTime=0),
                         SimpleNamespace(input_overflow=False))

        stream.start.side_effect = feed_quiet_frames
        device = Mock(RawInputStream=Mock(side_effect=open_stream))
        capture = MicrophoneCapture(VADSettings(wait_timeout=.1, mode=3))
        ticket = Interaction("quiet", float("inf"))
        with patch.dict("sys.modules", sounddevice=device), \
             patch.object(VADSettings, "detector", return_value=lambda frame: False):
            with self.assertRaisesRegex(NoSpeech, r"input=4, VAD_MODE=3, PCM16 peak=2/32768"):
                capture.record(4, 4, ticket)
        self.assertEqual(ticket.trace.snapshot()["counts"]["vad_wait_voiced_ms"], 0)
        stream.close.assert_called_once()

    def test_stopped_input_handle_can_be_reused_without_reading_stale_frames(self):
        marker = 0
        stream = Mock()

        def start():
            nonlocal marker
            marker += 1
            callback(bytes((marker, 0)) * 320, 320, SimpleNamespace(inputBufferAdcTime=0),
                     SimpleNamespace(input_overflow=False))

        def input_stream(**kwargs):
            nonlocal callback
            callback = kwargs["callback"]
            return stream

        callback = None
        stream.start.side_effect = start
        device = Mock(RawInputStream=Mock(side_effect=input_stream))
        capture = MicrophoneCapture(reuse=True)
        self.addCleanup(capture.close)
        with patch.dict("sys.modules", sounddevice=device):
            for sample in (1, 2):
                ticket = Interaction(str(sample), float("inf"))
                audio = capture.record(0, .02, ticket)
                self.assertEqual(audio, pcm_wav(bytes((sample, 0)) * 320))
        device.RawInputStream.assert_called_once()
        self.assertEqual(stream.abort.call_count, 2)
        stream.close.assert_not_called()



class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.events = asyncio.Queue()
        self.sent = []

    async def send(self, message):
        event = json.loads(message)
        self.sent.append(event)
        if event["type"] == "session.update":
            self.push("session.updated")
        await self.handler(self, event)

    async def recv(self):
        message = await self.events.get()
        if isinstance(message, Exception):
            raise message
        return json.dumps(message)

    def push(self, kind, **fields):
        self.events.put_nowait({"type": kind, **fields})


class ASRTests(unittest.TestCase):
    def setUp(self):
        self.capture = Mock(frame_ms=20, vad=True)
        self.llm = Mock()
        self.llm.decide.return_value = raw(speech="")
        self.motion = Mock(spec=MockMotion)
        self.registry = CapabilityRegistry(simulated=True)
        self.manager = BehaviorManager(self.motion)
        self.addCleanup(self.manager.close)
        self.dispatcher = CommandDispatcher(self.registry, self.manager, self.motion)
        self.runtime = AIRuntime(self.llm, self.registry, self.dispatcher, self.manager,
                                 asr=RealtimeASR("wss://fake", "test", final_timeout=.25),
                                 capture=self.capture, streaming_asr=True, output=Mock())
        self.uploaded = threading.Event()
        self.finalized = threading.Event()
        self.sessions = []
        self.session_closed = threading.Event()

    def run_turn(self, scenario):
        async def protocol(ws, event):
            if event["type"] == "input_audio_buffer.append":
                ws.push(TRANSCRIPTION + "text", stash="Nod", item_id="current")
                if scenario == "early-commit":
                    ws.push("input_audio_buffer.committed", item_id="old")
                self.uploaded.set()
            if event["type"] == "input_audio_buffer.commit":
                if scenario in ("ack-without-id", "both-without-id"):
                    ws.push("input_audio_buffer.committed")
                elif scenario != "no-ack":
                    ws.push("input_audio_buffer.committed", item_id="current")
            if event["type"] == "session.finish":
                if scenario == "partial":
                    ws.push("session.finished")
                elif scenario == "timeout":
                    return
                elif scenario == "disconnect":
                    ws.events.put_nowait(ConnectionError("closed"))
                else:
                    item = "old" if scenario == "other-item" else "current"
                    fields = {} if scenario == "both-without-id" else {"item_id": item}
                    final = "  Don't nod, turn on the light. "
                    ws.push(TRANSCRIPTION + "completed", transcript=final, **fields)
                    if scenario == "conflicting-final":
                        final = "Nod."
                    ws.push(TRANSCRIPTION + "completed", transcript=final, **fields)
                    if scenario == "missing-finish":
                        ws.events.put_nowait(ConnectionError("closed"))
                    else:
                        ws.push("session.finished")
                self.finalized.set()

        @asynccontextmanager
        async def connection(*args, **kwargs):
            ws = FakeSession(protocol)
            self.sessions.append(ws)
            try:
                yield ws
            finally:
                self.session_closed.set()

        def record(device, duration, ticket, consume, check):
            consume(bytes(640))
            consume(bytes(640))
            self.assertTrue(self.uploaded.wait(1))
            self.assertFalse(self.finalized.is_set())
            self.llm.decide.assert_not_called()
            if scenario == "early-commit":
                self.assertTrue(self.session_closed.wait(1))
                check()
            if scenario == "no-speech":
                raise NoSpeech("no valid utterance")
            if scenario == "cancel":
                self.manager.local_stop()
                check()

        self.capture.record.side_effect = record
        with patch("lelamp.audio.realtime_asr.connection", connection):
            result = self.runtime.voice(0, 4)
        self.assertTrue(self.session_closed.is_set())
        return result

    def test_upload_overlaps_capture_and_confirmed_final_is_forwarded_once_verbatim(self):
        self.assertEqual(self.run_turn("success").status, "success")
        self.llm.decide.assert_called_once()
        self.assertEqual(self.llm.decide.call_args.args[0], "  Don't nod, turn on the light. ")
        self.assertEqual([event["type"] for event in self.sessions[0].sent],
                         ["session.update", "input_audio_buffer.append",
                          "input_audio_buffer.commit", "session.finish"])
        chunk = self.sessions[0].sent[1]
        self.assertEqual(len(base64.b64decode(chunk["audio"])), 1280)

    def test_missing_ack_or_item_id_still_requires_session_final_and_submits_once(self):
        for scenario in ("ack-without-id", "both-without-id", "no-ack"):
            with self.subTest(scenario=scenario):
                self.llm.decide.reset_mock()
                self.uploaded.clear()
                self.finalized.clear()
                self.session_closed.clear()
                self.assertEqual(self.run_turn(scenario).status, "success")
                self.llm.decide.assert_called_once()
                self.assertEqual(self.llm.decide.call_args.args[0], "  Don't nod, turn on the light. ")

    def test_partial_and_late_or_unconfirmed_final_cannot_trigger_a_decision(self):
        invalid = ("partial", "other-item", "conflicting-final", "missing-finish", "disconnect",
                   "timeout", "early-commit", "no-speech", "cancel")
        for scenario in invalid:
            with self.subTest(scenario=scenario):
                self.uploaded.clear()
                self.finalized.clear()
                self.session_closed.clear()
                self.assertEqual(self.run_turn(scenario).status, "rejected")
                self.llm.decide.assert_not_called()
                self.motion.play.assert_not_called()
        self.assertEqual(len(self.sessions), len(invalid))


class FakeSpeaker:
    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    def __init__(self, cancel_after_block=None):
        self.written = []
        self.streams = 0
        self.cancel_after_block = cancel_after_block

    def RawOutputStream(self, *, callback, finished_callback, **settings):
        self.streams += 1
        event = threading.Event()
        worker = None

        def run():
            try:
                while not event.is_set():
                    data = bytearray(BLOCK_BYTES)
                    callback(data, BLOCK_FRAMES, None, SimpleNamespace(output_underflow=False))
                    self.written.append(bytes(data))
                    if self.cancel_after_block and any(data):
                        self.cancel_after_block.set()
                    time.sleep(.002)
            except (self.CallbackStop, self.CallbackAbort):
                pass
            finally:
                finished_callback()

        def start():
            nonlocal worker
            event.clear()
            worker = threading.Thread(target=run)
            worker.start()

        def abort():
            event.set()
            if worker is not None:
                worker.join()

        return SimpleNamespace(start=start, stop=abort, abort=abort, close=abort)


class TTSTests(unittest.TestCase):
    def connection(self, audio, speaker, *, scenario="success"):
        async def protocol(ws, event):
            if event["type"] == "input_text_buffer.commit":
                ws.push("response.audio.delta", delta=base64.b64encode(audio).decode())
            elif event["type"] == "session.finish":
                ws.push("session.finished")

        class Session(FakeSession):
            async def recv(self):
                if not self.events.empty():
                    return await super().recv()
                deadline = asyncio.get_running_loop().time() + .75
                while not any(any(block) for block in speaker.written):
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("speaker did not begin before final event")
                    await asyncio.sleep(.002)
                if scenario == "cancel":
                    await asyncio.Future()
                if scenario == "disconnect":
                    raise ConnectionError("closed")
                return json.dumps({"type": "response.done", "response": {"status": "completed"}})

        @asynccontextmanager
        async def connection(*args):
            yield Session(protocol)

        return connection

    def test_bounded_pcm_begins_playback_before_final_and_reuses_device(self):
        speaker = FakeSpeaker()
        audio = b"\x16\x00" * BLOCK_FRAMES * 8
        tts = RealtimeTTS("wss://fake", "test", "voice", 0, initial_ms=40, max_ms=60, reuse=True)
        self.addCleanup(tts.close)
        context = current_trace.set(Trace())
        self.addCleanup(current_trace.reset, context)
        with patch.dict("sys.modules", sounddevice=speaker), \
             patch("lelamp.audio.realtime_tts.connection", self.connection(audio, speaker)):
            for _ in range(2):
                self.assertTrue(tts.speak("Speak", threading.Event(), "id"))
        self.assertEqual(speaker.streams, 1)
        self.assertEqual(b"".join(block for block in speaker.written if any(block)), audio * 2)
        metrics = current_trace.get().snapshot()["metrics"]
        self.assertIsNotNone(metrics["tts_ttfa_ms"])

    def test_cancels_playback_and_next_task_does_not_play_queued_pcm_from_previous_task(self):
        cancel = threading.Event()
        speaker = FakeSpeaker(cancel)
        audio = b"\x17\x00" * BLOCK_FRAMES * 4
        tts = RealtimeTTS("wss://fake", "test", "voice", None, initial_ms=20, max_ms=100)
        with patch.dict("sys.modules", sounddevice=speaker), \
             patch("lelamp.audio.realtime_tts.connection", self.connection(audio, speaker, scenario="cancel")):
            self.assertFalse(tts.speak("Cancel", cancel, "old"))
        self.assertEqual(sum(any(block) for block in speaker.written), 1)
        speaker.written.clear()
        speaker.cancel_after_block = None
        fresh = b"\x02\x00" * BLOCK_FRAMES * 3
        with patch.dict("sys.modules", sounddevice=speaker), \
             patch("lelamp.audio.realtime_tts.connection", self.connection(fresh, speaker)):
            self.assertTrue(tts.speak("New", threading.Event(), "new"))
        self.assertEqual(b"".join(block for block in speaker.written if any(block)), fresh)

    def test_midstream_disconnect_does_not_replay_audio(self):
        speaker = FakeSpeaker()
        audio = b"\x16\x00" * BLOCK_FRAMES * 8
        tts = RealtimeTTS("wss://fake", "test", "voice", None, initial_ms=20, max_ms=40)
        with patch.dict("sys.modules", sounddevice=speaker), \
             patch("lelamp.audio.realtime_tts.connection", self.connection(audio, speaker, scenario="disconnect")):
            with self.assertRaises(ConnectionError):
                tts.speak("Disconnect", threading.Event(), "id")
        self.assertEqual(speaker.streams, 1)
        self.assertLessEqual(len(b"".join(block for block in speaker.written if any(block))), len(audio))


class CacheAndBenchmarkTests(unittest.TestCase):
    def test_cache_warm_play_reads_pose_and_rejects_changed_csv_or_authorization(self):
        profile = verified_profile()
        cache = RecordingCache()
        robot = SimpleNamespace(id="test", calibration={"test": Calibration()}, is_calibrated=True,
                                get_observation=Mock(return_value=dict.fromkeys(JOINTS, 2)), send_action=Mock())
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            recording = folder / "nod.csv"
            recording.write_bytes((AI_MOTION_DIR / "nod.csv").read_bytes())
            with patch("lelamp.behavior.capabilities.AI_MOTION_DIR", folder), \
                 patch("lelamp.behavior.motion.AI_MOTION_DIR", folder):
                registry = CapabilityRegistry(simulated=False, motion=True, profile=profile, recording_cache=cache)
                player = DemoPlayer(robot, profile, recording_cache=cache)
                motion = RecordedMotion(player, .25)
                token = threading.Event()
                robot.send_action.side_effect = lambda pose: token.set()
                with patch("lelamp.demo_recordings._adapt_csv") as adapt:
                    self.assertFalse(motion.play(registry.motions["nod"], token))
                    adapt.assert_not_called()
                robot.send_action.assert_called_once_with(dict.fromkeys(JOINTS, 2))
                recording.write_bytes(recording.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "verified file"):
                    motion.play(registry.motions["nod"], threading.Event())
                profile.data["ai_motion"]["actions"].clear()
                with self.assertRaisesRegex(RuntimeError, "authorized"):
                    motion.play(registry.motions["nod"], threading.Event())
        self.assertEqual(robot.send_action.call_count, 1)

    def test_benchmark_marks_failed_and_successful_turns_and_serializes_no_speech(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "perf.jsonl"
            benchmark = Benchmark("mock", path)
            llm, motion = Mock(), MockMotion()
            llm.decide.side_effect = [raw(speech="text only"), "invalid"]
            registry = CapabilityRegistry(simulated=True)
            manager = BehaviorManager(motion)
            self.addCleanup(manager.close)
            runtime = AIRuntime(llm, registry, CommandDispatcher(registry, manager, motion),
                                manager, benchmark=benchmark, output=Mock())
            runtime.text("secret input")
            runtime.text("bad")
            benchmark.report()
            payload = path.read_text()
        self.assertNotIn("secret input", payload)
        self.assertNotIn("text only", payload)
        rows = [json.loads(line) for line in payload.splitlines()]
        report = summarize_groups(rows)["mock"]
        self.assertEqual(report["cold"]["failures"], 0)
        self.assertEqual(report["warm"]["failure_rate"], 1)
        self.assertIn("llm_total_ms", report["cold"]["metrics"])
        self.assertIn("validation_finished", rows[1]["events_ms"])
        self.assertIsNone(rows[0]["metrics"]["end_to_end_ms"])
