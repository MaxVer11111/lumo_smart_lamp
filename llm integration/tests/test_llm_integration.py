import io
import json
import tempfile
import threading
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import httpx
from openai import APIConnectionError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from lelamp.ai_runtime import AIRuntime, _disconnect_robot
from lelamp.audio.tts import DashScopeTTS, MockTTS, OpenAITTS, tts_from_env
from lelamp.behavior.capabilities import CapabilityRegistry, profile_fingerprint, recording_fingerprint
from lelamp.behavior.dispatcher import CommandDispatcher
from lelamp.behavior.manager import BehaviorManager
from lelamp.behavior.motion import MockMotion, RecordedMotion
from lelamp.demo import main
from lelamp.demo_motion import JOINTS, DemoPlayer, MotionProfile
from lelamp.demo_recordings import AI_MOTION_DIR
from lelamp.demo_speech import BailianASR
from lelamp.intelligence.evaluate import actions_match, evaluate, read_cases
from lelamp.intelligence.llm_client import LLMClient, LLMError
from lelamp.intelligence.schemas import BehaviorEvent, InvalidEvent
from lelamp.lighting.controller import LightingController
from test_demo import Calibration, profile_data


def raw(intent="conversation", actions=(), speech="Okay.", emotion="neutral"):
    return json.dumps({"intent": intent, "user_emotion": emotion, "speech": speech, "actions": list(actions)})


def gesture(name="nod"):
    return {"type": "motion", "name": name}


def light(operation="turn_on", **kwargs):
    return {"type": "lighting", "operation": operation, **kwargs}


STOP = raw("stop", [{"type": "stop"}], speech="")


def response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                    finish_reason=finish_reason)])


def verified_profile(scale=0.25):
    data = profile_data()
    data["ai_motion"] = {
        "profile_sha256": profile_fingerprint(MotionProfile(data)),
        "actions": {"nod": {"recording_sha256": recording_fingerprint("nod.csv"), "max_scale": scale}},
    }
    return MotionProfile(data)


class Harness(unittest.TestCase):
    def build(self, motion=None, lighting=None, tts=None, registry=None, **options):
        self.motion = motion if motion is not None else Mock(spec=MockMotion)
        if motion is None:
            self.motion.play.return_value = True
            self.motion.stop.side_effect = lambda token: token.set()
        self.registry = registry or CapabilityRegistry(simulated=True, lighting=lighting)
        self.manager = BehaviorManager(self.motion, lighting, tts, **options)
        self.addCleanup(self.manager.close)
        self.dispatcher = CommandDispatcher(self.registry, self.manager, self.motion, lighting, tts)

    def submit(self, decision):
        _, result = self.dispatcher.dispatch(self.manager.begin_interaction(), decision)
        return result

    def outcomes(self, result):
        return [job.result(timeout=3).status for job in result.jobs]


class SchemaTests(Harness):
    def test_invalid_wire_protocol_never_dispatches(self):
        lighting = LightingController()
        self.build(lighting=lighting)
        invalid = (
            "",
            'unquoted string',
            '[]',
            '{"intent":"conversation","intent":"conversation","user_emotion":"neutral","speech":"","actions":[]}',
            '{}',
            raw("unknown"),
            raw(emotion="delighted"),
            raw(speech=True),
            raw(speech="a" * 1001),
            raw(speech="\ud800"),
            raw("motion_control", [{"type": "motion", "name": "nod", "angle": 47}]),
            raw("motion_control", [{"type": "motor", "id": 2}]),
            raw("motion_control", [gesture("../nod")]),
            raw("motion_control", [gesture("first_test")]),
            raw("motion_control", [gesture("unavailable")]),
            raw("motion_control", [gesture(), gesture("headshake")]),
            raw("motion_control", [gesture(), gesture(), gesture()]),
            raw("motion_control"),
            raw("conversation", [gesture()]),
            raw("combined_control", [gesture()]),
            raw("stop", [{"type": "stop"}, gesture()]),
            raw("no_action", [{"type": "stop"}]),
            raw("light_control", [light("set_brightness")]),
            raw("light_control", [light("set_brightness", value=-1)]),
            raw("light_control", [light("set_brightness", value=101)]),
            raw("light_control", [light("set_brightness", value=True)]),
            raw("light_control", [light("set_brightness", value="42")]),
            raw("light_control", [light("set_brightness", value=float("nan"))]),
            raw("light_control", [light("set_brightness", value=float("inf"))]),
            raw("light_control", [light("set_brightness", value=1)]).replace('"value": 1', '"value": 1e999'),
            raw("light_control", [light("increase_brightness", value=0)]),
            raw("light_control", [light("decrease_brightness", value=-10)]),
            raw("light_control", [light("turn_on", value=50)]),
            raw("light_control", [light("reading_mode", value=None)]),
            raw("light_control", [light("write_gpio")]),
        )
        for decision in invalid:
            with self.subTest(decision=decision):
                self.assertEqual(self.submit(decision).status, "rejected")
        self.motion.play.assert_not_called()
        self.assertEqual(lighting.state(), {"brightness": 65, "on": False})

    def test_unavailable_lighting_rejects_whole_event(self):
        self.build()
        result = self.submit(raw("combined_control", [gesture(), light()]))
        self.assertEqual(result.status, "unsupported")
        self.motion.play.assert_not_called()

    def test_lighting_fade_uses_local_output_and_cancellation(self):
        writer = Mock()
        controller = LightingController(write_output=writer, fade_seconds=0)
        writer.assert_called_once_with(0)
        self.assertTrue(controller.fade_to(40))
        self.assertEqual(controller.state(), {"brightness": 40, "on": True})
        token = threading.Event()
        token.set()
        self.assertFalse(controller.fade_to(0, cancel=token))
        self.assertEqual(writer.call_args_list, [call(0), call(40)])

    def test_permissions_by_intent_and_supported_combinations(self):
        lighting = LightingController()
        self.build(lighting=lighting)
        self.assertEqual(self.submit(raw("emotion_expression", [gesture("scanning")])).status, "unsupported")
        self.assertEqual(self.outcomes(self.submit(raw("combined_control", [gesture(), light()]))),
                         ["success", "success"])
        self.assertTrue(lighting.state()["on"])
        self.assertEqual(self.outcomes(self.submit(raw("combined_control", [light("turn_off"), gesture("headshake")]))),
                         ["success", "success"])
        self.assertFalse(lighting.state()["on"])
        self.assertEqual([call.args[0].name for call in self.motion.play.call_args_list], ["nod", "headshake"])
        for operation, value, expected in (("set_brightness", 50, 50),
                                           ("increase_brightness", 100, 100),
                                           ("decrease_brightness", 100, 0)):
            self.assertEqual(self.outcomes(self.submit(raw("light_control", [light(operation, value=value)]))),
                             ["success"])
            self.assertEqual(lighting.get_brightness(), expected)
        self.assertEqual(self.outcomes(self.submit(raw("light_control", [light("night_mode")]))), ["success"])
        self.assertEqual(lighting.get_brightness(), 12)


class CapabilityTests(Harness):
    def test_real_registry_requires_profile_verified_scale_and_known_content(self):
        with self.assertRaises(ValueError):
            CapabilityRegistry(simulated=False, motion=True, profile=MotionProfile(profile_data()))
        profile = verified_profile()
        registry = CapabilityRegistry(simulated=False, motion=True, profile=profile)
        self.assertEqual([m["name"] for m in registry.prompt_data()["motions"]], ["nod"])
        self.assertTrue(registry.motions["nod"].hardware_verified)
        self.assertFalse(registry.motions["nod"].allow_autonomous)
        profile.data["ai_motion"]["actions"]["nod"]["allow_autonomous"] = True
        registry = CapabilityRegistry(simulated=False, motion=True, profile=profile)
        self.assertTrue(registry.motions["nod"].allow_autonomous)
        self.assertFalse(registry.motions["excited"].enabled)
        self.assertNotIn("first_test", registry.motions)
        self.assertNotIn("recording_sha256", registry.prompt_data()["motions"][0])
        with self.assertRaisesRegex(ValueError, "scale"):
            CapabilityRegistry(simulated=False, motion=True, profile=profile, scale=0.5)
        profile.data["ai_motion"]["actions"]["nod"]["recording_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "verified file"):
            CapabilityRegistry(simulated=False, motion=True, profile=profile)
        profile.data["neutral"][JOINTS[0]] = 1
        with self.assertRaisesRegex(ValueError, "current profile"):
            CapabilityRegistry(simulated=False, motion=True, profile=profile)

    def test_motion_uses_real_profile_and_cancel_token_on_mock_bus(self):
        profile = verified_profile()
        registry = CapabilityRegistry(simulated=False, motion=True, profile=profile)
        robot = SimpleNamespace(id="test", calibration={"test": Calibration()}, is_calibrated=True,
                               get_observation=Mock(return_value=dict.fromkeys(JOINTS, 1)), send_action=Mock())
        player = DemoPlayer(robot, profile)
        motion = RecordedMotion(player, 0.25)
        cancelled = threading.Event()
        player.stop(cancel_event=cancelled)
        self.assertFalse(motion.play(registry.motions["nod"], cancelled))
        robot.get_observation.assert_not_called()
        robot.send_action.assert_not_called()
        token = threading.Event()
        robot.send_action.side_effect = lambda pose: player.stop(cancel_event=token)
        self.assertFalse(motion.play(registry.motions["nod"], token))
        robot.send_action.assert_called_once_with(dict.fromkeys(JOINTS, 1))
        robot.is_calibrated = False
        with self.assertRaises(RuntimeError):
            motion.play(registry.motions["nod"], threading.Event())
        self.assertEqual(robot.send_action.call_count, 1)

    def test_ai_playback_rechecks_the_same_motion_directory_at_execution(self):
        profile = verified_profile()
        robot = SimpleNamespace(id="test", calibration={"test": Calibration()}, is_calibrated=True,
                               get_observation=Mock(return_value=dict.fromkeys(JOINTS, 0)), send_action=Mock())
        motion = RecordedMotion(DemoPlayer(robot, profile), 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            recording = folder / "nod.csv"
            recording.write_bytes((AI_MOTION_DIR / "nod.csv").read_bytes())
            with patch("lelamp.behavior.capabilities.AI_MOTION_DIR", folder), \
                 patch("lelamp.behavior.motion.AI_MOTION_DIR", folder):
                registry = CapabilityRegistry(simulated=False, motion=True, profile=profile)
                capability = registry.motions["nod"]
                self.assertTrue(capability.enabled)
                # The unchanged copy in the legacy library must not be used.
                with recording.open("ab") as output:
                    output.write(b"\n")
                with self.assertRaisesRegex(ValueError, "verified file"):
                    motion.play(capability, threading.Event())
        robot.send_action.assert_not_called()

    def test_unverified_motion_is_unsupported_with_physical_registry(self):
        registry = CapabilityRegistry(simulated=False, motion=True, profile=verified_profile())
        self.build(registry=registry)
        self.assertEqual(self.submit(raw("motion_control", [gesture("excited")])).status, "unsupported")
        self.motion.play.assert_not_called()


class LLMTests(unittest.TestCase):
    def test_messages_are_fresh_and_provider_options_can_be_omitted(self):
        endpoint = Mock()
        endpoint.chat.completions.create.return_value = response(raw())
        llm = LLMClient(client=endpoint, model="model", temperature=None, token_parameter="")
        lighting = LightingController()
        capabilities = CapabilityRegistry(simulated=True, lighting=lighting)
        llm.decide("FIRST", capabilities.prompt_data())
        lighting.set_brightness(87)
        llm.decide("再来一次", capabilities.prompt_data())
        first, second = [request.kwargs for request in endpoint.chat.completions.create.call_args_list]
        for request, text in ((first, "FIRST"), (second, "再来一次")):
            self.assertEqual(set(request), {"messages", "model", "stream"})
            self.assertFalse(request["stream"])
            self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
            self.assertEqual(request["messages"][1]["content"], text)
        prompt = second["messages"][0]["content"]
        self.assertNotIn("FIRST", prompt)
        self.assertIn('"brightness": 87', prompt)
        self.assertIn("Stateless Interaction", prompt)
        self.assertIn("Use emotion_expression with one autonomous motion for ordinary spoken replies", prompt)
        self.assertNotIn("first_test", prompt)
        json_llm = LLMClient(client=endpoint, model="model", response_format="json_object",
                             token_parameter="max_completion_tokens")
        json_llm.decide("Hi", capabilities.prompt_data())
        request = endpoint.chat.completions.create.call_args.kwargs
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", request)

    def test_llm_failures_are_safe_and_unretried(self):
        client = Mock()
        llm = LLMClient(client=client, model="test")
        request = httpx.Request("POST", "https://example.invalid")
        caps = CapabilityRegistry(simulated=True).prompt_data()
        operation = client.chat.completions.create
        for error, message in ((APITimeoutError(request=request), "timed out"),
                               (APIConnectionError(request=request), "connect"),
                               (AuthenticationError("secret marker", response=httpx.Response(401, request=request),
                                                    body={}), "401"),
                               (RateLimitError("secret marker", response=httpx.Response(429, request=request),
                                               body={}), "429")):
            operation.reset_mock()
            operation.side_effect = error
            with self.assertRaisesRegex(LLMError, message) as caught:
                llm.decide("Hi", caps)
            self.assertNotIn("secret marker", str(caught.exception))
            operation.assert_called_once()
        operation.side_effect = None
        for result in (response(" "), response(None), response(raw(), "length"), SimpleNamespace(choices=[])):
            operation.return_value = result
            with self.assertRaises(LLMError):
                llm.decide("Hi", caps)


class SchedulingTests(Harness):
    def test_stop_cancels_running_and_queued_motion(self):
        self.build()
        entered = threading.Event()

        def play(capability, token):
            entered.set()
            return not token.wait(3)

        self.motion.play.side_effect = play
        first = self.submit(raw("motion_control", [gesture()]))
        self.assertTrue(entered.wait(2))
        queued = self.submit(raw("motion_control", [gesture("headshake")]))
        self.assertEqual(self.submit(STOP).status, "success")
        self.assertEqual(self.outcomes(first), ["cancelled"])
        self.assertEqual(self.outcomes(queued), ["cancelled"])
        self.motion.play.assert_called_once()
        self.motion.stop.assert_called_once()

    def test_preemption_waits_for_previous_writer_and_drops_lower_priority_queue(self):
        self.build()
        entered, cancelled, released, next_entered = (threading.Event() for _ in range(4))
        visits = []

        def play(capability, token):
            visits.append(capability.name)
            if capability.name == "excited":
                entered.set()
                if token.wait(3):
                    cancelled.set()
                released.wait(3)
                return False
            next_entered.set()
            return True

        self.motion.play.side_effect = play
        try:
            first = self.submit(raw("emotion_expression", [gesture("excited")]))
            self.assertTrue(entered.wait(2))
            queued = self.submit(raw("emotion_expression", [gesture("curious")]))
            high = self.submit(raw("motion_control", [gesture("headshake")]))
            self.assertTrue(cancelled.wait(2))
            self.assertFalse(next_entered.is_set())
            self.assertEqual(self.outcomes(queued), ["cancelled"])
        finally:
            released.set()
        self.assertEqual(self.outcomes(first), ["cancelled"])
        self.assertEqual(self.outcomes(high), ["success"])
        self.assertEqual(visits, ["excited", "headshake"])

    def test_duplicate_expired_and_late_events_cannot_recover_motion(self):
        clock = Mock(return_value=0)
        self.build(interaction_ttl=2, clock=clock)
        expired = self.manager.begin_interaction()
        clock.return_value = 3
        decision = raw("motion_control", [gesture()])
        self.assertEqual(self.dispatcher.dispatch(expired, decision)[1].status, "rejected")
        pending = self.manager.begin_interaction()
        self.manager.local_stop()
        self.assertFalse(self.manager.capabilities_ready()[0])
        self.assertEqual(self.dispatcher.dispatch(pending, decision)[1].status, "rejected")
        self.assertTrue(self.manager.resume())
        self.assertEqual(self.dispatcher.dispatch(pending, decision)[1].status, "rejected")
        ticket = self.manager.begin_interaction()
        accepted = self.dispatcher.dispatch(ticket, decision)[1]
        self.assertEqual(self.outcomes(accepted), ["success"])
        self.assertEqual(self.dispatcher.dispatch(ticket, decision)[1].status, "rejected")
        self.motion.play.assert_called_once()

    def test_motion_fault_cancels_queue_but_isolates_lighting(self):
        self.build(lighting=LightingController(), queue_size=1)
        entered, fail = threading.Event(), threading.Event()

        def play(capability, token):
            entered.set()
            fail.wait(3)
            raise ConnectionError("bus fault")

        self.motion.play.side_effect = play
        try:
            first = self.submit(raw("motion_control", [gesture()]))
            self.assertTrue(entered.wait(2))
            second = self.submit(raw("motion_control", [gesture("headshake")]))
            full = self.submit(raw("motion_control", [gesture()]))
            self.assertEqual(full.status, "rejected")
        finally:
            fail.set()
        self.assertEqual(self.outcomes(first), ["failed"])
        self.assertEqual(self.outcomes(second), ["cancelled"])
        self.assertEqual(self.manager.state()["motion"], "ERROR")
        self.assertEqual(self.submit(raw("motion_control", [gesture()])).status, "unsupported")
        self.assertEqual(self.outcomes(self.submit(raw("light_control", [light()]))), ["success"])

    def test_motion_and_speaker_parallel_half_duplex_and_local_stop(self):
        tts = Mock(spec=MockTTS)
        self.build(tts=tts)
        playing, speaking = threading.Event(), threading.Event()

        def play(capability, token):
            playing.set()
            return not token.wait(3)

        def speak(text, token, interaction):
            speaking.set()
            return not token.wait(3)

        self.motion.play.side_effect = play
        tts.speak.side_effect = speak
        result = self.submit(raw("motion_control", [gesture()]))
        try:
            self.assertTrue(playing.wait(2))
            self.assertTrue(speaking.wait(2))
            with self.manager.recording() as allowed:
                self.assertFalse(allowed)
        finally:
            self.manager.local_stop()
        self.assertEqual(self.outcomes(result), ["cancelled", "cancelled"])
        self.assertTrue(self.manager.resume())
        with self.manager.recording() as allowed:
            self.assertTrue(allowed)

    def test_motion_with_tts_disabled_accepts_recording(self):
        self.build()
        entered = threading.Event()

        def play(capability, token):
            entered.set()
            return not token.wait(3)

        self.motion.play.side_effect = play
        result = self.submit(raw("motion_control", [gesture()]))
        try:
            self.assertTrue(entered.wait(2))
            with self.manager.recording() as allowed:
                self.assertTrue(allowed)
                self.assertEqual(self.manager.state()["audio"], "LISTENING")
                self.assertEqual(self.manager.state()["motion"], "EXECUTING")
        finally:
            self.manager.local_stop()
        self.assertEqual(self.outcomes(result), ["cancelled"])


class RuntimeTests(Harness):
    def test_voice_transcript_reaches_llm_verbatim_without_fallback(self):
        self.build()
        asr, llm = Mock(), Mock()
        transcript = "  Could YOU nod for me？！  "
        recorder = Mock(return_value=b"wav")
        asr.transcribe.return_value = transcript
        llm.decide.side_effect = (raw("motion_control", [gesture()]), LLMError("timed out"))
        runtime = AIRuntime(llm, self.registry, self.dispatcher, self.manager, asr=asr,
                            recorder=recorder, output=Mock())
        with patch("lelamp.demo.match_command") as keyword:
            self.assertEqual(self.outcomes(runtime.voice(0, 4)), ["success"])
            self.assertEqual(runtime.voice(0, 4).status, "rejected")
            keyword.assert_not_called()
        self.assertEqual(recorder.call_args_list, [call(0, 4), call(0, 4)])
        self.assertEqual(asr.transcribe.call_args_list, [call(b"wav"), call(b"wav")])
        self.assertEqual([call.args[0] for call in llm.decide.call_args_list], [transcript, transcript])
        self.motion.play.assert_called_once()

    def test_asr_http_400_reports_safe_provider_details_before_llm(self):
        self.build()
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x00\x00" * 160)
        secret = "test-private-credential"
        seen = []

        def bailian(request):
            payload = json.loads(request.content)
            data = payload["messages"][0]["content"][0]["input_audio"]["data"]
            seen.append(data)
            return httpx.Response(
                400,
                json={
                    "request_id": "test-request-400",
                    "error": {
                        "code": "InvalidParameter",
                        "message": f"Workspace endpoint is invalid. {data} {secret} https://example.invalid/signed?token=abc",
                    },
                },
            )

        with httpx.Client(transport=httpx.MockTransport(bailian)) as transport:
            client = OpenAI(api_key=secret, base_url="https://example.invalid/compatible-mode/v1",
                            max_retries=0, http_client=transport)
            llm, output = Mock(), Mock()
            runtime = AIRuntime(llm, self.registry, self.dispatcher, self.manager, asr=BailianASR(client),
                                recorder=Mock(return_value=audio.getvalue()), output=output)
            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": secret}), self.assertLogs("lelamp.ai_runtime") as logs:
                result = runtime.voice(0, 1)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(len(seen), 1)
        for text in (result.detail, output.call_args.args[0], "\n".join(logs.output)):
            self.assertIn("HTTP 400", text)
            self.assertIn("code=InvalidParameter", text)
            self.assertIn("request_id=test-request-400", text)
            self.assertIn("Workspace endpoint is invalid", text)
            self.assertIn("Check ASR_BASE_URL", text)
            self.assertNotIn(secret, text)
            self.assertNotIn(seen[0], text)
            self.assertNotIn("token=abc", text)
        llm.decide.assert_not_called()
        self.motion.play.assert_not_called()

    def test_late_llm_completion_after_local_stop_is_rejected(self):
        self.build()
        llm, output = Mock(), Mock()
        waiting, return_response = threading.Event(), threading.Event()

        def decide(text, capabilities):
            waiting.set()
            return_response.wait(3)
            return raw("motion_control", [gesture()])

        llm.decide.side_effect = decide
        runtime = AIRuntime(llm, self.registry, self.dispatcher, self.manager, output=output)
        results = []
        turn = threading.Thread(target=lambda: results.append(runtime.text("nod")))
        turn.start()
        try:
            self.assertTrue(waiting.wait(2))
            self.manager.local_stop()
            self.assertTrue(self.manager.resume())
        finally:
            return_response.set()
            turn.join(3)
        self.assertEqual([result.status for result in results], ["rejected"])
        self.motion.play.assert_not_called()

    def test_dashscope_tts_uses_native_route_and_plays_downloaded_wav(self):
        pcm = b"\x03\x00" * 12
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        env = {"TTS_ENABLED": "true", "TTS_PROVIDER": "dashscope",
               "TTS_BASE_URL": "https://example.aliyuncs.com/api/v1",
               "TTS_API_KEY": "test-key", "TTS_MODEL": "qwen-audio-3.1-tts-plus",
               "TTS_VOICE": "test-voice", "TTS_DEVICE": "3", "TTS_TIMEOUT": "8"}
        with patch.dict("os.environ", env):
            tts = tts_from_env()
        self.assertIsInstance(tts, DashScopeTTS)
        self.assertEqual(tts.device, 3)

        def synthesize(request):
            if request.method == "GET":
                self.assertEqual(str(request.url),
                                 "https://example.oss-cn-beijing.aliyuncs.com/result.wav?signature=test")
                self.assertNotIn("authorization", request.headers)
                return httpx.Response(200, content=wav_bytes.getvalue())
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/api/v1/services/audio/tts/SpeechSynthesizer")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertEqual(json.loads(request.content), {
                "model": "qwen-audio-3.1-tts-plus",
                "input": {"text": "Hello", "voice": "test-voice", "format": "wav", "sample_rate": 24000},
            })
            return httpx.Response(200, json={"output": {"audio": {
                "url": "http://example.oss-cn-beijing.aliyuncs.com/result.wav?signature=test",
            }}})

        speaker = MagicMock()
        transport = httpx.MockTransport(synthesize)
        tts.close()
        with httpx.Client(transport=transport) as client, patch.dict("sys.modules", {"sounddevice": speaker}):
            tts.client = client
            tts.download_transport = transport
            self.assertTrue(tts.speak("Hello", threading.Event(), "test"))

        speaker.RawOutputStream.assert_called_once_with(
            samplerate=24000, channels=1, dtype="int16", device=3, blocksize=480, latency="low")
        speaker.RawOutputStream.return_value.__enter__.return_value.write.assert_called_once_with(pcm)

    def test_real_tts_interface_aborts_playback_and_falls_back_on_failure(self):
        client, device = Mock(), MagicMock()
        token = threading.Event()
        audio = b"\x00\x00" * 960
        client.audio.speech.create.return_value = SimpleNamespace(content=audio)
        stream = device.RawOutputStream.return_value.__enter__.return_value
        stream.write.side_effect = lambda frame: token.set()
        with patch.dict("sys.modules", {"sounddevice": device}):
            self.assertFalse(OpenAITTS(client, "voice-model", "voice").speak("Hi", token, "test"))
        stream.abort.assert_called_once()
        self.assertEqual(stream.write.call_count, 1)

        failed_tts = Mock(spec=MockTTS)
        failed_tts.speak.side_effect = TimeoutError
        self.build(tts=failed_tts)
        output, llm = Mock(), Mock()
        llm.decide.return_value = raw(speech="Hello")
        runtime = AIRuntime(llm, self.registry, self.dispatcher, self.manager, output=output)
        self.assertEqual(self.outcomes(runtime.text("Hi")), ["failed"])
        self.assertIn(call("LUMO: Hello"), output.call_args_list)

    def test_cli_preview_never_opens_motors_and_keeps_old_parser(self):
        sounddevice = Mock()
        with patch("lelamp.demo.connect_robot") as connect, \
             patch("lelamp.demo.match_command") as keywords, \
             patch("lelamp.ai_runtime.LLMClient.from_env") as llm, \
             patch("lelamp.ai_runtime.BailianASR") as asr, \
             patch("lelamp.ai_runtime.tts_from_env", return_value=None), \
             patch("lelamp.ai_runtime.env_flag", side_effect=lambda name: name == "LIGHTING_ENABLED"), \
             patch.dict("sys.modules", {"sounddevice": sounddevice}), \
             patch("builtins.input", side_effect=["", "q", "What is gravity?", "/quit"]), \
             redirect_stdout(io.StringIO()) as output:
            sounddevice.rec.return_value.tobytes.return_value = b"\x00\x00" * 16000
            asr.return_value.transcribe.return_value = "Could you nod?"
            llm.return_value.decide.side_effect = (raw("motion_control", [gesture()]), raw())
            self.assertEqual(main(["ai", "--dry-run", "--device", "0", "--duration", "1"]), 0)
            self.assertEqual(main(["chat", "--dry-run"]), 0)
            self.assertIn('"intent": "conversation"', output.getvalue())
            connect.assert_not_called()
            keywords.assert_not_called()
            self.assertEqual([req.args[0] for req in llm.return_value.decide.call_args_list],
                             ["Could you nod?", "What is gravity?"])
            with wave.open(io.BytesIO(asr.return_value.transcribe.call_args.args[0]), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            main(["ai", "--dry-run", "--enable-motion"])

    def test_physical_cli_authenticates_before_connecting_and_cleans_up_holding(self):
        robot = Mock()
        robot.bus.is_connected = True
        args = ["ai", "--enable-motion", "--id", "test", "--port", "mock", "--profile", "test.json"]
        with patch("lelamp.demo.connect_robot", return_value=robot) as connect, \
             patch("lelamp.ai_runtime.LLMClient.from_env"), patch("lelamp.ai_runtime.BailianASR"), \
             patch("lelamp.ai_runtime.tts_from_env", return_value=None), \
             patch("lelamp.ai_runtime.env_flag", return_value=False), \
             patch("lelamp.ai_runtime.MotionProfile.load") as load, redirect_stdout(io.StringIO()):
            load.return_value = MotionProfile(profile_data())
            self.assertEqual(main(args), 1)
            connect.assert_not_called()
            load.return_value = verified_profile()
            with patch("builtins.input", side_effect=["", KeyboardInterrupt, EOFError]):
                self.assertEqual(main(args), 130)
            connect.assert_called_once_with("test", "mock", load.return_value, hold_on_error=True)
            robot.bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_torque_release_requires_explicit_confirmation(self):
        robot = Mock()
        with patch("builtins.input", return_value="RELEASE"):
            _disconnect_robot(robot)
        robot.bus.disconnect.assert_called_once_with(disable_torque=True)


class DatasetTests(unittest.TestCase):
    def test_all_bilingual_fixtures_round_trip_as_mock_completions(self):
        cases = read_cases()
        replies = []
        for case in cases:
            actions = []
            for label in case["actions"][0]:
                parts = label.split(":")
                if parts[0] == "motion":
                    actions.append(gesture(parts[1]))
                elif parts[0] == "lighting":
                    operation = parts[1]
                    if len(parts) == 3:
                        actions.append(light(operation, value=int(parts[2])))
                    elif operation in ("increase_brightness", "decrease_brightness"):
                        actions.append(light(operation, value=10))
                    else:
                        actions.append(light(operation))
                else:
                    actions.append({"type": "stop"})
            replies.append(raw(case["intents"][0], actions))
        llm = Mock()
        llm.decide.side_effect = replies
        metrics = evaluate(llm, cases)
        self.assertEqual(metrics["intent_accuracy"], 1)
        self.assertEqual(metrics["action_selection_accuracy"], 1)
        self.assertEqual(metrics["json_validity_rate"], 1)
        self.assertEqual(metrics["false_action_trigger_rate"], 0)
        self.assertEqual([call.args[0] for call in llm.decide.call_args_list],
                         [case["text"] for case in cases])

    def test_bilingual_cases_and_evaluator_metrics_with_mock_llm(self):
        cases = read_cases()
        self.assertGreaterEqual(len(cases), 50)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertEqual(len({case["text"] for case in cases}), len(cases))
        self.assertEqual({case["id"].split("-")[0] for case in cases}, {"en", "zh"})
        llm = Mock()
        llm.decide.side_effect = (raw("motion_control", [gesture()]),
                                  raw("emotion_expression", [light("set_brightness", value=50), gesture()]),
                                  raw(), "{bad")
        selection = [next(item for item in cases if item["id"] == key)
                     for key in ("en-01", "en-23", "zh-16", "zh-19")]
        metrics = evaluate(llm, selection)
        self.assertEqual(metrics["intent_accuracy"], 0.75)
        self.assertEqual(metrics["action_selection_accuracy"], 0.75)
        self.assertEqual(metrics["json_validity_rate"], 0.75)
        self.assertEqual(metrics["false_action_trigger_rate"], 0.0)
        event = BehaviorEvent.parse(raw("combined_control", [gesture(), light("set_brightness", value=50)]))
        self.assertTrue(actions_match(event, [["lighting:set_brightness:50", "motion:nod"]]))
        self.assertFalse(actions_match(event, [["lighting:set_brightness:75", "motion:nod"]]))
        with self.assertRaises(InvalidEvent):
            BehaviorEvent.parse("garbage")


if __name__ == "__main__":
    unittest.main()
