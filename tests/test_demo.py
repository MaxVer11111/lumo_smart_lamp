import copy
import io
import json
import tempfile
import threading
import unittest
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from lelamp.demo_motion import ACTIONS, JOINTS, DemoPlayer, MotionProfile, calibration_digest, export_csv
from lelamp.demo_speech import BailianASR, match_command


@dataclass
class Calibration:
    id: int = 1
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = 4095


def profile_data():
    # Synthetic values for tests only; never a hardware profile.
    return {"units": "range_m100_100", "lamp_id": "test",
            "calibration_sha256": calibration_digest({"test": Calibration()}),
            "neutral": dict.fromkeys(JOINTS, 0),
            "limits": {j: [-20, 20] for j in JOINTS},
            "max_speed": dict.fromkeys(JOINTS, 10),
            "amplitudes": dict.fromkeys(ACTIONS, 2)}


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.profile = MotionProfile(profile_data())

    def robot(self):
        return SimpleNamespace(id="test", calibration={"test": Calibration()}, is_calibrated=True,
                               get_observation=lambda: dict.fromkeys(JOINTS, 1), send_action=Mock())

    def test_trajectories_speed_limits_and_endpoints(self):
        for name in ACTIONS:
            start = dict.fromkeys(JOINTS, 1)
            frames = self.profile.trajectory(start, name)
            self.assertEqual(frames[0], start)
            self.assertEqual(frames[-1], self.profile.neutral)
            for before, after in zip(frames, frames[1:]):
                self.profile.validate(after)
                for joint in JOINTS:
                    self.assertLessEqual(abs(after[joint] - before[joint]) * 30, 10 + 1e-9)

    def test_invalid_profiles_and_start_rejected(self):
        for change in (lambda d: d["max_speed"].update({JOINTS[0]: 0}),
                       lambda d: d["neutral"].update({JOINTS[0]: float("nan")}),
                       lambda d: d["amplitudes"].update({"greet": 100}),
                       lambda d: d.update(units="degrees")):
            data = copy.deepcopy(profile_data())
            change(data)
            with self.assertRaises(ValueError):
                MotionProfile(data)
        with self.assertRaises(ValueError):
            self.profile.trajectory(dict.fromkeys(JOINTS, 99), "nod")

    def test_calibration_guards_send_nothing(self):
        for mutate in (lambda r: setattr(r, "is_calibrated", False),
                       lambda r: setattr(r, "calibration", {}),
                       lambda r: setattr(r, "id", "other"),
                       lambda r: setattr(r, "calibration", {"test": Calibration(id=2)})):
            robot = self.robot()
            mutate(robot)
            with self.assertRaises((ValueError, RuntimeError)):
                DemoPlayer(robot, self.profile).play("nod")
            robot.send_action.assert_not_called()

    def test_stop_cancels_without_return_to_neutral(self):
        robot = self.robot()
        player = DemoPlayer(robot, self.profile)
        robot.send_action.side_effect = lambda pose: player.stop()
        self.assertFalse(player.play("nod"))
        self.assertEqual(robot.send_action.call_count, 1)

    def test_busy_rejected(self):
        robot = self.robot()
        player = DemoPlayer(robot, self.profile)
        entered, release = threading.Event(), threading.Event()
        def send(pose):
            entered.set()
            release.wait(2)
        robot.send_action.side_effect = send
        worker = threading.Thread(target=player.play, args=("nod",))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(RuntimeError):
                player.play("greet")
            player.stop()
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_export_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nod.csv"
            export_csv(path, self.profile.trajectory(self.profile.neutral, "nod"))
            self.assertTrue(path.read_text().startswith("timestamp,base_yaw.pos,"))


class SpeechTests(unittest.TestCase):
    def audio(self):
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 160)
        return data.getvalue()

    def test_exact_matching(self):
        for text, expected in ((" NOD your HEAD! ", "nod"), ("shake   your head.", "headshake"),
                               ("Hello", "greet"), ("say hello?", "greet")):
            self.assertEqual(match_command(text), expected)
        for text in ("", "don't nod", "nod and say hello", "hello robot", "please nod", "noise"):
            self.assertIsNone(match_command(text))

    def test_all_builtin_voice_mappings_and_generated_filter(self):
        from lelamp.demo_recordings import VOICE_RECORDINGS, available_recordings
        from lelamp.demo_speech import voice_commands
        expected = {"nod", "headshake", "curious", "excited", "happy_wiggle", "sad",
                    "scanning", "shock", "shy", "wake_up", "idle"}
        self.assertEqual(set(VOICE_RECORDINGS.values()), expected)
        self.assertTrue(expected <= set(available_recordings()))
        for phrase, action in voice_commands().items():
            self.assertEqual(match_command("  " + phrase.upper() + "!  "), action)
            self.assertEqual(match_command(phrase, source="generated"), action if action in ACTIONS else None)
        for phrase in ("don't be sad", "wake up and nod", "not excited", "first test", "play first_test"):
            self.assertIsNone(match_command(phrase))

    def test_voice_routes_every_builtin_to_adapted_recording(self):
        from lelamp.demo import main
        commands = {"nod": "nod", "shake your head": "headshake", "hello": "happy_wiggle",
                    "curious": "curious", "excited": "excited", "happy": "happy_wiggle",
                    "sad": "sad", "look around": "scanning", "shocked": "shock",
                    "shy": "shy", "wake up": "wake_up", "idle": "idle"}
        robot = Mock()
        robot.bus.is_connected = True
        with patch("lelamp.demo.connect_robot", return_value=robot), \
             patch("lelamp.demo.MotionProfile.load", return_value=MotionProfile(profile_data())), \
             patch("lelamp.demo.recording_trajectory", return_value=([], {})), \
             patch("lelamp.demo.DemoPlayer") as player, \
             patch("lelamp.demo.BailianASR") as asr, \
             patch("lelamp.demo.record_audio", return_value=self.audio()), \
             patch("builtins.input", side_effect=[""] * len(commands) + ["q", ""]):
            asr.return_value.transcribe.side_effect = list(commands)
            self.assertEqual(main(["voice", "--id", "test", "--port", "fake", "--profile", "fake.json"]), 0)
            self.assertEqual(player.return_value.play.call_args_list,
                             [call(name, source="recording", scale=0.25) for name in commands.values()])
            robot.disconnect.assert_called_once()

    def test_generated_voice_does_not_route_recording_only_commands(self):
        from lelamp.demo import main
        robot = Mock()
        robot.bus.is_connected = True
        with patch("lelamp.demo.connect_robot", return_value=robot), \
             patch("lelamp.demo.MotionProfile.load", return_value=MotionProfile(profile_data())), \
             patch("lelamp.demo.DemoPlayer") as player, \
             patch("lelamp.demo.BailianASR") as asr, \
             patch("lelamp.demo.record_audio", return_value=self.audio()), \
             patch("builtins.input", side_effect=["", "", "q", ""]):
            asr.return_value.transcribe.side_effect = ["excited", "hello"]
            self.assertEqual(main(["voice", "--source", "generated", "--id", "test", "--port", "fake", "--profile", "fake.json"]), 0)
            player.return_value.play.assert_called_once_with("greet", source="generated", scale=0.25)

    def test_bailian_protocol(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Nod."))])
        self.assertEqual(BailianASR(client).transcribe(self.audio()), "Nod.")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "qwen3-asr-flash")
        self.assertEqual(request["extra_body"]["asr_options"]["language"], "en")
        self.assertTrue(request["messages"][0]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))

    def test_cloud_failures_and_empty_results(self):
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError
        with self.assertRaises(TimeoutError):
            BailianASR(client).transcribe(self.audio())
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
        self.assertIsNone(match_command(BailianASR(client).transcribe(self.audio())))

    def test_authentication_failure_has_safe_actionable_message(self):
        import httpx
        from openai import AuthenticationError
        from lelamp.demo_speech import ASRAuthenticationError
        client = Mock()
        response = httpx.Response(401, request=httpx.Request("POST", "https://example.com"))
        client.chat.completions.create.side_effect = AuthenticationError(
            "secret-from-server", response=response, body={"message": "secret-from-server"})
        with self.assertRaises(ASRAuthenticationError) as caught:
            BailianASR(client).transcribe(self.audio())
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertIn("DASHSCOPE_API_KEY", str(caught.exception))
        self.assertNotIn("secret-from-server", str(caught.exception))

    def test_voice_authentication_failure_exits_without_playback(self):
        from lelamp.demo import main
        from lelamp.demo_speech import ASRAuthenticationError
        robot = Mock()
        robot.bus.is_connected = True
        with patch("lelamp.demo.connect_robot", return_value=robot), \
             patch("lelamp.demo.MotionProfile.load", return_value=MotionProfile(profile_data())), \
             patch("lelamp.demo.recording_trajectory", return_value=([], {})), \
             patch("lelamp.demo.DemoPlayer") as player, \
             patch("lelamp.demo.BailianASR") as asr, \
             patch("lelamp.demo.record_audio", return_value=self.audio()), \
             patch("builtins.input", side_effect=["", ""]):
            asr.return_value.transcribe.side_effect = ASRAuthenticationError("HTTP 401")
            self.assertEqual(main(["voice", "--id", "test", "--port", "fake", "--profile", "fake.json"]), 1)
            player.return_value.play.assert_not_called()
            robot.disconnect.assert_called_once()

    def test_voice_dry_run_and_failure_never_open_motors(self):
        from lelamp.demo import main
        with patch("lelamp.demo.connect_robot") as connect, \
             patch("lelamp.demo.BailianASR") as asr, \
             patch("lelamp.demo.record_audio", return_value=self.audio()), \
             patch("builtins.input", side_effect=["", "", "q"]):
            asr.return_value.transcribe.side_effect = [TimeoutError(), "nod"]
            self.assertEqual(main(["voice", "--dry-run"]), 0)
            connect.assert_not_called()

    def test_voice_cloud_failure_sends_no_action_and_closes_robot(self):
        from lelamp.demo import main
        robot = Mock()
        robot.bus.is_connected = True
        with patch("lelamp.demo.connect_robot", return_value=robot), \
             patch("lelamp.demo.MotionProfile.load", return_value=MotionProfile(profile_data())), \
             patch("lelamp.demo.DemoPlayer") as player, \
             patch("lelamp.demo.BailianASR") as asr, \
             patch("lelamp.demo.record_audio", return_value=self.audio()), \
             patch("builtins.input", side_effect=["", "q", ""]):
            asr.return_value.transcribe.side_effect = TimeoutError()
            self.assertEqual(main(["voice", "--id", "test", "--port", "fake", "--profile", "fake.json"]), 0)
            player.return_value.play.assert_not_called()
            robot.disconnect.assert_called_once()

    def test_play_preview_never_opens_motors(self):
        from lelamp.demo import main
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.json"
            path.write_text(json.dumps(profile_data()))
            with patch("lelamp.demo.connect_robot") as connect:
                self.assertEqual(main(["play", "--dry-run", "--profile", str(path), "--action", "nod"]), 0)
                connect.assert_not_called()


class ConnectionTests(unittest.TestCase):
    def test_failed_configuration_does_not_enable_torque(self):
        from lelamp.follower import LeLampFollower
        robot = object.__new__(LeLampFollower)
        robot.bus = Mock()
        robot.bus.motors = {"base_pitch": Mock()}
        for failure in ("disable_torque", "configure_motors", "write"):
            robot.bus.reset_mock(side_effect=True)
            getattr(robot.bus, failure).side_effect = ConnectionError("no status packet")
            with self.assertRaises(ConnectionError):
                robot.configure()
            robot.bus.enable_torque.assert_not_called()

    def test_configuration_uses_bounded_torque_retries(self):
        from lelamp.follower import LeLampFollower
        robot = object.__new__(LeLampFollower)
        robot.bus = Mock()
        robot.bus.motors = {"base_pitch": Mock()}
        robot.configure()
        robot.bus.disable_torque.assert_called_once_with(num_retry=3)
        robot.bus.enable_torque.assert_called_once_with(num_retry=3)
        self.assertTrue(all(c.kwargs.get("num_retry") == 3 for c in robot.bus.write.call_args_list))

    def test_cleanup_failure_preserves_original_startup_error_and_closes_port(self):
        from lelamp.demo import connect_robot
        robot = Mock()
        robot.calibration = {"test": Calibration()}
        robot.is_calibrated = True
        robot.bus.is_connected = True
        robot.get_observation.return_value = dict.fromkeys(JOINTS, 0)
        robot.bus.disable_torque.side_effect = ConnectionError("original startup failure")
        robot.bus.disconnect.side_effect = [ConnectionError("cleanup failure"), None]
        with patch("lelamp.follower.LeLampFollower", return_value=robot):
            with self.assertRaisesRegex(ConnectionError, "original startup failure"):
                connect_robot("test", "fake", MotionProfile(profile_data()))
        robot.bus.disconnect.assert_has_calls([call(disable_torque=True), call(disable_torque=False)])
        robot.configure.assert_not_called()

    def test_follower_missing_calibration_never_enables_torque(self):
        from lelamp.follower import LeLampFollower
        robot = object.__new__(LeLampFollower)
        robot.bus = Mock()
        # connect checks is_connected before bus.connect().
        robot.bus.is_connected = False
        robot.cameras = {}
        robot.calibration = {}
        robot.configure = Mock()
        with self.assertRaises(RuntimeError):
            robot.connect(calibrate=False)
        robot.configure.assert_not_called()
        robot.bus.sync_write.assert_not_called()
        robot.bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_calibration_only_does_not_configure(self):
        from lelamp.follower import LeLampFollower
        robot = object.__new__(LeLampFollower)
        robot.bus = Mock()
        robot.bus.is_connected = False
        robot.cameras = {}
        robot.configure = Mock()
        robot.connect(calibrate=False, calibration_only=True)
        robot.configure.assert_not_called()
        robot.bus.sync_write.assert_not_called()

    def test_demo_calibration_mismatch_closes_without_motion(self):
        from lelamp.demo import connect_robot
        robot = Mock()
        robot.calibration = {"test": Calibration()}
        robot.is_calibrated = False
        robot.bus.is_connected = True
        with patch("lelamp.follower.LeLampFollower", return_value=robot):
            with self.assertRaises(RuntimeError):
                connect_robot("test", "fake", MotionProfile(profile_data()))
        robot.configure.assert_not_called()
        robot.bus.sync_write.assert_not_called()
        robot.bus.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
