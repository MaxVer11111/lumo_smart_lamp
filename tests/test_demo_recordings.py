import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lelamp.demo_motion import JOINTS, MotionProfile
from lelamp.demo_recordings import adapt_recording, recording_trajectory
from test_demo import profile_data


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        patcher = patch("lelamp.demo_recordings.RECORDINGS_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.profile = MotionProfile(profile_data())

    def write(self, times=(10, 10.1, 10.4, 11), values=(0, 0.1, 0.1, 0)):
        with (self.root / "sample.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=["timestamp", *JOINTS])
            writer.writeheader()
            for t, v in zip(times, values):
                # Nonzero absolute pose proves that only deltas are reused.
                row = {j: 50 + v for j in JOINTS}
                writer.writerow({"timestamp": t, **row})

    def assert_limits_and_speeds(self, frames):
        for frame in frames:
            self.profile.validate(frame)
        for a, b in zip(frames, frames[1:]):
            for j in JOINTS:
                self.assertLessEqual(abs(b[j] - a[j]) * 30, self.profile.speeds[j] + 1e-8)

    def test_relative_motion_and_original_pause_timing(self):
        self.write()
        frames, info = adapt_recording(self.profile, "sample")
        self.assertEqual(frames[0], self.profile.neutral)
        self.assertAlmostEqual(info["source_seconds"], 1)
        self.assertEqual(len(frames), 31)
        self.assertAlmostEqual(info["actual_scale"], 0.25)
        for frame in frames[3:13]:
            self.assertAlmostEqual(frame[JOINTS[0]], 0.025)
        self.assert_limits_and_speeds(frames)

    def test_uniform_scale_respects_speed_without_slowing_timeline(self):
        self.write(values=(0, 100, 100, 0))
        frames, info = adapt_recording(self.profile, "sample", scale=1)
        self.assertAlmostEqual(info["actual_scale"], 0.009)
        self.assertEqual(len(frames), 31)
        self.assert_limits_and_speeds(frames)

    def test_position_margin_and_return_to_neutral(self):
        self.profile.speeds = dict.fromkeys(JOINTS, 10000)
        self.write(values=(0, 100, 100, 50))
        frames, info = recording_trajectory(self.profile, dict.fromkeys(JOINTS, 1), "sample", scale=1)
        self.assertAlmostEqual(info["actual_scale"], 0.16)
        self.assertEqual(frames[-1], self.profile.neutral)
        self.assertEqual(frames[0], dict.fromkeys(JOINTS, 1))
        self.assert_limits_and_speeds(frames)

    def test_bad_timestamps_and_nonfinite_values_rejected(self):
        for times, values in (((0, 0, 1, 2), (0, 1, 2, 3)),
                              ((0, 2, 1, 3), (0, 1, 2, 3)),
                              ((0, 1, 2, 3), (0, float("nan"), 2, 3))):
            self.write(times, values)
            with self.assertRaises(ValueError):
                adapt_recording(self.profile, "sample")

    def test_unknown_names_path_traversal_and_bad_scales_rejected(self):
        self.write()
        for name in ("../sample", "absent", "/sample"):
            with self.assertRaises(ValueError):
                adapt_recording(self.profile, name)
        for scale in (0, -1, 2, float("nan")):
            with self.assertRaises(ValueError):
                adapt_recording(self.profile, "sample", scale)

    def test_recording_preview_exports_without_motor_connection(self):
        from lelamp.demo import main
        self.write()
        with patch("lelamp.demo.MotionProfile.load", return_value=self.profile), \
             patch("lelamp.demo.connect_robot") as connect:
            output = self.root / "preview.csv"
            self.assertEqual(main(["play", "--source", "recording", "--profile", "fake.json",
                                   "--action", "sample", "--dry-run", "--export", str(output)]), 0)
            self.assertTrue(output.exists())
            connect.assert_not_called()

    def test_recording_player_can_be_cancelled(self):
        from test_demo import MotionTests
        from lelamp.demo_motion import DemoPlayer
        self.write()
        robot = MotionTests().robot()
        player = DemoPlayer(robot, self.profile)
        robot.send_action.side_effect = lambda pose: player.stop()
        self.assertFalse(player.play("sample", source="recording"))
        self.assertEqual(robot.send_action.call_count, 1)


if __name__ == "__main__":
    unittest.main()
