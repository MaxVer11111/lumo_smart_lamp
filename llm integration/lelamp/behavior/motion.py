from ..demo_recordings import AI_MOTION_DIR
from .capabilities import CapabilityRegistry


class MockMotion:
    def play(self, capability, cancel):
        return not cancel.is_set()

    def stop(self, cancel):
        cancel.set()


class RecordedMotion:
    def __init__(self, player, scale):
        self.player = player
        self.scale = scale
        self.approvals = CapabilityRegistry.motion_approvals(player.profile)

    def play(self, capability, cancel):
        if cancel.is_set():
            return False
        approval = CapabilityRegistry.motion_approvals(self.player.profile).get(capability.name)
        if (not capability.hardware_verified or not approval or approval != self.approvals.get(capability.name)
                or approval[0] != capability.recording_sha256 or self.scale > approval[1]):
            raise RuntimeError("The recording is no longer authorized for physical motion")
        return self.player.play(capability.recording.removesuffix(".csv"), source="recording",
                                scale=self.scale, cancel_event=cancel,
                                recording_sha256=capability.recording_sha256, recordings_dir=AI_MOTION_DIR)

    def stop(self, cancel):
        self.player.stop(cancel_event=cancel)
