from ..demo_recordings import AI_MOTION_DIR


class MockMotion:
    def play(self, capability, cancel):
        return not cancel.is_set()

    def stop(self, cancel):
        cancel.set()


class RecordedMotion:
    def __init__(self, player, scale):
        self.player = player
        self.scale = scale

    def play(self, capability, cancel):
        if not capability.hardware_verified or not capability.recording_sha256:
            raise RuntimeError("The recording is no longer authorized for physical motion")
        return self.player.play(capability.recording.removesuffix(".csv"), source="recording",
                                scale=self.scale, cancel_event=cancel,
                                recording_sha256=capability.recording_sha256, recordings_dir=AI_MOTION_DIR)

    def stop(self, cancel):
        self.player.stop(cancel_event=cancel)
