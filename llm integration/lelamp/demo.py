"""Run with python -m lelamp.demo --help. No hardware is opened for previews."""

import argparse
import json
import logging
import signal
import time
from pathlib import Path

from .demo_motion import ACTIONS, JOINTS, DemoPlayer, MotionProfile, calibration_digest, export_csv
from .demo_speech import ASRAuthenticationError, ASRServiceError, BailianASR, match_command, record_audio, voice_commands
from .demo_recordings import VOICE_RECORDINGS, available_recordings, recording_trajectory

logger = logging.getLogger(__name__)


def connect_robot(lamp_id, port, profile=None, *, hold_on_error=False):
    from .follower import LeLampFollower, LeLampFollowerConfig
    robot = LeLampFollower(LeLampFollowerConfig(id=lamp_id, port=port))
    if not robot.calibration:
        raise RuntimeError("No follower calibration; run lelamp.calibrate --follower-only first")
    try:
        # Bus-only connection does not configure or enable motors.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Motor calibration does not match the follower calibration file")
        if profile:
            if lamp_id != profile.data["lamp_id"] or calibration_digest(robot.calibration) != profile.data["calibration_sha256"]:
                raise ValueError("Motion profile does not match this lamp and calibration")
            current = robot.get_observation()
            profile.validate(current)
            # Seed the goal while torque is off, before configure() re-enables it.
            robot.bus.disable_torque(num_retry=3)
            robot.bus.sync_write("Goal_Position", {j.removesuffix(".pos"): v for j, v in current.items()})
            robot.configure()
        return robot
    except BaseException:
        if robot.bus.is_connected:
            try:
                robot.bus.disconnect(disable_torque=profile is not None and not hold_on_error)
            except Exception as cleanup_error:
                logger.error("Startup cleanup failed (%s); torque release is NOT confirmed. Support the lamp.",
                             type(cleanup_error).__name__)
                # A failed torque-off must not mask the original communication error
                # or leave the serial port owned by a failed startup.
                try:
                    robot.bus.disconnect(disable_torque=False)
                except Exception:
                    logger.error("Serial close also failed; check device access before restarting.")
        raise


def create_profile(args):
    robot = connect_robot(args.id, args.port)
    try:
        input("Support the lamp at its validated stable neutral pose. Press Enter to read it: ")
        neutral = robot.get_observation()
        data = {"units": "range_m100_100", "lamp_id": args.id,
                "calibration_sha256": calibration_digest(robot.calibration),
                "neutral": neutral, "limits": {}, "max_speed": {}, "amplitudes": {}}
        print("Values below use calibrated -100..100 units, NOT degrees.")
        for joint in JOINTS:
            print(f"{joint}: current neutral = {neutral[joint]:.3f}")
            data["limits"][joint] = [float(input("Validated soft minimum: ")),
                                      float(input("Validated soft maximum: "))]
            data["max_speed"][joint] = float(input("Validated maximum units/second: "))
        for name in ACTIONS:
            data["amplitudes"][name] = float(input(f"Validated {name} amplitude in units: "))
        MotionProfile(data)
        # Never overwrite an existing validated profile silently.
        with open(args.profile, "x") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
        print(f"Profile saved to {args.profile}")
    finally:
        robot.bus.disconnect(disable_torque=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("list", "commands", "devices", "inspect", "profile", "play", "jog", "voice", "chat", "ai"))
    p.add_argument("--id")
    p.add_argument("--port")
    p.add_argument("--profile", help="Validated local motion profile JSON")
    p.add_argument("--action", help="Generated action name or CSV recording name")
    p.add_argument("--source", choices=("generated", "recording"),
                   help="Default: recording for voice/commands; generated for other modes")
    p.add_argument("--recordings-dir", type=Path, help="CSV directory for play --source recording")
    p.add_argument("--scale", type=float, default=0.25,
                   help="Recording amplitude scale ceiling (0 < scale <= 1; default 0.25)")
    p.add_argument("--joint", choices=JOINTS)
    p.add_argument("--delta", type=float, help="Single-joint test offset in normalized units")
    p.add_argument("--device", type=int, help="Microphone device index")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--dry-run", action="store_true", help="Never connect to motors")
    p.add_argument("--enable-motion", action="store_true", help="Allow locally verified physical motion in ai mode")
    p.add_argument("--debug-ai", action="store_true", help="Include transcripts and raw events in AI debug logs")
    for flag, description in (
        ("vad", "End recording after local voice activity detection"),
        ("streaming-asr", "Send PCM to Qwen ASR as it is captured"),
        ("streaming-tts", "Play Qwen TTS PCM as it arrives"),
        ("reuse-audio", "Reuse stopped input and streaming output device handles"),
        ("motion-cache", "Cache validated recording data for authorized physical motion"),
    ):
        p.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, help=description)
    p.add_argument("--benchmark", action="store_true", help="Collect per-interaction latency statistics")
    p.add_argument("--benchmark-output", help="Append benchmark interactions as JSONL")
    p.add_argument("--benchmark-label", help="Label used to group benchmark runs")
    p.add_argument("--export", help="Export preview trajectory CSV (play --dry-run)")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.recordings_dir is not None and (args.mode != "play" or args.source != "recording"):
        p.error("--recordings-dir requires play --source recording")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv(override=False)
    if not args.benchmark and (args.benchmark_output or args.benchmark_label):
        p.error("--benchmark-output and --benchmark-label require --benchmark")
    if args.mode in ("chat", "ai"):
        from .ai_runtime import run_cli
        return run_cli(args, p, connect_robot)
    if (args.enable_motion or args.debug_ai or args.benchmark
            or any(option is not None for option in (args.vad, args.streaming_asr, args.streaming_tts,
                                                    args.reuse_audio, args.motion_cache))):
        p.error("AI pipeline options require chat or ai mode")
    args.source = args.source or ("recording" if args.mode in ("voice", "commands") else "generated")
    if args.mode == "commands":
        for phrase, action in voice_commands(args.source).items():
            motion = VOICE_RECORDINGS[action] + ".csv" if args.source == "recording" else action
            print(f"{phrase} -> {motion}")
        return 0
    if args.mode == "list":
        print("\n".join(available_recordings() if args.source == "recording" else ACTIONS))
        return 0
    if args.mode == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    if args.dry_run and args.mode not in ("play", "voice"):
        p.error("--dry-run supports play and voice only")
    if args.mode in ("play", "jog", "profile") or (args.mode == "voice" and not args.dry_run):
        if not args.profile:
            p.error("--profile is required")
    if not args.dry_run and (not args.id or not args.port):
        p.error("Hardware modes require --id and --port")
    if args.mode == "play" and not args.action:
        p.error("play requires --action")
    if not 0 < args.scale <= 1:
        p.error("--scale must be > 0 and <= 1")
    if args.mode == "jog" and (not args.joint or args.delta is None):
        p.error("jog requires --joint and --delta")
    if not 0 < args.duration <= 30:
        p.error("--duration must be between 0 and 30 seconds")
    robot = asr = None
    player = None
    previous_sigint = None
    try:
        if args.mode == "profile":
            create_profile(args)
            return 0
        if args.mode == "inspect":
            robot = connect_robot(args.id, args.port)
            print(json.dumps({"position": robot.get_observation(),
                              "calibration_sha256": calibration_digest(robot.calibration)}, indent=2))
            return 0
        profile = MotionProfile.load(args.profile) if args.profile else None
        if args.mode == "play":
            # Preflight before opening the serial port or enabling torque.
            if args.source == "recording":
                frames, info = recording_trajectory(
                    profile, profile.neutral, args.action, args.scale, recordings_dir=args.recordings_dir)
                print(json.dumps(info, indent=2))
            else:
                frames = profile.trajectory(profile.neutral, args.action)
        if args.mode == "voice" and args.source == "recording" and profile:
            for recording in sorted(set(VOICE_RECORDINGS.values())):
                _, info = recording_trajectory(profile, profile.neutral, recording, args.scale)
                print(json.dumps(info, indent=2))
        if args.mode == "play" and args.dry_run:
            print(f"{args.action}: {len(frames)} frames, {len(frames)/30:.2f}s, no motor connection")
            if args.export:
                export_csv(args.export, frames)
            return 0
        # Check cloud configuration before enabling motors.
        asr = BailianASR() if args.mode == "voice" else None
        if args.mode == "voice":
            print(f"Voice motion source: {args.source}. Run 'python -m lelamp.demo commands' for English commands.")
        if not args.dry_run:
            robot = connect_robot(args.id, args.port, profile)
            player = DemoPlayer(robot, profile)

        def interrupt(signum, frame):
            if player:
                player.stop()
            raise KeyboardInterrupt

        previous_sigint = signal.signal(signal.SIGINT, interrupt)
        if args.mode == "jog":
            start = robot.get_observation()
            target = dict(start)
            target[args.joint] += args.delta
            profile.validate(target)
            player.play("jog", targets=[target, start])
            return 0
        if args.mode == "play":
            logger.info("action=%s completed=%s", args.action,
                        player.play(args.action, source=args.source, scale=args.scale, recordings_dir=args.recordings_dir))
            return 0
        while True:
            if input("Enter to record; q then Enter to exit: ").strip().lower() == "q":
                break
            try:
                print(f"Recording {args.duration:g} seconds...")
                audio = record_audio(args.device, args.duration)
                started = time.monotonic()
                text = asr.transcribe(audio)
                action = match_command(text, source=args.source)
                motion = (VOICE_RECORDINGS[action] if args.source == "recording" else action) if action else None
                logger.info("transcript=%r action=%s motion=%s source=%s recognition_seconds=%.3f",
                            text, action, motion, args.source, time.monotonic() - started)
            except ASRAuthenticationError as exc:
                logger.error("%s", exc)
                return 1
            except ASRServiceError as exc:
                logger.error("%s", exc)
                continue
            except Exception as exc:
                logger.error("Voice cycle failed (%s)", type(exc).__name__)
                continue
            if action and player:
                started = time.monotonic()
                completed = player.play(motion, source=args.source, scale=args.scale)
                logger.info("action=%s completed=%s playback_seconds=%.3f",
                            action, completed, time.monotonic() - started)
            elif not action:
                print("No matching command; no action.")
        return 0
    except (KeyboardInterrupt, EOFError):
        if player:
            player.stop()
        print("Stopped. No further trajectory commands will be sent.")
        return 130
    except Exception as exc:
        logger.error("Demo failed: %s", exc)
        return 1
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if robot and robot.bus.is_connected:
            if player:
                # No background writer exists. Hold until the user supports the lamp.
                try:
                    input("Support the lamp, then Enter to release torque and disconnect: ")
                except (EOFError, KeyboardInterrupt):
                    print("Disconnecting and releasing torque; support the lamp.")
                finally:
                    robot.disconnect()
            else:
                robot.bus.disconnect(disable_torque=False)
        if asr:
            try:
                asr.client.close()
            except Exception as exc:
                logger.warning("ASR cleanup failed (%s)", type(exc).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
