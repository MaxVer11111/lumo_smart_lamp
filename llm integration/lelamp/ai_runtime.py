"""Press-to-record or text input → stateless LLM → validated, scheduled behavior."""

import logging
import signal
import time

from .ai_config import env_flag, redact_secrets
from .audio.tts import tts_from_env
from .behavior.capabilities import CapabilityRegistry
from .behavior.dispatcher import CommandDispatcher
from .behavior.manager import BehaviorManager, DispatchResult
from .behavior.motion import MockMotion, RecordedMotion
from .demo_motion import DemoPlayer, MotionProfile
from .demo_speech import ASRServiceError, BailianASR, record_audio
from .intelligence.llm_client import LLMClient, LLMError
from .lighting.controller import LightingController

logger = logging.getLogger(__name__)


class AIRuntime:
    def __init__(self, llm, registry, dispatcher, manager, *, asr=None, recorder=record_audio,
                 output=print, show_json=False):
        self.llm, self.registry, self.dispatcher, self.manager = llm, registry, dispatcher, manager
        self.asr, self.recorder, self.output, self.show_json = asr, recorder, output, show_json

    def text(self, text):
        if not text.strip():
            return DispatchResult("rejected", "Empty input")
        ticket = self.manager.begin_interaction()
        return self._decide(ticket, text, time.monotonic())

    def voice(self, device, duration):
        with self.manager.recording() as allowed:
            if not allowed:
                detail = "Recording blocked while speech is pending/playing or local stop is active"
                self.output(detail)
                return DispatchResult("rejected", detail)
            ticket = self.manager.begin_interaction()
            started = time.monotonic()
            try:
                self.output(f"Recording {duration:g} seconds...")
                audio = self.recorder(device, duration)
            except Exception as exc:
                return self._failure(ticket, "audio", type(exc).__name__)
        self.manager.set_runtime("THINKING")
        try:
            asr_started = time.monotonic()
            try:
                text = self.asr.transcribe(audio)
            finally:
                logger.info("interaction=%s asr_seconds=%.3f", ticket.id, time.monotonic() - asr_started)
        except ASRServiceError as exc:
            return self._failure(ticket, "ASR", str(exc))
        except Exception as exc:
            return self._failure(ticket, "ASR", type(exc).__name__)
        if not isinstance(text, str) or not text.strip():
            return self._failure(ticket, "ASR", "Empty transcription")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("interaction=%s transcript=%r", ticket.id, redact_secrets(text))
        return self._decide(ticket, text, started)

    def _decide(self, ticket, text, started):
        self.manager.set_runtime("THINKING")
        llm_started = time.monotonic()
        try:
            motion, lighting = self.manager.capabilities_ready()
            capabilities = self.registry.prompt_data(motion_ready=motion, lighting_ready=lighting)
            raw = self.llm.decide(text, capabilities)
        except LLMError as exc:
            return self._failure(ticket, "LLM", str(exc))
        except Exception as exc:
            return self._failure(ticket, "LLM", type(exc).__name__)
        finally:
            logger.info("interaction=%s llm_seconds=%.3f", ticket.id, time.monotonic() - llm_started)
        event, result = self.dispatcher.dispatch(ticket, raw)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("interaction=%s behavior_json=%r", ticket.id, redact_secrets(str(raw)))
        if event:
            logger.info("interaction=%s intent=%s emotion=%s schema=passed",
                        ticket.id, event.intent, event.user_emotion)
        else:
            logger.info("interaction=%s validation=%s", ticket.id, result.status)
        if result.status in ("accepted", "success"):
            if event.speech:
                self.output(f"LUMO: {redact_secrets(event.speech)}")
            if self.show_json:
                self.output(redact_secrets(raw))
        elif event and self.show_json:
            self.output(redact_secrets(raw))
        self.output(f"[{ticket.id}] {result.status} {result.detail}".rstrip())
        logger.info("interaction=%s interaction_seconds=%.3f", ticket.id, time.monotonic() - started)
        self.manager.set_runtime("IDLE" if result.status in ("accepted", "success") else "ERROR")
        return result

    def _failure(self, ticket, stage, detail):
        self.manager.discard(ticket)
        self.manager.set_runtime("ERROR")
        detail = redact_secrets(detail)
        logger.error("interaction=%s stage=%s failed: %s", ticket.id, stage, detail)
        self.output(f"[{ticket.id}] {stage} failed: {detail}")
        return DispatchResult("rejected", detail)


def run_cli(args, parser, connect_robot, *, lighting_factory=None):
    if args.enable_motion and (args.mode != "ai" or args.dry_run):
        parser.error("--enable-motion requires ai without --dry-run")
    if args.enable_motion and not all((args.id, args.port, args.profile)):
        parser.error("--enable-motion requires --id, --port and --profile")
    if args.source not in (None, "recording") or args.action or args.export or args.joint or args.delta is not None:
        parser.error("chat and ai select registered recording actions through the LLM")
    if not 0 < args.scale <= 1 or not 0 < args.duration <= 30:
        parser.error("Expected 0 < --scale <= 1 and 0 < --duration <= 30")
    if args.debug_ai:
        for module in (__name__, "lelamp.behavior"):
            logging.getLogger(module).setLevel(logging.DEBUG)
    robot = manager = None
    code = 0
    try:
        llm = LLMClient.from_env()
        asr = BailianASR() if args.mode == "ai" else None
        try:
            tts = tts_from_env()
        except Exception as exc:
            logger.error("TTS configuration failed (%s). Replies will appear in the terminal.", type(exc).__name__)
            tts = None
        simulated = not args.enable_motion
        lighting_enabled = env_flag("LIGHTING_ENABLED")
        lighting = LightingController() if simulated else None
        if lighting_enabled and args.mode == "ai" and not args.dry_run:
            lighting = _initialize_lighting(lighting_factory, fallback=lighting)
        profile = MotionProfile.load(args.profile) if args.enable_motion else None
        if profile and profile.data["lamp_id"] != args.id:
            raise ValueError("Motion profile belongs to a different lamp")
        registry = CapabilityRegistry(simulated=simulated, motion=args.enable_motion, profile=profile,
                                      scale=args.scale, lighting=lighting, tts_mode=tts.mode if tts else None)
        motion = MockMotion()
        if args.enable_motion:
            input("Support the lamp and clear its travel area. Press Enter to arm verified motion: ")
            robot = connect_robot(args.id, args.port, profile, hold_on_error=True)
            motion = RecordedMotion(DemoPlayer(robot, profile), args.scale)
        else:
            print("Motion is simulated. Lighting is simulated unless a verified backend is initialized.")
        manager = BehaviorManager(motion, lighting, tts)
        dispatcher = CommandDispatcher(registry, manager, motion, lighting, tts)
        runtime = AIRuntime(llm, registry, dispatcher, manager, asr=asr, show_json=args.mode == "chat")
        _input_loop(args, runtime, manager)
    except (KeyboardInterrupt, EOFError):
        if manager:
            manager.local_stop()
        print("Stopped. Pending behaviors have been cancelled.")
        code = 130
    except Exception as exc:
        # I/O exceptions and SDK errors can contain device or endpoint details.
        detail = redact_secrets(str(exc)) if isinstance(exc, ValueError) else type(exc).__name__
        logger.error("AI startup or runtime error: %s", detail)
        code = 1
    finally:
        if manager:
            # Once cancelling, wait for the serial writer to exit before releasing the bus.
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                manager.close()
            finally:
                signal.signal(signal.SIGINT, previous)
        if robot:
            try:
                _disconnect_robot(robot)
            except Exception as exc:
                logger.error("Serial cleanup failed (%s). Support the lamp and check the hardware.",
                             type(exc).__name__)
                code = 1
    return code


def _initialize_lighting(factory, *, fallback):
    if factory is None:
        logger.warning("No verified LED backend is installed")
        return fallback
    try:
        controller = factory()
        if controller is None or controller.simulated:
            raise ValueError("Expected an initialized physical lighting controller")
        return controller
    except Exception as exc:
        logger.error("Lighting initialization failed (%s)", type(exc).__name__)
        return fallback


def _input_loop(args, runtime, manager):
    text_mode = args.mode == "chat"
    while True:
        prompt = ("You (/stop, /resume, /quit): " if text_mode else
                  "Enter to record; /stop, /resume, q to quit: ")
        try:
            text = input(prompt)
        except EOFError:
            return
        command = text.strip().lower()
        if command == "/quit" or (not text_mode and command == "q"):
            return
        if command == "/stop":
            manager.local_stop()
            print("Local stop latched.")
        elif command == "/resume":
            print("Local stop cleared." if manager.resume() else "Resume refused. Check runtime state.")
        elif manager.state()["runtime"] == "STOPPED":
            print("Local stop is active. Use /resume when safe.")
        elif text_mode:
            runtime.text(text)
        elif not command:
            runtime.voice(args.device, args.duration)
        else:
            print("Press Enter to record or enter a displayed local control.")


def _disconnect_robot(robot):
    release = False
    try:
        if robot.bus.is_connected:
            release = input("Support the lamp. Type RELEASE to release torque, or Enter to close while holding: ") == "RELEASE"
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if robot.bus.is_connected:
            try:
                robot.bus.disconnect(disable_torque=release)
            except Exception:
                logger.error("Torque release was not confirmed.")
                if robot.bus.is_connected:
                    robot.bus.disconnect(disable_torque=False)
                raise
            if not release:
                print("Serial closed with torque unchanged. Support the lamp before removing power.")
