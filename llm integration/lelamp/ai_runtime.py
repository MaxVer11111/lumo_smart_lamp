"""Press-to-record or text input → stateless LLM → validated, scheduled behavior."""

import logging
import os
import signal
import time
from contextlib import ExitStack

from .ai_config import env_flag, redact_secrets
from .audio.capture import CaptureError, MicrophoneCapture, VADSettings
from .audio.realtime_asr import RealtimeASR
from .audio.realtime_tts import RealtimeTTS
from .audio.tts import tts_from_env
from .behavior.capabilities import CapabilityRegistry
from .behavior.dispatcher import CommandDispatcher
from .behavior.manager import BehaviorManager, DispatchResult
from .behavior.motion import MockMotion, RecordedMotion
from .behavior.performance import Benchmark, observe
from .demo_motion import DemoPlayer, MotionProfile
from .demo_recordings import RecordingCache
from .demo_speech import ASRServiceError, BailianASR, record_audio
from .intelligence.llm_client import LLMClient, LLMError
from .lighting.controller import LightingController

logger = logging.getLogger(__name__)


class AIRuntime:
    def __init__(self, llm, registry, dispatcher, manager, *, asr=None, recorder=record_audio,
                 output=print, show_json=False, capture=None, streaming_asr=False, benchmark=None):
        self.llm, self.registry, self.dispatcher, self.manager = llm, registry, dispatcher, manager
        self.asr, self.recorder, self.output, self.show_json = asr, recorder, output, show_json
        self.capture, self.streaming_asr, self.benchmark = capture, streaming_asr, benchmark

    def text(self, text):
        if not text.strip():
            return DispatchResult("rejected", "Empty input")
        ticket = self.manager.begin_interaction()
        return self._decide(ticket, text, time.monotonic())

    def voice(self, device, duration):
        with ExitStack() as session:
            with self.manager.recording() as allowed:
                if not allowed:
                    detail = "Recording blocked while speech is pending/playing or local stop is active"
                    self.output(detail)
                    return DispatchResult("rejected", detail)
                ticket = self.manager.begin_interaction()
                started = time.monotonic()
                trace = ticket.trace
                realtime = None
                try:
                    if self.streaming_asr:
                        realtime = session.enter_context(self.asr.session(ticket, self.capture.frame_ms))
                    if self.capture:
                        self.output("Speak now..." if self.capture.vad else f"Recording {duration:g} seconds...")
                        options = {"consume": realtime.append, "check": realtime.check} if realtime else {}
                        audio = self.capture.record(device, duration, ticket, **options)
                    else:
                        self.output(f"Recording {duration:g} seconds...")
                        trace.mark("recording_started")
                        try:
                            audio = self.recorder(device, duration)
                        finally:
                            trace.mark("recording_ended")
                except KeyboardInterrupt:
                    self._interrupted(ticket)
                    raise
                except ASRServiceError as exc:
                    return self._failure(ticket, "ASR", str(exc))
                except CaptureError as exc:
                    return self._failure(ticket, "audio", str(exc))
                except Exception as exc:
                    return self._failure(ticket, "audio", type(exc).__name__)
            self.manager.set_runtime("THINKING")
            if not self.manager.accepts(ticket):
                return self._failure(ticket, "ASR", "Interaction cancelled or expired")
            try:
                asr_started = time.monotonic()
                try:
                    if realtime:
                        text = realtime.final()
                    else:
                        trace.mark("asr_requested")
                        text = self.asr.transcribe(audio)
                        trace.mark("asr_final")
                finally:
                    logger.info("interaction=%s asr_seconds=%.3f", ticket.id, time.monotonic() - asr_started)
            except KeyboardInterrupt:
                self._interrupted(ticket)
                raise
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
        if not self.manager.accepts(ticket):
            return self._failure(ticket, "LLM", "Interaction cancelled or expired")
        self.manager.set_runtime("THINKING")
        llm_started = time.monotonic()
        try:
            motion, lighting = self.manager.capabilities_ready()
            capabilities = self.registry.prompt_data(motion_ready=motion, lighting_ready=lighting)
            ticket.trace.mark("llm_requested")
            raw = self.llm.decide(text, capabilities)
            ticket.trace.mark("llm_returned")
        except KeyboardInterrupt:
            self._interrupted(ticket)
            raise
        except LLMError as exc:
            return self._failure(ticket, "LLM", str(exc))
        except Exception as exc:
            return self._failure(ticket, "LLM", type(exc).__name__)
        finally:
            logger.info("interaction=%s llm_seconds=%.3f", ticket.id, time.monotonic() - llm_started)
        event, result = self.dispatcher.dispatch(ticket, raw)
        self._observed(ticket, result)
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

    def _interrupted(self, ticket):
        self.manager.local_stop()
        self._observed(ticket, DispatchResult("cancelled", "Local interrupt"))

    def _failure(self, ticket, stage, detail):
        self.manager.discard(ticket)
        self.manager.set_runtime("ERROR")
        detail = redact_secrets(detail)
        logger.error("interaction=%s stage=%s failed: %s", ticket.id, stage, detail)
        self.output(f"[{ticket.id}] {stage} failed: {detail}")
        return self._observed(ticket, DispatchResult("rejected", detail))

    def _observed(self, ticket, result):
        observe(ticket, result, self.benchmark)
        return result


def _option(cli, env):
    return env_flag(env) if cli is None else cli


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
        for module in (__name__, "lelamp.behavior", "lelamp.audio.realtime_asr"):
            logging.getLogger(module).setLevel(logging.DEBUG)
    if args.mode == "chat" and (args.vad is not None or args.streaming_asr is not None):
        parser.error("--vad and --streaming-asr require ai mode")
    robot = manager = llm = asr = tts = capture = benchmark = None
    code = 0
    try:
        voice = args.mode == "ai"
        provider = os.getenv("ASR_PROVIDER", "bailian_http").strip().lower()
        if voice and args.streaming_asr is None and provider not in ("bailian_http", "dashscope_realtime"):
            raise ValueError("ASR_PROVIDER must be bailian_http or dashscope_realtime")
        streaming_asr = voice and (provider == "dashscope_realtime" if args.streaming_asr is None
                                   else args.streaming_asr)
        vad = voice and _option(args.vad, "VAD_ENABLED")
        streaming_tts = _option(args.streaming_tts, "TTS_STREAMING_ENABLED")
        reuse = _option(args.reuse_audio, "AUDIO_REUSE_ENABLED")
        cache_motion = args.enable_motion and _option(args.motion_cache, "MOTION_CACHE_ENABLED")
        recording_cache = RecordingCache() if cache_motion else None
        if voice and (vad or streaming_asr or reuse):
            settings = VADSettings.from_env() if vad else None
            if settings:
                settings.detector()  # Fail at startup if the optional dependency is absent.
            capture = MicrophoneCapture(settings, reuse=reuse)
        label = args.benchmark_label or "/".join((
            "vad" if vad else "fixed", "realtime-asr" if streaming_asr else "http-asr",
            "realtime-tts" if streaming_tts else "configured-tts", "reuse" if reuse else "per-turn",
            "motion-cache" if cache_motion else "no-motion-cache"))
        if args.benchmark:
            benchmark = Benchmark(label, args.benchmark_output)
        llm = LLMClient.from_env()
        if voice:
            asr = RealtimeASR.from_env() if streaming_asr else BailianASR()
        try:
            tts = RealtimeTTS.from_env(reuse=reuse) if streaming_tts else tts_from_env()
        except Exception as exc:
            detail = redact_secrets(str(exc)) if isinstance(exc, ValueError) else type(exc).__name__
            logger.error("TTS configuration failed (%s). Replies will appear in the terminal.", detail)
            if args.benchmark:
                raise ValueError("Requested TTS could not be configured for the benchmark") from None
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
                                      scale=args.scale, lighting=lighting, tts_mode=tts.mode if tts else None,
                                      recording_cache=recording_cache)
        motion = MockMotion()
        if args.enable_motion:
            input("Support the lamp and clear its travel area. Press Enter to arm verified motion: ")
            robot = connect_robot(args.id, args.port, profile, hold_on_error=True)
            motion = RecordedMotion(DemoPlayer(robot, profile, recording_cache=recording_cache), args.scale)
        else:
            print("Motion is simulated. Lighting is simulated unless a verified backend is initialized.")
        manager = BehaviorManager(motion, lighting, tts)
        dispatcher = CommandDispatcher(registry, manager, motion, lighting, tts)
        runtime = AIRuntime(llm, registry, dispatcher, manager, asr=asr, show_json=args.mode == "chat",
                            capture=capture, streaming_asr=streaming_asr, benchmark=benchmark)
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
        for resource in (capture, tts, getattr(asr, "client", None), getattr(llm, "client", None)):
            close = getattr(resource, "close", None)
            if close:
                try:
                    close()
                except Exception as exc:
                    logger.error("Speech resource cleanup failed (%s)", type(exc).__name__)
                    code = 1
        if benchmark:
            try:
                benchmark.report()
            except OSError as exc:
                logger.error("Benchmark output could not be written (%s)", type(exc).__name__)
                code = 1
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
