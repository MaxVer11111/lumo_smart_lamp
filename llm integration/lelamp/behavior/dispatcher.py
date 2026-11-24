"""Validate the entire event before constructing any executable job."""

from ..intelligence.schemas import BehaviorEvent, InvalidEvent
from .manager import DispatchResult, Job, Priority


class UnsupportedAction(ValueError):
    pass


class CommandDispatcher:
    def __init__(self, registry, manager, motion, lighting=None, tts=None):
        self.registry = registry
        self.manager = manager
        self.motion = motion
        self.lighting = lighting
        self.tts = tts

    def validate(self, raw, *, motion_ready=True, lighting_ready=True):
        event = BehaviorEvent.parse(raw)
        for action in event.actions:
            if action.type == "motion":
                capability = self.registry.motions.get(action.name)
                if capability is None:
                    raise InvalidEvent("Unregistered motion")
                if not capability.enabled or not motion_ready:
                    raise UnsupportedAction("Motion is unavailable or lacks local authorization")
                flag = capability.allow_autonomous if event.intent == "emotion_expression" else capability.allow_explicit
                if not flag:
                    raise UnsupportedAction("Motion is unavailable for that kind of interaction")
            if action.type == "lighting" and (self.registry.lighting is None or not lighting_ready):
                raise UnsupportedAction("Lighting is unavailable")
        return event

    def dispatch(self, ticket, raw):
        if not self.manager.accepts(ticket):
            return None, DispatchResult("rejected", "Interaction is stale, expired or already dispatched")
        motion_ready, lighting_ready = self.manager.capabilities_ready()
        try:
            event = self.validate(raw, motion_ready=motion_ready, lighting_ready=lighting_ready)
        except InvalidEvent as exc:
            self.manager.discard(ticket)
            return None, DispatchResult("rejected", str(exc))
        except UnsupportedAction as exc:
            self.manager.discard(ticket)
            return None, DispatchResult("unsupported", str(exc))

        finally:
            ticket.trace.mark("validation_finished")

        ticket.trace.mark("validated")
        if event.intent == "stop":
            return event, self.manager.submit(ticket, (), stop=True)
        if event.intent == "emotion_expression":
            priority = Priority.P2
        elif event.intent in ("conversation", "no_action"):
            priority = Priority.P3
        else:
            priority = Priority.P1
        jobs = []
        for action in event.actions:
            if action.type == "motion":
                capability = self.registry.motions[action.name]

                def play(cancel, capability=capability):
                    return self.motion.play(capability, cancel)

                jobs.append(Job(ticket, "motion", action.name, priority, play))
            elif action.type == "lighting":

                def light(cancel, action=action):
                    if cancel.is_set():
                        return False
                    self.lighting.execute(action.operation, action.value)
                    return True

                jobs.append(Job(ticket, "lighting", action.operation, priority, light))
        if self.tts and event.speech.strip():
            def speak(cancel):
                if cancel.is_set():
                    return False
                return self.tts.speak(event.speech, cancel, ticket.id)

            jobs.append(Job(ticket, "audio", "speech", priority, speak))
        return event, self.manager.submit(ticket, jobs)
