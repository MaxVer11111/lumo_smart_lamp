"""Bounded, cancellable scheduling with a single writer per hardware lane."""

import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum

from .performance import Trace, current_trace

logger = logging.getLogger(__name__)


def _trace_lane(lane):
    return "tts" if lane == "audio" else lane


class Priority(IntEnum):
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


@dataclass(frozen=True)
class Interaction:
    id: str
    expires: float
    cancel: threading.Event = field(default_factory=threading.Event, compare=False)
    trace: Trace = field(default_factory=Trace, compare=False)


@dataclass(frozen=True)
class DispatchResult:
    status: str
    detail: str = ""
    jobs: tuple[Future, ...] = ()


@dataclass
class Job:
    ticket: Interaction
    lane: str
    label: str
    priority: Priority
    run: object
    cancel: threading.Event = field(default_factory=threading.Event)
    result: Future = field(default_factory=Future)


class BehaviorManager:
    def __init__(self, motion, lighting=None, tts=None, *, queue_size=2, interaction_ttl=90,
                 clock=time.monotonic):
        if queue_size < 1 or interaction_ttl <= 0:
            raise ValueError("Queue size and interaction lifetime must be positive")
        self._motion = motion
        self._devices = {"motion": motion, "lighting": lighting, "audio": tts}
        self._pending = {lane: deque() for lane in self._devices}
        self._active = {}
        self._faults = set()
        self._condition = threading.Condition()
        self._clock = clock
        self._ttl = interaction_ttl
        self._queue_size = queue_size
        self._latest = None
        self._used = True
        self._closed = self._stopped = self._listening = False
        self._runtime = "IDLE"
        self._workers = []
        try:
            for lane, device in self._devices.items():
                if device is not None:
                    worker = threading.Thread(target=self._worker, args=(lane,), name=f"lumo-{lane}")
                    worker.start()
                    self._workers.append(worker)
        except BaseException:
            self.close()
            raise

    def begin_interaction(self):
        with self._condition:
            if self._latest is not None:
                self._latest.cancel.set()
            ticket = Interaction(uuid.uuid4().hex[:12], self._clock() + self._ttl)
            self._latest = ticket if not self._stopped and not self._closed else None
            self._used = False
            return ticket

    def capabilities_ready(self):
        with self._condition:
            return tuple(self._ready(lane) for lane in ("motion", "lighting"))

    def _ready(self, lane):
        return (self._devices[lane] is not None and lane not in self._faults
                and not self._stopped and not self._closed)

    def state(self):
        with self._condition:
            return {
                "runtime": "STOPPED" if self._stopped else self._runtime,
                "motion": self._lane_state("motion", "EXECUTING"),
                "lighting": self._lane_state("lighting", "EXECUTING"),
                "audio": "LISTENING" if self._listening else self._lane_state("audio", "SPEAKING"),
            }

    def _lane_state(self, lane, active):
        if lane in self._faults:
            return "ERROR"
        if self._stopped:
            return "STOPPED"
        return active if lane in self._active else "IDLE"

    def set_runtime(self, state):
        with self._condition:
            self._runtime = state

    @contextmanager
    def recording(self):
        with self._condition:
            allowed = (not self._closed and not self._stopped and not self._listening
                       and "audio" not in self._active and not self._pending["audio"])
            if allowed:
                self._listening = True
        try:
            yield allowed
        finally:
            if allowed:
                with self._condition:
                    self._listening = False
                    self._condition.notify_all()

    def _accepts(self, ticket):
        return (not self._closed and not self._stopped and ticket is self._latest
                and not self._used and not ticket.cancel.is_set() and self._clock() < ticket.expires)

    def accepts(self, ticket):
        with self._condition:
            return self._accepts(ticket)

    def discard(self, ticket):
        with self._condition:
            if self._accepts(ticket):
                self._used = True
                ticket.cancel.set()

    def submit(self, ticket, jobs, *, stop=False):
        dropped = []
        with self._condition:
            if self._closed or self._stopped:
                return DispatchResult("rejected", "Local stop is active")
            if not self._accepts(ticket):
                return DispatchResult("rejected", "Interaction is stale, expired or already dispatched")
            self._used = True
            if stop:
                ticket.trace.mark("dispatched")
                dropped = self._stop_locked()
                result = DispatchResult("success", "P0 stop requested")
            else:
                queues = self._pending.copy()
                accepted = []
                for job in jobs:
                    if not self._ready(job.lane):
                        return DispatchResult("unsupported", f"{job.lane} unavailable")
                    now = self._clock()
                    queue = queues[job.lane]
                    retained = deque(task for task in queue if task.ticket.expires > now
                                     and not self._supersedes(job, task))
                    if len(retained) >= self._queue_size:
                        if job.lane == "audio":
                            ticket.trace.increment("tts_queue_skipped")
                            continue
                        return DispatchResult("rejected", f"{job.lane} queue is full")
                    dropped.extend(task for task in queue if task not in retained)
                    retained.append(job)
                    queues[job.lane] = retained
                    accepted.append(job)
                self._pending = queues
                ticket.trace.mark("dispatched")
                for job in accepted:
                    ticket.trace.mark(f"{_trace_lane(job.lane)}_submitted")
                    if active := self._active.get(job.lane):
                        if active.priority > job.priority or (
                                job.lane == "audio" and self._supersedes(job, active)):
                            self._cancel_active(active)
                self._condition.notify_all()
                result = DispatchResult("accepted" if accepted else "success",
                                        jobs=tuple(job.result for job in accepted))
        self._finish_dropped(dropped)
        return result

    @staticmethod
    def _supersedes(job, queued):
        return (queued.priority > job.priority or
                (queued.ticket is not job.ticket and queued.priority == job.priority
                 and (job.lane == "audio" or
                      (job.lane == "motion" and job.priority == Priority.P2))))

    def local_stop(self):
        with self._condition:
            self._stopped = True
            dropped = self._stop_locked()
        self._finish_dropped(dropped)
        logger.info("priority=P0 local_stop=latched")

    def resume(self):
        with self._condition:
            if self._closed or self._active or self._listening or "motion" in self._faults:
                return False
            self._stopped = False
            return True

    def _stop_locked(self):
        if self._latest is not None:
            self._latest.cancel.set()
        self._latest = None
        self._used = True
        for active in self._active.values():
            self._cancel_active(active)
        dropped = [job for queue in self._pending.values() for job in queue]
        for queue in self._pending.values():
            queue.clear()
        self._condition.notify_all()
        return dropped

    def _cancel_active(self, job):
        if job.cancel.is_set():
            return
        if job.lane == "motion":
            self._motion.stop(job.cancel)
        else:
            job.cancel.set()

    def _finish_dropped(self, dropped):
        for job in dropped:
            job.cancel.set()
            self._finish(job, "cancelled", "removed from queue")

    @staticmethod
    def _finish(job, status, detail=""):
        job.ticket.trace.mark(f"{_trace_lane(job.lane)}_finished")
        logger.info("interaction=%s %s=%s priority=%s status=%s %s", job.ticket.id, job.lane,
                    job.label, job.priority.name, status, detail)
        try:
            job.result.set_result(DispatchResult(status, detail))
        except InvalidStateError:
            # Cancelling the result handle only stops the caller's observation.
            pass

    def _worker(self, lane):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (
                    self._pending[lane] and (lane != "audio" or not self._listening)))
                if self._closed:
                    return
                queue = self._pending[lane]
                job = min(queue, key=lambda task: task.priority)
                queue.remove(job)
                self._active[lane] = job

            started = self._clock()
            dropped = []
            detail = ""
            trace_context = current_trace.set(job.ticket.trace)
            try:
                if job.cancel.is_set() or started >= job.ticket.expires:
                    status, detail = "cancelled", "stopped or expired before execution"
                else:
                    job.ticket.trace.mark(f"{_trace_lane(lane)}_started")
                    completed = job.run(job.cancel)
                    status = "success" if completed is not False and not job.cancel.is_set() else "cancelled"
            except Exception as exc:
                status, detail = "failed", type(exc).__name__
                if lane == "audio":
                    detail += " (reply available in terminal)"
                else:
                    with self._condition:
                        self._faults.add(lane)
                        dropped = list(self._pending[lane])
                        self._pending[lane].clear()
            finally:
                current_trace.reset(trace_context)
                with self._condition:
                    del self._active[lane]
                    self._condition.notify_all()
            logger.info("interaction=%s %s_seconds=%.3f", job.ticket.id, lane, self._clock() - started)
            self._finish(job, status, detail)
            self._finish_dropped(dropped)

    def wait_idle(self, timeout=None):
        with self._condition:
            return self._condition.wait_for(lambda: not self._active and not any(self._pending.values()), timeout)

    def close(self):
        with self._condition:
            self._closed = self._stopped = True
            dropped = self._stop_locked()
        self._finish_dropped(dropped)
        for worker in self._workers:
            worker.join()
