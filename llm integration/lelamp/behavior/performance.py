"""Monotonic interaction tracing and JSONL benchmark summaries."""

import argparse
import json
import logging
import math
import statistics
import threading
import time
from contextvars import ContextVar

logger = logging.getLogger(__name__)
current_trace = ContextVar("interaction_trace", default=None)

METRICS = {
    "recording_ms": ("recording_started", "recording_ended"),
    "endpoint_latency_ms": ("speech_ended", "endpoint_detected"),
    "asr_final_latency_ms": ("recording_ended", "asr_final"),
    "asr_total_ms": ("asr_requested", "asr_final"),
    "audio_after_capture_ms": ("recording_ended", "tts_first_output"),
    "motion_after_capture_ms": ("recording_ended", "motion_first_command"),
    "trajectory_preparation_ms": ("trajectory_started", "motion_prepared"),
    "recording_adaptation_ms": ("adaptation_started", "adaptation_finished"),
    "llm_total_ms": ("llm_requested", "llm_returned"),
    "dispatch_latency_ms": ("llm_returned", "dispatched"),
    "motion_queue_ms": ("motion_submitted", "motion_started"),
    "motion_preparation_ms": ("motion_started", "motion_prepared"),
    "tts_queue_ms": ("tts_submitted", "tts_started"),
    "tts_ttfa_ms": ("tts_requested", "tts_first_pcm"),
    "first_motion_command_latency_ms": ("speech_ended", "motion_first_command"),
    "first_audio_output_latency_ms": ("speech_ended", "tts_first_output"),
}


class Trace:
    def __init__(self):
        self.started = time.monotonic()
        self.events = {}
        self.counts = {}
        self.lock = threading.Lock()

    def mark(self, event, when=None):
        with self.lock:
            self.events.setdefault(event, time.monotonic() if when is None else when)

    def increment(self, name, amount=1):
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + amount

    def snapshot(self):
        with self.lock:
            events = self.events.copy()
            counts = self.counts.copy()

        def elapsed(start, end):
            if start in events and end in events:
                return round((events[end] - events[start]) * 1000, 3)
            return None

        metrics = {name: elapsed(*pair) for name, pair in METRICS.items()}
        outputs = [value for key, value in metrics.items()
                   if key in ("first_motion_command_latency_ms", "first_audio_output_latency_ms")
                   and value is not None]
        metrics["end_to_end_ms"] = min(outputs, default=None)
        return {
            "events_ms": {key: round((value - self.started) * 1000, 3) for key, value in events.items()},
            "metrics": metrics, "counts": counts,
        }


def mark(event, when=None):
    if trace := current_trace.get():
        trace.mark(event, when)


def summarize(rows):
    metrics = {}
    for name in (*METRICS, "end_to_end_ms"):
        values = sorted(row["metrics"][name] for row in rows if row["metrics"].get(name) is not None)
        if values:
            metrics[name] = {"n": len(values), "p50": round(statistics.median(values), 3),
                             "p95": values[math.ceil(len(values) * .95) - 1],
                             "mean": round(statistics.fmean(values), 3)}
    failures = sum(row["status"] not in ("accepted", "success")
                   or any(job != "success" for job in row["jobs"])
                   or bool(row.get("counts", {}).get("tts_queue_skipped")) for row in rows)
    return {"n": len(rows), "failures": failures, "failure_rate": failures / len(rows) if rows else None,
            "metrics": metrics}


class Benchmark:
    def __init__(self, label, output=None):
        self.label, self.output = label, output
        self.rows = []
        self.lock = threading.Lock()
        self.sequence = 0

    def reserve(self):
        with self.lock:
            self.sequence += 1
            return self.sequence

    def add(self, row, sequence):
        with self.lock:
            self.rows.append({
                **row, "label": self.label, "sequence": sequence,
                "phase": "cold" if sequence == 1 else "warm",
            })

    def report(self):
        with self.lock:
            rows = sorted(self.rows, key=lambda row: row["sequence"])
        if self.output:
            with open(self.output, "a", encoding="utf-8") as output:
                for row in rows:
                    print(json.dumps(row), file=output)
        logger.info("benchmark=%s", json.dumps(summarize_groups(rows)))


def observe(ticket, result, benchmark=None):
    sequence = benchmark.reserve() if benchmark else None
    pending = set(result.jobs)
    lock = threading.Lock()

    def finish():
        row = {
            "interaction_id": ticket.id, "status": result.status,
            "jobs": [job.result().status if not job.cancelled() else "unobserved" for job in result.jobs],
            **ticket.trace.snapshot(),
        }
        logger.info("performance=%s", json.dumps(row, separators=(",", ":")))
        if benchmark:
            benchmark.add(row, sequence)

    def completed(job):
        with lock:
            pending.discard(job)
            if not pending:
                finish()

    if pending:
        for job in result.jobs:
            job.add_done_callback(completed)
    else:
        finish()


def summarize_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], {}).setdefault(row["phase"], []).append(row)
    return {label: {phase: summarize(group) for phase, group in phases.items()}
            for label, phases in groups.items()}


def comparisons(groups, baseline):
    comparison = {}
    for label, phases in groups.items():
        if label == baseline:
            continue
        for phase, optimized in phases.items():
            original = groups[baseline].get(phase)
            if original is None:
                continue
            metrics = {}
            for name in original["metrics"].keys() & optimized["metrics"].keys():
                before, after = original["metrics"][name], optimized["metrics"][name]
                metrics[name] = {
                    stat: round(100 * (before[stat] - after[stat]) / before[stat], 2)
                    if before[stat] else None for stat in ("p50", "p95", "mean")
                }
            comparison.setdefault(label, {})[phase] = {
                "improvement_pct": metrics,
                "failure_rate_delta_pp": round(100 * (optimized["failure_rate"] - original["failure_rate"]), 2),
            }
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Summarize LUMO benchmark JSONL files")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--baseline", help="Calculate changes against this run label")
    args = parser.parse_args()
    rows = []
    for path in args.files:
        with open(path, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    groups = summarize_groups(rows)
    if args.baseline and args.baseline not in groups:
        parser.error("Baseline label was not found in the input files")
    report = {"runs": groups}
    if args.baseline:
        report["comparison"] = comparisons(groups, args.baseline)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
