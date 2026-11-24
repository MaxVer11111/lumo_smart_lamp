"""Opt-in live model evaluation with simulated behavior execution."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from ..behavior.capabilities import CapabilityRegistry
from ..behavior.dispatcher import CommandDispatcher
from ..behavior.manager import BehaviorManager
from ..behavior.motion import MockMotion
from ..lighting.controller import LightingController
from .llm_client import LLMClient, LLMError
from .schemas import BehaviorEvent, InvalidEvent

CASES = Path(__file__).with_name("eval_cases.jsonl")


def read_cases(path=CASES):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_labels(action):
    if action.type == "motion":
        return {f"motion:{action.name}"}
    if action.type == "lighting":
        label = f"lighting:{action.operation}"
        return {label} if action.value is None else {label, f"{label}:{action.value:g}"}
    return {"stop"}


def actions_match(event, alternatives):
    actual = [action_labels(action) for action in event.actions]
    return any(len(expected) == len(actual) and all(any(label in labels for labels in actual)
                                                   for label in expected)
               for expected in alternatives)


def evaluate(llm, cases, verbose=False):
    valid = intent_correct = action_correct = false_actions = no_action_cases = request_errors = 0
    llm_times, end_to_end = [], []
    motion, lighting = MockMotion(), LightingController()
    registry = CapabilityRegistry(simulated=True, lighting=lighting)
    manager = BehaviorManager(motion, lighting)
    dispatcher = CommandDispatcher(registry, manager, motion, lighting)
    try:
        for case in cases:
            lighting.set_mode("default")
            lighting.turn_off()
            if case.get("no_lighting"):
                registry.lighting = None
            else:
                registry.lighting = lighting
            began = time.monotonic()
            ticket = manager.begin_interaction()
            llm_start = time.monotonic()
            event = None
            status = "llm_error"
            try:
                raw = llm.decide(case["text"], registry.prompt_data())
            except LLMError:
                request_errors += 1
            else:
                try:
                    event = BehaviorEvent.parse(raw)
                except InvalidEvent:
                    status = "invalid_json_or_schema"
                else:
                    valid += 1
            finally:
                llm_times.append(time.monotonic() - llm_start)
            if event:
                _, result = dispatcher.dispatch(ticket, raw)
                statuses = [job.result(timeout=5).status for job in result.jobs]
                status = result.status
                if status in ("accepted", "success") and all(s == "success" for s in statuses):
                    action_correct += actions_match(event, case["actions"])
                intent_correct += event.intent in case["intents"]
            if case["actions"] == [[]]:
                no_action_cases += 1
                if event is not None and event.actions:
                    false_actions += 1
            end_to_end.append(time.monotonic() - began)
            if verbose:
                print(f"{case['id']} {status}")
    finally:
        manager.close()
    return {
        "cases": len(cases),
        "intent_accuracy": intent_correct / len(cases),
        "action_selection_accuracy": action_correct / len(cases),
        "json_validity_rate": valid / len(cases),
        "false_action_trigger_rate": false_actions / max(1, no_action_cases),
        "average_llm_latency_seconds": mean(llm_times),
        "average_end_to_end_interaction_latency_seconds": mean(end_to_end),
        "llm_request_errors": request_errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize configured cloud API requests")
    parser.add_argument("--verbose", action="store_true", help="Show per-case validation status")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("--live is required to send requests to the configured LLM endpoint")
    load_dotenv(override=False)
    try:
        metrics = evaluate(LLMClient.from_env(), read_cases(), args.verbose)
    except ValueError:
        parser.error("Check LLM settings and the evaluation cases")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["intent_accuracy"] >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
