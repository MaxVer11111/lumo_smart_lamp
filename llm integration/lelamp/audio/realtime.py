"""Shared transport for the DashScope realtime speech APIs."""

import asyncio
import json
import math
import os
import uuid
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException


class RealtimeError(RuntimeError):
    pass


class RealtimeCancelled(RealtimeError):
    pass


def timeout_from_env(name, default):
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not 0 < value <= 120:
        raise ValueError(f"{name} must be positive and at most 120 seconds")
    return value


def endpoint(url, base_url, model):
    if not model.strip():
        raise ValueError("A DashScope realtime model is required")
    if not url:
        host = urlsplit(base_url).netloc or "dashscope.aliyuncs.com"
        url = f"wss://{host}/api-ws/v1/realtime"
    parts = urlsplit(url)
    if (parts.scheme != "wss" or not parts.hostname
            or parts.username is not None or parts.fragment):
        raise ValueError("Expected a secure DashScope WebSocket URL")
    query = dict(parse_qsl(parts.query))
    query["model"] = model
    return urlunsplit(parts._replace(query=urlencode(query)))


@asynccontextmanager
async def connection(url, key, timeout, *, retry=False):
    deadline = asyncio.get_running_loop().time() + timeout
    headers = {"Authorization": f"Bearer {key}", "OpenAI-Beta": "realtime=v1"}
    for attempt in range(2 if retry else 1):
        try:
            ws = await connect(url, additional_headers=headers, open_timeout=max(
                .001, deadline - asyncio.get_running_loop().time()),
                close_timeout=.25, max_size=256 * 1024, max_queue=4)
        except OSError:
            if not retry or attempt or asyncio.get_running_loop().time() >= deadline:
                raise RealtimeError("Could not connect to DashScope realtime") from None
        except (TimeoutError, WebSocketException, ValueError):
            raise RealtimeError("DashScope realtime connection failed; check endpoint and API key") from None
        else:
            break
    try:
        yield ws
    except (OSError, WebSocketException):
        raise RealtimeError("DashScope realtime connection was interrupted") from None
    finally:
        await ws.close()


async def send(ws, kind, **fields):
    await ws.send(json.dumps({"event_id": uuid.uuid4().hex, "type": kind, **fields}))


async def receive(ws):
    try:
        event = json.loads(await ws.recv())
    except (TypeError, ValueError):
        raise RealtimeError("DashScope returned a malformed realtime event") from None
    if not isinstance(event, dict) or event.get("type") == "error":
        raise RealtimeError("DashScope rejected the realtime speech request")
    return event


async def wait_for_event(ws, kind):
    while (event := await receive(ws)).get("type") != kind:
        if event.get("type") == "session.finished":
            raise RealtimeError("DashScope ended the session before configuration completed")
    return event


async def cancellable(operation, cancelled):
    task = asyncio.create_task(operation)
    try:
        while not cancelled():
            done, _ = await asyncio.wait((task,), timeout=.025)
            if done:
                if not cancelled():
                    return task.result()
                break
        raise RealtimeCancelled("Realtime speech task cancelled")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
