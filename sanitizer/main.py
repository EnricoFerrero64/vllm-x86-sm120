"""
Tool-call JSON sanitizer proxy for vLLM.

Sits between LiteLLM (or any OpenAI-compatible client) and vLLM.
Fixes malformed tool_call arguments in conversation history before they
reach vLLM's json.loads() call.

Problem: Qwen3.6 / qwen3_coder parser occasionally emits trailing garbage:
    {"workflowId": "abc"} }
                          ^-- extra brace
When this gets stored in history and replayed, vLLM raises:
    json.JSONDecodeError: Extra data: line 1 column 35

Flow: LiteLLM → sanitizer:8001 → vLLM:8000
"""

import json
import logging
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

VLLM_BASE = os.getenv("VLLM_BASE_URL", "http://aeon_vllm:8000")
LOG_FIXES = os.getenv("LOG_FIXES", "1") == "1"

app = FastAPI(title="vllm-tool-sanitizer")
log = logging.getLogger("sanitizer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── JSON sanitizer ────────────────────────────────────────────────────────────

def _fix_json(s: str) -> str:
    """Strip trailing garbage after valid JSON.

    json.JSONDecodeError for 'Extra data' has .pos pointing to the start
    of the extra data.  Truncating there and stripping whitespace gives
    the valid JSON prefix.
    """
    if not s:
        return s
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError as e:
        if e.msg == "Extra data" and e.pos > 0:
            candidate = s[: e.pos].rstrip()
            try:
                json.loads(candidate)
                if LOG_FIXES:
                    log.warning("Fixed malformed tool_call args: %r → %r", s, candidate)
                return candidate
            except json.JSONDecodeError:
                pass
    return s  # can't fix — return as-is, let vLLM surface the error


def _sanitize_messages(messages: list) -> tuple[list, int]:
    """Walk messages and fix malformed tool_call arguments. Returns (messages, fix_count)."""
    fixes = 0
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            raw = fn.get("arguments")
            if isinstance(raw, str):
                fixed = _fix_json(raw)
                if fixed != raw:
                    fn["arguments"] = fixed
                    fixes += 1
    return messages, fixes


# ── Request forwarding ────────────────────────────────────────────────────────

async def _stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in response.aiter_bytes():
        yield chunk


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    # Only sanitize chat completions POST requests
    if request.method == "POST" and "chat/completions" in path and body:
        try:
            data = json.loads(body)
            if "messages" in data:
                data["messages"], n = _sanitize_messages(data["messages"])
                if n:
                    body = json.dumps(data).encode()
        except (json.JSONDecodeError, TypeError):
            pass  # not JSON or unexpected shape — forward as-is

    url = f"{VLLM_BASE}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    async with httpx.AsyncClient(timeout=None) as client:
        req = client.build_request(
            method=request.method,
            url=url,
            content=body,
            headers=headers,
        )
        response = await client.send(req, stream=True)

        # Preserve streaming for SSE (text/event-stream)
        if "text/event-stream" in response.headers.get("content-type", ""):
            return StreamingResponse(
                _stream_response(response),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="text/event-stream",
            )

        content = await response.aread()
        return Response(
            content=content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
