"""Small, auditable Jev image-goal evaluation client (no training or control).

Uses TypeSafe's documented HTTP multipart API without an SDK or automatic
retries. Images and typed decisions are experimental as of 2026-10-18.
Official contracts: https://docs.typesafe.ai/images,
https://docs.typesafe.ai/api, https://docs.typesafe.ai/primitives/noul,
https://docs.typesafe.ai/primitives/choice and https://docs.typesafe.ai/models.

The receipt never contains authentication headers. The caller must provide
only images and permitted observations: benchmark labels are not inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
import socket
import struct
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib

MODEL = "jev-1.14.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INPUT_USD_PER_MILLION = 0.042
PRICE_CHECKED_DATE = "2026-10-18"
PRESENCE_NO = 0.2
PRESENCE_YES = 0.8
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024

# Keep prompts unchanged across evaluated cases and modalities. Different task
# contracts should be frozen as separate benchmark protocols before API calls.
QUESTIONS = {
    "presence": {
        "type": "noul",
        "instructions": (
            "Is the target object specified by `task` and shown in `goal_image` "
            "present in the current scene? Use `current_image` and any supplied "
            "`measured_state`. Do not confuse a different object with the target. "
            "Occlusion or missing measurements can leave presence uncertain."
        ),
        "criteria": {
            "true": "The target object is present in the current scene.",
            "false": "The target object is absent from the current scene.",
        },
    },
    "completion": {
        "type": "noul",
        "instructions": (
            "Has the physical goal in `task` actually been achieved now? "
            "Compare `current_image` with `goal_image` and use any supplied "
            "`measured_state`. Matching appearance alone does not prove hidden "
            "support, release, stability, identity or dwell requirements. "
            "Return the probability of physical completion; express uncertainty "
            "when the provided evidence cannot establish those requirements."
        ),
        "criteria": {
            "true": "All physical requirements explicitly defined in `task` are satisfied.",
            "false": "At least one physical requirement explicitly defined in `task` is not satisfied.",
        },
    },
    "verdict": {
        "type": "choice",
        "instructions": (
            "What conclusion about completion of `task` is justified by "
            "`current_image`, `goal_image` and any supplied `measured_state`? "
            "Assess only the supplied evidence. Do not invent unseen telemetry "
            "or assume that a single image establishes stability over time."
        ),
        "criteria": {
            "complete": "Available evidence establishes every physical requirement in `task`.",
            "incomplete": "Available evidence establishes that at least one physical requirement in `task` is not satisfied.",
            "unknown": "No observed failure is established, but the available evidence cannot establish every physical requirement in `task`.",
        },
    },
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class PreparedRequest:
    request: dict[str, Any]
    body: bytes
    content_type: str
    images: dict[str, dict[str, Any]]
    request_sha256: str


def _png_metadata(data: bytes) -> dict[str, Any]:
    """Check still-PNG framing, CRCs and service size limits before uploading."""
    if not isinstance(data, bytes) or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be PNG bytes no larger than 16 MiB")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Only PNG inputs are accepted by this benchmark client")
    position, width, height, saw_idat, saw_end = 8, 0, 0, False, False
    while position + 12 <= len(data):
        length = struct.unpack_from(">I", data, position)[0]
        end = position + 12 + length
        if end > len(data):
            raise ValueError("Truncated PNG chunk")
        kind = data[position + 4:position + 8]
        chunk = data[position + 8:position + 8 + length]
        actual_crc = struct.unpack_from(">I", data, position + 8 + length)[0]
        if zlib.crc32(kind + chunk) & 0xffffffff != actual_crc:
            raise ValueError("Invalid PNG checksum")
        if position == 8:
            if kind != b"IHDR" or length != 13:
                raise ValueError("PNG must start with a valid IHDR chunk")
            width, height = struct.unpack_from(">II", chunk)
        elif kind == b"IHDR":
            raise ValueError("Duplicate PNG image header")
        if kind in (b"acTL", b"fcTL", b"fdAT"):
            raise ValueError("Animated PNG is not supported by Jev images")
        if kind == b"IDAT":
            saw_idat = True
        if kind == b"IEND":
            if length or end != len(data):
                raise ValueError("Invalid PNG end marker")
            saw_end = True
            break
        position = end
    if not saw_idat or not saw_end:
        raise ValueError("Incomplete PNG image")
    if not (280 <= width <= 1568 and 280 <= height <= 1568):
        raise ValueError("Jev image width and height must each be 280 through 1568 pixels")
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data),
            "width": width, "height": height, "format": "png"}


def build_goal_request(current_png: bytes, goal_png: bytes, *, task: str,
                       measured_state: dict[str, Any] | None = None) -> PreparedRequest:
    """Build a deterministic, credential-free request for freezing and review."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A nonempty physical task contract is required")
    if measured_state is not None and not isinstance(measured_state, dict):
        raise ValueError("measured_state must be an observation dictionary or None")
    images = {"current": _png_metadata(current_png), "goal": _png_metadata(goal_png)}
    refs = {name: f"dvidia-image-{name}-{value['sha256']}" for name, value in images.items()}
    state: dict[str, Any] = {"task": task, "current_image": refs["current"],
                             "goal_image": refs["goal"]}
    if measured_state is not None:
        state["measured_state"] = measured_state
    payload = {"model": MODEL, "state": state,
               "assets": {refs[name]: {"type": "image", "part": name,
                                        "format": "png"} for name in images},
               "questions": QUESTIONS}
    # JSON round-trip freezes caller-owned structures and rejects NaN/Infinity.
    request_bytes = _json_bytes(payload)
    payload = json.loads(request_bytes)
    boundary = "dvidia-" + sha256(request_bytes).hexdigest()
    pieces = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
              "Content-Type: application/json\r\n\r\n".encode(), request_bytes, b"\r\n"]
    for name, data in (("current", current_png), ("goal", goal_png)):
        if boundary.encode() in data:
            raise ValueError("Multipart boundary collides with an image")
        pieces.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                       f"filename=\"{name}.png\"\r\nContent-Type: image/png\r\n\r\n".encode(),
                       data, b"\r\n"])
    pieces.append(f"--{boundary}--\r\n".encode())
    body = b"".join(pieces)
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("Multipart request exceeds 32 MiB")
    return PreparedRequest(payload, body, f"multipart/form-data; boundary={boundary}",
                           images, sha256(body).hexdigest())


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be finite and between zero and one")
    return value


def parse_goal_response(response: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed responses instead of silently coercing them to negatives."""
    if not isinstance(response, dict) or response.get("model") != MODEL:
        raise ValueError("Response does not identify the pinned Jev model")
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("Missing answer map")
    probabilities = {}
    for name in ("presence", "completion"):
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError(f"Missing {name} Noul answer")
        probabilities[name] = _probability(answer.get("noul"), name)
    verdict = answers.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("type") != "choice":
        raise ValueError("Missing verdict Choice answer")
    choices = verdict.get("probabilities")
    options = {"complete", "incomplete", "unknown"}
    if not isinstance(choices, dict) or set(choices) != options:
        raise ValueError("Verdict probabilities must cover complete, incomplete and unknown")
    choices = {name: _probability(value, name) for name, value in choices.items()}
    if not math.isclose(sum(choices.values()), 1.0, abs_tol=0.005):
        raise ValueError("Verdict probabilities do not sum to one")
    selected = verdict.get("choice")
    if selected not in options or choices[selected] < max(choices.values()) - 0.005:
        raise ValueError("Verdict selection disagrees with probabilities")
    confidence = _probability(verdict.get("confidence"), "verdict confidence")
    usage_fields = _usage_fields(response)
    p_present = probabilities["presence"]
    return {"p_present": p_present, "p_complete": probabilities["completion"],
            "presence": ("present" if p_present >= PRESENCE_YES else
                         "absent" if p_present <= PRESENCE_NO else "unknown"),
            "verdict": selected, "verdict_probabilities": choices,
            "verdict_confidence": confidence, **usage_fields}


def _usage_fields(response: dict[str, Any]) -> dict[str, Any]:
    """Usage can remain valid even when a billed response cannot be scored."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        raise ValueError("Missing token usage")
    for name in ("input_tokens", "output_tokens"):
        if type(usage.get(name)) is not int or usage[name] < 0:
            raise ValueError(f"Invalid {name} usage")
    return {"usage": {name: usage[name] for name in ("input_tokens", "output_tokens")},
            "estimated_input_cost_usd": usage["input_tokens"] * INPUT_USD_PER_MILLION / 1e6}


def _retain_usage(receipt: dict[str, Any], response: Any) -> None:
    try:
        receipt.update(_usage_fields(response))
    except ValueError:
        pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {_redact(k, secret): _redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secret) for v in value]
    return value


def load_api_key(*, keychain_service: str | None = None,
                 keychain_account: str | None = None) -> str | None:
    """Read env first, then only an explicitly named macOS Keychain item.

    Never enumerates Keychain entries, writes credentials, or logs output.
    Interactive Keychain access denial is reported without revealing stderr.
    """
    configured = os.environ.get("TYPESAFE_API_KEY")
    if configured:
        return configured
    if keychain_service is None and keychain_account is None:
        return None
    if not keychain_service or not keychain_account:
        raise ValueError("Both Keychain service and account are required")
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", keychain_service,
             "-a", keychain_account, "-w"], capture_output=True, text=True,
            timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout.strip() or None


class JevClient:
    """One call per evaluate, never retry or follow redirects.

    timeout_s is the socket-operation timeout (including connect/read), not a
    real-time deadline guarantee. The receipt measures actual wall time. DNS,
    TLS, upload and network/server delay are all inside the HTTP timing.
    """
    def __init__(self, api_key: str | None = None, *, timeout_s: float = 20.0):
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError("timeout_s must be positive and at most 60 seconds")
        self._api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        if not isinstance(self._api_key, str) or any(c in self._api_key for c in "\r\n"):
            raise ValueError("Invalid API credential")
        self.timeout_s = float(timeout_s)
        self._opener = build_opener(_NoRedirect())

    def evaluate(self, current_png: bytes, goal_png: bytes, *, task: str,
                 measured_state: dict[str, Any] | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        receipt: dict[str, Any] = {
            "status": "error", "model": MODEL,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": None, "request_sha256": None, "images": None,
            "response": None, "response_sha256": None, "http_status": None,
            "latency_ms": {"prepare": None, "http": None, "total": None},
            "p_present": None, "p_complete": None, "presence": "unknown",
            "verdict": "unknown", "verdict_probabilities": None,
            "verdict_confidence": None, "usage": None,
            "estimated_input_cost_usd": None, "error": None,
            "socket_timeout_s": self.timeout_s, "attempts": 0,
        }
        try:
            prepared = build_goal_request(current_png, goal_png, task=task,
                                          measured_state=measured_state)
            receipt.update(request=prepared.request, request_sha256=prepared.request_sha256,
                           images=prepared.images, request_bytes=len(prepared.body))
            receipt["latency_ms"]["prepare"] = (time.perf_counter() - start) * 1000
            if not self._api_key:
                receipt["status"] = "missing_credentials"
                receipt["error"] = {"kind": "missing_credentials", "message": "TYPESAFE_API_KEY is not configured"}
                return receipt
            request = Request(ENDPOINT, data=prepared.body, method="POST", headers={
                "Authorization": f"Bearer {self._api_key}",
                "TypeSafe-Danger-Zone-Images": "true",
                "Content-Type": prepared.content_type,
                "Accept": "application/json", "Accept-Encoding": "identity",
                "User-Agent": "DVIDIA-Image-Goal-Benchmark/0.1",
            })
            receipt["attempts"] = 1
            http_start = time.perf_counter()
            try:
                with self._opener.open(request, timeout=self.timeout_s) as response:
                    receipt["http_status"] = response.status
                    body = response.read(MAX_RESPONSE_BYTES + 1)
            finally:
                receipt["latency_ms"]["http"] = (time.perf_counter() - http_start) * 1000
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Provider response exceeds the 1 MiB benchmark limit")
            receipt["response_sha256"] = sha256(body).hexdigest()
            decoded = json.loads(body)
            receipt["response"] = _redact(decoded, self._api_key)
            _retain_usage(receipt, decoded)
            receipt.update(parse_goal_response(decoded))
            receipt["status"] = "ok"
        except HTTPError as error:
            receipt["http_status"] = error.code
            receipt["error"] = {"kind": "http_error", "message": f"TypeSafe returned HTTP {error.code}; no automatic retry"}
            # Preserve bounded JSON error evidence, but never server text echoing auth.
            try:
                body = error.read(MAX_RESPONSE_BYTES + 1)
                if len(body) <= MAX_RESPONSE_BYTES:
                    receipt["response_sha256"] = sha256(body).hexdigest()
                    decoded = json.loads(body)
                    receipt["response"] = _redact(decoded, self._api_key)
                    _retain_usage(receipt, decoded)
            except (ValueError, OSError):
                pass
            receipt["latency_ms"]["http"] = (time.perf_counter() - http_start) * 1000
        except (TimeoutError, socket.timeout):
            receipt["error"] = {"kind": "timeout", "message": "TypeSafe socket operation timed out; no automatic retry"}
        except URLError as error:
            kind = "timeout" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "network_error"
            receipt["error"] = {"kind": kind, "message": "TypeSafe request could not complete; no automatic retry"}
        except (ValueError, TypeError, UnicodeError) as error:
            receipt["error"] = {"kind": "invalid_response" if receipt["attempts"] else "invalid_request",
                                "message": _redact(str(error), self._api_key)}
        except OSError:
            receipt["error"] = {"kind": "network_error", "message": "TypeSafe transport failed; no automatic retry"}
        finally:
            receipt["latency_ms"]["total"] = (time.perf_counter() - start) * 1000
        return receipt
