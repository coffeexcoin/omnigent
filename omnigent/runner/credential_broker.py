"""Process-local credential broker for Git and GitHub CLI wrappers.

The trusted runner owns this bridge. Agent subprocesses receive only an opaque
session capability and a loopback endpoint; they never receive actor identity
in their environment. Each request is rebound to the currently active turn
before an addon credential provider is called.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import secrets
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from omnigent.policies.schema import ActorContext

BROKER_ENDPOINT_ENV = "OMNIGENT_CREDENTIAL_BROKER_ENDPOINT"
BROKER_CAPABILITY_ENV = "OMNIGENT_CREDENTIAL_BROKER_CAPABILITY"
REAL_GIT_ENV = "OMNIGENT_REAL_GIT"
REAL_GH_ENV = "OMNIGENT_REAL_GH"
WRAPPER_PYTHON_ENV = "OMNIGENT_CREDENTIAL_WRAPPER_PYTHON"
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_CREDENTIAL_TTL_SECONDS = 15 * 60
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveCredentialTurn:
    """Immutable identity used for one brokered command request."""

    session_id: str
    turn_id: str
    actor: ActorContext


@dataclass(frozen=True)
class CredentialRequest:
    """Secret-free command metadata supplied to a credential provider."""

    tool: Literal["git", "gh"]
    action: str
    operation: Literal["identity", "credential"]
    protocol: str | None = None
    host: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class CredentialGrant:
    """Short-lived provider response. Secret values are excluded from repr."""

    username: str
    secret: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    git_user_name: str | None = None
    git_user_email: str | None = None


@dataclass(frozen=True)
class CredentialAuditEvent:
    """Secret-free audit record emitted for every authorized broker request."""

    session_id: str
    turn_id: str
    actor: ActorContext
    tool: Literal["git", "gh"]
    action: str
    operation: Literal["identity", "credential"]
    outcome: Literal["allowed", "denied", "error"]


class CredentialProvider(Protocol):
    """Addon seam that mints credentials for the active actor."""

    async def issue(
        self, context: ActiveCredentialTurn, request: CredentialRequest
    ) -> CredentialGrant: ...


AuditSink = Callable[[CredentialAuditEvent], Awaitable[None] | None]


class CredentialBrokerBridge:
    """Loopback broker binding opaque session capabilities to active turns."""

    def __init__(
        self,
        provider: CredentialProvider,
        *,
        audit_sink: AuditSink | None = None,
        wrapper_dir: Path | None = None,
    ) -> None:
        self._provider = provider
        self._audit_sink = audit_sink
        self._active_turns: dict[str, ActiveCredentialTurn] = {}
        self._session_capabilities: dict[str, str] = {}
        self._capability_sessions: dict[str, str] = {}
        self._server: asyncio.AbstractServer | None = None
        self._endpoint: str | None = None
        self._start_lock = asyncio.Lock()
        self._owned_wrapper_dir = wrapper_dir is None
        self._wrapper_dir = wrapper_dir or Path(tempfile.mkdtemp(prefix="omnigent-vcs-wrappers-"))

    @property
    def endpoint(self) -> str | None:
        """Return the bound loopback endpoint after :meth:`start`."""

        return self._endpoint

    async def start(self) -> None:
        """Bind the private loopback request server and materialize wrappers."""

        if self._server is not None:
            return
        async with self._start_lock:
            if self._server is not None:
                return
            self._write_wrappers()
            self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
            socket = self._server.sockets[0]
            host, port = socket.getsockname()[:2]
            self._endpoint = f"{host}:{port}"

    async def close(self) -> None:
        """Close the bridge and erase capabilities and generated wrappers."""

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._server = None
        self._endpoint = None
        self._active_turns.clear()
        self._session_capabilities.clear()
        self._capability_sessions.clear()
        if self._owned_wrapper_dir:
            shutil.rmtree(self._wrapper_dir, ignore_errors=True)

    def bind_turn(self, context: ActiveCredentialTurn) -> None:
        """Make *context* authoritative for subsequent session requests."""

        self._active_turns[context.session_id] = ActiveCredentialTurn(
            session_id=context.session_id,
            turn_id=context.turn_id,
            actor=context.actor.copy(),
        )

    def clear_turn(self, session_id: str, *, turn_id: str | None = None) -> None:
        """Clear a turn without allowing stale cleanup to remove its successor."""

        current = self._active_turns.get(session_id)
        if current is not None and (turn_id is None or current.turn_id == turn_id):
            self._active_turns.pop(session_id, None)

    def wrapper_environment(
        self,
        session_id: str,
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return environment overrides that expose wrappers, never credentials."""

        if self._endpoint is None:
            raise RuntimeError("credential broker bridge is not started")
        capability = self._session_capabilities.get(session_id)
        if capability is None:
            capability = secrets.token_urlsafe(32)
            self._session_capabilities[session_id] = capability
            self._capability_sessions[capability] = session_id

        source = base_env or os.environ
        original_path = source.get("PATH", os.defpath)
        env = {
            "PATH": f"{self._wrapper_dir}{os.pathsep}{original_path}",
            BROKER_ENDPOINT_ENV: self._endpoint,
            BROKER_CAPABILITY_ENV: capability,
            WRAPPER_PYTHON_ENV: sys.executable,
        }
        real_git = shutil.which("git", path=original_path)
        real_gh = shutil.which("gh", path=original_path)
        if real_git is not None:
            env[REAL_GIT_ENV] = real_git
        if real_gh is not None:
            env[REAL_GH_ENV] = real_gh
        return env

    def revoke_session(self, session_id: str) -> None:
        """Revoke a session capability and clear any active turn."""

        self._active_turns.pop(session_id, None)
        capability = self._session_capabilities.pop(session_id, None)
        if capability is not None:
            self._capability_sessions.pop(capability, None)

    def _write_wrappers(self) -> None:
        self._wrapper_dir.mkdir(parents=True, exist_ok=True)
        for tool in ("git", "gh"):
            target = self._wrapper_dir / tool
            target.write_text(
                "#!/bin/sh\n"
                f'exec "${{{WRAPPER_PYTHON_ENV}:-python3}}" '
                f'-m omnigent.runner.credential_wrapper {tool} "$@"\n',
                encoding="utf-8",
            )
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not raw or len(raw) > _MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                raise ValueError("invalid broker request framing")
            payload = json.loads(raw)
            response = await self._dispatch(payload)
        except PermissionError as exc:
            response = {"ok": False, "error": str(exc)[:300]}
        except Exception:  # noqa: BLE001 - never reflect provider errors or secrets
            response = {"ok": False, "error": "credential broker request failed"}
        writer.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        with contextlib.suppress(ConnectionError, BrokenPipeError):
            await writer.drain()
        writer.close()
        with contextlib.suppress(ConnectionError, BrokenPipeError):
            await writer.wait_closed()

    async def _dispatch(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("broker request must be an object")
        capability = payload.get("capability")
        if not isinstance(capability, str):
            raise PermissionError("missing broker capability")
        session_id = self._capability_sessions.get(capability)
        if session_id is None:
            raise PermissionError("invalid broker capability")
        context = self._active_turns.get(session_id)
        if context is None:
            raise PermissionError("no active turn for credential request")

        request = _parse_request(payload)
        outcome: Literal["allowed", "denied", "error"] = "error"
        try:
            try:
                provider_context = ActiveCredentialTurn(
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    actor=context.actor.copy(),
                )
                grant = await self._provider.issue(provider_context, request)
            except PermissionError:
                outcome = "denied"
                raise PermissionError("credential provider denied request") from None
            except Exception as exc:
                raise RuntimeError("credential provider failed") from exc
            _validate_grant(request, grant)
            outcome = "allowed"
            return {
                "ok": True,
                "grant": {
                    "username": grant.username,
                    "secret": grant.secret,
                    "expires_at": grant.expires_at,
                    "git_user_name": grant.git_user_name,
                    "git_user_email": grant.git_user_email,
                },
            }
        finally:
            event = CredentialAuditEvent(
                session_id=context.session_id,
                turn_id=context.turn_id,
                actor=context.actor.copy(),
                tool=request.tool,
                action=request.action,
                operation=request.operation,
                outcome=outcome,
            )
            _logger.info(
                "credential_broker action session=%r turn=%r actor=%r tool=%s "
                "action=%r operation=%s outcome=%s",
                event.session_id,
                event.turn_id,
                event.actor.get("run_as", ""),
                event.tool,
                event.action,
                event.operation,
                event.outcome,
            )
            if self._audit_sink is not None:
                result = self._audit_sink(event)
                if inspect.isawaitable(result):
                    await result


def _parse_request(payload: dict[str, object]) -> CredentialRequest:
    tool = payload.get("tool")
    operation = payload.get("operation")
    action = payload.get("action")
    if tool not in ("git", "gh"):
        raise ValueError("unsupported credential tool")
    if operation not in ("identity", "credential"):
        raise ValueError("unsupported credential operation")
    if not isinstance(action, str) or not action or len(action) > 128:
        raise ValueError("credential action must be 1-128 characters")

    optional: dict[str, str | None] = {}
    for key in ("protocol", "host", "path"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 2048):
            raise ValueError(f"credential {key} must be a bounded string")
        optional[key] = value
    return CredentialRequest(
        tool=tool,
        operation=operation,
        action=action,
        protocol=optional["protocol"],
        host=optional["host"],
        path=optional["path"],
    )


def _validate_grant(request: CredentialRequest, grant: CredentialGrant) -> None:
    if not grant.username:
        raise ValueError("credential provider returned an empty username")
    if request.operation == "credential" and request.action == "get" and not grant.secret:
        raise ValueError("credential provider returned no secret")
    if request.tool == "gh" and not grant.secret:
        raise ValueError("credential provider returned no GitHub token")
    if grant.secret is not None:
        now = time.time()
        if grant.expires_at is None or not (
            now < grant.expires_at <= now + _MAX_CREDENTIAL_TTL_SECONDS
        ):
            raise ValueError("credential provider returned a stale or long-lived credential")
