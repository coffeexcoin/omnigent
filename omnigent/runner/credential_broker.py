"""Process-local credential broker for Git and GitHub CLI wrappers.

The trusted runner owns this bridge. Agent subprocesses receive only an opaque
actor-scoped capability and a loopback endpoint; they never receive actor
identity, credentials, or trusted executable paths in their environment. Each
request is rebound to the currently active turn before an addon credential
provider is called.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

from omnigent.policies.schema import ActorContext

BROKER_ENDPOINT_ENV = "OMNIGENT_CREDENTIAL_BROKER_ENDPOINT"
BROKER_CAPABILITY_ENV = "OMNIGENT_CREDENTIAL_BROKER_CAPABILITY"
WRAPPER_PYTHON_ENV = "OMNIGENT_CREDENTIAL_WRAPPER_PYTHON"
_MAX_REQUEST_BYTES = 3 * 1024 * 1024
_MAX_STDIN_BYTES = 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_ARGUMENTS = 512
_MAX_ARGUMENT_BYTES = 64 * 1024
_MAX_CREDENTIAL_TTL_SECONDS = 15 * 60
_DEFAULT_PROVIDER_TIMEOUT_SECONDS = 10.0
_DEFAULT_EXECUTION_TIMEOUT_SECONDS = 120.0
_GIT_NETWORK_ACTIONS = frozenset({"clone", "fetch", "ls-remote", "push"})
_GIT_LOCAL_ACTIONS = frozenset(
    {
        "add",
        "apply",
        "bisect",
        "blame",
        "branch",
        "cat-file",
        "checkout",
        "cherry-pick",
        "clean",
        "commit",
        "describe",
        "diff",
        "diff-tree",
        "for-each-ref",
        "format-patch",
        "grep",
        "init",
        "log",
        "merge",
        "merge-base",
        "mv",
        "name-rev",
        "rebase",
        "reflog",
        "remote",
        "reset",
        "restore",
        "rev-list",
        "rev-parse",
        "rm",
        "show",
        "show-ref",
        "sparse-checkout",
        "stash",
        "status",
        "switch",
        "symbolic-ref",
        "tag",
        "update-index",
        "worktree",
    }
)
_GH_ACTIONS = frozenset(
    {
        "api",
        "attestation",
        "cache",
        "gist",
        "issue",
        "label",
        "org",
        "pr",
        "project",
        "release",
        "repo",
        "ruleset",
        "run",
        "search",
        "status",
        "variable",
        "workflow",
    }
)
_GITHUB_HOST = "github.com"
_GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DENIED_GH_FLAGS = frozenset(
    {
        "--browser",
        "--clone",
        "--editor",
        "--git-protocol",
        "--pager",
        "--ssh-key",
        "--web",
        "-w",
    }
)
_DENIED_GH_SUBCOMMANDS = frozenset(
    {("gist", "clone"), ("pr", "checkout"), ("repo", "clone"), ("repo", "fork")}
)
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
    actor: ActorContext | None = None
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


@dataclass(frozen=True)
class _ExecutionRequest:
    tool: Literal["git", "gh"]
    action: str
    argv: tuple[str, ...]
    cwd: str
    stdin: bytes
    host: str
    provider_request: CredentialRequest


@dataclass(frozen=True)
class _CapabilityBinding:
    """Principal that originally received one opaque subprocess capability."""

    session_id: str
    turn_id: str | None
    actor: ActorContext | None


class CredentialBrokerBridge:
    """Loopback broker binding opaque session capabilities to active turns."""

    def __init__(
        self,
        provider: CredentialProvider,
        *,
        audit_sink: AuditSink | None = None,
        wrapper_dir: Path | None = None,
        provider_timeout: float = _DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        execution_timeout: float = _DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        if provider_timeout <= 0 or execution_timeout <= 0:
            raise ValueError("broker timeouts must be positive")
        self._provider = provider
        self._audit_sink = audit_sink
        self._provider_timeout = provider_timeout
        self._execution_timeout = execution_timeout
        self._base_path = os.environ.get("PATH", os.defpath)
        self._active_turns: dict[str, ActiveCredentialTurn] = {}
        self._session_capabilities: dict[str, str] = {}
        self._capability_bindings: dict[str, _CapabilityBinding] = {}
        self._sessions_requiring_rotation: set[str] = set()
        self._running_processes: dict[str, set[subprocess.Popen[bytes]]] = {}
        self._state_lock = threading.RLock()
        self._executables = {
            tool: executable
            for tool in ("git", "gh")
            if (executable := shutil.which(tool)) is not None
        }
        self._server: asyncio.AbstractServer | None = None
        self._endpoint: str | None = None
        self._start_lock = asyncio.Lock()
        self._owned_wrapper_dir = wrapper_dir is None
        self._wrapper_dir = wrapper_dir or Path(tempfile.mkdtemp(prefix="omnigent-vcs-wrappers-"))
        self._artifact_dir = self._wrapper_dir / "invocations"

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
        with self._state_lock:
            for session_id in list(self._running_processes):
                self._terminate_session_processes(session_id)
        self._active_turns.clear()
        self._session_capabilities.clear()
        self._capability_bindings.clear()
        if self._owned_wrapper_dir:
            shutil.rmtree(self._wrapper_dir, ignore_errors=True)
        else:
            shutil.rmtree(self._artifact_dir, ignore_errors=True)

    def bind_turn(self, context: ActiveCredentialTurn) -> None:
        """Make *context* authoritative for subsequent session requests."""

        with self._state_lock:
            session_id = context.session_id
            self._terminate_session_processes(session_id)
            if session_id in self._session_capabilities:
                self._sessions_requiring_rotation.add(session_id)
            self._revoke_capabilities(session_id)
            self._active_turns[session_id] = ActiveCredentialTurn(
                session_id=context.session_id,
                turn_id=context.turn_id,
                actor=context.actor.copy(),
            )

    def clear_turn(self, session_id: str, *, turn_id: str) -> None:
        """Clear a turn without allowing stale cleanup to remove its successor."""

        with self._state_lock:
            current = self._active_turns.get(session_id)
            if current is not None and current.turn_id == turn_id:
                self._terminate_session_processes(session_id)
                self._active_turns.pop(session_id, None)
                if session_id in self._session_capabilities:
                    self._sessions_requiring_rotation.add(session_id)
                self._revoke_capabilities(session_id)

    def deactivate_session(self, session_id: str) -> None:
        """Remove credential authority when a new actorless turn begins."""

        with self._state_lock:
            self._terminate_session_processes(session_id)
            self._active_turns.pop(session_id, None)
            if session_id in self._session_capabilities:
                self._sessions_requiring_rotation.add(session_id)
            self._revoke_capabilities(session_id)

    def active_turn_id(self, session_id: str) -> str | None:
        """Return the current turn id for compatibility status fencing."""

        current = self._active_turns.get(session_id)
        return current.turn_id if current is not None else None

    def owns_turn(self, session_id: str, turn_id: str) -> bool:
        """Return whether *turn_id* is still authoritative for the session."""

        current = self._active_turns.get(session_id)
        return current is not None and current.turn_id == turn_id

    def wrapper_environment(
        self,
        session_id: str,
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return environment overrides that expose wrappers, never credentials."""

        if self._endpoint is None:
            raise RuntimeError("credential broker bridge is not started")
        with self._state_lock:
            context = self._active_turns.get(session_id)
            turn_id = context.turn_id if context is not None else None
            actor = context.actor.copy() if context is not None else None
            capability = self._session_capabilities.get(session_id)
            binding = self._capability_bindings.get(capability) if capability is not None else None
            if (
                capability is None
                or binding is None
                or binding.turn_id != turn_id
                or binding.actor != actor
            ):
                self._revoke_capabilities(session_id)
                capability = secrets.token_urlsafe(32)
                self._session_capabilities[session_id] = capability
                self._capability_bindings[capability] = _CapabilityBinding(
                    session_id=session_id,
                    turn_id=turn_id,
                    actor=actor,
                )
                self._sessions_requiring_rotation.discard(session_id)

        source = base_env or os.environ
        original_path = source.get("PATH", os.defpath)
        return {
            "PATH": f"{self._wrapper_dir}{os.pathsep}{original_path}",
            BROKER_ENDPOINT_ENV: self._endpoint,
            BROKER_CAPABILITY_ENV: capability,
            WRAPPER_PYTHON_ENV: sys.executable,
        }

    def requires_process_rotation(self, session_id: str) -> bool:
        """Return whether the live subprocess capability belongs to another actor."""

        return session_id in self._sessions_requiring_rotation

    def revoke_session(self, session_id: str) -> None:
        """Revoke a session capability and clear any active turn."""

        self._active_turns.pop(session_id, None)
        self._revoke_capabilities(session_id)
        self._sessions_requiring_rotation.discard(session_id)

    def _revoke_capabilities(self, session_id: str) -> None:
        """Revoke the current subprocess capability for one session."""

        capability = self._session_capabilities.pop(session_id, None)
        if capability is not None:
            self._capability_bindings.pop(capability, None)
        stale_capabilities = [
            capability
            for capability, binding in self._capability_bindings.items()
            if binding.session_id == session_id
        ]
        for capability in stale_capabilities:
            self._capability_bindings.pop(capability, None)

    def _write_wrappers(self) -> None:
        self._wrapper_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dir.mkdir(mode=0o700, exist_ok=True)
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
        with self._state_lock:
            binding = self._capability_bindings.get(capability)
            if binding is None:
                raise PermissionError("invalid or expired broker capability")
            session_id = binding.session_id
            context = self._active_turns.get(session_id)
            if context is None:
                raise PermissionError("no active turn for credential request")
            if binding.turn_id != context.turn_id or binding.actor != context.actor:
                raise PermissionError("broker capability does not own the active turn")

        execution = _parse_execution_request(payload)
        request = execution.provider_request
        outcome: Literal["allowed", "denied", "error"] = "error"
        try:
            try:
                provider_context = ActiveCredentialTurn(
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    actor=context.actor.copy(),
                )
                grant = await asyncio.wait_for(
                    self._provider.issue(provider_context, request),
                    timeout=self._provider_timeout,
                )
            except PermissionError:
                outcome = "denied"
                raise PermissionError("credential provider denied request") from None
            except Exception as exc:
                raise RuntimeError("credential provider failed") from exc
            _validate_grant(request, grant, expected_actor=context.actor)
            executable = self._executables.get(request.tool)
            if executable is None:
                raise RuntimeError(f"trusted {request.tool} executable is unavailable")
            if not self._authorization_is_current(capability, context):
                raise PermissionError("active credential turn changed during authorization")
            try:
                result = await asyncio.to_thread(
                    self._execute,
                    execution,
                    grant,
                    executable,
                    capability,
                    context,
                )
            except asyncio.CancelledError:
                with self._state_lock:
                    self._terminate_session_processes(context.session_id)
                raise
            outcome = "allowed"
            return {"ok": True, "result": result}
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

    def _execute(
        self,
        execution: _ExecutionRequest,
        grant: CredentialGrant,
        executable: str,
        capability: str,
        context: ActiveCredentialTurn,
    ) -> dict[str, object]:
        """Run one bounded command inside the trusted broker boundary."""

        with tempfile.TemporaryDirectory(
            prefix=f"{execution.tool}-",
            dir=self._artifact_dir,
        ) as root:
            invocation_dir = Path(root)
            env = _sanitized_environment(self._base_path)
            env["HOME"] = str(invocation_dir)
            env["TMPDIR"] = str(invocation_dir)
            if execution.tool == "git":
                command = _git_command(executable, execution, grant, invocation_dir, env)
            else:
                command = _gh_command(executable, execution, grant, invocation_dir, env)
            stdout_path = invocation_dir / "stdout"
            stderr_path = invocation_dir / "stderr"
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    with self._state_lock:
                        if not self._authorization_is_current(capability, context):
                            raise PermissionError(
                                "active credential turn changed before command execution"
                            )
                        if execution.provider_request.operation == "credential":
                            process = subprocess.Popen(
                                command,
                                cwd=execution.cwd,
                                env=env,
                                stdin=subprocess.PIPE,
                                stdout=stdout,
                                stderr=stderr,
                                start_new_session=True,
                            )
                            self._running_processes.setdefault(context.session_id, set()).add(
                                process
                            )
                        else:
                            process = None
                    if process is not None:
                        try:
                            process.communicate(
                                input=execution.stdin,
                                timeout=self._execution_timeout,
                            )
                        except subprocess.TimeoutExpired:
                            self._terminate_process(process)
                            process.communicate()
                            raise
                        finally:
                            with self._state_lock:
                                processes = self._running_processes.get(context.session_id)
                                if processes is not None:
                                    processes.discard(process)
                                    if not processes:
                                        self._running_processes.pop(context.session_id, None)
                        returncode = process.returncode
                    else:
                        completed = subprocess.run(
                            command,
                            cwd=execution.cwd,
                            env=env,
                            input=execution.stdin,
                            stdout=stdout,
                            stderr=stderr,
                            check=False,
                            timeout=self._execution_timeout,
                        )
                        returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("brokered command timed out") from exc
            if stdout_path.stat().st_size > _MAX_OUTPUT_BYTES:
                raise RuntimeError("brokered stdout exceeded limit")
            if stderr_path.stat().st_size > _MAX_OUTPUT_BYTES:
                raise RuntimeError("brokered stderr exceeded limit")
            stdout_bytes = stdout_path.read_bytes()
            stderr_bytes = stderr_path.read_bytes()
            if grant.secret:
                secret = grant.secret.encode()
                stdout_bytes = stdout_bytes.replace(secret, b"[REDACTED]")
                stderr_bytes = stderr_bytes.replace(secret, b"[REDACTED]")
        return {
            "returncode": returncode,
            "stdout": base64.b64encode(stdout_bytes).decode("ascii"),
            "stderr": base64.b64encode(stderr_bytes).decode("ascii"),
        }

    def _authorization_is_current(
        self,
        capability: str,
        context: ActiveCredentialTurn,
    ) -> bool:
        """Atomically revalidate one exact-turn authorization snapshot."""

        with self._state_lock:
            binding = self._capability_bindings.get(capability)
            return (
                binding is not None
                and binding.session_id == context.session_id
                and binding.turn_id == context.turn_id
                and binding.actor == context.actor
                and self._active_turns.get(context.session_id) == context
            )

    def _terminate_session_processes(self, session_id: str) -> None:
        """Terminate every credentialed child still running for one session."""

        for process in tuple(self._running_processes.get(session_id, ())):
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        """Terminate a brokered process group without exposing its environment."""

        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            with contextlib.suppress(OSError):
                process.kill()


def _parse_execution_request(payload: dict[str, object]) -> _ExecutionRequest:
    tool = payload.get("tool")
    operation = payload.get("operation")
    if tool not in ("git", "gh"):
        raise ValueError("unsupported credential tool")
    if operation != "execute":
        raise PermissionError("raw credential requests are unavailable")
    raw_argv = payload.get("argv")
    if not isinstance(raw_argv, list) or len(raw_argv) > _MAX_ARGUMENTS:
        raise ValueError("broker argv must be a bounded list")
    if any(not isinstance(value, str) or "\x00" in value for value in raw_argv):
        raise ValueError("broker argv contains an invalid argument")
    argv = tuple(raw_argv)
    if sum(len(value.encode()) for value in argv) > _MAX_ARGUMENT_BYTES:
        raise ValueError("broker argv is too large")
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not cwd or len(cwd) > 4096:
        raise ValueError("broker cwd must be a bounded path")
    cwd_path = Path(cwd)
    if not cwd_path.is_absolute() or not cwd_path.is_dir():
        raise ValueError("broker cwd must be an existing absolute directory")
    raw_stdin = payload.get("stdin", "")
    if not isinstance(raw_stdin, str):
        raise ValueError("broker stdin must be base64 text")
    try:
        stdin = base64.b64decode(raw_stdin, validate=True)
    except ValueError as exc:
        raise ValueError("broker stdin must be valid base64") from exc
    if len(stdin) > _MAX_STDIN_BYTES:
        raise ValueError("broker stdin is too large")
    raw_host = payload.get("host", _GITHUB_HOST)
    if raw_host != _GITHUB_HOST:
        raise PermissionError("only github.com credentials are available")
    host = _GITHUB_HOST
    action = _git_action(argv) if tool == "git" else _gh_action(argv)
    if tool == "git":
        if action not in _GIT_LOCAL_ACTIONS | _GIT_NETWORK_ACTIONS:
            raise PermissionError(f"git {action} is unavailable through the credential broker")
        provider_operation: Literal["identity", "credential"] = (
            "credential" if action in _GIT_NETWORK_ACTIONS else "identity"
        )
        repository = _git_repository(argv, action) if provider_operation == "credential" else None
    else:
        if action not in _GH_ACTIONS:
            raise PermissionError(f"gh {action} is unavailable through the credential broker")
        provider_operation = "credential"
        repository = _validate_gh_arguments(argv, action)
    provider_request = CredentialRequest(
        tool=tool,
        operation=provider_operation,
        action=action,
        protocol="https" if provider_operation == "credential" else None,
        host=host if provider_operation == "credential" else None,
        path=repository,
    )
    return _ExecutionRequest(tool, action, argv, cwd, stdin, host, provider_request)


def _git_repository(argv: Sequence[str], action: str) -> str:
    """Require credentialed Git operations to name an explicit GitHub HTTPS URL."""

    _, command = _split_git_global_args(argv)
    value_options = {"--branch", "-b", "--depth", "--filter", "--upload-pack", "-u"}
    skip = False
    target: str | None = None
    for value in command[1:]:
        if skip:
            skip = False
            continue
        if value in value_options:
            skip = True
            continue
        if value.startswith("-"):
            if value.startswith(("--upload-pack=", "--receive-pack=", "--exec=")):
                raise PermissionError("custom Git transport executables are unavailable")
            continue
        target = value
        break
    if target is None:
        raise ValueError(f"git {action} requires an explicit GitHub HTTPS URL")
    parsed = urlparse(target)
    if parsed.scheme != "https" or parsed.hostname != _GITHUB_HOST:
        raise PermissionError("credentialed Git remotes must use https://github.com")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise PermissionError("GitHub remote URL contains unsupported components")
    repository = parsed.path.strip("/")
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not _GITHUB_REPOSITORY.fullmatch(repository):
        raise PermissionError("GitHub remote must identify one owner/repository")
    return repository


def _validate_gh_arguments(argv: Sequence[str], action: str) -> str | None:
    """Reject host switching, interactive escapes, and malformed repo selectors."""

    repository: str | None = None
    action_index = argv.index(action)
    if action_index + 1 < len(argv) and (action, argv[action_index + 1]) in _DENIED_GH_SUBCOMMANDS:
        raise PermissionError("Git-spawning gh subcommands are unavailable")
    for index, value in enumerate(argv):
        flag = value.split("=", 1)[0]
        if flag in _DENIED_GH_FLAGS:
            raise PermissionError(f"gh option {flag} is unavailable")
        if flag == "--hostname":
            hostname = (
                value.split("=", 1)[1]
                if "=" in value
                else (argv[index + 1] if index + 1 < len(argv) else "")
            )
            if hostname != _GITHUB_HOST:
                raise PermissionError("gh may only address github.com")
        if flag in {"-R", "--repo"}:
            repository = (
                value.split("=", 1)[1]
                if "=" in value
                else (argv[index + 1] if index + 1 < len(argv) else "")
            )
            if not _GITHUB_REPOSITORY.fullmatch(repository):
                raise PermissionError("gh repository must be owner/repository")
    return repository


def _split_git_global_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Allow only non-configuring Git global options before the subcommand."""

    index = 0
    while index < len(argv):
        item = argv[index]
        if not item.startswith("-") or item == "-":
            break
        if item == "-C":
            if index + 1 >= len(argv):
                raise ValueError("git -C requires a path")
            index += 2
            continue
        if item in {"--no-pager", "--literal-pathspecs", "--no-optional-locks"}:
            index += 1
            continue
        raise PermissionError(f"git global option {item[:128]} is unavailable")
    return list(argv[:index]), list(argv[index:])


def _git_action(argv: Sequence[str]) -> str:
    _, command = _split_git_global_args(argv)
    return command[0][:128] if command else "unknown"


def _gh_action(argv: Sequence[str]) -> str:
    """Return a gh subcommand without treating a global option value as one."""

    index = 0
    while index < len(argv):
        item = argv[index]
        if not item.startswith("-") or item == "-":
            return item[:128]
        if item in {"-R", "--repo", "--hostname"}:
            if index + 1 >= len(argv):
                raise ValueError(f"gh {item} requires a value")
            index += 2
            continue
        if item.startswith(("--repo=", "--hostname=")):
            index += 1
            continue
        if item in {"--help", "--version"}:
            return "unknown"
        raise PermissionError(f"gh global option {item[:128]} is unavailable")
    return "unknown"


def _sanitized_environment(path: str) -> dict[str, str]:
    """Build a minimal child environment with no runner or caller credentials."""

    env = {"PATH": path}
    for key in ("LANG", "LC_ALL"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


def _set_git_config(env: dict[str, str], entries: Sequence[tuple[str, str]]) -> None:
    env["GIT_CONFIG_COUNT"] = str(len(entries))
    for index, (key, value) in enumerate(entries):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value


def _git_command(
    executable: str,
    execution: _ExecutionRequest,
    grant: CredentialGrant,
    invocation_dir: Path,
    env: dict[str, str],
) -> list[str]:
    hooks_dir = invocation_dir / "hooks"
    hooks_dir.mkdir(mode=0o700)
    global_config = invocation_dir / "gitconfig"
    global_config.touch(mode=0o600)
    entries = [
        ("core.hooksPath", str(hooks_dir)),
        ("credential.helper", ""),
        ("protocol.allow", "never"),
        ("protocol.https.allow", "always"),
    ]
    if grant.git_user_name:
        entries.append(("user.name", grant.git_user_name))
    if grant.git_user_email:
        entries.append(("user.email", grant.git_user_email))
    if execution.action in _GIT_NETWORK_ACTIONS:
        if not grant.secret:
            raise ValueError("credential provider returned no Git credential")
        helper = invocation_dir / "credential-helper"
        helper.write_text(
            "#!/bin/sh\n"
            "host=\n"
            "while IFS='=' read -r key value && [ -n \"$key\" ]; do\n"
            '  [ "$key" = host ] && host=$value\n'
            "done\n"
            '[ "$host" = "$OMNIGENT_GIT_HOST" ] || exit 0\n'
            "printf 'username=%s\\npassword=%s\\n' "
            '"$OMNIGENT_GIT_USERNAME" "$OMNIGENT_GIT_SECRET"\n',
            encoding="utf-8",
        )
        helper.chmod(0o700)
        entries.extend(
            [
                ("credential.helper", str(helper)),
                ("credential.useHttpPath", "true"),
            ]
        )
        env["OMNIGENT_GIT_HOST"] = execution.host
        env["OMNIGENT_GIT_USERNAME"] = grant.username
        env["OMNIGENT_GIT_SECRET"] = grant.secret
    _set_git_config(env, entries)
    env["GIT_CONFIG_GLOBAL"] = str(global_config)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"
    global_args, command_args = _split_git_global_args(execution.argv)
    return [executable, *global_args, *command_args]


def _gh_command(
    executable: str,
    execution: _ExecutionRequest,
    grant: CredentialGrant,
    invocation_dir: Path,
    env: dict[str, str],
) -> list[str]:
    if not grant.secret:
        raise ValueError("credential provider returned no GitHub token")
    config_dir = invocation_dir / "gh-config"
    hooks_dir = invocation_dir / "hooks"
    git_config = invocation_dir / "gitconfig"
    config_dir.mkdir(mode=0o700)
    hooks_dir.mkdir(mode=0o700)
    git_config.touch(mode=0o600)
    _set_git_config(
        env,
        [
            ("core.hooksPath", str(hooks_dir)),
            ("protocol.allow", "never"),
            ("protocol.https.allow", "always"),
        ],
    )
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = str(git_config)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"
    env["GH_CONFIG_DIR"] = str(config_dir)
    env["GH_EDITOR"] = "true"
    env["GH_PROMPT_DISABLED"] = "1"
    env["GH_PAGER"] = "cat"
    env["BROWSER"] = "false"
    token_env = "GH_TOKEN" if execution.host == "github.com" else "GH_ENTERPRISE_TOKEN"
    env[token_env] = grant.secret
    return [executable, *execution.argv]


def _validate_grant(
    request: CredentialRequest,
    grant: CredentialGrant,
    *,
    expected_actor: ActorContext,
) -> None:
    if grant.actor != expected_actor:
        raise ValueError("credential provider returned an actor that does not own the active turn")
    if not grant.username:
        raise ValueError("credential provider returned an empty username")
    if request.operation == "credential" and not grant.secret:
        raise ValueError("credential provider returned no secret")
    if request.tool == "gh" and not grant.secret:
        raise ValueError("credential provider returned no GitHub token")
    if grant.secret is not None:
        now = time.time()
        if grant.expires_at is None or not (
            now < grant.expires_at <= now + _MAX_CREDENTIAL_TTL_SECONDS
        ):
            raise ValueError("credential provider returned a stale or long-lived credential")
