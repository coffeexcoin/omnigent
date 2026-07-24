"""Tests for active-turn Git/GitHub credential brokerage."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from omnigent.runner import app as runner_app_module
from omnigent.runner import create_runner_app
from omnigent.runner.credential_broker import (
    BROKER_CAPABILITY_ENV,
    BROKER_ENDPOINT_ENV,
    ActiveCredentialTurn,
    CredentialAuditEvent,
    CredentialBrokerBridge,
    CredentialGrant,
    CredentialRequest,
)
from omnigent.runner.credential_wrapper import (
    _broker_request,
    run_gh,
    run_git,
    run_git_credential,
)
from omnigent.runtime.harnesses.process_manager import HarnessProcessManager
from tests.runner.helpers import NullServerClient


class _Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[ActiveCredentialTurn, CredentialRequest]] = []

    async def issue(
        self, context: ActiveCredentialTurn, request: CredentialRequest
    ) -> CredentialGrant:
        self.calls.append((context, request))
        actor = context.actor.get("run_as", "")
        return CredentialGrant(
            username="x-access-token",
            actor=context.actor.copy(),
            secret=(f"short-lived-for-{actor}" if request.operation == "credential" else None),
            expires_at=(time.time() + 300 if request.operation == "credential" else None),
            git_user_name=actor.split("@", 1)[0].title(),
            git_user_email=actor,
        )


@pytest.fixture
async def broker() -> AsyncIterator[
    tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]]
]:
    provider = _Provider()
    audit: list[CredentialAuditEvent] = []
    bridge = CredentialBrokerBridge(provider, audit_sink=audit.append)
    await bridge.start()
    try:
        yield bridge, provider, audit
    finally:
        await bridge.close()


def _install_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)


async def test_wrapper_environment_contains_capability_but_no_actor_or_credential(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
) -> None:
    bridge, _, _ = broker

    env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})

    assert env[BROKER_ENDPOINT_ENV].startswith("127.0.0.1:")
    assert env[BROKER_CAPABILITY_ENV]
    assert "alice" not in repr(env)
    assert "short-lived" not in repr(env)
    wrapper_dir = Path(env["PATH"].split(os.pathsep, 1)[0])
    assert (wrapper_dir / "git").stat().st_mode & 0o111
    assert (wrapper_dir / "gh").stat().st_mode & 0o111


async def test_request_fails_closed_without_active_turn(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, audit = broker
    env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})
    _install_env(monkeypatch, env)
    monkeypatch.setattr("sys.stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))

    with pytest.raises(RuntimeError, match="no active turn"):
        await asyncio.to_thread(run_git_credential, "get")

    assert provider.calls == []
    assert audit == []


async def test_git_credential_uses_active_actor_and_audits_turn(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, audit = broker
    bridge.bind_turn(
        ActiveCredentialTurn(
            session_id="conv_1",
            turn_id="turn_1",
            actor={"run_as": "alice@example.com"},
        )
    )
    env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})
    _install_env(monkeypatch, env)
    stdin = io.StringIO(
        "protocol=https\nhost=github.com\npath=acme/repo.git\npassword=must-not-round-trip\n\n"
    )
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = await asyncio.to_thread(run_git_credential, "get")

    assert result == 0
    assert stdout.getvalue().endswith(
        "username=x-access-token\npassword=short-lived-for-alice@example.com\n\n"
    )
    context, request = provider.calls[0]
    assert context.actor == {"run_as": "alice@example.com"}
    assert request == CredentialRequest(
        tool="git",
        action="get",
        operation="credential",
        protocol="https",
        host="github.com",
        path="acme/repo.git",
    )
    assert "must-not-round-trip" not in repr(provider.calls)
    assert audit == [
        CredentialAuditEvent(
            session_id="conv_1",
            turn_id="turn_1",
            actor={"run_as": "alice@example.com"},
            tool="git",
            action="get",
            operation="credential",
            outcome="allowed",
        )
    ]


async def test_actor_takeover_only_affects_next_bound_turn(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, _ = broker
    first = ActiveCredentialTurn("conv_1", "turn_1", {"run_as": "alice@example.com"})
    second = ActiveCredentialTurn("conv_1", "turn_2", {"run_as": "bob@example.com"})
    bridge.bind_turn(first)
    _install_env(
        monkeypatch,
        bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]}),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))
    monkeypatch.setattr("sys.stdout", io.StringIO())

    await asyncio.to_thread(run_git_credential, "get")
    bridge.bind_turn(second)
    bridge.clear_turn("conv_1", turn_id="turn_1")
    _install_env(
        monkeypatch,
        bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]}),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    await asyncio.to_thread(run_git_credential, "get")

    assert [call[0] for call in provider.calls] == [first, second]


async def test_git_wrapper_uses_process_local_identity_and_global_config(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bridge, _, audit = broker
    bridge._executables["git"] = "/usr/bin/git"
    bridge.bind_turn(
        ActiveCredentialTurn("conv_1", "turn_commit", {"run_as": "alice@example.com"})
    )
    env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})
    _install_env(monkeypatch, env)
    monkeypatch.setenv("GIT_ASKPASS", "/tmp/untrusted-askpass")
    monkeypatch.setenv("SSH_ASKPASS", "/tmp/untrusted-ssh-askpass")
    home_config = tmp_path / ".gitconfig"
    home_config.write_text("original\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    captured: dict[str, Any] = {}

    def fake_run(
        command: Sequence[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        captured.update(command=list(command), env=dict(env), check=check)
        assert Path(env["GIT_CONFIG_GLOBAL"]).parent != tmp_path
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = await asyncio.to_thread(
        run_git,
        ["-c", "user.email=mallory@example.com", "commit", "-m", "message"],
    )

    assert result == 0
    command = captured["command"]
    assert command[:3] == [
        "/usr/bin/git",
        "-c",
        "user.email=mallory@example.com",
    ]
    assert command[3:7] == [
        "-c",
        "user.name=Alice",
        "-c",
        "user.email=alice@example.com",
    ]
    assert "credential.helper=" in command
    assert command[-3:] == ["commit", "-m", "message"]
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_ASKPASS" not in captured["env"]
    assert "SSH_ASKPASS" not in captured["env"]
    assert home_config.read_text(encoding="utf-8") == "original\n"
    assert not Path(captured["env"]["GIT_CONFIG_GLOBAL"]).parent.exists()
    assert audit[0].action == "commit"
    assert audit[0].turn_id == "turn_commit"


async def test_generated_git_wrapper_executes_real_git(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    tmp_path: Path,
) -> None:
    """The generated PATH entry is executable, not only a unit-level shim."""

    bridge, _, audit = broker
    real_git = bridge._executables.get("git")
    if real_git is None:
        pytest.skip("git is not installed")
    subprocess.run(
        [real_git, "init", "--quiet", str(tmp_path)],
        check=True,
        env=dict(os.environ),
    )
    bridge.bind_turn(
        ActiveCredentialTurn(
            "conv_real_git",
            "turn_status",
            {"run_as": "alice@example.com"},
        )
    )
    env_updates = bridge.wrapper_environment("conv_real_git", os.environ)
    env = dict(os.environ)
    env.update(env_updates)

    result = await asyncio.to_thread(
        subprocess.run,
        ["git", "status", "--short"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert audit[0].action == "status"
    assert audit[0].turn_id == "turn_status"


async def test_gh_wrapper_scopes_token_and_config_to_child_process(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bridge, _, audit = broker
    bridge._executables["gh"] = "/usr/bin/gh"
    bridge.bind_turn(ActiveCredentialTurn("conv_1", "turn_pr", {"run_as": "bob@example.com"}))
    env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})
    _install_env(monkeypatch, env)
    monkeypatch.setenv("GH_TOKEN", "stale-token")
    monkeypatch.setenv("GITHUB_TOKEN", "also-stale")
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "stale-enterprise-token")
    monkeypatch.setenv("HOME", str(tmp_path))
    captured: dict[str, Any] = {}

    def fake_run(
        command: Sequence[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        captured.update(command=list(command), env=dict(env), check=check)
        assert Path(env["GH_CONFIG_DIR"]).parent != tmp_path
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = await asyncio.to_thread(
        run_gh,
        ["--repo", "acme/widgets", "pr", "view", "123"],
    )

    assert result == 0
    assert captured["command"] == [
        "/usr/bin/gh",
        "--repo",
        "acme/widgets",
        "pr",
        "view",
        "123",
    ]
    assert captured["env"]["GH_TOKEN"] == "short-lived-for-bob@example.com"
    assert "GITHUB_TOKEN" not in captured["env"]
    assert "GH_ENTERPRISE_TOKEN" not in captured["env"]
    assert not (tmp_path / ".config" / "gh").exists()
    assert not Path(captured["env"]["GH_CONFIG_DIR"]).exists()
    assert audit[0].session_id == "conv_1"
    assert audit[0].turn_id == "turn_pr"
    assert audit[0].actor == {"run_as": "bob@example.com"}
    assert audit[0].action == "pr"


async def test_native_runner_keeps_actor_bound_until_forwarded_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An instant native proxy retains actor authority until terminal idle."""

    provider = _Provider()
    audit: list[CredentialAuditEvent] = []
    captured_env: dict[str, str] = {}

    class _Stream:
        status_code = 200

        async def __aenter__(self) -> _Stream:
            saved = {
                key: os.environ.get(key) for key in (BROKER_ENDPOINT_ENV, BROKER_CAPABILITY_ENV)
            }
            try:
                for key in saved:
                    os.environ[key] = captured_env[key]
                await asyncio.to_thread(
                    _broker_request,
                    {"tool": "git", "operation": "identity", "action": "status"},
                )
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def aiter_text(self) -> AsyncIterator[str]:
            yield (
                "event: response.completed\ndata: "
                '{"type":"response.completed","response":'
                '{"id":"resp_broker","status":"completed"}}\n\n'
            )

    class _Client:
        def stream(self, *args: object, **kwargs: object) -> _Stream:
            del args, kwargs
            return _Stream()

    class _Manager:
        async def get_client(
            self,
            conversation_id: str,
            harness: str,
            env: dict[str, str] | None = None,
        ) -> _Client:
            del conversation_id, harness
            assert env is not None
            captured_env.update(env)
            return _Client()

        def mark_in_flight(self, conversation_id: str, response_id: str) -> None:
            del conversation_id, response_id

        def clear_in_flight(self, conversation_id: str) -> None:
            del conversation_id

    app = create_runner_app(
        process_manager=cast(HarnessProcessManager, _Manager()),
        server_client=NullServerClient(),  # type: ignore[arg-type]
        credential_provider=provider,
        credential_audit_sink=audit.append,
    )
    monkeypatch.setattr(runner_app_module, "is_native_harness", lambda _name: True)
    conversation_id = "conv_broker_actor"
    runner_app_module._session_histories_ref[conversation_id] = []
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://runner",
        ) as client:
            response = await client.post(
                f"/v1/sessions/{conversation_id}/events?stream=true",
                json={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "inspect repo"}],
                    "actor": {"run_as": "alice@example.com"},
                    "persisted_item_id": "item_turn_42",
                },
            )
            _install_env(monkeypatch, captured_env)
            await asyncio.to_thread(
                _broker_request,
                {"tool": "git", "operation": "identity", "action": "status"},
            )
            idle = await client.post(
                f"/v1/sessions/{conversation_id}/events",
                json={"type": "external_session_status", "data": {"status": "idle"}},
            )
            with pytest.raises(RuntimeError, match="no active turn"):
                await asyncio.to_thread(
                    _broker_request,
                    {"tool": "git", "operation": "identity", "action": "status"},
                )
    finally:
        bridge = app.state.credential_broker
        await bridge.close()
        runner_app_module._session_histories_ref.pop(conversation_id, None)

    assert response.status_code == 200
    assert idle.status_code == 204
    assert captured_env[BROKER_ENDPOINT_ENV]
    assert captured_env[BROKER_CAPABILITY_ENV]
    assert provider.calls[0][0] == ActiveCredentialTurn(
        conversation_id,
        "item_turn_42",
        {"run_as": "alice@example.com"},
    )
    assert len(provider.calls) == 2
    assert audit[0].session_id == conversation_id
    assert audit[0].turn_id == "item_turn_42"
    assert audit[0].actor == {"run_as": "alice@example.com"}


async def test_wrapper_environment_does_not_trust_real_executable_overrides(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
) -> None:
    bridge, _, _ = broker

    env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})

    assert not any(key.startswith("OMNIGENT_REAL_") for key in env)


async def test_stale_actor_capability_cannot_borrow_takeover_credentials(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, _ = broker
    first = ActiveCredentialTurn("conv_1", "turn_1", {"run_as": "alice@example.com"})
    second = ActiveCredentialTurn("conv_1", "turn_2", {"run_as": "bob@example.com"})
    bridge.bind_turn(first)
    old_env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})
    _install_env(monkeypatch, old_env)
    monkeypatch.setattr("sys.stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    await asyncio.to_thread(run_git_credential, "get")

    bridge.bind_turn(second)
    monkeypatch.setattr("sys.stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    with pytest.raises(RuntimeError, match="origin actor"):
        await asyncio.to_thread(run_git_credential, "get")

    assert [call[0] for call in provider.calls] == [first]


async def test_actorless_capability_cannot_adopt_a_later_turn_actor(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, audit = broker
    old_env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})
    bridge.bind_turn(ActiveCredentialTurn("conv_1", "turn_1", {"run_as": "alice@example.com"}))
    assert bridge.requires_process_rotation("conv_1") is True
    _install_env(monkeypatch, old_env)
    monkeypatch.setattr("sys.stdin", io.StringIO("protocol=https\nhost=github.com\n\n"))

    with pytest.raises(RuntimeError, match="origin actor"):
        await asyncio.to_thread(run_git_credential, "get")

    assert provider.calls == []
    assert audit == []


@pytest.mark.parametrize(
    "grant_actor",
    [None, {"run_as": "mallory@example.com"}],
    ids=["missing", "mismatched"],
)
async def test_provider_grant_actor_must_match_active_turn(
    grant_actor: dict[str, str] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WrongActorProvider:
        async def issue(
            self, context: ActiveCredentialTurn, request: CredentialRequest
        ) -> CredentialGrant:
            del context, request
            return CredentialGrant(
                username="x-access-token",
                actor=grant_actor,  # type: ignore[arg-type]
                git_user_name="Alice",
                git_user_email="alice@example.com",
            )

    audit: list[CredentialAuditEvent] = []
    bridge = CredentialBrokerBridge(_WrongActorProvider(), audit_sink=audit.append)
    await bridge.start()
    try:
        turn = ActiveCredentialTurn(
            "conv_wrong_actor",
            "turn_wrong_actor",
            {"run_as": "alice@example.com"},
        )
        bridge.bind_turn(turn)
        _install_env(
            monkeypatch,
            bridge.wrapper_environment("conv_wrong_actor", {"PATH": os.environ["PATH"]}),
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("Git must not run for an invalid actor grant"),
        )

        with pytest.raises(RuntimeError, match="credential broker request failed"):
            await asyncio.to_thread(run_git, ["status"])
    finally:
        await bridge.close()

    assert audit == [
        CredentialAuditEvent(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            actor=turn.actor,
            tool="git",
            action="status",
            operation="identity",
            outcome="error",
        )
    ]


async def test_provider_exception_is_secret_safe_over_the_broker_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_secret = "provider-secret-must-not-cross-boundary"

    class _FailingProvider:
        async def issue(
            self, context: ActiveCredentialTurn, request: CredentialRequest
        ) -> CredentialGrant:
            del context, request
            raise RuntimeError(leaked_secret)

    audit: list[CredentialAuditEvent] = []
    bridge = CredentialBrokerBridge(_FailingProvider(), audit_sink=audit.append)
    await bridge.start()
    try:
        turn = ActiveCredentialTurn(
            "conv_provider_error",
            "turn_provider_error",
            {"run_as": "alice@example.com"},
        )
        bridge.bind_turn(turn)
        _install_env(
            monkeypatch,
            bridge.wrapper_environment("conv_provider_error", {"PATH": os.environ["PATH"]}),
        )

        with pytest.raises(RuntimeError) as error:
            await asyncio.to_thread(
                _broker_request,
                {"tool": "git", "operation": "identity", "action": "status"},
            )
    finally:
        await bridge.close()

    assert leaked_secret not in str(error.value)
    assert leaked_secret not in repr(audit)
    assert audit[0].outcome == "error"
    assert audit[0].actor == turn.actor


@pytest.mark.parametrize("argv", [["auth", "token"], ["extension", "exec", "untrusted"]])
async def test_gh_secret_revealing_commands_fail_before_requesting_a_credential(
    argv: list[str],
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, audit = broker
    _install_env(
        monkeypatch,
        bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]}),
    )

    with pytest.raises(PermissionError, match=r"gh (auth|extension)"):
        await asyncio.to_thread(run_gh, argv)

    assert provider.calls == []
    assert audit == []


async def test_provider_timeout_fails_closed() -> None:
    class _HangingProvider:
        async def issue(
            self, context: ActiveCredentialTurn, request: CredentialRequest
        ) -> CredentialGrant:
            del context, request
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    bridge = CredentialBrokerBridge(_HangingProvider(), provider_timeout=0.01)
    await bridge.start()
    try:
        bridge.bind_turn(
            ActiveCredentialTurn("conv_timeout", "turn_timeout", {"run_as": "alice@example.com"})
        )
        env = bridge.wrapper_environment("conv_timeout", {"PATH": os.environ["PATH"]})
        payload = {
            "capability": env[BROKER_CAPABILITY_ENV],
            "tool": "git",
            "operation": "identity",
            "action": "status",
        }

        with pytest.raises(RuntimeError, match="provider failed"):
            await bridge._dispatch(payload)
    finally:
        await bridge.close()


async def test_concurrent_git_invocations_use_isolated_ephemeral_config(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _, _ = broker
    bridge._executables["git"] = "/usr/bin/git"
    bridge.bind_turn(
        ActiveCredentialTurn(
            "conv_concurrent",
            "turn_concurrent",
            {"run_as": "alice@example.com"},
        )
    )
    _install_env(
        monkeypatch,
        bridge.wrapper_environment("conv_concurrent", {"PATH": os.environ["PATH"]}),
    )
    barrier = threading.Barrier(2)
    config_dirs: list[Path] = []
    config_dirs_lock = threading.Lock()

    def fake_run(
        command: Sequence[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        config_dir = Path(env["GIT_CONFIG_GLOBAL"]).parent
        with config_dirs_lock:
            config_dirs.append(config_dir)
        barrier.wait(timeout=5)
        assert config_dir.exists()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = await asyncio.gather(
        asyncio.to_thread(run_git, ["status", "--short"]),
        asyncio.to_thread(run_git, ["status", "--short"]),
    )

    assert results == [0, 0]
    assert len(set(config_dirs)) == 2
    assert all(not config_dir.exists() for config_dir in config_dirs)
