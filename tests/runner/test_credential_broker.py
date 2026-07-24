"""Tests for active-turn Git/GitHub credential brokerage."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
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

    with pytest.raises(RuntimeError, match="no active turn"):
        await asyncio.to_thread(run_git, ["status", "--short"])

    assert provider.calls == []
    assert audit == []


async def test_git_execution_uses_active_actor_and_audits_turn(
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
    result = await asyncio.to_thread(run_git, ["status", "--short"])

    assert result == 0
    context, request = provider.calls[0]
    assert context.actor == {"run_as": "alice@example.com"}
    assert request == CredentialRequest(
        tool="git",
        action="status",
        operation="identity",
    )
    assert audit == [
        CredentialAuditEvent(
            session_id="conv_1",
            turn_id="turn_1",
            actor={"run_as": "alice@example.com"},
            tool="git",
            action="status",
            operation="identity",
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

    await asyncio.to_thread(run_git, ["status", "--short"])
    bridge.bind_turn(second)
    bridge.clear_turn("conv_1", turn_id="turn_1")
    _install_env(
        monkeypatch,
        bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]}),
    )
    await asyncio.to_thread(run_git, ["status", "--short"])

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

    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        env = kwargs["env"]
        captured.update(command=list(command), env=dict(env), check=kwargs["check"])
        assert Path(env["GIT_CONFIG_GLOBAL"]).parent != tmp_path
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = await asyncio.to_thread(run_git, ["commit", "-m", "message"])

    assert result == 0
    command = captured["command"]
    assert command == ["/usr/bin/git", "commit", "-m", "message"]
    config = {
        captured["env"][f"GIT_CONFIG_KEY_{index}"]: captured["env"][f"GIT_CONFIG_VALUE_{index}"]
        for index in range(int(captured["env"]["GIT_CONFIG_COUNT"]))
    }
    assert config["user.name"] == "Alice"
    assert config["user.email"] == "alice@example.com"
    assert Path(config["core.hooksPath"]).parent != tmp_path
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
    for key in (
        "BROWSER",
        "GH_BROWSER",
        "GH_EDITOR",
        "GH_PAGER",
        "GIT_ASKPASS",
        "GIT_EDITOR",
        "HTTPS_PROXY",
        "PAGER",
        "SSH_ASKPASS",
    ):
        monkeypatch.setenv(key, "/tmp/untrusted-callback")
    monkeypatch.setenv("HOME", str(tmp_path))
    captured: dict[str, Any] = {}

    class _Popen:
        pid = 12345
        returncode = 0

        def __init__(self, command: Sequence[str], **kwargs: Any) -> None:
            env = kwargs["env"]
            captured.update(command=list(command), env=dict(env))
            assert Path(env["GH_CONFIG_DIR"]).parent != tmp_path

        def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
        ) -> tuple[None, None]:
            del input, timeout
            return None, None

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", _Popen)

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
    assert not {"GH_BROWSER", "HTTPS_PROXY", "SSH_ASKPASS"} & captured["env"].keys()
    assert captured["env"]["BROWSER"] == "false"
    assert captured["env"]["GH_EDITOR"] == "true"
    assert captured["env"]["GH_PAGER"] == "cat"
    assert captured["env"]["GIT_EDITOR"] == "true"
    assert "PAGER" not in captured["env"]
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
                await asyncio.to_thread(run_git, ["status", "--short"])
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
            await asyncio.to_thread(run_git, ["status", "--short"])
            idle = await client.post(
                f"/v1/sessions/{conversation_id}/events",
                json={
                    "type": "external_session_status",
                    "data": {"status": "idle", "turn_id": "item_turn_42"},
                },
            )
            with pytest.raises(RuntimeError, match="invalid or expired broker capability"):
                await asyncio.to_thread(run_git, ["status", "--short"])
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
    await asyncio.to_thread(run_git, ["status", "--short"])

    bridge.bind_turn(second)
    with pytest.raises(RuntimeError, match="invalid or expired broker capability"):
        await asyncio.to_thread(run_git, ["status", "--short"])

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

    with pytest.raises(RuntimeError, match="invalid or expired broker capability"):
        await asyncio.to_thread(run_git, ["status", "--short"])

    assert provider.calls == []
    assert audit == []


async def test_every_turn_revokes_prior_capability_including_same_actor_and_aba(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capabilities are exact-turn grants, not reusable actor identities."""

    bridge, provider, _ = broker
    turns = (
        ActiveCredentialTurn("conv_1", "turn_a1", {"run_as": "alice@example.com"}),
        ActiveCredentialTurn("conv_1", "turn_a2", {"run_as": "alice@example.com"}),
        ActiveCredentialTurn("conv_1", "turn_b", {"run_as": "bob@example.com"}),
        ActiveCredentialTurn("conv_1", "turn_a3", {"run_as": "alice@example.com"}),
    )
    bridge.bind_turn(turns[0])
    stale_env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})

    for successor in turns[1:]:
        bridge.bind_turn(successor)
        _install_env(monkeypatch, stale_env)
        with pytest.raises(RuntimeError, match="invalid or expired broker capability"):
            await asyncio.to_thread(run_git, ["status", "--short"])
        stale_env = bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]})

    assert provider.calls == []


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
            await asyncio.to_thread(run_git, ["status"])
    finally:
        await bridge.close()

    assert leaked_secret not in str(error.value)
    assert leaked_secret not in repr(audit)
    assert audit[0].outcome == "error"
    assert audit[0].actor == turn.actor


@pytest.mark.parametrize(
    "argv",
    [
        ["auth", "token"],
        ["extension", "exec", "untrusted"],
        ["gist", "clone", "1"],
        ["pr", "checkout", "1"],
        ["repo", "clone", "acme/widgets"],
        ["repo", "fork", "acme/widgets"],
        ["repo", "create", "acme/widgets", "--clone=true"],
    ],
)
async def test_gh_secret_revealing_commands_fail_before_requesting_a_credential(
    argv: list[str],
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, audit = broker
    bridge.bind_turn(
        ActiveCredentialTurn("conv_1", "turn_denied", {"run_as": "alice@example.com"})
    )
    _install_env(
        monkeypatch,
        bridge.wrapper_environment("conv_1", {"PATH": os.environ["PATH"]}),
    )

    with pytest.raises(RuntimeError, match=r"(gh (auth|extension)|unavailable)"):
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
            "operation": "execute",
            "argv": ["status"],
            "cwd": os.getcwd(),
            "stdin": "",
        }

        with pytest.raises(RuntimeError, match="provider failed"):
            await bridge._dispatch(payload)
    finally:
        await bridge.close()


async def test_turn_change_during_provider_await_prevents_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider issuance is revalidated against the exact turn after awaiting."""

    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockingProvider(_Provider):
        async def issue(
            self,
            context: ActiveCredentialTurn,
            request: CredentialRequest,
        ) -> CredentialGrant:
            started.set()
            await release.wait()
            return await super().issue(context, request)

    provider = _BlockingProvider()
    bridge = CredentialBrokerBridge(provider)
    await bridge.start()
    first = ActiveCredentialTurn("conv_race", "turn_a", {"run_as": "alice@example.com"})
    second = ActiveCredentialTurn("conv_race", "turn_b", {"run_as": "bob@example.com"})
    try:
        bridge.bind_turn(first)
        env = bridge.wrapper_environment("conv_race", {"PATH": os.environ["PATH"]})
        payload = {
            "capability": env[BROKER_CAPABILITY_ENV],
            "tool": "gh",
            "operation": "execute",
            "argv": ["--repo", "acme/widgets", "pr", "view", "1"],
            "cwd": os.getcwd(),
            "stdin": "",
            "host": "github.com",
        }
        monkeypatch.setattr(
            subprocess,
            "Popen",
            lambda *args, **kwargs: pytest.fail("stale authorization executed"),
        )
        request = asyncio.create_task(bridge._dispatch(payload))
        await started.wait()
        bridge.bind_turn(second)
        release.set()
        with pytest.raises(PermissionError, match="changed during authorization"):
            await request
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

    def fake_run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        env = kwargs["env"]
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


async def test_raw_credential_requests_are_rejected_at_broker_boundary(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Possession of a wrapper capability must not make credentials extractable."""

    bridge, provider, audit = broker
    turn = ActiveCredentialTurn("conv_1", "turn_1", {"run_as": "alice@example.com"})
    bridge.bind_turn(turn)
    _install_env(monkeypatch, bridge.wrapper_environment("conv_1", os.environ))

    with pytest.raises(RuntimeError, match="credential broker request failed"):
        await asyncio.to_thread(
            _broker_request,
            {
                "tool": "git",
                "operation": "credential",
                "action": "get",
                "protocol": "https",
                "host": "github.com",
            },
        )

    assert provider.calls == []
    assert audit == []


async def test_direct_wrapper_module_credential_fill_cannot_extract_secret(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, provider, audit = broker
    bridge.bind_turn(ActiveCredentialTurn("conv_1", "turn_1", {"run_as": "alice@example.com"}))
    _install_env(monkeypatch, bridge.wrapper_environment("conv_1", os.environ))

    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "omnigent.runner.credential_wrapper", "git", "credential", "fill"],
        input=b"protocol=https\nhost=github.com\n\n",
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert b"short-lived-for" not in result.stdout + result.stderr
    assert provider.calls == []
    assert audit == []


async def test_successful_broker_response_contains_only_process_results(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The socket protocol never returns provider grants to the wrapper."""

    bridge, _, _ = broker
    bridge.bind_turn(
        ActiveCredentialTurn("conv_1", "turn_result", {"run_as": "alice@example.com"})
    )
    _install_env(monkeypatch, bridge.wrapper_environment("conv_1", os.environ))

    response = await asyncio.to_thread(
        _broker_request,
        {
            "tool": "git",
            "operation": "execute",
            "argv": ["status", "--short"],
            "cwd": os.getcwd(),
            "stdin": "",
        },
    )

    assert set(response) == {"ok", "result"}
    result = response["result"]
    assert isinstance(result, dict)
    assert set(result) == {"returncode", "stdout", "stderr"}
    assert "short-lived-for" not in repr(response)


async def test_delayed_untagged_idle_cannot_clear_new_actor_turn() -> None:
    """A delayed actor-A terminal edge must not revoke actor B's authority."""

    provider = _Provider()

    class _Manager:
        pass

    app = create_runner_app(
        process_manager=cast(HarnessProcessManager, _Manager()),
        server_client=NullServerClient(),  # type: ignore[arg-type]
        credential_provider=provider,
    )
    bridge = app.state.credential_broker
    conversation_id = "conv_delayed_idle"
    first = ActiveCredentialTurn(
        conversation_id,
        "turn_actor_a",
        {"run_as": "alice@example.com"},
    )
    second = ActiveCredentialTurn(
        conversation_id,
        "turn_actor_b",
        {"run_as": "bob@example.com"},
    )
    bridge.bind_turn(first)
    bridge.bind_turn(second)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://runner",
        ) as client:
            response = await client.post(
                f"/v1/sessions/{conversation_id}/events",
                json={"type": "external_session_status", "data": {"status": "idle"}},
            )
        assert response.status_code == 204
        assert bridge.active_turn_id(conversation_id) == second.turn_id
    finally:
        await bridge.close()


async def test_native_response_status_keeps_its_originating_credential_turn() -> None:
    """A response-tagged actor-A idle resolves to A, never the later actor B."""

    class _Manager:
        pass

    app = create_runner_app(
        process_manager=cast(HarnessProcessManager, _Manager()),
        server_client=NullServerClient(),  # type: ignore[arg-type]
        credential_provider=_Provider(),
    )
    bridge = app.state.credential_broker
    conversation_id = "conv_response_fence"
    first = ActiveCredentialTurn(
        conversation_id,
        "turn_actor_a",
        {"run_as": "alice@example.com"},
    )
    second = ActiveCredentialTurn(
        conversation_id,
        "turn_actor_b",
        {"run_as": "bob@example.com"},
    )
    bridge.bind_turn(first)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://runner",
        ) as client:
            running = await client.post(
                f"/v1/sessions/{conversation_id}/events",
                json={
                    "type": "external_session_status",
                    "data": {"status": "running", "response_id": "native_response_a"},
                },
            )
            bridge.bind_turn(second)
            idle = await client.post(
                f"/v1/sessions/{conversation_id}/events",
                json={
                    "type": "external_session_status",
                    "data": {"status": "idle", "response_id": "native_response_a"},
                },
            )
        assert running.status_code == 204
        assert idle.status_code == 204
        assert bridge.active_turn_id(conversation_id) == second.turn_id
    finally:
        await bridge.close()


async def test_gh_execution_disables_repository_hooks_that_can_exfiltrate_token(
    broker: tuple[CredentialBrokerBridge, _Provider, list[CredentialAuditEvent]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Git processes launched by gh must not run repository-controlled hooks."""

    bridge, _, _ = broker
    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    subprocess.run([real_git, "init", "--quiet", str(repo)], check=True)
    subprocess.run(
        [real_git, "-C", str(repo), "commit", "--allow-empty", "-m", "initial"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    leaked = tmp_path / "leaked-token"
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\nprintf '%s' \"$GH_TOKEN\" > {leaked}\n", encoding="utf-8")
    hook.chmod(0o700)
    fake_gh = tmp_path / "gh"
    fake_gh.write_text(
        f"#!/bin/sh\n{real_git} checkout -q -b broker-hook-test\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    bridge._executables["gh"] = str(fake_gh)
    bridge.bind_turn(ActiveCredentialTurn("conv_1", "turn_gh", {"run_as": "alice@example.com"}))
    _install_env(monkeypatch, bridge.wrapper_environment("conv_1", os.environ))
    monkeypatch.chdir(repo)

    result = await asyncio.to_thread(run_gh, ["pr", "view", "1"])

    assert result == 0
    assert not leaked.exists()
