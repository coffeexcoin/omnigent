from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent.entities.session_resources import SessionResourceView
from omnigent.runner import create_runner_app
from omnigent.runner.credential_broker import CredentialAuditEvent
from omnigent.runner.model_credentials import (
    MODEL_CREDENTIAL_SCOPE_ENV,
    ModelCredentialGrant,
    ModelCredentialRequest,
)
from omnigent.spec.types import AgentSpec, ExecutorSpec
from tests.runner.helpers import NullServerClient


class _Provider:
    def __init__(self) -> None:
        self.requests: list[ModelCredentialRequest] = []
        self.released_sessions: list[str] = []
        self.close_calls = 0

    async def issue(self, request: ModelCredentialRequest) -> ModelCredentialGrant:
        self.requests.append(request)
        actor = request.actor.get("run_as")
        assert actor is not None
        return ModelCredentialGrant(
            environment={"OPENAI_API_KEY": f"credential-for-{actor}"},
            provider_id="test-provider",
            billing_account_id=f"billing-{actor}",
            generation=f"generation-for-{actor}",
        )

    async def release_session(self, session_id: str) -> None:
        self.released_sessions.append(session_id)

    async def close(self) -> None:
        self.close_calls += 1


class _HarnessClient:
    def __init__(self) -> None:
        self.posted_bodies: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, *, json: dict[str, Any], timeout: Any) -> Any:
        del method, url, timeout
        self.posted_bodies.append(json)

        class _Stream:
            status_code = 200

            async def __aenter__(self) -> _Stream:
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def aiter_text(self) -> AsyncIterator[str]:
                yield (
                    'event: response.created\ndata: {"type":"response.created",'
                    '"response":{"id":"resp_test"}}\n\n'
                )
                yield (
                    'event: response.completed\ndata: {"type":"response.completed",'
                    '"response":{"id":"resp_test"}}\n\n'
                )

        return _Stream()


class _ProcessManager:
    handles_tool_dispatch = True

    def __init__(self) -> None:
        self.client = _HarnessClient()
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []

    async def get_client(
        self,
        conversation_id: str,
        harness: str,
        env: dict[str, str] | None = None,
    ) -> _HarnessClient:
        self.calls.append((conversation_id, harness, env))
        return self.client

    def mark_in_flight(self, conversation_id: str, response_id: str) -> None:
        del conversation_id, response_id

    def clear_in_flight(self, conversation_id: str) -> None:
        del conversation_id

    async def release(self, conversation_id: str) -> None:
        del conversation_id

    async def forward_cancel(self, conversation_id: str) -> None:
        del conversation_id


@pytest.mark.asyncio
async def test_two_actors_on_one_session_receive_distinct_model_credentials() -> None:
    provider = _Provider()
    process_manager = _ProcessManager()
    audit: list[CredentialAuditEvent] = []
    spec = AgentSpec(
        spec_version=1,
        name="credential-test",
        executor=ExecutorSpec(type="omnigent", config={"harness": "openai-agents"}),
    )

    async def _resolve(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return spec

    app = create_runner_app(
        process_manager=process_manager,  # type: ignore[arg-type]
        spec_resolver=_resolve,
        server_client=NullServerClient(),  # type: ignore[arg-type]
        model_credential_provider=provider,
        credential_audit_sink=audit.append,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:
        for index, (actor, item_id) in enumerate(
            (
                ("alice@example.com", "item_alice"),
                ("bob@example.com", "item_bob"),
            ),
            start=1,
        ):
            response = await client.post(
                "/v1/sessions/conv_shared/events",
                json={
                    "type": "message",
                    "content": [{"type": "input_text", "text": f"hello from {actor}"}],
                    "agent_id": "agent_test",
                    "actor": {"run_as": actor},
                    "persisted_item_id": item_id,
                },
            )
            assert response.status_code == 202, response.text
            async with asyncio.timeout(2):
                while len(process_manager.calls) < index:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0)

        deleted = await client.delete("/v1/sessions/conv_shared")
        assert deleted.status_code == 200, deleted.text

    await app.state.close_credential_resources()

    assert [request.actor for request in provider.requests] == [
        {"run_as": "alice@example.com"},
        {"run_as": "bob@example.com"},
    ]
    assert [request.turn_id for request in provider.requests] == ["item_alice", "item_bob"]
    assert [call[0] for call in process_manager.calls] == ["conv_shared", "conv_shared"]
    alice_env = process_manager.calls[0][2]
    bob_env = process_manager.calls[1][2]
    assert alice_env is not None and bob_env is not None
    assert alice_env["OPENAI_API_KEY"] == "credential-for-alice@example.com"
    assert bob_env["OPENAI_API_KEY"] == "credential-for-bob@example.com"
    assert alice_env[MODEL_CREDENTIAL_SCOPE_ENV] != bob_env[MODEL_CREDENTIAL_SCOPE_ENV]
    assert "alice@example.com" not in process_manager.client.posted_bodies[0].values()
    assert "bob@example.com" not in process_manager.client.posted_bodies[1].values()
    assert [(event.tool, event.actor, event.outcome) for event in audit] == [
        ("model", {"run_as": "alice@example.com"}, "allowed"),
        ("model", {"run_as": "bob@example.com"}, "allowed"),
    ]
    assert [(event.provider_id, event.billing_account_id) for event in audit] == [
        ("test-provider", "billing-alice@example.com"),
        ("test-provider", "billing-bob@example.com"),
    ]
    assert "credential-for-" not in repr(audit)
    assert provider.released_sessions == ["conv_shared"]
    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_missing_model_credential_fails_closed_without_secret_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _MissingProvider:
        async def issue(self, request: ModelCredentialRequest) -> ModelCredentialGrant:
            del request
            raise LookupError("missing credential: top-secret")

    process_manager = _ProcessManager()
    spec = AgentSpec(
        spec_version=1,
        name="credential-test",
        executor=ExecutorSpec(type="omnigent", config={"harness": "openai-agents"}),
    )

    async def _resolve(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return spec

    app = create_runner_app(
        process_manager=process_manager,  # type: ignore[arg-type]
        spec_resolver=_resolve,
        server_client=NullServerClient(),  # type: ignore[arg-type]
        model_credential_provider=_MissingProvider(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:
        response = await client.post(
            "/v1/sessions/conv_missing/events?stream=true",
            json={
                "type": "message",
                "content": [{"type": "input_text", "text": "hello"}],
                "agent_id": "agent_test",
                "actor": {"run_as": "alice@example.com"},
                "persisted_item_id": "item_missing",
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": "model_credential_resolution_failed",
        "detail": "Failed to resolve model credentials for this turn.",
    }
    assert "top-secret" not in response.text
    assert "top-secret" not in caplog.text
    assert process_manager.calls == []


@pytest.mark.asyncio
async def test_no_model_provider_preserves_actor_optional_behavior() -> None:
    process_manager = _ProcessManager()
    spec = AgentSpec(
        spec_version=1,
        name="legacy-test",
        executor=ExecutorSpec(type="omnigent", config={"harness": "openai-agents"}),
    )

    async def _resolve(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return spec

    app = create_runner_app(
        process_manager=process_manager,  # type: ignore[arg-type]
        spec_resolver=_resolve,
        server_client=NullServerClient(),  # type: ignore[arg-type]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:
        response = await client.post(
            "/v1/sessions/conv_legacy/events",
            json={
                "type": "message",
                "content": [{"type": "input_text", "text": "hello"}],
                "agent_id": "agent_test",
                "persisted_item_id": "item_legacy",
            },
        )
        assert response.status_code == 202, response.text
        async with asyncio.timeout(2):
            while not process_manager.calls:
                await asyncio.sleep(0.01)

    legacy_env = process_manager.calls[0][2]
    assert legacy_env is not None
    assert MODEL_CREDENTIAL_SCOPE_ENV not in legacy_env


class _NativeResourceRegistry:
    """Minimal native registry that records actor-terminal rotation."""

    def __init__(self) -> None:
        self.terminal_registry = self
        self.current: Any | None = None
        self.close_calls = 0

    def set_terminal_activity_publisher(self, _publisher: Any) -> None:
        return None

    def set_session_status_publisher(self, _publisher: Any) -> None:
        return None

    def set_terminal_exit_publisher(self, _publisher: Any) -> None:
        return None

    def get(self, session_id: str, terminal_name: str, session_key: str) -> Any | None:
        del session_id, terminal_name, session_key
        return self.current

    async def close(self, session_id: str, terminal_name: str, session_key: str) -> bool:
        del session_id, terminal_name, session_key
        self.close_calls += 1
        self.current = None
        return True

    async def get_terminal_resource(self, session_id: str, terminal_id: str) -> Any | None:
        del session_id, terminal_id
        return self.current

    def terminal_resource_role(self, session_id: str, terminal_id: str) -> str | None:
        del session_id, terminal_id
        return "codex-native"


@pytest.mark.asyncio
async def test_native_takeover_rotates_actor_home_without_changing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    registry = _NativeResourceRegistry()
    launch_envs: list[dict[str, str]] = []

    async def _fake_auto_create(
        session_id: str,
        _resource_registry: Any,
        _publish_event: Any,
        **kwargs: Any,
    ) -> Any:
        env = dict(kwargs["model_credential_env"])
        launch_envs.append(env)
        registry.current = SessionResourceView(
            id="terminal_claude_main",
            type="terminal",
            session_id=session_id,
            name="claude:main",
            metadata={"terminal_name": "claude", "session_key": "main"},
            environment=None,
        )
        return registry.current

    async def _issue(request: ModelCredentialRequest) -> ModelCredentialGrant:
        actor = request.actor.get("run_as", "unknown")
        return ModelCredentialGrant(
            environment={
                "CLAUDE_CONFIG_DIR": str(tmp_path / actor),
                "ANTHROPIC_API_KEY": f"credential-for-{actor}",
            },
            provider_id="test-provider",
            generation=actor,
        )

    provider.issue = _issue  # type: ignore[method-assign]
    monkeypatch.setattr("omnigent.runner.app._auto_create_claude_terminal", _fake_auto_create)
    app = create_runner_app(
        server_client=NullServerClient(),  # type: ignore[arg-type]
        resource_registry=registry,  # type: ignore[arg-type]
        model_credential_provider=provider,
    )
    transport = httpx.ASGITransport(app=app)
    responses = []
    async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:
        for actor in ("alice@example.com", "bob@example.com"):
            responses.append(
                await client.post(
                    "/v1/sessions/conv_native/resources/terminals",
                    json={
                        "terminal": "claude",
                        "session_key": "main",
                        "ensure_native_terminal": True,
                        "actor": {"run_as": actor},
                    },
                )
            )

    assert [response.status_code for response in responses] == [200, 200]
    assert [env["CLAUDE_CONFIG_DIR"] for env in launch_envs] == [
        str(tmp_path / "alice@example.com"),
        str(tmp_path / "bob@example.com"),
    ]
    assert launch_envs[0]["ANTHROPIC_API_KEY"] != launch_envs[1]["ANTHROPIC_API_KEY"]
    assert registry.close_calls == 1
    assert {response.json()["session_id"] for response in responses} == {"conv_native"}
    assert "credential-for-" not in "".join(response.text for response in responses)


@pytest.mark.asyncio
async def test_codex_native_takeover_rotates_actor_home_without_changing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    registry = _NativeResourceRegistry()
    launch_envs: list[dict[str, str]] = []

    async def _fake_auto_create(
        session_id: str,
        _resource_registry: Any,
        _publish_event: Any,
        **kwargs: Any,
    ) -> SessionResourceView:
        env = dict(kwargs["model_credential_env"])
        launch_envs.append(env)
        registry.current = SessionResourceView(
            id="terminal_codex_main",
            type="terminal",
            session_id=session_id,
            name="codex:main",
            metadata={"terminal_name": "codex", "session_key": "main"},
        )
        return registry.current

    async def _issue(request: ModelCredentialRequest) -> ModelCredentialGrant:
        actor = request.actor.get("run_as", "unknown")
        return ModelCredentialGrant(
            environment={
                "CODEX_HOME": str(tmp_path / actor),
                "OPENAI_API_KEY": f"credential-for-{actor}",
            },
            provider_id="test-provider",
            generation=actor,
        )

    provider.issue = _issue  # type: ignore[method-assign]
    monkeypatch.setattr("omnigent.runner.app._auto_create_codex_terminal", _fake_auto_create)
    app = create_runner_app(
        server_client=NullServerClient(),  # type: ignore[arg-type]
        resource_registry=registry,  # type: ignore[arg-type]
        model_credential_provider=provider,
    )
    transport = httpx.ASGITransport(app=app)
    responses = []
    async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:
        for actor in ("alice@example.com", "bob@example.com"):
            responses.append(
                await client.post(
                    "/v1/sessions/conv_codex_native/resources/terminals",
                    json={
                        "terminal": "codex",
                        "session_key": "main",
                        "ensure_native_terminal": True,
                        "actor": {"run_as": actor},
                    },
                )
            )

    assert [response.status_code for response in responses] == [200, 200]
    assert [env["CODEX_HOME"] for env in launch_envs] == [
        str(tmp_path / "alice@example.com"),
        str(tmp_path / "bob@example.com"),
    ]
    assert launch_envs[0]["OPENAI_API_KEY"] != launch_envs[1]["OPENAI_API_KEY"]
    assert registry.close_calls == 1
    assert {response.json()["session_id"] for response in responses} == {"conv_codex_native"}
    assert "credential-for-" not in "".join(response.text for response in responses)


@pytest.mark.asyncio
async def test_stale_native_authorization_cannot_recreate_prior_actor_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    registry = _NativeResourceRegistry()
    alice_issued = asyncio.Event()
    release_alice = asyncio.Event()
    launched_actors: list[str] = []

    async def _issue(request: ModelCredentialRequest) -> ModelCredentialGrant:
        actor = request.actor.get("run_as", "unknown")
        if actor == "alice@example.com":
            alice_issued.set()
            await release_alice.wait()
        return ModelCredentialGrant(
            environment={"CLAUDE_CONFIG_DIR": str(tmp_path / actor)},
            provider_id="test-provider",
            generation=actor,
        )

    async def _fake_auto_create(
        session_id: str,
        _resource_registry: Any,
        _publish_event: Any,
        **kwargs: Any,
    ) -> SessionResourceView:
        launched_actors.append(Path(kwargs["model_credential_env"]["CLAUDE_CONFIG_DIR"]).name)
        registry.current = SessionResourceView(
            id="terminal_claude_main",
            type="terminal",
            session_id=session_id,
            name="claude:main",
            metadata={"terminal_name": "claude", "session_key": "main"},
        )
        return registry.current

    provider.issue = _issue  # type: ignore[method-assign]
    monkeypatch.setattr("omnigent.runner.app._auto_create_claude_terminal", _fake_auto_create)
    app = create_runner_app(
        server_client=NullServerClient(),  # type: ignore[arg-type]
        resource_registry=registry,  # type: ignore[arg-type]
        model_credential_provider=provider,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runner") as client:
        alice_request = asyncio.create_task(
            client.post(
                "/v1/sessions/conv_native/resources/terminals",
                json={
                    "terminal": "claude",
                    "session_key": "main",
                    "ensure_native_terminal": True,
                    "actor": {"run_as": "alice@example.com"},
                },
            )
        )
        await asyncio.wait_for(alice_issued.wait(), timeout=1)
        bob_response = await client.post(
            "/v1/sessions/conv_native/resources/terminals",
            json={
                "terminal": "claude",
                "session_key": "main",
                "ensure_native_terminal": True,
                "actor": {"run_as": "bob@example.com"},
            },
        )
        release_alice.set()
        alice_response = await alice_request

    assert bob_response.status_code == 200
    assert alice_response.status_code == 503
    assert launched_actors == ["bob@example.com"]
