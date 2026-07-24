"""Actor-aware credential tests for the runner MCP connection pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pytest
from mcp.types import Tool as McpToolDef

from omnigent.policies.schema import ActorContext
from omnigent.runner import mcp_manager as _mcp_manager_module
from omnigent.runner.mcp_credentials import McpCredential
from omnigent.runner.mcp_manager import RunnerMcpManager
from omnigent.spec.types import AgentSpec, MCPServerConfig


@dataclass
class _CreatedConnection:
    config: MCPServerConfig
    calls: list[tuple[str, dict[str, Any]]]
    closed: bool = False


class _CredentialResolver:
    def __init__(
        self,
        *,
        transport: Literal["http", "stdio"],
        scope: Literal["actor", "service"] = "actor",
    ) -> None:
        self.transport = transport
        self.scope = scope
        self.generations: dict[str, str] = {}
        self.resolve_calls: list[tuple[str, str]] = []

    async def resolve(
        self,
        config: MCPServerConfig,
        actor: ActorContext | None,
    ) -> McpCredential:
        principal = (actor or {}).get("run_as", "anonymous")
        self.resolve_calls.append((config.name, principal))
        generation = self.generations.get(principal, "1")
        token_identity = "org-service" if self.scope == "service" else principal
        if self.transport == "http":
            return McpCredential(
                scope=self.scope,
                generation=generation,
                identity="org-service" if self.scope == "service" else None,
                headers={"Authorization": f"Bearer token-{token_identity}-{generation}"},
            )
        return McpCredential(
            scope=self.scope,
            generation=generation,
            identity="org-service" if self.scope == "service" else None,
            env={"MCP_TOKEN": f"token-{token_identity}-{generation}"},
        )


def _patch_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_CreatedConnection]:
    created: list[_CreatedConnection] = []

    class _Connection:
        def __init__(self, *, config: MCPServerConfig, **_kwargs: Any) -> None:
            self.record = _CreatedConnection(config=config, calls=[])
            created.append(self.record)

        async def connect(self) -> list[McpToolDef]:
            return [
                McpToolDef(
                    name="whoami",
                    description="Return the authenticated principal",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            **_kwargs: Any,
        ) -> str:
            self.record.calls.append((name, arguments))
            return "ok"

        async def close(self) -> None:
            self.record.closed = True

    monkeypatch.setattr(_mcp_manager_module, "McpServerConnection", _Connection)
    return created


def _spec(config: MCPServerConfig) -> AgentSpec:
    return AgentSpec(spec_version=1, name="actor-mcp", mcp_servers=[config])


@pytest.mark.asyncio
async def test_http_credentials_resolve_per_request_and_pool_by_actor_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_connections(monkeypatch)
    resolver = _CredentialResolver(transport="http")
    manager = RunnerMcpManager(credential_resolver=resolver)
    spec = _spec(
        MCPServerConfig(
            name="github",
            transport="http",
            url="https://mcp.example.test/sse",
            headers={"X-Service": "github"},
        )
    )
    alice: ActorContext = {"run_as": "alice@example.com"}
    bob: ActorContext = {"run_as": "bob@example.com"}

    try:
        await manager.schemas_for(spec, actor=alice, session_id="session-1")
        await manager.call_tool(
            spec, "github__whoami", {}, actor=alice, session_id="session-1"
        )
        await manager.call_tool(
            spec, "github__whoami", {}, actor=bob, session_id="session-1"
        )

        assert len(created) == 2
        assert created[0].config.headers == {
            "X-Service": "github",
            "Authorization": "Bearer token-alice@example.com-1",
        }
        assert created[1].config.headers == {
            "X-Service": "github",
            "Authorization": "Bearer token-bob@example.com-1",
        }
        assert created[0].calls == [("whoami", {})]
        assert created[1].calls == [("whoami", {})]
        assert resolver.resolve_calls == [
            ("github", "alice@example.com"),
            ("github", "alice@example.com"),
            ("github", "bob@example.com"),
        ]

        resolver.generations["alice@example.com"] = "2"
        await manager.call_tool(
            spec, "github__whoami", {}, actor=alice, session_id="session-1"
        )

        assert len(created) == 3
        assert created[2].config.headers["Authorization"] == (
            "Bearer token-alice@example.com-2"
        )
        assert created[2].calls == [("whoami", {})]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_stdio_processes_are_scoped_by_session_actor_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_connections(monkeypatch)
    resolver = _CredentialResolver(transport="stdio")
    manager = RunnerMcpManager(credential_resolver=resolver)
    spec = _spec(
        MCPServerConfig(
            name="github",
            transport="stdio",
            command="mcp-github",
            env={"LOG_LEVEL": "warning"},
        )
    )
    alice: ActorContext = {"run_as": "alice@example.com"}
    bob: ActorContext = {"run_as": "bob@example.com"}

    try:
        await manager.call_tool(
            spec, "github__whoami", {}, actor=alice, session_id="session-1"
        )
        await manager.call_tool(
            spec, "github__whoami", {}, actor=alice, session_id="session-2"
        )
        await manager.call_tool(
            spec, "github__whoami", {}, actor=bob, session_id="session-1"
        )
        assert created[0].closed is True
        assert created[1].closed is False

        resolver.generations["bob@example.com"] = "2"
        await manager.call_tool(
            spec, "github__whoami", {}, actor=bob, session_id="session-1"
        )

        assert [connection.config.env for connection in created] == [
            {"LOG_LEVEL": "warning", "MCP_TOKEN": "token-alice@example.com-1"},
            {"LOG_LEVEL": "warning", "MCP_TOKEN": "token-alice@example.com-1"},
            {"LOG_LEVEL": "warning", "MCP_TOKEN": "token-bob@example.com-1"},
            {"LOG_LEVEL": "warning", "MCP_TOKEN": "token-bob@example.com-2"},
        ]
        assert all(connection.calls == [("whoami", {})] for connection in created)
        assert created[2].closed is True
        assert created[3].closed is False
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_explicit_service_identity_remains_shared_across_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_connections(monkeypatch)
    resolver = _CredentialResolver(transport="http", scope="service")
    manager = RunnerMcpManager(credential_resolver=resolver)
    spec = _spec(
        MCPServerConfig(
            name="org-search",
            transport="http",
            url="https://mcp.example.test/sse",
        )
    )

    try:
        await manager.call_tool(
            spec,
            "org-search__whoami",
            {},
            actor={"run_as": "alice@example.com"},
        )
        await manager.call_tool(
            spec,
            "org-search__whoami",
            {},
            actor={"run_as": "bob@example.com"},
        )

        assert len(created) == 1
        assert created[0].calls == [("whoami", {}), ("whoami", {})]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_actor_credential_fails_closed_without_active_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_connections(monkeypatch)
    manager = RunnerMcpManager(
        credential_resolver=_CredentialResolver(transport="http")
    )
    spec = _spec(
        MCPServerConfig(
            name="github",
            transport="http",
            url="https://mcp.example.test/sse",
        )
    )

    try:
        with pytest.raises(ValueError, match="require an active actor"):
            await manager.schemas_for(spec)
        assert created == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_actor_stdio_credential_fails_closed_without_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_connections(monkeypatch)
    manager = RunnerMcpManager(
        credential_resolver=_CredentialResolver(transport="stdio")
    )
    spec = _spec(
        MCPServerConfig(
            name="github",
            transport="stdio",
            command="mcp-github",
        )
    )

    try:
        with pytest.raises(ValueError, match="require an active session"):
            await manager.call_tool(
                spec,
                "github__whoami",
                {},
                actor={"run_as": "alice@example.com"},
            )
        assert created == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_credential_material_is_absent_from_repr_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_connections(monkeypatch)
    resolver = _CredentialResolver(transport="http")
    manager = RunnerMcpManager(credential_resolver=resolver)
    spec = _spec(
        MCPServerConfig(
            name="github",
            transport="http",
            url="https://mcp.example.test/sse",
        )
    )
    credential = McpCredential(
        scope="actor",
        generation="7",
        headers={"Authorization": "Bearer super-secret-token"},
    )

    try:
        await manager.schemas_for(spec, actor={"run_as": "alice@example.com"})
        rendered = repr(credential) + repr(manager.status_snapshot())
        assert "super-secret-token" not in rendered
        assert "token-alice@example.com" not in rendered
    finally:
        await manager.shutdown()
