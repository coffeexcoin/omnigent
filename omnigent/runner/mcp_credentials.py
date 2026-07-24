"""Provider-neutral actor-aware credentials for runner-owned MCP connectors.

A deployment may attach an :class:`McpCredentialResolver` to
:class:`~omnigent.runner.mcp_manager.RunnerMcpManager`. The resolver maps the
active turn actor and one MCP server config to transport-specific credential
material plus a non-secret generation. The manager resolves on every MCP
request and partitions live connectors by the declared identity and generation,
so a takeover cannot reuse a process or HTTP session authenticated as the prior
actor.

Resolvers must increment ``generation`` whenever returned credential material
changes. Actor-scoped credentials are partitioned by the canonical actor mapping;
service-scoped credentials use the explicit ``identity`` and may be shared across
actors. Raw credential values are merged only into the effective connection
config and are never included in pool keys, status snapshots, or logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol

from omnigent.policies.schema import ActorContext
from omnigent.spec.types import MCPServerConfig


@dataclass(frozen=True, repr=False)
class McpCredential:
    """Credential material and non-secret connector-pooling metadata.

    :param scope: ``"actor"`` isolates connectors by active turn actor;
        ``"service"`` shares an explicit organization/service identity.
    :param generation: Non-secret version that changes whenever the material
        changes, e.g. a token version or lease generation.
    :param identity: Stable non-secret service identity. Required for service
        scope and ignored for actor scope.
    :param headers: Headers merged into HTTP MCP config headers.
    :param env: Environment variables merged into stdio MCP config env.
    """

    scope: Literal["actor", "service"]
    generation: str
    identity: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in {"actor", "service"}:
            raise ValueError(f"unsupported MCP credential scope {self.scope!r}")
        if not self.generation:
            raise ValueError("MCP credential generation must be non-empty")
        if self.scope == "service" and not self.identity:
            raise ValueError("service-scoped MCP credentials require an identity")
        if self.scope == "actor" and self.identity is not None:
            raise ValueError("actor-scoped MCP credentials derive identity from the active actor")
        if self.headers and self.env:
            raise ValueError("MCP credentials must target exactly one transport (headers or env)")

    def __repr__(self) -> str:
        """Return non-secret metadata only."""
        return (
            f"McpCredential(scope={self.scope!r}, generation={self.generation!r}, "
            f"identity={self.identity!r})"
        )


class McpCredentialResolver(Protocol):
    """Async deployment hook for resolving one MCP server's active credential."""

    async def resolve(
        self,
        config: MCPServerConfig,
        actor: ActorContext | None,
    ) -> McpCredential | None:
        """Return credentials for this request, or ``None`` for static config identity."""
        ...
