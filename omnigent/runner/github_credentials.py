"""Owner-scoped GitHub credentials backed by a managed launch broker.

The managed credential hook projects a launch-scoped broker capability into the
runner process. This module consumes that configuration before agent processes
start, removes it from the inherited environment, and implements the runner's
existing :class:`CredentialProvider` contract. Long-lived GitHub credentials
remain in the external broker; only bounded grants enter the trusted command
broker for one Git or GitHub CLI invocation.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

import httpx

from omnigent.runner.credential_broker import (
    ActiveCredentialTurn,
    CredentialGrant,
    CredentialRequest,
)

GITHUB_BROKER_URL_ENV = "OMNIGENT_GITHUB_CREDENTIAL_BROKER_URL"
GITHUB_BROKER_CAPABILITY_ENV = "OMNIGENT_GITHUB_CREDENTIAL_BROKER_CAPABILITY"
GITHUB_BROKER_OWNER_ENV = "OMNIGENT_GITHUB_CREDENTIAL_OWNER"
GITHUB_BROKER_SESSION_ENV = "OMNIGENT_GITHUB_CREDENTIAL_SESSION_ID"

_GITHUB_BROKER_ENV = (
    GITHUB_BROKER_URL_ENV,
    GITHUB_BROKER_CAPABILITY_ENV,
    GITHUB_BROKER_OWNER_ENV,
    GITHUB_BROKER_SESSION_ENV,
)
_STATIC_GITHUB_CREDENTIAL_ENV = (
    "GH_TOKEN",
    "OMNIGENT_GH_TOKEN",
    "GITHUB_TOKEN",
    "OMNIGENT_GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "OMNIGENT_GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
    "OMNIGENT_GITHUB_ENTERPRISE_TOKEN",
    "GIT_TOKEN",
    "OMNIGENT_GIT_TOKEN",
)
_MAX_CREDENTIAL_TTL_SECONDS = 15 * 60
_MAX_BROKER_RESPONSE_BYTES = 64 * 1024
_DEFAULT_BROKER_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class BrokeredGitHubCredentialConfig:
    """Launch-bound, secret-safe configuration for the GitHub broker client."""

    broker_url: str
    capability: str = field(repr=False)
    owner: str
    session_id: str

    def __post_init__(self) -> None:
        broker_url = self.broker_url.strip()
        owner = self.owner.strip()
        session_id = self.session_id.strip()
        capability = self.capability.strip()
        if not broker_url or not owner or not session_id or not capability:
            raise ValueError("GitHub broker configuration values must be non-empty")
        if len(owner) > 320 or len(session_id) > 512 or len(capability) > 4096:
            raise ValueError("GitHub broker configuration value is too long")
        parsed = urlparse(broker_url)
        is_loopback_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "::1",
            "localhost",
        }
        if (
            (parsed.scheme != "https" and not is_loopback_http)
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "GitHub broker URL must be HTTPS or loopback HTTP without credentials"
            )
        object.__setattr__(self, "broker_url", broker_url)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "capability", capability)


class BrokeredGitHubCredentialProvider:
    """Resolve bounded GitHub grants for one managed session owner.

    The class satisfies :class:`omnigent.runner.credential_broker.CredentialProvider`.
    The broker capability is fixed at managed launch and never returned to agent
    subprocesses. Every issue call is rebound to the exact active turn by
    :class:`CredentialBrokerBridge` before this provider validates the immutable
    session/owner pair.
    """

    def __init__(
        self,
        config: BrokeredGitHubCredentialConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(_DEFAULT_BROKER_TIMEOUT_SECONDS),
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._revoked = False
        self._revocation_complete = False
        self._lock = asyncio.Lock()

    @property
    def config(self) -> BrokeredGitHubCredentialConfig:
        """Return launch metadata with the capability excluded from ``repr``."""

        return self._config

    def __repr__(self) -> str:
        """Describe the provider without exposing its broker capability."""

        return (
            f"{type(self).__name__}(broker_url={self._config.broker_url!r}, "
            f"owner={self._config.owner!r}, session_id={self._config.session_id!r}, "
            f"revoked={self._revoked!r})"
        )

    async def issue(
        self,
        context: ActiveCredentialTurn,
        request: CredentialRequest,
    ) -> CredentialGrant:
        """Request one short-lived grant for the fixed launch owner."""

        if (
            context.session_id != self._config.session_id
            or context.actor.get("run_as") != self._config.owner
        ):
            raise PermissionError("GitHub credentials are restricted to the managed launch owner")
        if request.operation == "credential" and (
            request.host != "github.com" or request.protocol != "https"
        ):
            raise PermissionError("managed GitHub credentials are restricted to github.com HTTPS")

        async with self._lock:
            if self._revoked:
                raise PermissionError("managed GitHub credential capability is revoked")
            try:
                async with self._client.stream(
                    "POST",
                    self._config.broker_url,
                    headers={"Authorization": f"Bearer {self._config.capability}"},
                    json={
                        "session_id": self._config.session_id,
                        "turn_id": context.turn_id,
                        "owner": self._config.owner,
                        "request": asdict(request),
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError("broker request failed")
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=16 * 1024):
                        body.extend(chunk)
                        if len(body) > _MAX_BROKER_RESPONSE_BYTES:
                            raise RuntimeError("broker response is too large")
                payload = json.loads(bytes(body))
                grant = self._grant_from_payload(payload, request, context)
            except PermissionError:
                raise
            except Exception:  # noqa: BLE001 - broker errors may contain secret values
                raise RuntimeError("GitHub credential broker returned an invalid grant") from None
            if self._revoked:
                raise PermissionError("managed GitHub credential capability is revoked")
            return grant

    async def revoke(self) -> None:
        """Revoke this launch capability once and deny every later issue call."""

        async with self._lock:
            if self._revocation_complete:
                return
            self._revoked = True
            try:
                async with self._client.stream(
                    "DELETE",
                    self._config.broker_url,
                    headers={"Authorization": f"Bearer {self._config.capability}"},
                    json={
                        "session_id": self._config.session_id,
                        "owner": self._config.owner,
                    },
                ) as response:
                    if not 200 <= response.status_code < 300:
                        raise RuntimeError("GitHub credential broker revocation failed")
                self._revocation_complete = True
            except Exception:  # noqa: BLE001 - broker errors may contain secret values
                raise RuntimeError("GitHub credential broker revocation failed") from None

    async def aclose(self) -> None:
        """Revoke the launch capability and close an owned HTTP client."""

        try:
            await self.revoke()
        finally:
            if self._owns_client:
                await self._client.aclose()

    @staticmethod
    def _grant_from_payload(
        payload: object,
        request: CredentialRequest,
        context: ActiveCredentialTurn,
    ) -> CredentialGrant:
        if not isinstance(payload, dict):
            raise ValueError("grant must be an object")
        allowed_keys = {
            "username",
            "secret",
            "expires_at",
            "git_user_name",
            "git_user_email",
        }
        if not set(payload).issubset(allowed_keys):
            raise ValueError("grant contains unsupported fields")
        username = payload.get("username")
        secret = payload.get("secret")
        expires_at = payload.get("expires_at")
        git_user_name = payload.get("git_user_name")
        git_user_email = payload.get("git_user_email")
        if not isinstance(username, str) or not username or len(username) > 256:
            raise ValueError("grant username is invalid")
        if secret is not None and (
            not isinstance(secret, str) or not secret or len(secret) > 8192
        ):
            raise ValueError("grant secret is invalid")
        for value in (git_user_name, git_user_email):
            if value is not None and (not isinstance(value, str) or not value or len(value) > 320):
                raise ValueError("grant identity is invalid")
        if request.operation == "credential":
            if (
                secret is None
                or isinstance(expires_at, bool)
                or not isinstance(expires_at, (int, float))
            ):
                raise ValueError("credential grant requires a numeric expiry")
            now = time.time()
            if not now < float(expires_at) <= now + _MAX_CREDENTIAL_TTL_SECONDS:
                raise ValueError("credential grant is stale or long-lived")
        elif secret is not None or expires_at is not None:
            raise ValueError("identity grants cannot contain credentials")
        return CredentialGrant(
            username=username,
            actor=context.actor.copy(),
            secret=secret,
            expires_at=float(expires_at) if expires_at is not None else None,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )


def consume_github_credential_provider_from_environment(
    environment: MutableMapping[str, str] | None = None,
) -> BrokeredGitHubCredentialProvider | None:
    """Consume managed-launch settings before child processes can inherit them.

    A complete broker configuration activates the provider. Partial
    configuration fails closed. Broker settings and legacy static GitHub token
    variables are removed in either case; only the returned provider retains the
    launch capability in process memory.
    """

    source = environment if environment is not None else os.environ
    configured = {key: source.pop(key, None) for key in _GITHUB_BROKER_ENV}
    any_configured = any(value is not None for value in configured.values())
    if not any_configured:
        return None
    for key in _STATIC_GITHUB_CREDENTIAL_ENV:
        source.pop(key, None)
    if any(not isinstance(value, str) or not value.strip() for value in configured.values()):
        raise RuntimeError("managed GitHub credential broker configuration is incomplete")
    try:
        config = BrokeredGitHubCredentialConfig(
            broker_url=configured[GITHUB_BROKER_URL_ENV] or "",
            capability=configured[GITHUB_BROKER_CAPABILITY_ENV] or "",
            owner=configured[GITHUB_BROKER_OWNER_ENV] or "",
            session_id=configured[GITHUB_BROKER_SESSION_ENV] or "",
        )
    except ValueError:
        raise RuntimeError("managed GitHub credential broker configuration is invalid") from None
    return BrokeredGitHubCredentialProvider(config)
