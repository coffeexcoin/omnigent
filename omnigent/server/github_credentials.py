"""Concrete managed-launch GitHub credential control-plane integration.

The hook calls a deployment-owned control plane that resolves the launch owner,
creates a generation-scoped credential projection (for Kubernetes, a Secret),
and returns only its non-secret launcher reference. Omnigent never receives the
GitHub credential or broker capability. The projected values are the runner
broker URL, launch capability, owner, and session id documented by
:mod:`omnigent.runner.github_credentials`.

Wire :class:`BrokeredGitHubCredentialHook` into
``ManagedSandboxConfig.credential_hook``. The endpoint must implement an
idempotent ``POST``/``DELETE`` contract keyed by the ``Idempotency-Key`` header:

* ``POST`` receives the secret-free launch identity and generation, provisions
  the owner-scoped projection, and returns ``201`` with ``reference`` and
  ``expires_at``.
* ``DELETE`` receives the same identity plus the optional stored reference and
  revokes/deletes the generation. ``404`` and ``410`` mean cleanup is already
  complete.

A supplied ``httpx.AsyncClient`` may carry deployment authentication (for
example mTLS). Authentication is intentionally outside the payload so no
control-plane secret can enter managed-host records or API responses.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from omnigent.server.managed_credentials import (
    ManagedCredentialHook,
    ManagedCredentialLease,
    ManagedCredentialReleaseContext,
    ManagedLaunchContext,
)

_DEFAULT_TIMEOUT_S = 10.0
_MAX_RESPONSE_BYTES = 16 * 1024
_DNS_SUBDOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")


@dataclass(frozen=True, slots=True)
class BrokeredGitHubCredentialHookConfig:
    """Non-secret connection and lifetime policy for the launch control plane."""

    endpoint: str
    max_lease_ttl_s: float = 24 * 60 * 60

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip()
        parsed = urlparse(endpoint)
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
                "managed GitHub credential endpoint must be HTTPS or loopback HTTP "
                "without credentials"
            )
        if self.max_lease_ttl_s <= 0:
            raise ValueError("managed GitHub credential lease TTL must be positive")
        object.__setattr__(self, "endpoint", endpoint)


class _BrokeredGitHubCredentialLease(ManagedCredentialLease):
    """Secret-free live handle for one projected launch generation."""

    def __init__(
        self,
        hook: BrokeredGitHubCredentialHook,
        context: ManagedLaunchContext,
        generation: int,
        reference: str,
        expires_at: float,
    ) -> None:
        self._hook = hook
        self._context = context
        self._generation = generation
        self._reference = reference
        self._expires_at = expires_at
        self._released = False
        self._release_lock = asyncio.Lock()

    @property
    def reference(self) -> str:
        return self._reference

    @property
    def expires_at(self) -> float:
        """Return the control-plane expiry for observability and tests."""

        return self._expires_at

    async def release(self) -> None:
        """Release once after success; leave transient failures retryable."""

        async with self._release_lock:
            if self._released:
                return
            await self._hook._release_identity(
                self._context,
                self._generation,
                self._reference,
            )
            self._released = True


class BrokeredGitHubCredentialHook(ManagedCredentialHook):
    """Provision owner-scoped launch projections through an HTTP control plane."""

    def __init__(
        self,
        config: BrokeredGitHubCredentialHookConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT_S),
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None

    async def acquire(
        self,
        context: ManagedLaunchContext,
        generation: int,
    ) -> ManagedCredentialLease:
        """Provision one bounded projection without receiving its secret values."""

        payload = self._identity_payload(context, generation)
        try:
            async with self._client.stream(
                "POST",
                self._config.endpoint,
                headers={"Idempotency-Key": self._idempotency_key(context, generation)},
                json=payload,
            ) as response:
                if response.status_code != 201:
                    raise RuntimeError("managed GitHub credential acquisition failed")
                body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=4 * 1024):
                    body.extend(chunk)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise RuntimeError("managed GitHub credential response is invalid")
            response_payload = json.loads(body)
            reference, expires_at = self._validate_response(response_payload)
        except Exception:  # noqa: BLE001 - remote errors may contain secret values
            raise RuntimeError("managed GitHub credential acquisition failed") from None
        return _BrokeredGitHubCredentialLease(
            self,
            context,
            generation,
            reference,
            expires_at,
        )

    async def release(
        self,
        context: ManagedCredentialReleaseContext,
        generation: int,
        reference: str | None,
    ) -> None:
        """Revoke/delete one generation; raise a sanitized error for durable retry."""

        await self._release_identity(context, generation, reference)

    async def _release_identity(
        self,
        context: ManagedLaunchContext | ManagedCredentialReleaseContext,
        generation: int,
        reference: str | None,
    ) -> None:
        payload = self._identity_payload(context, generation)
        payload["reference"] = reference
        try:
            async with self._client.stream(
                "DELETE",
                self._config.endpoint,
                headers={"Idempotency-Key": self._idempotency_key(context, generation)},
                json=payload,
            ) as response:
                if response.status_code not in {200, 202, 204, 404, 410}:
                    raise RuntimeError("managed GitHub credential release failed")
        except Exception:  # noqa: BLE001 - remote errors may contain secret values
            raise RuntimeError("managed GitHub credential release failed") from None

    async def aclose(self) -> None:
        """Close the internally-created HTTP client, if any."""

        if self._owns_client:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(endpoint={self._config.endpoint!r})"

    @staticmethod
    def _identity_payload(
        context: ManagedLaunchContext | ManagedCredentialReleaseContext,
        generation: int,
    ) -> dict[str, object]:
        if generation < 1:
            raise ValueError("managed GitHub credential generation must be positive")
        return {
            "owner": context.owner,
            "host_id": context.host_id,
            "host_name": context.host_name,
            "generation": generation,
            "session_id": context.session_id,
            "repo_url": context.repo_url,
            "repo_branch": context.repo_branch,
            "repo_name": context.repo_name,
        }

    @staticmethod
    def _idempotency_key(
        context: ManagedLaunchContext | ManagedCredentialReleaseContext,
        generation: int,
    ) -> str:
        identity = f"{context.owner}\0{context.host_id}\0{generation}".encode()
        return f"omnigent-github-{hashlib.sha256(identity).hexdigest()}"

    def _validate_response(self, payload: Any) -> tuple[str, float]:
        if not isinstance(payload, dict) or set(payload) != {"reference", "expires_at"}:
            raise RuntimeError("managed GitHub credential response is invalid")
        reference = payload["reference"]
        expires_at = payload["expires_at"]
        if (
            not isinstance(reference, str)
            or len(reference) > 253
            or not _DNS_SUBDOMAIN_RE.fullmatch(reference)
        ):
            raise RuntimeError("managed GitHub credential response is invalid")
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
            raise RuntimeError("managed GitHub credential response is invalid")
        now = time.time()
        normalized_expiry = float(expires_at)
        if not now < normalized_expiry <= now + self._config.max_lease_ttl_s:
            raise RuntimeError("managed GitHub credential response is invalid")
        return reference, normalized_expiry
