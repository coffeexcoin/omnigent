"""Tests for the managed-launch GitHub credential control-plane hook."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from omnigent.server.github_credentials import (
    BrokeredGitHubCredentialHook,
    BrokeredGitHubCredentialHookConfig,
)
from omnigent.server.managed_credentials import (
    ManagedCredentialReleaseContext,
    ManagedLaunchContext,
)

_OWNER = "owner@example.com"
_ENDPOINT = "https://credentials.example.test/v1/managed-github-leases"


def _launch_context() -> ManagedLaunchContext:
    return ManagedLaunchContext(
        owner=_OWNER,
        host_id="host_owner",
        host_name="owner-host",
        session_id="conv_owner",
        repo_url="https://github.com/acme/widgets.git",
        repo_branch="main",
        repo_name="widgets",
    )


def _release_context() -> ManagedCredentialReleaseContext:
    context = _launch_context()
    return ManagedCredentialReleaseContext(
        owner=context.owner,
        host_id=context.host_id,
        host_name=context.host_name,
        sandbox_provider="kubernetes",
        sandbox_id="omnigent-host-owner",
        session_id=context.session_id,
        repo_url=context.repo_url,
        repo_branch=context.repo_branch,
        repo_name=context.repo_name,
    )


def _config() -> BrokeredGitHubCredentialHookConfig:
    return BrokeredGitHubCredentialHookConfig(endpoint=_ENDPOINT, max_lease_ttl_s=3600)


async def test_hook_acquires_owner_generation_scoped_projected_reference_and_releases_it() -> None:
    requests: list[tuple[str, dict[str, object], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.method, payload, request.headers["idempotency-key"]))
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "reference": "omnigent-github-host-owner-7",
                    "expires_at": time.time() + 300,
                },
            )
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        hook = BrokeredGitHubCredentialHook(_config(), client=client)
        lease = await hook.acquire(_launch_context(), generation=7)
        await lease.release()
        await lease.release()

    expected = {
        "owner": _OWNER,
        "host_id": "host_owner",
        "host_name": "owner-host",
        "generation": 7,
        "session_id": "conv_owner",
        "repo_url": "https://github.com/acme/widgets.git",
        "repo_branch": "main",
        "repo_name": "widgets",
    }
    assert lease.reference == "omnigent-github-host-owner-7"
    assert lease.expires_at is not None and time.time() < lease.expires_at <= time.time() + 300
    assert requests == [
        ("POST", expected, requests[0][2]),
        (
            "DELETE",
            {**expected, "reference": "omnigent-github-host-owner-7"},
            requests[0][2],
        ),
    ]
    assert _OWNER not in requests[0][2]
    assert _OWNER not in repr(lease)
    assert "omnigent-github-host-owner-7" not in repr(lease)


async def test_hook_release_without_live_lease_is_deterministic_and_retryable() -> None:
    attempts = 0
    keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        keys.append(request.headers["idempotency-key"])
        assert request.method == "DELETE"
        assert json.loads(request.content)["generation"] == 9
        return httpx.Response(503 if attempts == 1 else 404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        hook = BrokeredGitHubCredentialHook(_config(), client=client)
        with pytest.raises(RuntimeError, match="release failed") as error:
            await hook.release(_release_context(), 9, None)
        await hook.release(_release_context(), 9, None)

    assert attempts == 2
    assert keys[0] == keys[1]
    assert _OWNER not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"reference": "Bad_Secret", "expires_at": lambda: time.time() + 60},
        {"reference": "valid-secret", "expires_at": None},
        {"reference": "valid-secret", "expires_at": lambda: time.time() - 1},
        {"reference": "valid-secret", "expires_at": lambda: time.time() + 3601},
    ],
)
async def test_hook_rejects_invalid_or_unbounded_projected_lease(
    payload: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        resolved = {key: value() if callable(value) else value for key, value in payload.items()}
        return httpx.Response(201, json=resolved)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        hook = BrokeredGitHubCredentialHook(_config(), client=client)
        with pytest.raises(RuntimeError, match="acquisition failed"):
            await hook.acquire(_launch_context(), generation=1)


async def test_hook_sanitizes_untrusted_control_plane_exceptions() -> None:
    secret = "control-plane-secret-value"

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"transport exposed {secret}: {request.method}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        hook = BrokeredGitHubCredentialHook(_config(), client=client)
        with pytest.raises(RuntimeError, match="acquisition failed") as acquire_error:
            await hook.acquire(_launch_context(), generation=1)
        with pytest.raises(RuntimeError, match="release failed") as release_error:
            await hook.release(_release_context(), 1, None)

    assert secret not in str(acquire_error.value)
    assert secret not in str(release_error.value)


async def test_hook_scopes_idempotency_to_owner_host_and_generation() -> None:
    keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["idempotency-key"])
        payload = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "reference": f"lease-{payload['generation']}",
                "expires_at": time.time() + 60,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        hook = BrokeredGitHubCredentialHook(_config(), client=client)
        await hook.acquire(_launch_context(), generation=1)
        await hook.acquire(_launch_context(), generation=1)
        await hook.acquire(_launch_context(), generation=2)

    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
