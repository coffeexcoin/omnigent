"""Tests for managed owner-scoped GitHub credential brokerage."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from omnigent.runner.credential_broker import (
    BROKER_CAPABILITY_ENV,
    ActiveCredentialTurn,
    CredentialAuditEvent,
    CredentialBrokerBridge,
    CredentialRequest,
)
from omnigent.runner.github_credentials import (
    GITHUB_BROKER_CAPABILITY_ENV,
    GITHUB_BROKER_OWNER_ENV,
    GITHUB_BROKER_SESSION_ENV,
    GITHUB_BROKER_URL_ENV,
    BrokeredGitHubCredentialConfig,
    BrokeredGitHubCredentialProvider,
    consume_github_credential_provider_from_environment,
)

_OWNER = "owner@example.com"
_SESSION_ID = "conv_owner"
_CAPABILITY = "launch-capability-must-stay-private"
_TOKEN = "short-lived-github-token-must-stay-private"


def _config() -> BrokeredGitHubCredentialConfig:
    return BrokeredGitHubCredentialConfig(
        broker_url="https://credentials.example.test/v1/github/grants",
        capability=_CAPABILITY,
        owner=_OWNER,
        session_id=_SESSION_ID,
    )


def _credential_request() -> CredentialRequest:
    return CredentialRequest(
        tool="gh",
        action="pr",
        operation="credential",
        protocol="https",
        host="github.com",
        path="acme/widgets",
    )


async def test_provider_resolves_short_lived_grant_for_launch_owner() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "username": "x-access-token",
                "secret": _TOKEN,
                "expires_at": time.time() + 300,
                "git_user_name": "Repository Owner",
                "git_user_email": _OWNER,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        grant = await provider.issue(
            ActiveCredentialTurn(_SESSION_ID, "turn_1", {"run_as": _OWNER}),
            _credential_request(),
        )

    assert seen == {
        "authorization": f"Bearer {_CAPABILITY}",
        "payload": {
            "session_id": _SESSION_ID,
            "turn_id": "turn_1",
            "owner": _OWNER,
            "request": {
                "tool": "gh",
                "action": "pr",
                "operation": "credential",
                "protocol": "https",
                "host": "github.com",
                "path": "acme/widgets",
            },
        },
    }
    assert grant.actor == {"run_as": _OWNER}
    assert grant.secret == _TOKEN
    assert grant.expires_at is not None and time.time() < grant.expires_at <= time.time() + 300
    assert _CAPABILITY not in repr(provider)
    assert _CAPABILITY not in repr(provider.config)
    assert _TOKEN not in repr(grant)
    assert _TOKEN not in repr(seen)


@pytest.mark.parametrize(
    ("session_id", "actor"),
    [
        ("conv_other", {"run_as": _OWNER}),
        (_SESSION_ID, {"run_as": "collaborator@example.com"}),
    ],
)
async def test_provider_denies_non_owner_context_before_contacting_broker(
    session_id: str,
    actor: dict[str, str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"broker must not be called for a non-owner context: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        with pytest.raises(PermissionError, match="launch owner"):
            await provider.issue(
                ActiveCredentialTurn(session_id, "turn_other", actor),  # type: ignore[arg-type]
                _credential_request(),
            )


@pytest.mark.parametrize(
    "expires_at",
    [
        None,
        0,
        pytest.param("tomorrow", id="non-numeric"),
        pytest.param(lambda: time.time() - 1, id="expired"),
        pytest.param(lambda: time.time() + 901, id="too-long"),
    ],
)
async def test_provider_rejects_missing_stale_or_long_lived_credentials(
    expires_at: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        value = expires_at() if callable(expires_at) else expires_at
        return httpx.Response(
            200,
            json={
                "username": "x-access-token",
                "secret": _TOKEN,
                "expires_at": value,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        with pytest.raises(RuntimeError, match="invalid grant") as error:
            await provider.issue(
                ActiveCredentialTurn(_SESSION_ID, "turn_expiry", {"run_as": _OWNER}),
                _credential_request(),
            )

    assert _TOKEN not in str(error.value)


async def test_provider_stops_streaming_oversized_broker_response() -> None:
    """The response cap is enforced while reading, not after full buffering."""

    class _OversizedStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.chunks_read = 0

        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for _ in range(3):
                self.chunks_read += 1
                if self.chunks_read > 2:
                    raise AssertionError("provider buffered beyond the response limit")
                yield b"x" * (40 * 1024)

    stream = _OversizedStream()

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        with pytest.raises(RuntimeError, match="invalid grant"):
            await provider.issue(
                ActiveCredentialTurn(_SESSION_ID, "turn_oversized", {"run_as": _OWNER}),
                _credential_request(),
            )

    assert stream.chunks_read == 2


async def test_provider_revocation_is_idempotent_and_fails_closed() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        assert request.headers["authorization"] == f"Bearer {_CAPABILITY}"
        assert json.loads(request.content) == {
            "session_id": _SESSION_ID,
            "owner": _OWNER,
        }
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        await provider.revoke()
        await provider.revoke()
        with pytest.raises(PermissionError, match="revoked"):
            await provider.issue(
                ActiveCredentialTurn(_SESSION_ID, "turn_revoked", {"run_as": _OWNER}),
                _credential_request(),
            )

    assert methods == ["DELETE"]


async def test_failed_revocation_can_retry_but_credentials_stay_revoked() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        with pytest.raises(RuntimeError, match="revocation failed"):
            await provider.revoke()
        with pytest.raises(PermissionError, match="revoked"):
            await provider.issue(
                ActiveCredentialTurn(_SESSION_ID, "turn_revoked", {"run_as": _OWNER}),
                _credential_request(),
            )
        await provider.revoke()
        await provider.revoke()

    assert attempts == 2


async def test_failed_close_leaves_remote_revocation_retryable() -> None:
    """A failed shutdown attempt denies locally without marking remote cleanup done."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        with pytest.raises(RuntimeError, match="revocation failed"):
            await provider.aclose()
        with pytest.raises(PermissionError, match="revoked"):
            await provider.issue(
                ActiveCredentialTurn(_SESSION_ID, "turn_after_close", {"run_as": _OWNER}),
                _credential_request(),
            )
        await provider.aclose()

    assert attempts == 2


def test_launch_environment_is_consumed_without_pat_or_capability_inheritance() -> None:
    environment = {
        "PATH": os.defpath,
        GITHUB_BROKER_URL_ENV: "https://credentials.example.test/v1/github/grants",
        GITHUB_BROKER_CAPABILITY_ENV: _CAPABILITY,
        GITHUB_BROKER_OWNER_ENV: _OWNER,
        GITHUB_BROKER_SESSION_ENV: _SESSION_ID,
        "GH_TOKEN": "static-pat-must-not-be-inherited",
        "GITHUB_TOKEN": "other-static-pat-must-not-be-inherited",
        "GIT_TOKEN": "legacy-static-pat-must-not-be-inherited",
        "OMNIGENT_GIT_TOKEN": "prefixed-static-pat-must-not-be-inherited",
    }

    provider = consume_github_credential_provider_from_environment(environment)

    assert provider is not None
    assert environment == {"PATH": os.defpath}
    assert "static-pat" not in repr(provider)
    assert _CAPABILITY not in repr(provider)


def test_partial_launch_environment_fails_closed_after_scrubbing_secrets() -> None:
    environment = {
        GITHUB_BROKER_URL_ENV: "https://credentials.example.test/v1/github/grants",
        GITHUB_BROKER_CAPABILITY_ENV: _CAPABILITY,
        "GH_TOKEN": "static-pat-must-not-be-inherited",
    }

    with pytest.raises(RuntimeError, match="incomplete"):
        consume_github_credential_provider_from_environment(environment)

    assert GITHUB_BROKER_CAPABILITY_ENV not in environment
    assert "GH_TOKEN" not in environment


def test_unmanaged_runner_environment_keeps_operator_github_token() -> None:
    environment = {"GH_TOKEN": "local-operator-token"}

    provider = consume_github_credential_provider_from_environment(environment)

    assert provider is None
    assert environment == {"GH_TOKEN": "local-operator-token"}


async def test_owner_scoped_provider_uses_existing_secret_free_audit_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "username": "x-access-token",
                "secret": _TOKEN,
                "expires_at": time.time() + 300,
            },
        )

    audit: list[CredentialAuditEvent] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        bridge = CredentialBrokerBridge(provider, audit_sink=audit.append)
        await bridge.start()
        bridge._executables["gh"] = "/usr/bin/gh"
        turn = ActiveCredentialTurn(_SESSION_ID, "turn_audit", {"run_as": _OWNER})
        bridge.bind_turn(turn)

        class _Popen:
            pid = 12345
            returncode = 0

            def __init__(self, command: Sequence[str], **kwargs: Any) -> None:
                del command
                assert kwargs["env"]["GH_TOKEN"] == _TOKEN

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
        try:
            environment = bridge.wrapper_environment(_SESSION_ID, {"PATH": os.defpath})
            payload = {
                "capability": environment[BROKER_CAPABILITY_ENV],
                "tool": "gh",
                "operation": "execute",
                "argv": ["--repo", "acme/widgets", "pr", "view", "1"],
                "cwd": str(tmp_path),
                "stdin": "",
                "host": "github.com",
            }
            response = await bridge._dispatch(payload)
        finally:
            await bridge.close()

    assert response["ok"] is True
    assert audit == [
        CredentialAuditEvent(
            session_id=_SESSION_ID,
            turn_id="turn_audit",
            actor={"run_as": _OWNER},
            tool="gh",
            action="pr",
            operation="credential",
            outcome="allowed",
        )
    ]
    assert _TOKEN not in repr(response)
    assert _TOKEN not in repr(audit)


async def test_non_owner_denial_emits_secret_free_audit_event(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"owner mismatch must fail before broker request: {request.url}")

    audit: list[CredentialAuditEvent] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = BrokeredGitHubCredentialProvider(_config(), client=client)
        bridge = CredentialBrokerBridge(provider, audit_sink=audit.append)
        await bridge.start()
        turn = ActiveCredentialTurn(
            _SESSION_ID,
            "turn_collaborator",
            {"run_as": "collaborator@example.com"},
        )
        bridge.bind_turn(turn)
        try:
            environment = bridge.wrapper_environment(_SESSION_ID, {"PATH": os.defpath})
            with pytest.raises(PermissionError, match="provider denied"):
                await bridge._dispatch(
                    {
                        "capability": environment[BROKER_CAPABILITY_ENV],
                        "tool": "git",
                        "operation": "execute",
                        "argv": ["status", "--short"],
                        "cwd": str(tmp_path),
                        "stdin": "",
                    }
                )
        finally:
            await bridge.close()

    assert audit == [
        CredentialAuditEvent(
            session_id=_SESSION_ID,
            turn_id="turn_collaborator",
            actor={"run_as": "collaborator@example.com"},
            tool="git",
            action="status",
            operation="identity",
            outcome="denied",
        )
    ]
    assert _CAPABILITY not in repr(audit)
