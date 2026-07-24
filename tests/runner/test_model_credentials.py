from __future__ import annotations

from dataclasses import fields

import pytest

from omnigent.runner.model_credentials import (
    MODEL_CREDENTIAL_SCOPE_ENV,
    ModelCredentialGrant,
    ModelCredentialRequest,
    apply_model_credential,
    credential_scope,
    validate_model_credential_grant,
)


def _request(run_as: str = "alice@example.com") -> ModelCredentialRequest:
    return ModelCredentialRequest(
        session_id="conv_test",
        turn_id="turn_test",
        actor={"run_as": run_as},
        harness="claude-sdk",
        model="claude-sonnet",
    )


def test_apply_model_credential_merges_secret_without_exposing_actor() -> None:
    request = _request()
    grant = ModelCredentialGrant(
        environment={"ANTHROPIC_API_KEY": "top-secret"},
        provider_id="anthropic",
        generation="v1",
    )

    merged = apply_model_credential({"EXISTING": "value"}, request, grant)

    assert merged["EXISTING"] == "value"
    assert merged["ANTHROPIC_API_KEY"] == "top-secret"
    assert merged[MODEL_CREDENTIAL_SCOPE_ENV] == credential_scope(request, grant)
    assert request.actor.get("run_as") not in merged.values()


def test_credential_scope_separates_actors_and_provider_generations() -> None:
    alice_v1 = credential_scope(
        _request("alice@example.com"),
        ModelCredentialGrant(
            environment={"TOKEN": "alice-v1"}, provider_id="gateway", generation="v1"
        ),
    )
    alice_v2 = credential_scope(
        _request("alice@example.com"),
        ModelCredentialGrant(
            environment={"TOKEN": "alice-v2"}, provider_id="gateway", generation="v2"
        ),
    )
    bob_v1 = credential_scope(
        _request("bob@example.com"),
        ModelCredentialGrant(
            environment={"TOKEN": "bob-v1"}, provider_id="gateway", generation="v1"
        ),
    )
    alice_other_billing_account = credential_scope(
        _request("alice@example.com"),
        ModelCredentialGrant(
            environment={"TOKEN": "alice-v1"},
            provider_id="gateway",
            billing_account_id="team-b",
            generation="v1",
        ),
    )

    scopes = (alice_v1, alice_v2, bob_v1, alice_other_billing_account)
    assert len(set(scopes)) == 4
    assert all(len(scope) == 64 for scope in scopes)


def test_model_credential_secrets_are_excluded_from_repr() -> None:
    grant = ModelCredentialGrant(
        environment={"OPENAI_API_KEY": "top-secret"},
        provider_id="openai",
        generation="v1",
    )

    assert "top-secret" not in repr(grant)
    assert fields(ModelCredentialGrant)[0].repr is False


@pytest.mark.parametrize(
    "environment",
    [
        {"": "secret"},
        {"BAD=KEY": "secret"},
        {"OPENAI_API_KEY": "secret\x00suffix"},
        {"OPENAI_API_KEY": 123},
    ],
)
def test_invalid_model_credential_grant_does_not_expose_values(
    environment: dict[str, object],
) -> None:
    grant = ModelCredentialGrant(
        environment=environment,  # type: ignore[arg-type]
        provider_id="test",
        generation="v1",
    )

    with pytest.raises((TypeError, ValueError)) as exc_info:
        validate_model_credential_grant(grant)

    assert "secret" not in str(exc_info.value)


def test_model_credential_grant_requires_contract_type() -> None:
    with pytest.raises(TypeError, match="invalid grant"):
        validate_model_credential_grant({"environment": {"OPENAI_API_KEY": "secret"}})


def test_model_credential_grant_rejects_missing_and_expired_credentials() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_model_credential_grant(ModelCredentialGrant())

    with pytest.raises(ValueError, match="provider id must not be empty"):
        validate_model_credential_grant(
            ModelCredentialGrant(environment={"TOKEN": "secret"}, generation="v1")
        )

    with pytest.raises(ValueError, match="generation must not be empty"):
        validate_model_credential_grant(
            ModelCredentialGrant(environment={"TOKEN": "secret"}, provider_id="gateway")
        )

    expired = ModelCredentialGrant(
        environment={"OPENAI_API_KEY": "top-secret"},
        provider_id="openai",
        generation="v1",
        expires_at=99.0,
    )
    with pytest.raises(ValueError, match="expired") as exc_info:
        validate_model_credential_grant(expired, now=100.0)
    assert "top-secret" not in str(exc_info.value)
