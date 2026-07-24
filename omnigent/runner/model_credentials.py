"""Actor-aware model credential resolution for runner harnesses.

The runner is the trust boundary: addon providers receive the authenticated
actor context for the active turn and return process-local environment values.
Harness subprocesses receive credentials but never the actor identity itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from omnigent.policies.schema import ActorContext

MODEL_CREDENTIAL_SCOPE_ENV = "OMNIGENT_MODEL_CREDENTIAL_SCOPE"


@dataclass(frozen=True)
class ModelCredentialRequest:
    """Secret-free metadata describing the model process being authorized."""

    session_id: str
    turn_id: str
    actor: ActorContext
    harness: str
    model: str | None = None


@dataclass(frozen=True)
class ModelCredentialGrant:
    """Actor-scoped model process configuration.

    ``environment`` can contain API keys, gateway coordinates, or native CLI
    home selectors such as ``CLAUDE_CONFIG_DIR`` and ``CODEX_HOME``. Values are
    excluded from repr so logs and assertion failures do not expose secrets.
    ``generation`` is a non-secret provider revision used to rotate a cached
    harness when credentials change without an actor change. ``provider_id``
    and ``billing_account_id`` are non-secret attribution keys used for audit
    and process isolation.
    """

    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    provider_id: str = ""
    billing_account_id: str = ""
    generation: str = ""
    expires_at: float | None = None


def validate_model_credential_grant(
    grant: object,
    *,
    now: float | None = None,
) -> ModelCredentialGrant:
    """Validate an addon grant without including credential values in errors."""

    if not isinstance(grant, ModelCredentialGrant):
        raise TypeError("model credential provider returned an invalid grant")
    if not isinstance(grant.generation, str):
        raise TypeError("model credential grant generation must be a string")
    if not isinstance(grant.provider_id, str):
        raise TypeError("model credential grant provider id must be a string")
    if not isinstance(grant.billing_account_id, str):
        raise TypeError("model credential grant billing account id must be a string")
    if not grant.provider_id.strip():
        raise ValueError("model credential grant provider id must not be empty")
    if not grant.generation.strip():
        raise ValueError("model credential grant generation must not be empty")
    if not isinstance(grant.environment, Mapping):
        raise TypeError("model credential grant environment must be a mapping")
    if not grant.environment:
        raise ValueError("model credential grant environment must not be empty")
    if grant.expires_at is not None:
        if not isinstance(grant.expires_at, (int, float)):
            raise TypeError("model credential grant expiry must be numeric")
        if grant.expires_at <= (time.time() if now is None else now):
            raise ValueError("model credential grant has expired")
    for key, value in grant.environment.items():
        if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
            raise ValueError("model credential grant contains an invalid environment key")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("model credential grant contains an invalid environment value")
    return grant


def credential_scope(request: ModelCredentialRequest, grant: ModelCredentialGrant) -> str:
    """Return a stable, non-secret process-isolation key for a grant."""

    payload = {
        "actor": request.actor,
        "billing_account_id": grant.billing_account_id,
        "generation": grant.generation,
        "harness": request.harness,
        "model": request.model,
        "provider_id": grant.provider_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def apply_model_credential(
    base_env: Mapping[str, str] | None,
    request: ModelCredentialRequest,
    grant: ModelCredentialGrant,
) -> dict[str, str]:
    """Merge a grant into spawn env and attach its non-secret scope key."""

    grant = validate_model_credential_grant(grant)
    merged = dict(base_env or {})
    merged.update(grant.environment)
    merged[MODEL_CREDENTIAL_SCOPE_ENV] = credential_scope(request, grant)
    return merged
