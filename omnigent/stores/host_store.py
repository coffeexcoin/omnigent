"""
Persistent store for host registrations.

Hosts are machines connected via ``omnigent host``. The store
tracks which hosts have ever connected, their names, user_ids, and
online/offline status. The ``hosts`` table is the source of truth
for ``GET /v1/hosts`` — all server replicas query it. Live WebSocket
connection state is tracked separately in the in-memory
``HostRegistry`` (one per replica).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, and_, func, or_, select, update
from sqlalchemy import delete as sql_delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from omnigent.db.db_models import (
    SqlConversationMetadata,
    SqlHost,
    SqlManagedCredentialLease,
    current_workspace_id,
)
from omnigent.db.enum_codecs import (
    decode_host_status,
    decode_managed_credential_lease_state,
    encode_host_status,
    encode_managed_credential_lease_state,
)
from omnigent.db.utils import get_or_create_engine, make_managed_session_maker, now_epoch
from omnigent.harness_availability import HarnessAvailability, is_harness_availability

# A host is considered live only if its row was touched (connect or
# heartbeat) within this window. The host tunnel's ping loop writes a
# heartbeat every PING_INTERVAL_S (30s); three missed heartbeats means
# the host is gone. This freshness gate is the safety net for every
# path that never runs set_offline — hard crash, OOM, deploy/replica
# restart, silent network drop, or a connect that died after the online
# upsert. It must stay >= the tunnel's ping-miss window
# (PING_INTERVAL_S * PING_MISS_THRESHOLD) so a healthy host that is
# still heart-beating is never falsely aged out.
HOST_LIVENESS_TTL_S = 90


@dataclass
class Host:
    """
    A registered host machine.

    :param host_id: Stable identifier from the host's local
        ``~/.omnigent/config.yaml``, e.g. ``"host_a1b2c3d4..."``.
    :param name: Human-readable name, e.g. ``"corey-laptop"``.
    :param user_id: User ID from the Databricks auth Bearer token,
        e.g. ``"corey.zumar@databricks.com"``.
    :param status: ``"online"`` or ``"offline"``.
    :param created_at: Unix epoch seconds of first registration.
    :param updated_at: Unix epoch seconds the row was last touched —
        a status change (connect/disconnect) or a tunnel heartbeat.
        Used as the host's last-seen for the liveness freshness gate
        (see :data:`HOST_LIVENESS_TTL_S`).
    :param sandbox_provider: Sandbox provider backing a SERVER-MANAGED
        host (``host_type="managed"`` sessions), e.g. ``"modal"``.
        ``None`` for external (user-connected) hosts — non-``None``
        marks the host as server-managed.
    :param sandbox_id: Provider-assigned id of the sandbox currently
        backing a managed host, e.g. ``"sb-a1b2c3"`` — what
        termination is issued against. ``None`` for external hosts.
    :param configured_harnesses: Per-harness readiness reported in the
        host's last ``host.hello`` frame, e.g.
        ``{"claude-sdk": True, "codex": False}``. ``None`` when the
        host has never reported it (older host build) — unknown, not
        "nothing configured".
    """

    host_id: str
    name: str
    user_id: str
    status: str
    created_at: int
    updated_at: int
    sandbox_provider: str | None = None
    sandbox_id: str | None = None
    configured_harnesses: dict[str, HarnessAvailability] | None = None


@dataclass(frozen=True)
class CredentialLeaseRecord:
    """Non-secret durable identity for one managed credential generation."""

    host_id: str
    generation: int
    user_id: str
    host_name: str
    sandbox_provider: str
    sandbox_id: str
    session_id: str | None
    repo_url: str | None
    repo_branch: str | None
    repo_name: str | None
    reference: str | None
    launch_owner_id: str
    owner_expires_at: int
    claim_owner: str | None
    claim_expires_at: int | None
    state: str
    created_at: int
    updated_at: int


def host_is_live(host: Host, now: int | None = None) -> bool:
    """
    Return whether a :class:`Host` is online and recently seen.

    Pure helper over an already-loaded entity (no DB access), so
    callers that already hold a :class:`Host` — or a list of them —
    don't re-query per row. A host is live only when its ``status`` is
    ``"online"`` **and** its last-seen (``updated_at``) is within
    :data:`HOST_LIVENESS_TTL_S`; the freshness half is what catches a
    host that died without a graceful disconnect.

    :param host: The host entity to evaluate.
    :param now: Unix epoch seconds to measure freshness against;
        defaults to the current time. Pass an explicit value to
        classify many hosts against one consistent clock.
    :returns: ``True`` when the host is online and fresh.
    """
    ref = now if now is not None else now_epoch()
    return host.status == "online" and host.updated_at >= ref - HOST_LIVENESS_TTL_S


_logger = logging.getLogger(__name__)


def _rowcount(result: Any) -> int:
    """Return affected rows for SQLAlchemy DML results."""
    return int(getattr(result, "rowcount", 0))


def _parse_configured_harnesses(raw: str | None) -> dict[str, HarnessAvailability] | None:
    """
    Parse the JSON-encoded ``hosts.configured_harnesses`` column.

    Tolerant: ``NULL``, malformed JSON, or a non-object payload all
    map to ``None`` ("unknown") — a corrupt column value must degrade
    to no-warning in the UI, never break host listing. Entries with a
    unsupported readiness value are dropped for the same reason.

    :param raw: The raw column value, e.g.
        ``'{"claude-sdk": true, "codex": false}'`` or ``None``.
    :returns: The readiness map, or ``None`` when absent or unparseable.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _logger.warning("Ignoring malformed hosts.configured_harnesses value")
        return None
    if not isinstance(parsed, dict):
        return None
    return {k: v for k, v in parsed.items() if isinstance(k, str) and is_harness_availability(v)}


def _row_to_host(row: SqlHost) -> Host:
    """
    Convert a :class:`SqlHost` ORM row to a :class:`Host` entity.

    :param row: The SQLAlchemy ORM row to convert.
    :returns: A :class:`Host` dataclass instance.
    """
    return Host(
        host_id=row.host_id,
        name=row.name,
        user_id=row.user_id,
        status=decode_host_status(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        sandbox_provider=row.sandbox_provider,
        sandbox_id=row.sandbox_id,
        configured_harnesses=_parse_configured_harnesses(row.configured_harnesses),
    )


def _row_to_credential_lease(row: SqlManagedCredentialLease) -> CredentialLeaseRecord:
    """Convert a persisted lease row to its non-secret store entity."""
    return CredentialLeaseRecord(
        host_id=row.host_id,
        generation=row.generation,
        user_id=row.user_id,
        host_name=row.host_name,
        sandbox_provider=row.sandbox_provider,
        sandbox_id=row.sandbox_id,
        session_id=row.session_id,
        repo_url=row.repo_url,
        repo_branch=row.repo_branch,
        repo_name=row.repo_name,
        reference=row.reference,
        launch_owner_id=row.launch_owner_id,
        owner_expires_at=row.owner_expires_at,
        claim_owner=row.claim_owner,
        claim_expires_at=row.claim_expires_at,
        state=decode_managed_credential_lease_state(row.state),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _is_credential_generation_conflict(exc: IntegrityError) -> bool:
    """Return whether an insert lost the generation primary-key race."""
    diagnostic = getattr(exc.orig, "diag", None)
    if getattr(diagnostic, "constraint_name", None) == ("pk_managed_credential_leases"):
        return True
    message = str(exc.orig).lower()
    return (
        "managed_credential_leases" in message
        and "generation" in message
        and ("unique" in message or "duplicate" in message)
    )


def hash_host_launch_token(token: str) -> str:
    """
    Digest a managed-host launch token for storage / lookup.

    Only the digest is ever persisted (``hosts.token_hash``), so a
    database leak does not leak usable credentials, and the
    tunnel-side lookup is by digest — the raw token never touches a
    query.

    :param token: The raw launch token, e.g. the value of
        ``secrets.token_urlsafe(32)``.
    :returns: Hex SHA-256 digest, e.g. ``"9f86d08..."`` (64 chars).
    """
    return hashlib.sha256(token.encode()).hexdigest()


class HostStore:
    """
    Persistent store for host registrations backed by SQLAlchemy.

    :param storage_location: SQLAlchemy database URI, e.g.
        ``"sqlite:///hosts.db"``.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the host store.

        :param storage_location: SQLAlchemy database URI, e.g.
            ``"sqlite:///hosts.db"``.
        """
        self._engine: Engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)
        self._write_session = make_managed_session_maker(self._engine, immediate=True)

    def record_credential_lease(
        self,
        *,
        host_id: str,
        user_id: str,
        host_name: str,
        sandbox_provider: str,
        sandbox_id: str,
        session_id: str | None,
        repo_url: str | None,
        repo_branch: str | None,
        repo_name: str | None,
        reference: str | None,
        owner_token: str,
        owner_expires_at: int | None = None,
    ) -> CredentialLeaseRecord:
        """Atomically reserve one pending generation before provider work.

        ``owner_token`` is a random fencing identity, not an authentication
        token. It grants no provider or application access.
        """
        workspace_id = current_workspace_id()
        if owner_expires_at is None:
            # Legacy callers that do not renew ownership are recoverable immediately.
            owner_expires_at = now_epoch()
        deadline = time.monotonic() + 10.0
        attempt = 0
        # BEGIN IMMEDIATE serializes SQLite writers before MAX is read. The
        # primary key remains the final fence on PostgreSQL; a collision is
        # retried in a fresh transaction to a deadline rather than a replica
        # count, so a synchronized burst cannot exhaust a fixed retry budget.
        while True:
            try:
                return self._record_credential_lease_once(
                    workspace_id=workspace_id,
                    now=now_epoch(),
                    host_id=host_id,
                    user_id=user_id,
                    host_name=host_name,
                    sandbox_provider=sandbox_provider,
                    sandbox_id=sandbox_id,
                    session_id=session_id,
                    repo_url=repo_url,
                    repo_branch=repo_branch,
                    repo_name=repo_name,
                    reference=reference,
                    owner_token=owner_token,
                    owner_expires_at=owner_expires_at,
                )
            except IntegrityError as exc:
                if not _is_credential_generation_conflict(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.001 * (2 ** min(attempt, 6)), 0.05))
                attempt += 1

    def _record_credential_lease_once(
        self,
        *,
        workspace_id: int,
        now: int,
        host_id: str,
        user_id: str,
        host_name: str,
        sandbox_provider: str,
        sandbox_id: str,
        session_id: str | None,
        repo_url: str | None,
        repo_branch: str | None,
        repo_name: str | None,
        reference: str | None,
        owner_token: str,
        owner_expires_at: int,
    ) -> CredentialLeaseRecord:
        """Insert one generation inside a single writer transaction."""
        with self._write_session() as session:
            latest = session.execute(
                select(func.max(SqlManagedCredentialLease.generation)).where(
                    SqlManagedCredentialLease.workspace_id == workspace_id,
                    SqlManagedCredentialLease.host_id == host_id,
                )
            ).scalar_one()
            row = SqlManagedCredentialLease(
                workspace_id=workspace_id,
                host_id=host_id,
                generation=(latest or 0) + 1,
                user_id=user_id,
                host_name=host_name,
                sandbox_provider=sandbox_provider,
                sandbox_id=sandbox_id,
                session_id=session_id,
                repo_url=repo_url,
                repo_branch=repo_branch,
                repo_name=repo_name,
                reference=reference,
                launch_owner_id=owner_token,
                owner_expires_at=owner_expires_at,
                state=encode_managed_credential_lease_state("pending"),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.flush()
            return _row_to_credential_lease(row)

    def set_credential_lease_reference(
        self,
        host_id: str,
        generation: int,
        reference: str | None,
        owner_token: str,
    ) -> bool:
        """CAS-persist the non-secret provider handle for the launch owner."""
        now = now_epoch()
        with self._session() as session:
            result = session.execute(
                update(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == current_workspace_id(),
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.generation == generation,
                    SqlManagedCredentialLease.launch_owner_id == owner_token,
                    SqlManagedCredentialLease.owner_expires_at > now,
                    SqlManagedCredentialLease.state
                    == encode_managed_credential_lease_state("pending"),
                )
                .values(reference=reference, updated_at=now)
            )
            return _rowcount(result) == 1

    def renew_credential_lease_owner(
        self,
        host_id: str,
        generation: int,
        *,
        owner_token: str,
        owner_expires_at: int,
    ) -> bool:
        """Extend an unexpired pending launch-owner lease by CAS."""
        now = now_epoch()
        if owner_expires_at <= now:
            return False
        with self._session() as session:
            result = session.execute(
                update(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == current_workspace_id(),
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.generation == generation,
                    SqlManagedCredentialLease.launch_owner_id == owner_token,
                    SqlManagedCredentialLease.owner_expires_at > now,
                    SqlManagedCredentialLease.owner_expires_at < owner_expires_at,
                    SqlManagedCredentialLease.state
                    == encode_managed_credential_lease_state("pending"),
                )
                .values(owner_expires_at=owner_expires_at, updated_at=now)
            )
            return _rowcount(result) == 1

    def list_credential_leases(
        self,
        host_id: str | None = None,
        *,
        include_released: bool = False,
    ) -> list[CredentialLeaseRecord]:
        """List durable leases, omitting released tombstones by default."""
        query = select(SqlManagedCredentialLease).where(
            SqlManagedCredentialLease.workspace_id == current_workspace_id()
        )
        if host_id is not None:
            query = query.where(SqlManagedCredentialLease.host_id == host_id)
        if not include_released:
            query = query.where(
                SqlManagedCredentialLease.state
                != encode_managed_credential_lease_state("released")
            )
        query = query.order_by(
            SqlManagedCredentialLease.host_id,
            SqlManagedCredentialLease.generation,
        )
        with self._session() as session:
            return [_row_to_credential_lease(row) for row in session.scalars(query).all()]

    def activate_credential_lease(
        self,
        host_id: str,
        generation: int,
        owner_token: str,
        *,
        expected_sandbox_id: str | None = None,
    ) -> bool:
        """CAS-promote only the pending generation whose sandbox is online."""
        now = now_epoch()
        exact_online_binding = (
            select(SqlHost.host_id)
            .where(
                SqlHost.workspace_id == current_workspace_id(),
                SqlHost.host_id == host_id,
                SqlHost.sandbox_id == SqlManagedCredentialLease.sandbox_id,
                SqlHost.status == encode_host_status("online"),
            )
            .exists()
        )
        with self._session() as session:
            result = session.execute(
                update(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == current_workspace_id(),
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.generation == generation,
                    SqlManagedCredentialLease.launch_owner_id == owner_token,
                    SqlManagedCredentialLease.owner_expires_at > now,
                    (
                        SqlManagedCredentialLease.sandbox_id.is_not(None)
                        if expected_sandbox_id is None
                        else SqlManagedCredentialLease.sandbox_id == expected_sandbox_id
                    ),
                    SqlManagedCredentialLease.state
                    == encode_managed_credential_lease_state("pending"),
                    exact_online_binding,
                )
                .values(
                    state=encode_managed_credential_lease_state("active"),
                    updated_at=now,
                )
            )
            return _rowcount(result) == 1

    def claim_active_credential_leases(
        self,
        host_id: str,
        *,
        claim_owner: str,
        claim_expires_at: int,
    ) -> list[CredentialLeaseRecord]:
        """Atomically claim active generations before relaunch or teardown."""
        workspace_id = current_workspace_id()
        now = now_epoch()
        if claim_expires_at <= now:
            return []
        with self._write_session() as session:
            rows = session.scalars(
                select(SqlManagedCredentialLease).where(
                    SqlManagedCredentialLease.workspace_id == workspace_id,
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.state
                    == encode_managed_credential_lease_state("active"),
                )
            ).all()
            claimed: list[CredentialLeaseRecord] = []
            for row in rows:
                result = session.execute(
                    update(SqlManagedCredentialLease)
                    .where(
                        SqlManagedCredentialLease.workspace_id == workspace_id,
                        SqlManagedCredentialLease.host_id == row.host_id,
                        SqlManagedCredentialLease.generation == row.generation,
                        SqlManagedCredentialLease.state
                        == encode_managed_credential_lease_state("active"),
                    )
                    .values(
                        state=encode_managed_credential_lease_state("retiring"),
                        claim_owner=claim_owner,
                        claim_expires_at=claim_expires_at,
                        updated_at=now,
                    )
                )
                if _rowcount(result) == 1:
                    row.state = encode_managed_credential_lease_state("retiring")
                    row.claim_owner = claim_owner
                    row.claim_expires_at = claim_expires_at
                    row.updated_at = now
                    claimed.append(_row_to_credential_lease(row))
            return claimed

    def claim_pending_credential_lease(
        self,
        host_id: str,
        generation: int,
        *,
        owner_token: str,
        claim_owner: str,
        claim_expires_at: int,
    ) -> CredentialLeaseRecord | None:
        """Claim this launch's pending generation for failure cleanup."""
        workspace_id = current_workspace_id()
        now = now_epoch()
        if claim_expires_at <= now:
            return None
        with self._write_session() as session:
            result = session.execute(
                update(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == workspace_id,
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.generation == generation,
                    SqlManagedCredentialLease.launch_owner_id == owner_token,
                    SqlManagedCredentialLease.owner_expires_at > now,
                    SqlManagedCredentialLease.state
                    == encode_managed_credential_lease_state("pending"),
                )
                .values(
                    state=encode_managed_credential_lease_state("retiring"),
                    claim_owner=claim_owner,
                    claim_expires_at=claim_expires_at,
                    updated_at=now,
                )
            )
            if _rowcount(result) != 1:
                return None
            row = session.get(SqlManagedCredentialLease, (workspace_id, host_id, generation))
            assert row is not None
            return _row_to_credential_lease(row)

    def renew_credential_lease_claim(
        self,
        host_id: str,
        generation: int,
        *,
        claim_owner: str,
        claim_expires_at: int,
    ) -> bool:
        """Extend an unexpired cleanup claim only while the caller owns it."""
        now = now_epoch()
        if claim_expires_at <= now:
            return False
        with self._session() as session:
            result = session.execute(
                update(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == current_workspace_id(),
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.generation == generation,
                    SqlManagedCredentialLease.claim_owner == claim_owner,
                    SqlManagedCredentialLease.claim_expires_at > now,
                    SqlManagedCredentialLease.claim_expires_at < claim_expires_at,
                    SqlManagedCredentialLease.state.in_(
                        (
                            encode_managed_credential_lease_state("retiring"),
                            encode_managed_credential_lease_state("recovering"),
                        )
                    ),
                )
                .values(
                    claim_expires_at=claim_expires_at,
                    updated_at=now,
                )
            )
            return _rowcount(result) == 1

    def release_credential_lease(
        self,
        host_id: str,
        generation: int,
        *,
        claim_owner: str,
    ) -> bool:
        """CAS-tombstone cleanup only while the caller owns a live claim."""
        now = now_epoch()
        with self._session() as session:
            result = session.execute(
                update(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == current_workspace_id(),
                    SqlManagedCredentialLease.host_id == host_id,
                    SqlManagedCredentialLease.generation == generation,
                    SqlManagedCredentialLease.claim_owner == claim_owner,
                    SqlManagedCredentialLease.claim_expires_at > now,
                    SqlManagedCredentialLease.state.in_(
                        (
                            encode_managed_credential_lease_state("retiring"),
                            encode_managed_credential_lease_state("recovering"),
                        )
                    ),
                )
                .values(
                    state=encode_managed_credential_lease_state("released"),
                    reference=None,
                    claim_owner=None,
                    claim_expires_at=None,
                    updated_at=now,
                )
            )
            return _rowcount(result) == 1

    def claim_recoverable_credential_leases(
        self,
        *,
        claim_owner: str,
        stale_before: int,
        claim_expires_at: int,
        limit: int = 100,
    ) -> list[CredentialLeaseRecord]:
        """Atomically claim stale interrupted work without touching live leases."""
        del stale_before  # Launch-owner expiry is the authoritative liveness fence.
        workspace_id = current_workspace_id()
        now = now_epoch()
        if claim_expires_at <= now:
            return []
        pending = encode_managed_credential_lease_state("pending")
        active = encode_managed_credential_lease_state("active")
        retiring = encode_managed_credential_lease_state("retiring")
        recovering = encode_managed_credential_lease_state("recovering")
        exact_host_binding = (
            select(SqlHost.host_id)
            .where(
                SqlHost.workspace_id == workspace_id,
                SqlHost.host_id == SqlManagedCredentialLease.host_id,
                SqlHost.sandbox_id == SqlManagedCredentialLease.sandbox_id,
            )
            .exists()
        )
        eligible = or_(
            and_(
                SqlManagedCredentialLease.state == pending,
                SqlManagedCredentialLease.owner_expires_at <= now,
            ),
            and_(
                SqlManagedCredentialLease.state.in_((retiring, recovering)),
                or_(
                    SqlManagedCredentialLease.claim_expires_at.is_(None),
                    SqlManagedCredentialLease.claim_expires_at <= now,
                ),
            ),
            and_(SqlManagedCredentialLease.state == active, ~exact_host_binding),
        )
        with self._write_session() as session:
            rows = session.scalars(
                select(SqlManagedCredentialLease)
                .where(
                    SqlManagedCredentialLease.workspace_id == workspace_id,
                    eligible,
                )
                .order_by(
                    SqlManagedCredentialLease.updated_at,
                    SqlManagedCredentialLease.host_id,
                    SqlManagedCredentialLease.generation,
                )
                .limit(limit)
            ).all()
            claimed: list[CredentialLeaseRecord] = []
            for row in rows:
                expected_claim_owner = (
                    SqlManagedCredentialLease.claim_owner.is_(None)
                    if row.claim_owner is None
                    else SqlManagedCredentialLease.claim_owner == row.claim_owner
                )
                expected_claim_expiry = (
                    SqlManagedCredentialLease.claim_expires_at.is_(None)
                    if row.claim_expires_at is None
                    else SqlManagedCredentialLease.claim_expires_at == row.claim_expires_at
                )
                result = session.execute(
                    update(SqlManagedCredentialLease)
                    .where(
                        SqlManagedCredentialLease.workspace_id == workspace_id,
                        SqlManagedCredentialLease.host_id == row.host_id,
                        SqlManagedCredentialLease.generation == row.generation,
                        SqlManagedCredentialLease.state == row.state,
                        SqlManagedCredentialLease.updated_at == row.updated_at,
                        expected_claim_owner,
                        expected_claim_expiry,
                        eligible,
                    )
                    .values(
                        state=recovering,
                        claim_owner=claim_owner,
                        claim_expires_at=claim_expires_at,
                        updated_at=now,
                    )
                )
                if _rowcount(result) == 1:
                    row.state = recovering
                    row.claim_owner = claim_owner
                    row.claim_expires_at = claim_expires_at
                    row.updated_at = now
                    claimed.append(_row_to_credential_lease(row))
            return claimed

    def is_host_bound(self, host_id: str) -> bool:
        """Return whether any durable conversation remains bound to a host."""
        with self._session() as session:
            return (
                session.execute(
                    select(SqlConversationMetadata.id)
                    .where(
                        SqlConversationMetadata.workspace_id == current_workspace_id(),
                        SqlConversationMetadata.host_id == host_id,
                    )
                    .limit(1)
                ).scalar_one_or_none()
                is not None
            )

    def upsert_on_connect(
        self,
        host_id: str,
        name: str,
        user_id: str,
        *,
        allow_host_id_reown: bool = False,
        configured_harnesses: dict[str, HarnessAvailability] | None = None,
    ) -> Host:
        """
        Register or update a host on WebSocket connect.

        Inserts a new row if ``host_id`` does not exist, otherwise
        updates ``name``, ``user_id``, ``status``, and ``updated_at``.
        Called by the host tunnel endpoint when a host sends its
        ``host.hello`` frame.

        The upsert keys on the ``(user_id, name)`` primary key, but
        ``host_id`` carries its own UNIQUE constraint. When the same
        physical host re-registers under a *different* user_id (e.g. a
        local server respawned with a flipped auth posture changes the
        user_id between an accounts user and the reserved ``local`` user),
        the ``(user_id, name)`` lookup misses and a plain INSERT would
        collide on ``host_id``. That collision is a deliberate W2-class
        boundary in shared deployments — a different user must not be
        able to claim another user's host_id — so re-owning is gated
        behind *allow_host_id_reown*, which the server sets only for the
        loopback single-user local server. Remote / multi-user servers
        never set it, so the hijack boundary stays intact (the INSERT
        raises ``IntegrityError`` and fails the handshake closed).

        :param host_id: Stable host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        :param name: Human-readable name from ``config.yaml``, e.g.
            ``"corey-laptop"``.
        :param user_id: Authenticated user ID from the Bearer token,
            e.g. ``"corey.zumar@databricks.com"``.
        :param allow_host_id_reown: When ``True`` and a row already
            exists for *host_id* under a different ``(user_id, name)``,
            re-own that row in place (preserving the ``host_id`` and its
            conversation bindings) instead of inserting. Intended solely
            for the single-user loopback local server.
        :param configured_harnesses: Per-harness readiness from the
            host's ``host.hello`` frame, e.g. ``{"claude-sdk": True}``.
            Written on every connect — including ``None`` from an older
            host that doesn't report it, which correctly resets any
            stale value back to "unknown".
        :returns: The upserted :class:`Host`.
        """
        now = now_epoch()
        harnesses_json = (
            json.dumps(configured_harnesses) if configured_harnesses is not None else None
        )
        with self._session() as session:
            # Primary lookup: by (workspace_id, host_id) — the new PK.
            row = session.get(SqlHost, (current_workspace_id(), host_id))
            if row is not None:
                # W2-class boundary: a different user must not claim another
                # user's host_id. Raise the same IntegrityError the old UNIQUE
                # constraint produced so the tunnel handler rejects the hijack.
                if row.user_id != user_id and not allow_host_id_reown:
                    raise IntegrityError(
                        "host_id already owned by a different user",
                        params={"host_id": host_id, "user_id": user_id},
                        orig=Exception("UNIQUE constraint failed: hosts.host_id"),
                    )
                # Known host_id (same user_id, or reown opted in): update
                # user_id/name in case they changed, then refresh status and timestamp.
                row.user_id = user_id
                row.name = name
                row.status = encode_host_status("online")
                row.updated_at = now
                row.configured_harnesses = harnesses_json
                return _row_to_host(row)

            # host_id is new — check whether (workspace_id, user_id, name)
            # already exists. If it does, the same machine regenerated its
            # identity file: this is a host_id rotation. If allow_host_id_reown
            # is set, also check if any row holds this host_id under a different
            # user_id and re-own it instead of inserting.
            if allow_host_id_reown:
                reowned = self._reown_host_id(
                    session,
                    host_id=host_id,
                    name=name,
                    user_id=user_id,
                    configured_harnesses_json=harnesses_json,
                )
                if reowned is not None:
                    return reowned

            existing_by_name = session.execute(
                select(SqlHost).where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.user_id == user_id,
                    SqlHost.name == name,
                )
            ).scalar_one_or_none()
            if existing_by_name is not None:
                # Same (user_id, name), different host_id: identity rotation.
                # host_id is now part of the PK, so we can't UPDATE it via the
                # ORM — delete the old row and insert a fresh one that carries
                # the new host_id while preserving created_at.
                row = self._rotate_host_id(session, existing_by_name, host_id, now, harnesses_json)
                return _row_to_host(row)

            # Genuinely new host: plain INSERT.
            row = SqlHost(
                user_id=user_id,
                name=name,
                host_id=host_id,
                status=encode_host_status("online"),
                created_at=now,
                updated_at=now,
                configured_harnesses=harnesses_json,
            )
            session.add(row)
            return _row_to_host(row)

    @staticmethod
    def _rotate_host_id(
        session: Session,
        row: SqlHost,
        new_host_id: str,
        now: int,
        harnesses_json: str | None,
    ) -> SqlHost:
        """Replace a host row's host_id while repointing its conversations.

        ``host_id`` is now part of the PK, so an in-place UPDATE is not
        possible via the ORM. The rotation is:

        1. Capture the conversation ids bound to the old host_id.
        2. NULL them so nothing references the old PK value.
        3. DELETE the old row (host_id was the PK member being changed).
        4. INSERT a new row with the new host_id, preserving ``created_at``.
        5. Reattach the captured conversations to the new host_id.

        All steps run inside the caller's transaction so a failure rolls
        the whole upsert back.

        :param session: The active SQLAlchemy session.
        :param row: The existing host row whose ``host_id`` rotates.
        :param new_host_id: The host_id the host reconnected with.
        :param now: Unix epoch seconds for the updated_at timestamp.
        :param harnesses_json: JSON-encoded harness readiness, or None.
        :returns: The newly inserted :class:`SqlHost` row.
        """
        old_host_id = row.host_id
        # Preserve durable fields from the outgoing row before deletion.
        created_at = row.created_at
        user_id = row.user_id
        name = row.name
        token_hash = row.token_hash
        token_expires_at = row.token_expires_at
        sandbox_provider = row.sandbox_provider
        sandbox_id = row.sandbox_id

        bound_ids = list(
            session.execute(
                select(SqlConversationMetadata.id).where(
                    SqlConversationMetadata.workspace_id == current_workspace_id(),
                    SqlConversationMetadata.host_id == old_host_id,
                )
            ).scalars()
        )
        if bound_ids:
            session.execute(
                update(SqlConversationMetadata)
                .where(
                    SqlConversationMetadata.workspace_id == current_workspace_id(),
                    SqlConversationMetadata.host_id == old_host_id,
                )
                .values(host_id=None)
            )
            session.flush()

        # Delete the old PK row and insert a new one with the rotated host_id.
        session.execute(
            sql_delete(SqlHost).where(
                SqlHost.workspace_id == current_workspace_id(),
                SqlHost.host_id == old_host_id,
            )
        )
        session.flush()

        new_row = SqlHost(
            workspace_id=current_workspace_id(),
            host_id=new_host_id,
            user_id=user_id,
            name=name,
            status=encode_host_status("online"),
            created_at=created_at,
            updated_at=now,
            token_hash=token_hash,
            token_expires_at=token_expires_at,
            sandbox_provider=sandbox_provider,
            sandbox_id=sandbox_id,
            configured_harnesses=harnesses_json,
        )
        session.add(new_row)
        session.flush()

        if bound_ids:
            session.execute(
                update(SqlConversationMetadata)
                .where(
                    SqlConversationMetadata.workspace_id == current_workspace_id(),
                    SqlConversationMetadata.id.in_(bound_ids),
                )
                .values(host_id=new_host_id)
            )
            session.flush()

        return new_row

    def _reown_host_id(
        self,
        session: Session,
        *,
        host_id: str,
        name: str,
        user_id: str,
        configured_harnesses_json: str | None = None,
    ) -> Host | None:
        """Re-own an existing host_id row under a new ``(user_id, name)``.

        Used only when ``upsert_on_connect`` opts in via
        ``allow_host_id_reown`` (the single-user loopback local server).
        Updates ``user_id``, ``name``, ``status``, and ``updated_at`` on the
        row that already holds *host_id*, leaving ``host_id`` itself
        unchanged so the ``conversations.host_id`` foreign-key bindings
        survive the user_id change. ``(workspace_id, user_id, name)`` is a
        unique constraint (the PK is ``(workspace_id, host_id)``), so the
        change is issued as a Core ``UPDATE`` rather than loading and
        mutating the ORM object in place.

        :param session: The active SQLAlchemy session.
        :param host_id: Host identifier whose row should be re-owned,
            e.g. ``"host_a1b2c3d4..."``.
        :param name: New host name to record, e.g. ``"corey-laptop"``.
        :param user_id: New user_id to record, e.g. ``"local"`` or
            ``"corey.zumar@databricks.com"``.
        :param configured_harnesses_json: JSON-encoded readiness map from
            the connecting host's hello, e.g.
            ``'{"claude-sdk": true}'``, or ``None`` when unreported.
            Written like the normal connect paths so a re-owned row
            carries fresh (not stale) readiness.
        :returns: The re-owned :class:`Host`, or ``None`` if no row holds
            *host_id* (caller falls through to a normal insert).
        """
        existing = session.execute(
            select(SqlHost).where(
                SqlHost.workspace_id == current_workspace_id(), SqlHost.host_id == host_id
            )
        ).scalar_one_or_none()
        if existing is None:
            return None
        created_at = existing.created_at
        now = now_epoch()
        session.execute(
            update(SqlHost)
            .where(
                SqlHost.workspace_id == current_workspace_id(),
                SqlHost.host_id == host_id,
            )
            .values(
                user_id=user_id,
                name=name,
                status=encode_host_status("online"),
                updated_at=now,
                configured_harnesses=configured_harnesses_json,
            )
        )
        return Host(
            host_id=host_id,
            name=name,
            user_id=user_id,
            status="online",
            created_at=created_at,
            updated_at=now,
            sandbox_provider=existing.sandbox_provider,
            sandbox_id=existing.sandbox_id,
            configured_harnesses=_parse_configured_harnesses(configured_harnesses_json),
        )

    def set_offline(self, host_id: str) -> None:
        """
        Mark a host as offline when its WebSocket disconnects.

        No-op if the host does not exist (the disconnect callback
        may fire after a failed registration).

        :param host_id: Host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        """
        with self._session() as session:
            row = session.execute(
                select(SqlHost).where(
                    SqlHost.workspace_id == current_workspace_id(), SqlHost.host_id == host_id
                )
            ).scalar_one_or_none()
            if row is not None:
                row.status = encode_host_status("offline")
                row.updated_at = now_epoch()

    def update_harness_readiness(
        self,
        host_id: str,
        configured_harnesses: dict[str, HarnessAvailability],
    ) -> None:
        """Replace a connected host's live per-harness readiness map.

        :param host_id: Host identifier, e.g. ``"host_a1b2c3d4..."``.
        :param configured_harnesses: Current readiness keyed by harness spelling.
        """
        with self._session() as session:
            session.execute(
                update(SqlHost)
                .where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.host_id == host_id,
                )
                .values(
                    configured_harnesses=json.dumps(configured_harnesses),
                    updated_at=now_epoch(),
                )
            )

    def heartbeat(self, host_id: str) -> None:
        """
        Refresh a host's last-seen timestamp while its tunnel is alive.

        Bumps ``updated_at`` to now so the liveness freshness gate
        (see :data:`HOST_LIVENESS_TTL_S`) keeps treating the host as
        online. Called from the host tunnel's ping loop every
        ``PING_INTERVAL_S``. Does not change ``status`` — a host whose
        ping loop is running is, by construction, still ``"online"``.

        No-op if the host does not exist.

        :param host_id: Host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        """
        # Single UPDATE rather than SELECT-then-mutate: this runs every
        # ping interval for every connected host, so the extra read is
        # pure overhead. A missing host simply matches no rows (a no-op).
        with self._session() as session:
            session.execute(
                update(SqlHost)
                .where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.host_id == host_id,
                )
                .values(updated_at=now_epoch())
            )

    def is_online(self, host_id: str) -> bool:
        """
        Return whether a host is currently live, cross-replica.

        A host counts as live only when its row is ``status="online"``
        **and** its last-seen (``updated_at``) is within
        :data:`HOST_LIVENESS_TTL_S`. The freshness check is what
        catches a host that died without a graceful disconnect: the
        ``status`` flag alone stays ``"online"`` forever in that case
        (set_offline only runs on a clean tunnel close), so a stale
        timestamp is the only reliable signal that the host is gone.

        :param host_id: Host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        :returns: ``True`` when the host is online and its last-seen is
            fresh; ``False`` if unknown, offline, or stale.
        """
        row = self.get_host(host_id)
        return row is not None and host_is_live(row)

    def online_host_ids(self, host_ids: list[str]) -> set[str]:
        """
        Return the subset of ``host_ids`` that are currently live.

        Bulk variant of :meth:`is_online` for the sidebar online-dot
        batch path: one ``SELECT ... WHERE host_id IN (...)`` instead
        of a per-host query. Liveness applies the same
        status-plus-freshness gate as :meth:`is_online`, classifying
        every row against one consistent clock.

        :param host_ids: Host identifiers to check, e.g.
            ``["host_abc123", "host_def456"]``. Duplicates are
            tolerated; empty input returns an empty set without
            touching the database.
        :returns: The set of ids whose host row is online and fresh.
            Unknown, offline, or stale ids are absent.
        """
        if not host_ids:
            return set()
        unique_ids = list(set(host_ids))
        ref = now_epoch()
        with self._session() as session:
            rows = session.execute(
                select(SqlHost.host_id, SqlHost.status, SqlHost.updated_at).where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.host_id.in_(unique_ids),
                )
            ).all()
        online_code = encode_host_status("online")
        return {
            row.host_id
            for row in rows
            if row.status == online_code and row.updated_at >= ref - HOST_LIVENESS_TTL_S
        }

    def list_hosts(self, user_id: str) -> list[Host]:
        """
        List all hosts owned by a specific user.

        Returns both online and offline hosts, ordered by
        ``updated_at`` descending (most recently active first).

        :param user_id: User ID to filter by, e.g.
            ``"corey.zumar@databricks.com"``.
        :returns: List of :class:`Host` entities.
        """
        with self._session() as session:
            rows = (
                session.query(SqlHost)
                .filter(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.user_id == user_id,
                )
                .order_by(SqlHost.updated_at.desc())
                .all()
            )
            return [_row_to_host(row) for row in rows]

    def get_host(self, host_id: str) -> Host | None:
        """
        Fetch a single host by ID.

        :param host_id: Host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        :returns: The :class:`Host` if found, otherwise ``None``.
        """
        with self._session() as session:
            row = session.execute(
                select(SqlHost).where(
                    SqlHost.workspace_id == current_workspace_id(), SqlHost.host_id == host_id
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return _row_to_host(row)

    def register_managed_host(
        self,
        *,
        host_id: str,
        name: str,
        user_id: str,
        token: str,
        provider: str,
        sandbox_id: str,
        token_expires_at: int,
        expected_sandbox_id: str | None = None,
    ) -> Host:
        """
        Pre-register a server-managed sandbox host with its credential.

        Called by the managed-launch orchestration after the sandbox is
        provisioned and BEFORE the in-sandbox host process starts, so
        the launch token is resolvable by the time the host first dials
        the tunnel. The row is created ``"offline"``; the tunnel's
        normal ``upsert_on_connect`` flips it online when the host
        registers.

        If a row already exists for *host_id* (a RELAUNCH: the host
        identity is durable across sandbox generations so session
        bindings survive a dead sandbox), the credential and sandbox
        columns are overwritten in place — which atomically revokes the
        previous generation's token, since its digest no longer matches
        anything.

        :param host_id: Server-generated host identifier, e.g.
            ``"host_a1b2c3d4..."``.
        :param name: Display name for the host picker, e.g.
            ``"managed-a1b2c3d4"``. Part of the table's
            ``(user_id, name)`` primary key.
        :param user_id: User the managed host acts for, e.g.
            ``"alice@example.com"``.
        :param token: The RAW launch token (hashed here, never stored),
            e.g. the value of ``secrets.token_urlsafe(32)``.
        :param provider: Sandbox provider name, e.g. ``"modal"``.
        :param sandbox_id: Provider-assigned sandbox id, e.g.
            ``"sb-a1b2c3"``.
        :param token_expires_at: Unix epoch seconds after which the
            token no longer authenticates.
        :returns: The registered :class:`Host`.
        :raises ValueError: If a row for *host_id* exists under a
            DIFFERENT user_id — a relaunch may only re-credential a host
            the same user owns.
        """
        now = now_epoch()
        token_hash = hash_host_launch_token(token)
        with self._write_session() as session:
            existing = session.execute(
                select(SqlHost)
                .where(SqlHost.workspace_id == current_workspace_id(), SqlHost.host_id == host_id)
                .with_for_update()
            ).scalar_one_or_none()
            if existing is not None:
                if existing.user_id != user_id:
                    # Fail closed (W2-class boundary): re-crediting a host
                    # row hands its launch token holder the row owner's
                    # identity, so a cross-owner overwrite would be a host
                    # hijack. host_id is server-generated today (uuid4 per
                    # launch), so this can only fire on a bug or a forged
                    # id — refuse rather than re-own.
                    raise ValueError(
                        f"host {host_id!r} is registered to a different user; "
                        "refusing to re-credential it"
                    )
                if expected_sandbox_id is not None and existing.sandbox_id != expected_sandbox_id:
                    raise RuntimeError(f"managed host {host_id!r} changed during relaunch")
                sandbox_changed = existing.sandbox_id != sandbox_id
                existing.token_hash = token_hash
                existing.token_expires_at = token_expires_at
                existing.sandbox_provider = provider
                existing.sandbox_id = sandbox_id
                if sandbox_changed:
                    existing.status = encode_host_status("offline")
                existing.updated_at = now
                return _row_to_host(existing)
            row = SqlHost(
                user_id=user_id,
                name=name,
                host_id=host_id,
                status=encode_host_status("offline"),
                created_at=now,
                updated_at=now,
                token_hash=token_hash,
                token_expires_at=token_expires_at,
                sandbox_provider=provider,
                sandbox_id=sandbox_id,
            )
            session.add(row)
            return _row_to_host(row)

    def resolve_launch_token(self, host_id: str, token: str) -> Host | None:
        """
        Resolve a launch token presented for *host_id* to its managed host.

        The host tunnel's auth path for managed hosts, whose endpoint is
        ``/hosts/{host_id}/tunnel`` — so the connecting peer names the
        host it claims to be, and the token proves the claim. The row is
        fetched by its ``(workspace_id, host_id)`` primary key and the
        stored SHA-256 digest is compared to the presented token's digest
        with :func:`hmac.compare_digest`, so the equality is constant-time
        and leaks no timing oracle on the raw token. Presenting a token
        for the wrong ``host_id`` fails closed: the named row's digest
        won't match. Expired tokens do not authenticate.

        :param host_id: The host the peer claims to be, from the tunnel
            path, e.g. ``"host_a1b2c3d4..."``.
        :param token: The raw token presented by the connecting host.
        :returns: The matching :class:`Host` whose token is unexpired,
            or ``None`` when the host is unknown, the token does not match,
            or the token is expired.
        """
        with self._session() as session:
            row = session.execute(
                select(SqlHost).where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.host_id == host_id,
                )
            ).scalar_one_or_none()
            # token_expires_at is written together with token_hash, so a
            # credentialled row always carries both; a row with either
            # cleared (external host, or a revoked credential) never
            # authenticates.
            if row is None or row.token_hash is None or row.token_expires_at is None:
                return None
            if not hmac.compare_digest(row.token_hash, hash_host_launch_token(token)):
                return None
            if row.token_expires_at < now_epoch():
                return None
            return _row_to_host(row)

    def delete_host(
        self,
        host_id: str,
        *,
        expected_sandbox_id: str | None = None,
    ) -> bool:
        """
        Delete a host row entirely.

        Managed-host teardown: removes the host from the picker AND
        revokes its launch token in one operation (the row IS the
        credential). Explicitly nulls ``conversations.host_id`` for any
        sessions still bound to this host — the DB no longer cascades
        this via FK. No-op when the row does not exist — deletion is
        invoked from best-effort cleanup paths that may race.

        :param host_id: Host identifier, e.g. ``"host_a1b2c3d4..."``.
        """
        with self._write_session() as session:
            host = session.execute(
                select(SqlHost)
                .where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.host_id == host_id,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if host is None or (
                expected_sandbox_id is not None and host.sandbox_id != expected_sandbox_id
            ):
                return False
            session.execute(
                update(SqlConversationMetadata)
                .where(
                    SqlConversationMetadata.workspace_id == current_workspace_id(),
                    SqlConversationMetadata.host_id == host_id,
                )
                .values(host_id=None)
            )
            session.execute(
                sql_delete(SqlHost).where(
                    SqlHost.workspace_id == current_workspace_id(),
                    SqlHost.host_id == host_id,
                )
            )
            return True

    def revoke_launch_token(
        self,
        host_id: str,
        *,
        expected_sandbox_id: str | None = None,
    ) -> bool:
        """
        Clear a managed host's launch credential, keeping the row.

        Relaunch-failure cleanup: a failed sandbox RELAUNCH must revoke
        the token it armed (the new sandbox never came up to use it)
        without deleting the durable host row — the session binding
        survives, and the next relaunch attempt re-arms a fresh token
        via :meth:`register_managed_host`. Contrast :meth:`delete_host`,
        which is full teardown. No-op when the row does not exist.

        :param host_id: Host identifier, e.g. ``"host_a1b2c3d4..."``.
        """
        with self._write_session() as session:
            row = session.execute(
                select(SqlHost)
                .where(SqlHost.workspace_id == current_workspace_id(), SqlHost.host_id == host_id)
                .with_for_update()
            ).scalar_one_or_none()
            if row is None or (
                expected_sandbox_id is not None and row.sandbox_id != expected_sandbox_id
            ):
                return False
            row.token_hash = None
            row.token_expires_at = None
            row.updated_at = now_epoch()
            return True
