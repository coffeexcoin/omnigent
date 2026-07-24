"""Focused tests for durable managed credential lease state transitions."""

from __future__ import annotations

import asyncio
import threading

import pytest
import sqlalchemy as sa

from omnigent.db.db_models import SqlManagedCredentialLease
from omnigent.db.utils import get_or_create_engine, now_epoch
from omnigent.stores.host_store import CredentialLeaseRecord, HostStore, _rowcount

_HOST_ID = "112233445566478890abcdef12345678"
_USER_ID = "alice@example.com"
_HOST_NAME = "managed-foundation"


@pytest.mark.parametrize("value", [None, -1, "1"])
def test_unknown_dml_rowcounts_fail_closed(value: object) -> None:
    """Dialect-specific unknown rowcounts never crash or claim CAS success."""
    result = type("Result", (), {"rowcount": value})()
    assert _rowcount(result) == 0


def _record(
    store: HostStore,
    *,
    sandbox_id: str = "sb-generation-1",
    owner_token: str = "launch-owner-1",
    owner_expires_at: int | None = None,
) -> CredentialLeaseRecord:
    return store.record_credential_lease(
        host_id=_HOST_ID,
        user_id=_USER_ID,
        host_name=_HOST_NAME,
        sandbox_provider="kubernetes",
        sandbox_id=sandbox_id,
        session_id="session-1",
        repo_url="https://github.com/acme/repo.git",
        repo_branch="main",
        repo_name="repo",
        reference=None,
        owner_token=owner_token,
        owner_expires_at=owner_expires_at or now_epoch() + 300,
    )


def _register(store: HostStore, sandbox_id: str) -> None:
    store.register_managed_host(
        host_id=_HOST_ID,
        name=_HOST_NAME,
        user_id=_USER_ID,
        token=f"ephemeral-{sandbox_id}",
        provider="kubernetes",
        sandbox_id=sandbox_id,
        token_expires_at=now_epoch() + 3600,
    )


def _expire_owner(db_uri: str, generation: int) -> None:
    engine = get_or_create_engine(db_uri)
    with engine.begin() as connection:
        connection.execute(
            sa.update(SqlManagedCredentialLease)
            .where(
                SqlManagedCredentialLease.host_id == _HOST_ID,
                SqlManagedCredentialLease.generation == generation,
            )
            .values(owner_expires_at=0, updated_at=0)
        )


def _expire_claim(db_uri: str, generation: int) -> None:
    engine = get_or_create_engine(db_uri)
    with engine.begin() as connection:
        connection.execute(
            sa.update(SqlManagedCredentialLease)
            .where(
                SqlManagedCredentialLease.host_id == _HOST_ID,
                SqlManagedCredentialLease.generation == generation,
            )
            .values(claim_expires_at=0)
        )


def test_lease_identity_and_generation_survive_store_restart(db_uri: str) -> None:
    """The durable row reconstructs cleanup identity without secret material."""
    created = _record(HostStore(db_uri))

    persisted = HostStore(db_uri).list_credential_leases(_HOST_ID)

    assert persisted == [created]
    assert created.generation == 1
    assert created.user_id == _USER_ID
    assert created.host_name == _HOST_NAME
    assert created.sandbox_provider == "kubernetes"
    assert created.sandbox_id == "sb-generation-1"
    assert created.session_id == "session-1"
    assert created.repo_url == "https://github.com/acme/repo.git"
    assert created.repo_branch == "main"
    assert created.repo_name == "repo"
    assert created.reference is None
    assert created.state == "pending"


async def test_concurrent_generation_reservations_are_unique(db_uri: str) -> None:
    """Concurrent replicas reserve monotonically unique generations."""
    store = HostStore(db_uri)

    async def reserve(index: int) -> int:
        record = await asyncio.to_thread(
            _record,
            store,
            sandbox_id=f"sb-generation-{index}",
            owner_token=f"launch-owner-{index}",
        )
        return record.generation

    generations = await asyncio.gather(*(reserve(index) for index in range(32)))

    assert sorted(generations) == list(range(1, 33))


@pytest.mark.parametrize(
    "message",
    [
        "UNIQUE constraint failed: managed_credential_leases.workspace_id, "
        "managed_credential_leases.host_id, managed_credential_leases.generation",
        "(1062, Duplicate entry '0-host-2' for key 'managed_credential_leases.PRIMARY')",
    ],
)
def test_generation_collision_retries_beyond_fixed_replica_budget(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """A synchronized burst is bounded by time, not a small attempt count."""
    store = HostStore(db_uri)
    real_insert = store._record_credential_lease_once
    attempts = 0

    def flaky_insert(**kwargs: object) -> CredentialLeaseRecord:
        nonlocal attempts
        attempts += 1
        if attempts <= 9:
            cause = RuntimeError(message)
            raise sa.exc.IntegrityError("INSERT", {}, cause)
        return real_insert(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "_record_credential_lease_once", flaky_insert)

    record = _record(store)

    assert record.generation == 1
    assert attempts == 10


def test_live_launch_owner_blocks_recovery_and_expired_owner_cannot_renew(
    db_uri: str,
) -> None:
    """Recovery claims only an expired launch-owner lease."""
    store = HostStore(db_uri)
    record = _record(store)

    assert (
        store.claim_recoverable_credential_leases(
            claim_owner="recovery-1",
            stale_before=now_epoch() + 10_000,
            claim_expires_at=now_epoch() + 120,
        )
        == []
    )
    assert store.renew_credential_lease_owner(
        _HOST_ID,
        record.generation,
        owner_token="launch-owner-1",
        owner_expires_at=now_epoch() + 600,
    )

    _expire_owner(db_uri, record.generation)

    assert not store.renew_credential_lease_owner(
        _HOST_ID,
        record.generation,
        owner_token="launch-owner-1",
        owner_expires_at=now_epoch() + 600,
    )
    claimed = store.claim_recoverable_credential_leases(
        claim_owner="recovery-2",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert [(row.generation, row.claim_owner, row.state) for row in claimed] == [
        (record.generation, "recovery-2", "recovering")
    ]


def test_expired_launch_owner_cannot_claim_pending_before_recovery(
    db_uri: str,
) -> None:
    """An expired launcher cannot outrun recovery and seize cleanup authority."""
    store = HostStore(db_uri)
    record = _record(store)
    _expire_owner(db_uri, record.generation)

    assert (
        store.claim_pending_credential_lease(
            _HOST_ID,
            record.generation,
            owner_token="launch-owner-1",
            claim_owner="stale-launcher",
            claim_expires_at=now_epoch() + 120,
        )
        is None
    )
    claimed = store.claim_recoverable_credential_leases(
        claim_owner="recovery",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert [(row.generation, row.claim_owner, row.state) for row in claimed] == [
        (record.generation, "recovery", "recovering")
    ]


async def test_competing_recovery_claimers_have_one_winner(db_uri: str) -> None:
    """Concurrent replicas cannot both acquire cleanup authority."""
    record = _record(HostStore(db_uri))
    _expire_owner(db_uri, record.generation)
    barrier = threading.Barrier(2)

    def claim(owner: str) -> list[CredentialLeaseRecord]:
        barrier.wait()
        return HostStore(db_uri).claim_recoverable_credential_leases(
            claim_owner=owner,
            stale_before=now_epoch(),
            claim_expires_at=now_epoch() + 120,
        )

    first, second = await asyncio.gather(
        asyncio.to_thread(claim, "recovery-1"),
        asyncio.to_thread(claim, "recovery-2"),
    )

    winners = first + second
    assert len(winners) == 1
    assert winners[0].generation == record.generation
    assert winners[0].claim_owner in {"recovery-1", "recovery-2"}
    persisted = HostStore(db_uri).list_credential_leases(
        _HOST_ID,
        include_released=True,
    )
    assert len(persisted) == 1
    assert persisted[0].state == "recovering"
    assert persisted[0].claim_owner == winners[0].claim_owner


def test_non_future_claim_expiries_do_not_authorize_cleanup(db_uri: str) -> None:
    """Claim entry points reject work that would be stale before it is returned."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    active = _record(store)

    assert (
        store.claim_pending_credential_lease(
            _HOST_ID,
            active.generation,
            owner_token="launch-owner-1",
            claim_owner="cleanup",
            claim_expires_at=now_epoch(),
        )
        is None
    )
    assert store.set_credential_lease_reference(
        _HOST_ID, active.generation, "secret-ref", "launch-owner-1"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        active.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )
    assert (
        store.claim_active_credential_leases(
            _HOST_ID,
            claim_owner="cleanup",
            claim_expires_at=now_epoch(),
        )
        == []
    )

    recoverable = _record(
        store,
        sandbox_id="sb-generation-2",
        owner_token="launch-owner-2",
    )
    _expire_owner(db_uri, recoverable.generation)
    assert (
        store.claim_recoverable_credential_leases(
            claim_owner="recovery",
            stale_before=now_epoch(),
            claim_expires_at=now_epoch(),
        )
        == []
    )
    states = {
        row.generation: row.state
        for row in store.list_credential_leases(_HOST_ID, include_released=True)
    }
    assert states == {active.generation: "active", recoverable.generation: "pending"}


def test_claim_renewal_only_extends_expiry(db_uri: str) -> None:
    """A cleanup owner cannot accidentally shorten its own live claim."""
    store = HostStore(db_uri)
    record = _record(store)
    initial_expiry = now_epoch() + 120
    claimed = store.claim_pending_credential_lease(
        _HOST_ID,
        record.generation,
        owner_token="launch-owner-1",
        claim_owner="cleanup",
        claim_expires_at=initial_expiry,
    )
    assert claimed is not None

    assert not store.renew_credential_lease_claim(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup",
        claim_expires_at=initial_expiry,
    )
    assert not store.renew_credential_lease_claim(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup",
        claim_expires_at=initial_expiry - 1,
    )
    assert store.renew_credential_lease_claim(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup",
        claim_expires_at=initial_expiry + 120,
    )
    persisted = store.list_credential_leases(_HOST_ID, include_released=True)
    assert persisted[0].claim_expires_at == initial_expiry + 120


def test_active_claim_is_fenced_to_expected_sandbox_generation(db_uri: str) -> None:
    """A stale teardown cannot claim credentials for a replacement sandbox."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    first = _record(store)
    assert store.set_credential_lease_reference(
        _HOST_ID, first.generation, "secret-ref-1", "launch-owner-1"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        first.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )

    _register(store, "sb-generation-2")
    second = _record(
        store,
        sandbox_id="sb-generation-2",
        owner_token="launch-owner-2",
    )
    assert store.set_credential_lease_reference(
        _HOST_ID, second.generation, "secret-ref-2", "launch-owner-2"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        second.generation,
        "launch-owner-2",
        expected_sandbox_id="sb-generation-2",
    )

    claimed = store.claim_active_credential_leases(
        _HOST_ID,
        claim_owner="stale-cleanup",
        claim_expires_at=now_epoch() + 120,
        expected_sandbox_id="sb-generation-1",
    )

    assert [(row.generation, row.sandbox_id) for row in claimed] == [
        (first.generation, "sb-generation-1")
    ]
    persisted = {
        row.generation: row.state
        for row in store.list_credential_leases(_HOST_ID, include_released=True)
    }
    assert persisted == {first.generation: "retiring", second.generation: "active"}


def test_cleanup_claim_takes_over_expired_retiring_generation(db_uri: str) -> None:
    """A relaunch retry can reclaim an abandoned predecessor cleanup claim."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    record = _record(store)
    assert store.set_credential_lease_reference(
        _HOST_ID, record.generation, "secret-ref", "launch-owner-1"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )
    first_claim = store.claim_active_credential_leases(
        _HOST_ID,
        claim_owner="cleanup-1",
        claim_expires_at=now_epoch() + 120,
        expected_sandbox_id="sb-generation-1",
    )
    assert [row.generation for row in first_claim] == [record.generation]

    _expire_claim(db_uri, record.generation)
    replacement = store.claim_active_credential_leases(
        _HOST_ID,
        claim_owner="cleanup-2",
        claim_expires_at=now_epoch() + 120,
        expected_sandbox_id="sb-generation-1",
    )

    assert [(row.generation, row.claim_owner, row.state) for row in replacement] == [
        (record.generation, "cleanup-2", "retiring")
    ]


def test_reference_activation_and_release_are_cas_fenced(db_uri: str) -> None:
    """Wrong or stale owners cannot mutate another generation."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    record = _record(store)

    assert not store.release_credential_lease(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-1",
    )

    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert not store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )

    assert not store.set_credential_lease_reference(
        _HOST_ID, record.generation, "secret-ref", "wrong-owner"
    )
    assert store.set_credential_lease_reference(
        _HOST_ID, record.generation, "secret-ref", "launch-owner-1"
    )
    assert not store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-other",
    )
    assert store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )
    assert not store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )

    first_claim = store.claim_active_credential_leases(
        _HOST_ID,
        claim_owner="cleanup-1",
        claim_expires_at=now_epoch() + 120,
    )
    assert [row.generation for row in first_claim] == [record.generation]
    assert (
        store.claim_active_credential_leases(
            _HOST_ID,
            claim_owner="cleanup-2",
            claim_expires_at=now_epoch() + 120,
        )
        == []
    )

    _expire_claim(db_uri, record.generation)
    assert not store.renew_credential_lease_claim(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-1",
        claim_expires_at=now_epoch() + 120,
    )
    assert not store.release_credential_lease(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-1",
    )

    replacement = store.claim_recoverable_credential_leases(
        claim_owner="cleanup-2",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert [row.generation for row in replacement] == [record.generation]
    assert not store.release_credential_lease(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-1",
    )
    assert not store.release_credential_lease(
        _HOST_ID,
        record.generation + 1,
        claim_owner="cleanup-2",
    )
    assert store.release_credential_lease(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-2",
    )
    assert not store.release_credential_lease(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-2",
    )
    assert store.list_credential_leases(_HOST_ID) == []
    tombstone = store.list_credential_leases(_HOST_ID, include_released=True)
    assert len(tombstone) == 1
    assert tombstone[0].state == "released"
    assert tombstone[0].reference is None


def test_completed_provider_cleanup_survives_final_cas_retry(db_uri: str) -> None:
    """A failed tombstone CAS does not make recovery repeat provider cleanup."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    record = _record(store)
    assert store.set_credential_lease_reference(
        _HOST_ID, record.generation, "secret-ref", "launch-owner-1"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )
    claimed = store.claim_active_credential_leases(
        _HOST_ID,
        claim_owner="cleanup-1",
        claim_expires_at=now_epoch() + 120,
    )
    assert [row.generation for row in claimed] == [record.generation]

    assert store.complete_credential_lease_cleanup(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-1",
    )
    persisted = store.list_credential_leases(_HOST_ID)[0]
    assert persisted.state == "retiring"
    assert persisted.credential_cleanup_required is False
    assert persisted.reference is None

    _expire_claim(db_uri, record.generation)
    replacement = store.claim_recoverable_credential_leases(
        claim_owner="cleanup-2",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert len(replacement) == 1
    assert replacement[0].credential_cleanup_required is False
    assert store.release_credential_lease(
        _HOST_ID,
        record.generation,
        claim_owner="cleanup-2",
    )


def test_recovery_preserves_only_exact_active_sandbox_binding(db_uri: str) -> None:
    """An active lease is live only while the host points to its sandbox."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    record = _record(store)
    assert store.set_credential_lease_reference(
        _HOST_ID, record.generation, "secret-ref", "launch-owner-1"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )

    assert (
        store.claim_recoverable_credential_leases(
            claim_owner="recovery-1",
            stale_before=now_epoch(),
            claim_expires_at=now_epoch() + 120,
        )
        == []
    )

    _register(store, "sb-generation-2")
    replaced = store.get_host(_HOST_ID)
    assert replaced is not None
    assert replaced.sandbox_id == "sb-generation-2"
    assert replaced.status == "offline"

    claims = store.claim_recoverable_credential_leases(
        claim_owner="recovery-2",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert [row.generation for row in claims] == [record.generation]


def test_provider_change_with_reused_sandbox_id_does_not_preserve_old_lease(
    db_uri: str,
) -> None:
    """A provider is part of the durable sandbox binding, not just its id."""
    store = HostStore(db_uri)
    _register(store, "sb-generation-1")
    record = _record(store)
    assert store.set_credential_lease_reference(
        _HOST_ID, record.generation, "secret-ref", "launch-owner-1"
    )
    store.upsert_on_connect(_HOST_ID, _HOST_NAME, _USER_ID)
    assert store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )

    store.register_managed_host(
        host_id=_HOST_ID,
        name=_HOST_NAME,
        user_id=_USER_ID,
        token="replacement-token",
        provider="modal",
        sandbox_id="sb-generation-1",
        token_expires_at=now_epoch() + 3600,
        expected_sandbox_id="sb-generation-1",
    )
    pending = _record(store, owner_token="launch-owner-2")
    assert not store.activate_credential_lease(
        _HOST_ID,
        pending.generation,
        "launch-owner-2",
        expected_sandbox_id="sb-generation-1",
    )

    assert (
        store.claim_active_credential_leases(
            _HOST_ID,
            claim_owner="modal-cleanup",
            claim_expires_at=now_epoch() + 120,
            expected_sandbox_id="sb-generation-1",
            expected_provider="modal",
        )
        == []
    )
    claims = store.claim_recoverable_credential_leases(
        claim_owner="kubernetes-recovery",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert [row.generation for row in claims] == [record.generation]


def test_recovery_claim_fences_stale_launch_owner_writes(db_uri: str) -> None:
    """A recovery transition prevents a stale launcher from resurrecting a row."""
    store = HostStore(db_uri)
    record = _record(store)
    _expire_owner(db_uri, record.generation)

    claims = store.claim_recoverable_credential_leases(
        claim_owner="recovery",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert [row.generation for row in claims] == [record.generation]
    assert not store.set_credential_lease_reference(
        _HOST_ID,
        record.generation,
        "late-reference",
        "launch-owner-1",
    )
    assert not store.activate_credential_lease(
        _HOST_ID,
        record.generation,
        "launch-owner-1",
        expected_sandbox_id="sb-generation-1",
    )
