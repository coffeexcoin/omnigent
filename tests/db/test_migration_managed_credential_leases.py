"""Migration validation for the managed credential lease cleanup ledger."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import IntegrityError

from omnigent.db.utils import _build_alembic_config

_PRIOR_REVISION = "c2d3e4f5a6b7"
_THIS_REVISION = "d3e4f5a6b7c8"
_TABLE = "managed_credential_leases"


def _upgrade(engine: sa.Engine, uri: str, revision: str) -> None:
    config = _build_alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


def _downgrade(engine: sa.Engine, uri: str, revision: str) -> None:
    config = _build_alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, revision)


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "workspace_id": 0,
        "host_id": "112233445566478890abcdef12345678",
        "generation": 1,
        "user_id": "alice@example.com",
        "host_name": "managed-migration",
        "sandbox_provider": "kubernetes",
        "sandbox_id": "sandbox-1",
        "session_id": "session-1",
        "repo_url": "https://github.com/acme/repo.git",
        "repo_branch": "main",
        "repo_name": "repo",
        "reference": "non-secret-reference",
        "launch_owner_id": "non-secret-fence-id",
        "owner_expires_at": 2_000_000_000,
        "claim_owner": None,
        "claim_expires_at": None,
        "state": 1,
        "created_at": 1_900_000_000,
        "updated_at": 1_900_000_000,
    }
    row.update(overrides)
    if isinstance(row["host_id"], str):
        row["host_id"] = bytes.fromhex(row["host_id"])
    return row


def test_migration_schema_and_round_trip(tmp_path: Path) -> None:
    """Upgrade creates the durable schema and downgrade removes it safely."""
    uri = f"sqlite:///{tmp_path / 'lease-migration.db'}"
    engine = sa.create_engine(uri)
    try:
        _upgrade(engine, uri, _PRIOR_REVISION)
        assert _TABLE not in sa.inspect(engine).get_table_names()

        _upgrade(engine, uri, _THIS_REVISION)
        inspector = sa.inspect(engine)
        assert _TABLE in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns(_TABLE)}
        assert columns == {
            "workspace_id",
            "host_id",
            "generation",
            "user_id",
            "host_name",
            "sandbox_provider",
            "sandbox_id",
            "session_id",
            "repo_url",
            "repo_branch",
            "repo_name",
            "reference",
            "launch_owner_id",
            "owner_expires_at",
            "claim_owner",
            "claim_expires_at",
            "state",
            "created_at",
            "updated_at",
        }
        assert not {"credential", "secret", "token", "owner_token"} & set(columns)
        assert inspector.get_pk_constraint(_TABLE)["constrained_columns"] == [
            "workspace_id",
            "host_id",
            "generation",
        ]
        assert {index["name"] for index in inspector.get_indexes(_TABLE)} == {
            "ix_managed_credential_leases_claim_recovery",
            "ix_managed_credential_leases_owner_recovery",
        }

        _downgrade(engine, uri, _PRIOR_REVISION)
        assert _TABLE not in sa.inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_migration_constraints_fence_generation_and_state(tmp_path: Path) -> None:
    """The database rejects duplicate generations and impossible lifecycle rows."""
    uri = f"sqlite:///{tmp_path / 'lease-constraints.db'}"
    engine = sa.create_engine(uri)
    try:
        _upgrade(engine, uri, _THIS_REVISION)
        table = sa.Table(_TABLE, sa.MetaData(), autoload_with=engine)
        with engine.begin() as connection:
            connection.execute(table.insert().values(**_valid_row()))

        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(table.insert().values(**_valid_row()))

        invalid_rows = (
            _valid_row(generation=0, host_id="212233445566478890abcdef12345678"),
            _valid_row(state=99, host_id="312233445566478890abcdef12345678"),
            _valid_row(
                state=3,
                host_id="412233445566478890abcdef12345678",
                claim_owner=None,
                claim_expires_at=None,
            ),
            _valid_row(
                state=5,
                host_id="512233445566478890abcdef12345678",
                reference="must-be-cleared",
            ),
        )
        for row in invalid_rows:
            with pytest.raises(IntegrityError):
                with engine.begin() as connection:
                    connection.execute(table.insert().values(**row))
    finally:
        engine.dispose()
