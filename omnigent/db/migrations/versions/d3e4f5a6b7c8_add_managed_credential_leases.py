"""Add durable managed-host credential lease metadata.

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-07-23 18:30:00.000000

Only non-secret cleanup coordinates are stored. Released rows remain with a
NULL reference as generation tombstones, allowing teardown and recovery to use
an idempotent ``(host_id, generation)`` identity without ever persisting raw
credential payloads.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnigent.db.db_models import Uuid16

revision: str = "d3e4f5a6b7c8"
down_revision: str | None = "c2d3e4f5a6b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the durable lease cleanup ledger."""
    op.create_table(
        "managed_credential_leases",
        sa.Column("workspace_id", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("host_id", Uuid16(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("host_name", sa.String(length=64), nullable=False),
        sa.Column("sandbox_provider", sa.String(length=32), nullable=False),
        sa.Column("sandbox_id", sa.String(length=256), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("repo_url", sa.String(length=2048), nullable=True),
        sa.Column("repo_branch", sa.String(length=256), nullable=True),
        sa.Column("repo_name", sa.String(length=256), nullable=True),
        sa.Column("reference", sa.String(length=256), nullable=True),
        # Random fencing identity only; never a provider credential or auth token.
        sa.Column("launch_owner_id", sa.String(length=64), nullable=False),
        sa.Column("owner_expires_at", sa.Integer(), nullable=False),
        sa.Column("claim_owner", sa.String(length=64), nullable=True),
        sa.Column("claim_expires_at", sa.Integer(), nullable=True),
        sa.Column("state", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "state IN (1, 2, 3, 4, 5)",
            name="ck_managed_credential_leases_state",
        ),
        sa.CheckConstraint(
            "generation > 0",
            name="ck_managed_credential_leases_generation_positive",
        ),
        sa.CheckConstraint(
            "(state IN (1, 2) AND claim_owner IS NULL AND claim_expires_at IS NULL) "
            "OR (state IN (3, 4) AND claim_owner IS NOT NULL "
            "AND claim_expires_at IS NOT NULL) "
            "OR (state = 5 AND claim_owner IS NULL AND claim_expires_at IS NULL "
            "AND reference IS NULL)",
            name="ck_managed_credential_leases_lifecycle",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "host_id",
            "generation",
            name="pk_managed_credential_leases",
        ),
    )
    op.create_index(
        "ix_managed_credential_leases_owner_recovery",
        "managed_credential_leases",
        ["workspace_id", "state", "owner_expires_at", "host_id"],
        unique=False,
    )
    op.create_index(
        "ix_managed_credential_leases_claim_recovery",
        "managed_credential_leases",
        ["workspace_id", "state", "claim_expires_at", "host_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the durable lease cleanup ledger."""
    op.drop_index(
        "ix_managed_credential_leases_claim_recovery",
        table_name="managed_credential_leases",
    )
    op.drop_index(
        "ix_managed_credential_leases_owner_recovery",
        table_name="managed_credential_leases",
    )
    op.drop_table("managed_credential_leases")
