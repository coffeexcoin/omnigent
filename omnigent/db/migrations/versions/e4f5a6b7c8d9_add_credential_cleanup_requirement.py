"""Distinguish sandbox-only lifecycle rows from credential leases.

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-07-23 22:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e4f5a6b7c8d9"
down_revision: str | None = "d3e4f5a6b7c8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Mark whether a lifecycle row requires credential-provider cleanup."""
    op.add_column(
        "managed_credential_leases",
        sa.Column(
            "credential_cleanup_required",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Remove the credential cleanup discriminator."""
    with op.batch_alter_table("managed_credential_leases") as batch_op:
        batch_op.drop_column("credential_cleanup_required")
