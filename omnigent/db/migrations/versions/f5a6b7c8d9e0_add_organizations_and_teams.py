"""add organizations, teams, memberships, and session team scope

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-07-24 00:00:00.000000

First organization/team permission slice. Team scope is discovery metadata and
does not grant session access. Relationships are application validated rather
than foreign-key constrained (schema rule R032).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnigent.db.db_models import Uuid16

revision: str = "f5a6b7c8d9e0"
down_revision: str | None = "e4f5a6b7c8d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create organization/team rows and add the session team scope."""
    op.create_table(
        "organizations",
        sa.Column("workspace_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("id", Uuid16(), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("workspace_id", "id"),
    )
    op.create_index(
        "ix_organizations_name",
        "organizations",
        ["workspace_id", "name"],
        unique=True,
    )

    op.create_table(
        "organization_memberships",
        sa.Column("workspace_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("organization_id", Uuid16(), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("role", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.CheckConstraint("role IN (1, 2)", name="ck_organization_memberships_role"),
        sa.PrimaryKeyConstraint("workspace_id", "organization_id", "user_id"),
    )
    op.create_index(
        "ix_organization_memberships_user_id",
        "organization_memberships",
        ["workspace_id", "user_id", "organization_id"],
        unique=False,
    )

    op.create_table(
        "teams",
        sa.Column("workspace_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("id", Uuid16(), nullable=False),
        sa.Column("organization_id", Uuid16(), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("workspace_id", "id"),
    )
    op.create_index(
        "ix_teams_organization_name",
        "teams",
        ["workspace_id", "organization_id", "name"],
        unique=True,
    )

    op.create_table(
        "team_memberships",
        sa.Column("workspace_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("team_id", Uuid16(), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("role", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.CheckConstraint("role IN (1, 2)", name="ck_team_memberships_role"),
        sa.PrimaryKeyConstraint("workspace_id", "team_id", "user_id"),
    )
    op.create_index(
        "ix_team_memberships_user_id",
        "team_memberships",
        ["workspace_id", "user_id", "team_id"],
        unique=False,
    )

    with op.batch_alter_table("omnigent_conversation_metadata") as batch_op:
        batch_op.add_column(sa.Column("team_id", Uuid16(), nullable=True))
    op.create_index(
        "ix_conversation_metadata_team_id",
        "omnigent_conversation_metadata",
        ["workspace_id", "team_id", "id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove session scope and organization/team persistence."""
    op.drop_index(
        "ix_conversation_metadata_team_id",
        table_name="omnigent_conversation_metadata",
    )
    with op.batch_alter_table("omnigent_conversation_metadata") as batch_op:
        batch_op.drop_column("team_id")

    op.drop_index("ix_team_memberships_user_id", table_name="team_memberships")
    op.drop_table("team_memberships")
    op.drop_index("ix_teams_organization_name", table_name="teams")
    op.drop_table("teams")
    op.drop_index(
        "ix_organization_memberships_user_id",
        table_name="organization_memberships",
    )
    op.drop_table("organization_memberships")
    op.drop_index("ix_organizations_name", table_name="organizations")
    op.drop_table("organizations")
