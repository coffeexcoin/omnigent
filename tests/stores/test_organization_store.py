"""Tests for organization and team persistence."""

from __future__ import annotations

from typing import get_args
from uuid import uuid4

import pytest

from omnigent.db.db_models import workspace_scope
from omnigent.entities.organization import MembershipRole, ResourceCapability
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.stores.organization_store.sqlalchemy_store import SqlAlchemyOrganizationStore


def _id() -> str:
    return uuid4().hex


def test_membership_roles_and_resource_capabilities_are_explicit() -> None:
    assert get_args(MembershipRole) == ("member", "admin")
    assert get_args(ResourceCapability) == ("view", "edit", "drive", "fork", "admin")


def test_organization_and_team_memberships(db_uri: str) -> None:
    store = SqlAlchemyOrganizationStore(db_uri)
    organization_id = _id()
    alpha_team_id = _id()
    beta_team_id = _id()

    organization = store.create_organization(organization_id, "  Acme  ")
    assert organization.name == "Acme"
    assert store.get_organization(organization_id) == organization

    org_membership = store.add_organization_member(
        organization_id, "alice@example.com", role="admin"
    )
    assert org_membership.role == "admin"

    beta = store.create_team(beta_team_id, organization_id, "Beta")
    alpha = store.create_team(alpha_team_id, organization_id, "Alpha")
    team_membership = store.add_team_member(alpha.id, "alice@example.com", role="admin")
    assert team_membership.role == "admin"
    assert store.is_team_member(alpha.id, "alice@example.com") is True
    assert store.is_team_member(beta.id, "alice@example.com") is False
    assert store.list_teams_for_user("alice@example.com") == [alpha]

    # Membership upserts update the role rather than creating duplicates.
    updated = store.add_team_member(alpha.id, "alice@example.com", role="member")
    assert updated.role == "member"
    assert updated.created_at == team_membership.created_at


def test_team_membership_requires_organization_membership(db_uri: str) -> None:
    store = SqlAlchemyOrganizationStore(db_uri)
    organization_id = _id()
    team_id = _id()
    store.create_organization(organization_id, "Acme")
    store.create_team(team_id, organization_id, "Platform")

    with pytest.raises(OmnigentError) as exc_info:
        store.add_team_member(team_id, "outsider@example.com")

    assert exc_info.value.code == ErrorCode.INVALID_INPUT
    assert "organization" in str(exc_info.value).lower()


def test_names_are_unique_in_their_natural_scope(db_uri: str) -> None:
    store = SqlAlchemyOrganizationStore(db_uri)
    first_org_id = _id()
    second_org_id = _id()
    store.create_organization(first_org_id, "Acme")

    with pytest.raises(OmnigentError) as exc_info:
        store.create_organization(second_org_id, "Acme")
    assert exc_info.value.code == ErrorCode.ALREADY_EXISTS

    team_id = _id()
    store.create_team(team_id, first_org_id, "Platform")
    with pytest.raises(OmnigentError) as exc_info:
        store.create_team(_id(), first_org_id, "Platform")
    assert exc_info.value.code == ErrorCode.ALREADY_EXISTS


def test_organization_and_team_rows_are_workspace_scoped(db_uri: str) -> None:
    store = SqlAlchemyOrganizationStore(db_uri)
    organization_id = _id()
    team_id = _id()
    user_id = "alice@example.com"

    for workspace_id in (101, 202):
        with workspace_scope(workspace_id):
            store.create_organization(organization_id, "Acme")
            store.add_organization_member(organization_id, user_id)
            store.create_team(team_id, organization_id, "Platform")
            store.add_team_member(team_id, user_id)

    for workspace_id in (101, 202):
        with workspace_scope(workspace_id):
            assert store.get_organization(organization_id) is not None
            assert store.get_team(team_id) is not None
            assert store.is_team_member(team_id, user_id) is True
            assert [team.id for team in store.list_teams_for_user(user_id)] == [team_id]

    assert store.get_organization(organization_id) is None
    assert store.get_team(team_id) is None
    assert store.is_team_member(team_id, user_id) is False
