"""Integration coverage for team-scoped session discovery."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from omnigent.runtime.agent_cache import AgentCache
from omnigent.server.app import create_app
from omnigent.server.auth import LEVEL_EDIT, LEVEL_READ, UnifiedAuthProvider
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.artifact_store.local import LocalArtifactStore
from omnigent.stores.comment_store.sqlalchemy_store import SqlAlchemyCommentStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from omnigent.stores.organization_store.sqlalchemy_store import SqlAlchemyOrganizationStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import build_agent_bundle

pytestmark = pytest.mark.asyncio


def _id() -> str:
    return uuid.uuid4().hex


@pytest.fixture()
def team_app(runtime_init: None, db_uri: str, tmp_path: Path) -> FastAPI:
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=artifact_store,
        agent_cache=AgentCache(artifact_store=artifact_store, cache_dir=tmp_path / "cache"),
        comment_store=SqlAlchemyCommentStore(db_uri),
        permission_store=SqlAlchemyPermissionStore(db_uri),
        organization_store=SqlAlchemyOrganizationStore(db_uri),
        auth_provider=UnifiedAuthProvider(source="header", local_single_user=False),
    )


@pytest_asyncio.fixture()
async def team_client(
    team_app: FastAPI,
    mock_llm: ControllableMockClient,
    tmp_path: Path,
) -> AsyncIterator[httpx.AsyncClient]:
    from omnigent.runtime import set_harness_process_manager
    from omnigent.runtime.harnesses.process_manager import HarnessProcessManager

    manager = HarnessProcessManager(tmp_parent=tmp_path / "harness_pm")
    await manager.start()
    set_harness_process_manager(manager)
    transport = httpx.ASGITransport(app=team_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    mock_llm.release_all()
    set_harness_process_manager(None)
    await manager.shutdown()


async def _create_session(client: httpx.AsyncClient, user: str, title: str) -> dict[str, Any]:
    response = await client.post(
        "/v1/sessions",
        data={"metadata": json.dumps({"title": title})},
        files={
            "bundle": (
                "agent.tar.gz",
                build_agent_bundle(name="test-agent"),
                "application/gzip",
            )
        },
        headers={"X-Forwarded-Email": user},
    )
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]
    snapshot = await client.get(
        f"/v1/sessions/{session_id}",
        headers={"X-Forwarded-Email": user},
    )
    assert snapshot.status_code == 200, snapshot.text
    return snapshot.json()


async def test_team_filter_intersects_existing_session_acl(
    team_client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    alice = "alice@example.com"
    bob = "bob@example.com"
    outsider = "outsider@example.com"
    scoped = await _create_session(team_client, alice, "Scoped")
    await _create_session(team_client, alice, "Unscoped")

    organizations = SqlAlchemyOrganizationStore(db_uri)
    organization_id = _id()
    team_id = _id()
    organizations.create_organization(organization_id, "Acme")
    organizations.add_organization_member(organization_id, alice, role="admin")
    organizations.add_organization_member(organization_id, bob)
    organizations.create_team(team_id, organization_id, "Platform")
    organizations.add_team_member(team_id, alice, role="admin")
    organizations.add_team_member(team_id, bob)

    response = await team_client.patch(
        f"/v1/sessions/{scoped['id']}",
        json={"team_id": team_id},
        headers={"X-Forwarded-Email": alice},
    )
    assert response.status_code == 200, response.text
    assert response.json()["team_id"] == team_id

    alice_list = await team_client.get(
        "/v1/sessions",
        params={"team_id": team_id},
        headers={"X-Forwarded-Email": alice},
    )
    assert alice_list.status_code == 200
    assert [item["id"] for item in alice_list.json()["data"]] == [scoped["id"]]
    assert alice_list.json()["data"][0]["team_id"] == team_id

    # Team membership is classification only: Bob sees nothing until the
    # existing direct session ACL grants him read access.
    bob_list = await team_client.get(
        "/v1/sessions",
        params={"team_id": team_id},
        headers={"X-Forwarded-Email": bob},
    )
    assert bob_list.status_code == 200
    assert bob_list.json()["data"] == []

    permissions = SqlAlchemyPermissionStore(db_uri)
    permissions.grant(bob, scoped["id"], LEVEL_READ)
    bob_list = await team_client.get(
        "/v1/sessions",
        params={"team_id": team_id},
        headers={"X-Forwarded-Email": bob},
    )
    assert [item["id"] for item in bob_list.json()["data"]] == [scoped["id"]]
    assert bob_list.json()["data"][0]["team_id"] == team_id

    outsider_list = await team_client.get(
        "/v1/sessions",
        params={"team_id": team_id},
        headers={"X-Forwarded-Email": outsider},
    )
    assert outsider_list.status_code == 404
    assert outsider_list.json()["error"]["code"] == "not_found"

    # A direct session grant does not reveal a team's identity to a user who
    # is outside that team, through either snapshots or unfiltered discovery.
    permissions.grant(outsider, scoped["id"], LEVEL_READ)
    outsider_snapshot = await team_client.get(
        f"/v1/sessions/{scoped['id']}",
        headers={"X-Forwarded-Email": outsider},
    )
    assert outsider_snapshot.status_code == 200
    assert outsider_snapshot.json()["team_id"] is None

    outsider_plain_list = await team_client.get(
        "/v1/sessions",
        headers={"X-Forwarded-Email": outsider},
    )
    assert [item["id"] for item in outsider_plain_list.json()["data"]] == [scoped["id"]]
    assert "team_id" not in outsider_plain_list.json()["data"][0]

    unknown_list = await team_client.get(
        "/v1/sessions",
        params={"team_id": _id()},
        headers={"X-Forwarded-Email": alice},
    )
    assert unknown_list.status_code == 404
    assert unknown_list.json()["error"]["code"] == "not_found"

    malformed_list = await team_client.get(
        "/v1/sessions",
        params={"team_id": "not-a-team-id"},
        headers={"X-Forwarded-Email": alice},
    )
    assert malformed_list.status_code == 404


async def test_team_scope_is_owner_only_and_can_be_cleared(
    team_client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    alice = "alice@example.com"
    editor = "editor@example.com"
    scoped = await _create_session(team_client, alice, "Scoped")

    organizations = SqlAlchemyOrganizationStore(db_uri)
    organization_id = _id()
    team_id = _id()
    organizations.create_organization(organization_id, "Acme")
    for user in (alice, editor):
        organizations.add_organization_member(organization_id, user)
    organizations.create_team(team_id, organization_id, "Platform")
    for user in (alice, editor):
        organizations.add_team_member(team_id, user)

    permissions = SqlAlchemyPermissionStore(db_uri)
    permissions.grant(editor, scoped["id"], LEVEL_EDIT)
    editor_response = await team_client.patch(
        f"/v1/sessions/{scoped['id']}",
        json={"team_id": team_id},
        headers={"X-Forwarded-Email": editor},
    )
    assert editor_response.status_code == 403

    null_response = await team_client.patch(
        f"/v1/sessions/{scoped['id']}",
        json={"team_id": None},
        headers={"X-Forwarded-Email": alice},
    )
    assert null_response.status_code == 400

    unknown_response = await team_client.patch(
        f"/v1/sessions/{scoped['id']}",
        json={"team_id": _id()},
        headers={"X-Forwarded-Email": alice},
    )
    assert unknown_response.status_code == 404

    assigned = await team_client.patch(
        f"/v1/sessions/{scoped['id']}",
        json={"team_id": team_id},
        headers={"X-Forwarded-Email": alice},
    )
    assert assigned.status_code == 200

    cleared = await team_client.patch(
        f"/v1/sessions/{scoped['id']}",
        json={"team_id": ""},
        headers={"X-Forwarded-Email": alice},
    )
    assert cleared.status_code == 200
    assert cleared.json()["team_id"] is None
