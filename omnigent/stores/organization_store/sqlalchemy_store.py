"""SQLAlchemy-backed organization and team store."""

from __future__ import annotations

from sqlalchemy import asc, select
from sqlalchemy.exc import IntegrityError

from omnigent.db.db_models import (
    SqlOrganization,
    SqlOrganizationMembership,
    SqlTeam,
    SqlTeamMembership,
    current_workspace_id,
)
from omnigent.db.utils import get_or_create_engine, make_managed_session_maker, now_epoch
from omnigent.entities.organization import (
    MembershipRole,
    Organization,
    OrganizationMembership,
    Team,
    TeamMembership,
)
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.stores.organization_store import OrganizationStore

_ROLE_TO_CODE: dict[MembershipRole, int] = {"member": 1, "admin": 2}
_CODE_TO_ROLE: dict[int, MembershipRole] = {code: role for role, code in _ROLE_TO_CODE.items()}


def _clean_name(name: str, *, resource: str) -> str:
    value = name.strip()
    if not value:
        raise OmnigentError(f"{resource} name must not be empty", code=ErrorCode.INVALID_INPUT)
    if len(value) > 256:
        raise OmnigentError(
            f"{resource} name must be at most 256 characters",
            code=ErrorCode.INVALID_INPUT,
        )
    return value


def _encode_role(role: MembershipRole) -> int:
    try:
        return _ROLE_TO_CODE[role]
    except KeyError:
        raise OmnigentError(
            f"Unknown membership role: {role!r}",
            code=ErrorCode.INVALID_INPUT,
        ) from None


def _organization_entity(row: SqlOrganization) -> Organization:
    return Organization(
        id=row.id,
        name=row.name,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _organization_membership_entity(
    row: SqlOrganizationMembership,
) -> OrganizationMembership:
    return OrganizationMembership(
        organization_id=row.organization_id,
        user_id=row.user_id,
        role=_CODE_TO_ROLE[row.role],
        created_at=row.created_at,
    )


def _team_entity(row: SqlTeam) -> Team:
    return Team(
        id=row.id,
        organization_id=row.organization_id,
        name=row.name,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _team_membership_entity(row: SqlTeamMembership) -> TeamMembership:
    return TeamMembership(
        team_id=row.team_id,
        user_id=row.user_id,
        role=_CODE_TO_ROLE[row.role],
        created_at=row.created_at,
    )


class SqlAlchemyOrganizationStore(OrganizationStore):
    """Persist workspace-scoped organization and team membership data."""

    def __init__(self, storage_location: str) -> None:
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)

    def create_organization(self, organization_id: str, name: str) -> Organization:
        clean_name = _clean_name(name, resource="Organization")
        with self._session() as session:
            row = SqlOrganization(
                id=organization_id,
                name=clean_name,
                created_at=now_epoch(),
                updated_at=None,
            )
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise OmnigentError(
                    f"An organization named {clean_name!r} already exists",
                    code=ErrorCode.ALREADY_EXISTS,
                ) from exc
            return _organization_entity(row)

    def get_organization(self, organization_id: str) -> Organization | None:
        with self._session() as session:
            row = session.get(SqlOrganization, (current_workspace_id(), organization_id))
            return None if row is None else _organization_entity(row)

    def add_organization_member(
        self,
        organization_id: str,
        user_id: str,
        *,
        role: MembershipRole = "member",
    ) -> OrganizationMembership:
        role_code = _encode_role(role)
        with self._session() as session:
            organization = session.get(SqlOrganization, (current_workspace_id(), organization_id))
            if organization is None:
                raise OmnigentError("Organization not found", code=ErrorCode.NOT_FOUND)
            key = (current_workspace_id(), organization_id, user_id)
            row = session.get(SqlOrganizationMembership, key)
            if row is None:
                row = SqlOrganizationMembership(
                    organization_id=organization_id,
                    user_id=user_id,
                    role=role_code,
                    created_at=now_epoch(),
                )
                session.add(row)
            else:
                row.role = role_code
            session.flush()
            return _organization_membership_entity(row)

    def get_organization_membership(
        self, organization_id: str, user_id: str
    ) -> OrganizationMembership | None:
        with self._session() as session:
            row = session.get(
                SqlOrganizationMembership,
                (current_workspace_id(), organization_id, user_id),
            )
            return None if row is None else _organization_membership_entity(row)

    def create_team(self, team_id: str, organization_id: str, name: str) -> Team:
        clean_name = _clean_name(name, resource="Team")
        with self._session() as session:
            organization = session.get(SqlOrganization, (current_workspace_id(), organization_id))
            if organization is None:
                raise OmnigentError("Organization not found", code=ErrorCode.NOT_FOUND)
            row = SqlTeam(
                id=team_id,
                organization_id=organization_id,
                name=clean_name,
                created_at=now_epoch(),
                updated_at=None,
            )
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                raise OmnigentError(
                    f"A team named {clean_name!r} already exists in this organization",
                    code=ErrorCode.ALREADY_EXISTS,
                ) from exc
            return _team_entity(row)

    def get_team(self, team_id: str) -> Team | None:
        with self._session() as session:
            row = session.get(SqlTeam, (current_workspace_id(), team_id))
            return None if row is None else _team_entity(row)

    def add_team_member(
        self,
        team_id: str,
        user_id: str,
        *,
        role: MembershipRole = "member",
    ) -> TeamMembership:
        role_code = _encode_role(role)
        with self._session() as session:
            team = session.get(SqlTeam, (current_workspace_id(), team_id))
            if team is None:
                raise OmnigentError("Team not found", code=ErrorCode.NOT_FOUND)
            organization_member = session.get(
                SqlOrganizationMembership,
                (current_workspace_id(), team.organization_id, user_id),
            )
            if organization_member is None:
                raise OmnigentError(
                    "A team member must belong to the team's organization",
                    code=ErrorCode.INVALID_INPUT,
                )
            key = (current_workspace_id(), team_id, user_id)
            row = session.get(SqlTeamMembership, key)
            if row is None:
                row = SqlTeamMembership(
                    team_id=team_id,
                    user_id=user_id,
                    role=role_code,
                    created_at=now_epoch(),
                )
                session.add(row)
            else:
                row.role = role_code
            session.flush()
            return _team_membership_entity(row)

    def get_team_membership(self, team_id: str, user_id: str) -> TeamMembership | None:
        with self._session() as session:
            row = session.get(
                SqlTeamMembership,
                (current_workspace_id(), team_id, user_id),
            )
            return None if row is None else _team_membership_entity(row)

    def list_teams_for_user(self, user_id: str) -> list[Team]:
        with self._session() as session:
            stmt = (
                select(SqlTeam)
                .join(
                    SqlTeamMembership,
                    (SqlTeamMembership.workspace_id == SqlTeam.workspace_id)
                    & (SqlTeamMembership.team_id == SqlTeam.id),
                )
                .where(
                    SqlTeam.workspace_id == current_workspace_id(),
                    SqlTeamMembership.user_id == user_id,
                )
                .order_by(asc(SqlTeam.name), asc(SqlTeam.id))
            )
            return [_team_entity(row) for row in session.execute(stmt).scalars().all()]
