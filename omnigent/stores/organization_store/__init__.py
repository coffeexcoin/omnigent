"""Organization/team persistence interfaces.

Team membership is a subset of organization membership. Session scopes live on
conversation metadata and are managed by ``ConversationStore``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from omnigent.entities.organization import (
    MembershipRole,
    Organization,
    OrganizationMembership,
    Team,
    TeamMembership,
)


class OrganizationStore(ABC):
    """Persist organizations, teams, and their memberships."""

    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location

    @abstractmethod
    def create_organization(self, organization_id: str, name: str) -> Organization:
        """Create an organization with a workspace-unique name."""
        ...

    @abstractmethod
    def get_organization(self, organization_id: str) -> Organization | None:
        """Return an organization by id, or ``None``."""
        ...

    @abstractmethod
    def add_organization_member(
        self,
        organization_id: str,
        user_id: str,
        *,
        role: MembershipRole = "member",
    ) -> OrganizationMembership:
        """Upsert a user's organization membership."""
        ...

    @abstractmethod
    def get_organization_membership(
        self, organization_id: str, user_id: str
    ) -> OrganizationMembership | None:
        """Return a user's organization membership, or ``None``."""
        ...

    @abstractmethod
    def create_team(self, team_id: str, organization_id: str, name: str) -> Team:
        """Create a team inside an existing organization."""
        ...

    @abstractmethod
    def get_team(self, team_id: str) -> Team | None:
        """Return a team by id, or ``None``."""
        ...

    @abstractmethod
    def add_team_member(
        self,
        team_id: str,
        user_id: str,
        *,
        role: MembershipRole = "member",
    ) -> TeamMembership:
        """Upsert team membership for an organization member."""
        ...

    @abstractmethod
    def get_team_membership(self, team_id: str, user_id: str) -> TeamMembership | None:
        """Return a user's team membership, or ``None``."""
        ...

    def is_team_member(self, team_id: str, user_id: str) -> bool:
        """Return whether ``user_id`` belongs to ``team_id``."""
        return self.get_team_membership(team_id, user_id) is not None

    @abstractmethod
    def list_teams_for_user(self, user_id: str) -> list[Team]:
        """List teams the user belongs to, ordered by name and id."""
        ...
