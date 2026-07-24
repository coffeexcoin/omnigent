"""Organization and team collaboration entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MembershipRole = Literal["member", "admin"]
ResourceCapability = Literal["view", "edit", "drive", "fork", "admin"]


@dataclass
class Organization:
    """A top-level collaboration boundary."""

    id: str
    name: str
    created_at: int
    updated_at: int | None = None


@dataclass
class OrganizationMembership:
    """A user's role in an organization."""

    organization_id: str
    user_id: str
    role: MembershipRole
    created_at: int


@dataclass
class Team:
    """A named group within an organization."""

    id: str
    organization_id: str
    name: str
    created_at: int
    updated_at: int | None = None


@dataclass
class TeamMembership:
    """A user's role in a team."""

    team_id: str
    user_id: str
    role: MembershipRole
    created_at: int
