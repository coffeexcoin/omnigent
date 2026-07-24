# Organizations, teams, and resource permissions

Status: first implementation slice

## Problem

Omnigent currently has user identities and direct session grants, but no durable
organization or team model. A team-aware product needs two separate concepts:

1. a **scope** that groups resources for discovery and policy, and
2. an **authorization grant** that gives a principal capabilities on a resource.

Conflating those concepts would make moving a session between teams silently
change who can access it. This design keeps them separate.

## Current authentication and authorization path

Authentication is resolved at the HTTP boundary:

1. `create_auth_provider()` selects accounts, OIDC, header, or local mode.
2. Route helpers in `server/routes/_auth_helpers.py` resolve the request actor
   and ensure its `users` row exists.
3. Session routes call `_require_access*`, which delegates to
   `server.permissions.check_session_access()`.
4. `PermissionStore.resolve_access()` reads the actor's direct grant, the public
   grant, and the global-admin bit. Parent-session delegation is applied by the
   server policy layer.
5. `ConversationStore.list_conversations(accessible_by=...)` independently
   applies the same direct-grant boundary to discovery queries.

The two enforcement sites are intentional: object endpoints fail closed, while
collection endpoints must avoid disclosing resource existence in the first
place. Any future team grants must update both paths atomically in one release.

## Domain model

All rows are partitioned by `workspace_id`. Relationships are application
validated rather than database foreign keys, following schema rule R032.

- **Organization**: top-level tenant-owned collaboration boundary.
- **OrganizationMembership**: `(organization_id, user_id)` with role `member`
  or `admin`.
- **Team**: named group inside exactly one organization.
- **TeamMembership**: `(team_id, user_id)` with role `member` or `admin`.
  A team member must also be a member of the team's organization.
- **Session team scope**: nullable `team_id` on conversation metadata. It is
  classification/discovery metadata, not an ACL.

Membership roles answer who may administer containers. They do not encode
resource capabilities.

## Resource permission taxonomy

The target resource ACL uses orthogonal capabilities rather than overloading
membership roles:

- **view**: discover and read resource state/history.
- **edit**: mutate non-execution metadata and content.
- **drive**: submit input, start/stop/resume execution, and answer elicitations.
- **fork**: create a derived resource while preserving provenance.
- **admin**: grant/revoke access, change scope, archive/delete, and transfer
  ownership where supported.

The existing numeric session levels remain unchanged in this slice. A later
migration can map them conservatively:

- read -> view
- edit -> view + edit
- manage -> view + edit + drive + fork + admin
- owner -> all capabilities plus immutable ownership semantics

The missing distinction between edit and drive is why this design does not
rename the existing levels in place.

## Principal and evaluation model

A future grant identifies `(principal_type, principal_id, resource_type,
resource_id, capability_set)`, where principal type is `user` or `team`.
Effective access is the union of:

1. direct user grants,
2. grants to teams of which the actor is currently a member,
3. the existing public-read sentinel, and
4. explicit global-admin bypass.

Deny grants are intentionally excluded. Additive grants are easier to reason
about and avoid order-dependent policy. Removing a user from a team removes the
team-derived contribution immediately; it does not remove independent direct
grants.

Ownership remains separate from ACL grants. Membership in an organization or
team never implies access to every resource in that scope.

## First implementation slice

This slice deliberately establishes the boundary without changing existing
session authorization behavior:

- persist organizations, organization memberships, teams, and team memberships;
- enforce that team members are organization members;
- persist an optional session `team_id` scope;
- allow a session owner to assign a session only to a team they belong to;
- support `GET /v1/sessions?team_id=<id>` for team members;
- intersect the team filter with the existing `accessible_by` direct-session
  ACL filter.

Consequences:

- Team membership alone grants **no** session access.
- A non-member cannot use a team filter to discover scoped sessions.
- Moving a session into or out of a team does not change its grants.
- Existing direct/public/admin checks and permission levels are untouched.

Organization/team administration HTTP APIs and team-principal resource grants
are deferred. Until those land, deployments can provision membership through
the store from their control plane.

## Enforcement rules for the slice

- Unknown and non-member team scopes are both reported as not found at the
  route boundary to avoid team enumeration.
- Assigning/unassigning a team scope is owner-only, matching project filing and
  archive lifecycle operations.
- Explicit `null` is invalid; `""` clears a scope; omitting `team_id` leaves it
  unchanged.
- Store queries always include `workspace_id`.
- Names are trimmed, non-empty, and unique inside their natural parent
  (organization names per workspace; team names per organization).

## Follow-up sequence

1. Add organization/team administration routes with organization-admin and
   team-admin gates.
2. Introduce typed user/team resource grants and a shared effective-access
   resolver.
3. Update both object authorization and collection discovery in the same
   release, with tests proving no existence leak.
4. Split drive from edit at all execution endpoints before exposing the new
   taxonomy publicly.
5. Add project and agent resource types after session parity is complete.
