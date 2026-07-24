# Managed GitHub credential provider

## Status and decision

Implemented by:

- `omnigent.server.github_credentials` for concrete managed-launch acquisition;
- `ManagedCredentialHook` and the durable managed-host lease reconciler for
  projection lifecycle;
- `omnigent.runner.github_credentials` for owner-scoped broker grants; and
- the existing `CredentialProvider` / `CredentialBrokerBridge` seam for local
  `git` and `gh` command integration.

Managed hosts use a launch-scoped broker capability, not a static GitHub PAT.
The server resolves the authenticated session owner once for each durable host
generation, asks a deployment control plane to create a provider projection,
and passes only the projection's non-secret reference to the sandbox launcher.
The runner exchanges the capability for short-lived command grants and exposes
them only through local `git`/`gh` wrappers.

The first concrete path is Kubernetes:

```yaml
sandbox:
  provider: kubernetes
  server_url: https://omnigent.example.com
  github_credentials:
    endpoint: https://credentials.example.com/v1/managed-github-leases
    max_lease_ttl_s: 86400
    acquisition_timeout_s: 10
```

`parse_sandbox_config` builds `BrokeredGitHubCredentialHook` and wires it to
`ManagedSandboxConfig.credential_hook`. The Kubernetes launcher projects the
returned Secret name through `envFrom`; no Secret value is copied into a Pod
spec, host row, session row, log, exception, or API response.

## Trust boundaries

- The server authentication layer is authoritative for `owner`. A launch
  request cannot supply or override it.
- The managed-host store reserves `(owner, host_id, generation)` before
  credential acquisition. That tuple is the provider resource identity.
- The deployment credential control plane is trusted to map the owner to the
  right GitHub installation/user identity and create the provider Secret.
- The sandbox receives only a scoped launch capability. It never receives the
  control plane's own authentication or a long-lived user PAT.
- The runner's loopback credential bridge is the only component that exchanges
  the launch capability for command grants. Child processes do not inherit the
  launch capability or ambient GitHub token variables.
- GitHub and the credential control plane remain external trust boundaries.
  Their response and exception text is not propagated into logs or API errors.

## Managed-launch control-plane contract

The configured endpoint implements idempotent `POST` and `DELETE` operations.
Omnigent sends an `Idempotency-Key` derived from owner, host id, and generation;
the header contains only a SHA-256 digest.

`POST` receives secret-free launch identity:

```json
{
  "owner": "authenticated-owner",
  "host_id": "host_...",
  "host_name": "managed-...",
  "generation": 1,
  "session_id": "conversation-id-or-null",
  "repo_url": "https://github.com/org/repo-or-null",
  "repo_branch": "branch-or-null",
  "repo_name": "repo-or-null"
}
```

It resolves the owner and provisions a generation-scoped Kubernetes Secret.
The Secret supplies:

- `OMNIGENT_GITHUB_CREDENTIAL_BROKER_URL`
- `OMNIGENT_GITHUB_CREDENTIAL_BROKER_CAPABILITY`
- `OMNIGENT_GITHUB_CREDENTIAL_OWNER`
- `OMNIGENT_GITHUB_CREDENTIAL_SESSION_ID`

The response is exactly:

```json
{"reference": "rfc1123-kubernetes-secret-name", "expires_at": 1780000000}
```

The reference is non-secret. `expires_at` must be in the future and no later
than `max_lease_ttl_s`. The control plane may use mTLS or an otherwise
preconfigured `httpx.AsyncClient` for service authentication; service
credentials never appear in request JSON.

`DELETE` receives the same identity plus `reference`. `200`, `202`, `204`,
`404`, and `410` are terminal success. Other outcomes raise a sanitized error,
leaving the durable lease available for restart recovery. The control plane
must support deletion from owner, host id, and generation when `reference` is
null because the server may crash after provisioning but before persistence.

For a private repository clone performed by a Kubernetes init container, the
control plane may additionally project a short-lived, repo-scoped clone token.
It must not be a PAT. Prefer a TTL that only covers launch and revoke it when the
host becomes online. A static PAT in the shared Secret is not acceptable.

## Runner broker wire contract

The runner provider sends `POST` to the projected broker URL with its capability
in the `Authorization` header and secret-free JSON:

```json
{
  "session_id": "conv_...",
  "turn_id": "item_...",
  "owner": "owner@example.com",
  "request": {
    "tool": "gh",
    "action": "pr",
    "operation": "credential",
    "protocol": "https",
    "host": "github.com",
    "path": "owner/repository"
  }
}
```

The broker revalidates launch scope and returns:

```json
{
  "username": "x-access-token",
  "secret": "...",
  "expires_at": 1784869500.0,
  "git_user_name": "Repository Owner",
  "git_user_email": "owner@example.com"
}
```

Credential grants must expire in the future and no later than fifteen minutes
from validation. Identity-only Git requests omit `secret` and `expires_at`.
Unknown fields, stale grants, long-lived grants, redirects, oversized responses,
and non-2xx responses fail closed without reflecting broker response text.

Revocation sends `DELETE` to the same URL with `{session_id, owner}`. A broker
must atomically invalidate the capability and every unexpired token issued under
it. Repeated revocation is idempotent.

## Reusable provider interface and precedence

`CredentialProvider.issue(context, request)` is the canonical interface for this
and later actor-aware providers:

1. `context` is the immutable active-turn identity authenticated by the local
   wrapper capability. Authorize session, turn, and actor; never re-read mutable
   session ownership after an await.
2. `request` is secret-free, validated command metadata, not an arbitrary
   command execution API.
3. `CredentialGrant.actor` must exactly equal `context.actor`.
4. Credential secrets are `repr=False`, bounded to GitHub HTTPS, and expire
   within fifteen minutes.
5. Provider exceptions and audit events never contain grant or capability
   values.

The owner-scoped implementation intentionally denies collaborator actors even
when they can drive the session. A later actor-aware provider can reuse the same
interface and resolve `context.actor`; it must not return the session owner's
credential for a different actor.

Credential precedence and process behavior are explicit:

1. Managed broker configuration wins over ambient `GH_TOKEN`, `GITHUB_TOKEN`,
   and `GIT_TOKEN`.
2. Runner startup consumes and removes launch capability and static GitHub token
   variables before agent commands can inherit them.
3. Each `git` or `gh` command requests a grant bound to current session owner,
   turn actor, repository, operation, and command.
4. The wrapper provides the grant through Git credential-helper stdin or a
   command-local `GH_TOKEN`. It is removed immediately after launch and never
   written into git config.
5. Git author name/email are identity metadata, not authentication credentials.

## Expiry, revocation, and audit

- Launch capabilities are bounded by control-plane expiry and
  `max_lease_ttl_s`.
- Per-command GitHub grants are rejected unless they expire within 15 minutes.
- Runner shutdown closes the local bridge first, then revokes the launch
  capability before slower pane/process cleanup.
- A failed remote revocation remains retryable; local use is denied immediately
  after the first revoke attempt.
- Server teardown, relaunch, failed launch, timeout, and restart recovery use the
  durable generation record. Release is idempotent and fenced so an old cleanup
  worker cannot delete a newer generation.

The control plane should audit acquisition and deletion by owner, host, and
generation. The runner's existing `CredentialAuditEvent` records session, turn,
actor, tool, action, operation, and outcome for allowed, denied, and provider
error results. Audit events must never contain capabilities, PATs, grants,
authorization headers, or credential-helper payloads.

## Security test coverage

Regression and integration tests cover:

- owner propagation through concrete acquisition, launcher projection, durable
  persistence, process restart, and deletion;
- strict, bounded launch response parsing and lease expiry enforcement;
- deterministic and retryable release after process loss;
- owner mismatch rejection and 15-minute command-grant expiry;
- local Git credential-helper and `gh` wrapper isolation;
- bounded streaming of oversized broker responses;
- revocation retry after a transient DELETE failure;
- runner cleanup completion when every earlier shutdown step can fail; and
- absence of capabilities and token values from reprs, logs, persisted rows,
  API payloads, and audit events.
