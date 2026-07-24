# Actor-aware model credential broker

Status: implemented

## Problem

A session can be shared by multiple authenticated actors. Model credentials therefore cannot be selected once at runner startup or retained across actor takeovers: doing so can charge the wrong account or expose one actor's model access to another.

The runner needs one credential decision for each model invocation while preserving Omnigent's canonical conversation and session resource model.

## Contract

Addons may register a `CredentialProvider[ModelCredentialRequest, ModelCredentialGrant]` with the runner.

A request contains only attribution and routing data:

- session and turn IDs
- the validated `ActorContext`
- harness name
- optional model name

A grant contains:

- an allowlisted child-process environment
- non-secret provider and billing-account attribution IDs
- an opaque provider generation used for rotation
- an optional expiry timestamp

Providers must return a non-empty, unexpired grant. They should change `generation` whenever credential material or billing identity rotates. Resolution failures are fail-closed and client responses never include provider exception text or credential values.

`release_session(session_id)` and `close()` are optional provider lifecycle hooks. The runner invokes them during session release and shutdown when present.

Deployments register the addon object path through `OMNIGENT_MODEL_CREDENTIAL_PROVIDER`.

## Identity and lifetime

`ActorContext` remains the sole actor identity. The broker does not introduce a parallel identity model.

For each message turn, the runner:

1. validates the actor supplied by the authenticated server boundary;
2. binds a turn fence for `(session_id, turn_id, actor)`;
3. asks the provider for a fresh grant;
4. re-checks the fence after the asynchronous provider call;
5. launches or reuses a process only when its credential scope matches; and
6. clears the turn fence when the turn ends.

The non-secret credential scope is a SHA-256 digest over actor identity, harness, model, provider ID, billing-account ID, and provider generation. It never includes environment values. A changed actor, provider, billing account, model, harness, or generation forces `ProcessManager` to release the old subprocess before starting another.

A request without an actor is rejected only when a model credential provider is configured. Installations without the addon retain existing behavior.

## Harness behavior

### API-key-backed harnesses

The provider is resolved on every invocation. Credential environment values are supplied only to the harness subprocess. They are not added to `AgentSpec`, launch configuration, conversation items, resource metadata, audit events, or persisted runner state.

### Native Claude and Codex

The server forwards the validated actor when it asks the runner to ensure a native terminal. The runner resolves that actor's grant before terminal creation and rotates the terminal, auto-forwarder, and cached harness process when the scope changes.

Provider environments identify the actor's credential home (for example `CLAUDE_CONFIG_DIR` or `CODEX_HOME`). Native adapters use those locations without copying OAuth credential files into shared launch configuration. Codex may create a per-session runtime home that symlinks the actor's OAuth file while retaining the session rollout directory; Claude materializes the canonical Omnigent transcript in the selected actor home before resume. The Omnigent session ID and persisted conversation items remain canonical across takeover.

Only one actor can own an active model turn for a session. Terminal takeover during another actor's active turn returns a conflict. Setup that loses its turn fence after asynchronous credential resolution is denied; a native terminal that loses authority during launch is closed before it can be used.

## Persistence and observability

Raw credential values are process-local and intentionally absent from:

- audit events and logs;
- API error bodies;
- session resources and conversation items;
- agent specs and launch configuration; and
- credential scope digests.

Audit events contain session, turn, actor, harness, provider ID, billing-account ID, operation, and outcome only. Provider and audit-sink failures are logged by exception type, not message.

## Cleanup

Session release removes the actor-turn fence and native scope, releases cached subprocesses, closes terminal resources, deletes native bridge/runtime directories through existing harness cleanup, and calls the provider's optional session hook. Runner shutdown clears remaining broker state and calls the optional provider close hook.

## Security invariants

- Credentials are resolved from the current validated actor for every invocation.
- No credential value is used as a cache key, audit field, resource field, or persisted launch field.
- Cross-actor process reuse is impossible because actor identity is part of the scope digest.
- Provider generation changes rotate same-actor processes after credential refresh.
- Missing, malformed, or expired grants fail before process launch.
- Stale setup work cannot publish or retain a native terminal after losing its turn fence.
