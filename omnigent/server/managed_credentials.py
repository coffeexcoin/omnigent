"""Per-launch credential seam for server-managed sandbox hosts.

The managed-host flow (:mod:`omnigent.server.managed_hosts`) provisions a
sandbox, mints a launch token, and starts ``omnigent host`` inside it — the
user's own harness/VCS credentials never enter that sandbox by default. Some
deployments want the opposite: each managed host should act with the SESSION
OWNER's credentials (their GitHub token, their Claude/Codex login), resolved
per launch by a control plane sitting alongside the server.

This module is the generic, provider-neutral seam for that, and nothing more.
It carries no credential-provider behavior of its own: an addon implements
:class:`ManagedCredentialHook`, a deployment wires it onto
``ManagedSandboxConfig.credential_hook`` (the same direct-construction path
that injects a custom launcher factory), and the managed-launch orchestration
calls it once per launch. When no hook is configured the managed flow behaves
exactly as before — no credentials are resolved and no lease is acquired.

Three types make up the contract:

* :class:`ManagedLaunchContext` — the immutable launch identity handed to the
  hook (owner, host id/name, repository/workspace metadata, session id). It is
  everything the seam can honestly promise is available at the existing launch
  boundary; it never carries secrets.
* :class:`ManagedCredentialLease` — an addon-returned handle to the resolved
  credentials for ONE launch generation. It deliberately does NOT carry the
  secret material through the server: it exposes only a non-secret
  :attr:`~ManagedCredentialLease.reference` (e.g. the name of a Secret the
  addon created out of band) plus lifecycle hooks.
* :class:`ManagedCredentialHook` — the async resolver an addon implements.

**Ownership and cleanup (read before implementing a hook).** Before acquisition,
Omnigent durably reserves a generation and passes it to the hook. Implementations
must use the launch identity plus generation as a deterministic cleanup key so a
restart can release credentials created before :meth:`acquire` returned. Only
the non-secret reference is persisted after acquisition. A failed launch
releases the live lease object; successful and interrupted leases are released
through :meth:`ManagedCredentialHook.release` on teardown, relaunch, or restart
recovery. Both release methods must be idempotent; transient failures raise a
sanitized exception so the durable cleanup record remains retryable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ManagedLaunchContext:
    """
    Immutable launch identity handed to a :class:`ManagedCredentialHook`.

    Constructed by the managed-launch orchestration at the point a sandbox
    host is armed, so a hook can resolve the right user's credentials for the
    right launch generation. Every field is a plain identifier or repository
    coordinate that already exists at the launch boundary — the seam invents
    nothing. Frozen so a hook cannot mutate the identity mid-resolution, and
    secret-free by construction, so it is safe to log.

    :param owner: The user the managed host acts for — the session creator,
        e.g. ``"alice@example.com"`` (or the reserved local user on
        auth-disabled servers). The primary key a per-user credential hook
        resolves against.
    :param host_id: The server-chosen host identity for this managed host,
        e.g. ``"a1b2c3d4..."``. Durable across relaunches (a relaunch keeps
        the id and provisions a fresh sandbox generation under it).
    :param host_name: The host's display label, e.g. ``"managed-a1b2c3d4"``.
    :param session_id: The session/conversation this launch was kicked for,
        e.g. ``"conv_abc123"``, or ``None`` when a launch path has no session
        in hand. Present for both first-launch and relaunch through the
        background-provision task; never fabricated when genuinely absent.
    :param repo_url: Clone URL of the repository the session's workspace is
        materialized from, e.g. ``"https://github.com/org/repo.git"``, or
        ``None`` for an empty workspace. The coordinate a VCS-credential hook
        scopes a token to.
    :param repo_branch: Branch the workspace is cloned at, e.g.
        ``"release-1.2"``, or ``None`` for the default branch / no repo.
    :param repo_name: Directory the clone lands in under the sandbox
        workspace, e.g. ``"repo"``, or ``None`` when there is no repo.
    """

    owner: str
    host_id: str
    host_name: str
    session_id: str | None = None
    repo_url: str | None = None
    repo_branch: str | None = None
    repo_name: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedCredentialReleaseContext:
    """Durable identity available to cleanup after a process crash.

    This mirrors every non-secret coordinate from :class:`ManagedLaunchContext`
    and adds the concrete sandbox identity so a hook can deterministically
    reconstruct the exact provider resource after the acquiring process exits.
    """

    owner: str
    host_id: str
    host_name: str
    sandbox_provider: str
    sandbox_id: str
    session_id: str | None = None
    repo_url: str | None = None
    repo_branch: str | None = None
    repo_name: str | None = None


class ManagedCredentialLease(ABC):
    """
    A resolved per-launch credential lease returned by a hook.

    One lease corresponds to ONE managed-host launch generation. It is a
    HANDLE, not a secret container: the addon resolves and delivers the actual
    credential material out of band (today, into whatever store the launcher's
    provider reads (the concrete GitHub hook uses a Kubernetes Secret), and hands
    the server back only a non-secret :attr:`reference` plus lifecycle hooks.
    Keeping secret payloads off this object is what lets the managed-launch
    layer log, repr, and error around leases freely without leaking anything.

    Subclasses MUST NOT store raw secret material on the instance, and MUST NOT
    widen :meth:`__repr__` / ``__str__`` to expose any. The base
    :meth:`__repr__` intentionally reflects only the class name and the
    (non-secret) reference, so a subclass that adds private credential fields
    still reprs safely by default.

    This live object's release method handles failures in the process that
    acquired it; durable cleanup uses :meth:`ManagedCredentialHook.release`.
    """

    @property
    @abstractmethod
    def reference(self) -> str | None:
        """
        A non-secret handle the launcher/provider can resolve, or ``None``.

        Names WHERE the resolved credentials live for this launch (e.g. the
        name of a Kubernetes Secret the addon created), never the credentials
        themselves. ``None`` when the addon injects credentials by a channel
        that needs no server-visible handle. Must be safe to log.
        """

    async def release(self) -> None:  # noqa: B027 — intentional concrete no-op
        # default: addons whose credentials need no teardown inherit it, so it
        # must NOT be @abstractmethod.
        """
        Release / revoke this lease.

        Called when the launch this live object belongs to fails. Teardown after
        the acquiring process exits is reconstructed through the hook instead.

        MUST be idempotent — a second call, or a call on a lease whose
        credentials were never fully provisioned, is a no-op. Transient failures
        should raise a sanitized exception so durable cleanup remains retryable.
        The default implementation is a no-op for credentials needing no
        explicit teardown.
        """

    def __repr__(self) -> str:
        """Return a representation that never resolves or exposes provider state."""
        return f"{type(self).__name__}(reference=[REDACTED], secret=[REDACTED])"


class ManagedCredentialHook(ABC):
    """
    Resolver an addon implements to supply per-launch managed-host credentials.

    A control plane (e.g. a Switchyard addon) implements
    :meth:`acquire` to map a :class:`ManagedLaunchContext` to the owner's
    resolved credentials, delivered as a :class:`ManagedCredentialLease`. A
    deployment wires the instance onto ``ManagedSandboxConfig.credential_hook``;
    the managed-launch orchestration then invokes it exactly once per launch
    attempt (first launch and relaunch alike), immediately before the
    in-sandbox host process starts.

    Leaving ``credential_hook`` unset (the default) keeps the managed flow's
    original behavior: no hook is consulted and no lease is acquired.
    """

    @abstractmethod
    async def acquire(
        self,
        context: ManagedLaunchContext,
        generation: int,
    ) -> ManagedCredentialLease:
        """Resolve credentials for one launch and return their lease.

        Called exactly once per managed-host launch attempt, after *generation*
        is durably reserved and before the host process starts. Implementations
        must key provider resource identity only from ``context.owner``,
        ``context.host_id``, and *generation*. Repository coordinates may select
        credentials but must not affect resource identity. This lets
        :meth:`release` clean up even if the process dies while this method is
        creating a scoped token or provider Secret, before a reference can be
        persisted.

        Implementations must honor task cancellation. The server enforces its
        deadline without waiting for a cancellation-resistant implementation,
        runs deterministic release from the durable launch identity, and also
        releases any lease the late task eventually returns.

        Raising aborts the launch: the orchestration treats a failed acquire
        like any other post-provision failure — it tears the sandbox down and
        surfaces the error — so a hook that cannot resolve credentials should
        raise rather than return a half-formed lease.

        :param context: The immutable identity of the launch to resolve
            credentials for.
        :param generation: Durable launch generation reserved before this call.
        :returns: The acquired lease (its
            :attr:`~ManagedCredentialLease.reference` exposed to launcher
            startup).
        :raises Exception: When credentials cannot be resolved; the launch is
            aborted and the sandbox torn down.
        """
        ...

    async def release(  # noqa: B027 — intentional concrete no-op
        self,
        context: ManagedCredentialReleaseContext,
        generation: int,
        reference: str | None,
    ) -> None:
        """Release a persisted lease without its original in-memory object.

        Called during successful sandbox teardown, before a relaunch supersedes
        an old generation, and by restart recovery. ``reference`` may be ``None``
        after a crash during :meth:`acquire`; implementations must then derive
        the provider resource from ``context.owner``, ``context.host_id``, and
        *generation*. Implementations must be idempotent and treat missing
        provider resources as success. Transient failures should raise a
        sanitized exception so the orchestrator retains the durable record for
        retry. The default is a no-op for credentials needing no explicit
        cleanup.
        """

    async def aclose(self) -> None:  # noqa: B027 - intentional concrete no-op
        """Close hook-owned clients when the server shuts down."""
