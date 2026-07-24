"""Tests for :mod:`omnigent.server.managed_hosts`."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import click
import pytest
import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from omnigent.db.utils import now_epoch
from omnigent.onboarding.sandboxes.base import render_host_config_write_command
from omnigent.onboarding.sandboxes.e2b import managed_token_ttl_s as e2b_managed_token_ttl_s
from omnigent.runtime import _globals
from omnigent.runtime.agent_cache import AgentCache
from omnigent.server.app import create_app
from omnigent.server.managed_credentials import (
    ManagedCredentialHook,
    ManagedCredentialLease,
    ManagedCredentialReleaseContext,
    ManagedLaunchContext,
)
from omnigent.server.managed_hosts import (
    BOXLITE_MANAGED_TOKEN_TTL_S,
    DAYTONA_MANAGED_TOKEN_TTL_S,
    ISLO_MANAGED_TOKEN_TTL_S,
    KUBERNETES_MANAGED_TOKEN_TTL_S,
    MODAL_MANAGED_TOKEN_TTL_S,
    OPENSHELL_MANAGED_TOKEN_TTL_S,
    ManagedSandboxConfig,
    RepoWorkspace,
    _release_stored_credential_leases,
    host_resume_supported,
    launch_managed_host,
    parse_repo_workspace,
    parse_sandbox_config,
    recover_managed_credential_leases,
    relaunch_managed_host,
    resume_managed_host,
    terminate_managed_host,
)
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.artifact_store.local import LocalArtifactStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from omnigent.stores.host_store import HostStore
from omnigent.terminals import TerminalRegistry
from tests.server.helpers import (
    FakeSandboxLauncher,
    HostStartInvocation,
    install_fake_boxlite_launcher,
    install_fake_daytona_launcher,
    install_fake_e2b_launcher,
    install_fake_islo_launcher,
    install_fake_kubernetes_launcher,
    install_fake_modal_launcher,
    install_fake_openshell_launcher,
)

pytestmark = pytest.mark.asyncio

_OWNER = "alice@example.com"


def _injected_config(
    fake: FakeSandboxLauncher,
    *,
    server_url: str = "https://srv.example.com",
    token_ttl_s: int = 3600,
    host_config: dict[str, object] | None = None,
) -> ManagedSandboxConfig:
    """
    Build a config that injects *fake* through the launcher-factory seam
    — the same way an embedding deployment injects a custom launcher.

    :param fake: The launcher every launch should use.
    :param server_url: Server URL the sandbox host dials back to.
    :param token_ttl_s: Launch-token lifetime in seconds.
    :param host_config: In-sandbox config.yaml content to forward, or ``None``.
    :returns: A ready :class:`ManagedSandboxConfig`.
    """
    return ManagedSandboxConfig(
        server_url=server_url,
        launcher_factory=lambda: fake,
        token_ttl_s=token_ttl_s,
        host_config=host_config,
    )


# ── parse_sandbox_config ────────────────────────────────────


def test_parse_absent_section_disables_managed_hosts() -> None:
    """No ``sandbox:`` section → managed hosts simply not configured."""
    assert parse_sandbox_config(None) is None


def test_parse_valid_modal_config_builds_image_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented modal YAML shape parses into a config whose factory
    constructs Modal launchers carrying the configured image — the
    pre-baked-image thread that makes managed startup fast.
    """
    cfg = parse_sandbox_config(
        {
            "provider": "modal",
            # Trailing slash is normalized: the URL is interpolated into
            # `omnigent host --server <url>` and double slashes break joins.
            "server_url": "https://srv.example.com/",
            "modal": {"image": "docker.io/me/omnigent-host:latest"},
        }
    )
    assert cfg is not None
    assert cfg.server_url == "https://srv.example.com"
    assert cfg.token_ttl_s == MODAL_MANAGED_TOKEN_TTL_S
    # modal is in PROVIDERS_WITH_MANAGED_LAUNCH, so the parsed config
    # advertises managed launch (drives /v1/info's capability flag).
    assert cfg.managed_launch_supported is True
    # The parsed provider is carried through so /v1/info can label the
    # web UI's option ("Modal Sandbox").
    assert cfg.provider == "modal"
    # The factory resolves ModalSandboxLauncher at call time; substitute
    # the fake at that public seam to observe the constructor wiring.
    fake = FakeSandboxLauncher()
    install_fake_modal_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image == "docker.io/me/omnigent-host:latest"
    # No secrets configured → None reaches the launcher (its env-var
    # fallback applies), not an empty list.
    assert fake.secrets is None


def test_parse_modal_without_image_defaults_to_official(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provider: modal` + `server_url` is a complete config: the image is
    optional and defaults to the official prebaked host image (the
    launcher resolves env override / official default when constructed
    with image=None).
    """
    cfg = parse_sandbox_config({"provider": "modal", "server_url": "https://s.example.com"})
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_modal_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    # image=None → the launcher's own resolution (env var → official
    # default) applies, rather than a config-pinned ref.
    assert fake.image is None


def test_parse_non_modal_provider_yields_rejecting_factory() -> None:
    """
    lakebox configs parse (a deployment can stage config before
    managed-launch support lands), but their factory rejects with a 400
    naming the provider when a managed session is actually requested.
    """
    cfg = parse_sandbox_config({"provider": "lakebox", "server_url": "https://s.example.com"})
    assert cfg is not None
    # A staged provider must not advertise managed launch on /v1/info —
    # the web UI would offer a sandbox option every create rejects.
    assert cfg.managed_launch_supported is False
    # The provider is still parsed onto the config; /v1/info gates on
    # managed_launch_supported, so the name is not surfaced while staged.
    assert cfg.provider == "lakebox"
    with pytest.raises(HTTPException) as exc:
        cfg.launcher_factory()
    assert exc.value.status_code == 400
    assert "lakebox" in exc.value.detail


def test_parse_valid_daytona_config_builds_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented daytona YAML shape parses into a config whose
    factory constructs Daytona launchers carrying the configured image
    and env-passthrough names, with the daytona token TTL (no platform
    lifetime cap; 7-day policy bound).
    """
    cfg = parse_sandbox_config(
        {
            "provider": "daytona",
            "server_url": "https://srv.example.com/",
            "daytona": {
                "image": "docker.io/me/omnigent-host:latest",
                "env": ["OPENAI_API_KEY", "GIT_TOKEN"],
            },
        }
    )
    assert cfg is not None
    assert cfg.server_url == "https://srv.example.com"
    assert cfg.token_ttl_s == DAYTONA_MANAGED_TOKEN_TTL_S
    assert cfg.managed_launch_supported is True
    fake = FakeSandboxLauncher()
    install_fake_daytona_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image == "docker.io/me/omnigent-host:latest"
    assert fake.env == ["OPENAI_API_KEY", "GIT_TOKEN"]


def test_parse_daytona_without_section_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provider: daytona` + `server_url` is a complete config: image and
    env are optional and reach the launcher as None (its own env-var
    fallbacks / official-image default apply).
    """
    cfg = parse_sandbox_config({"provider": "daytona", "server_url": "https://s.example.com"})
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_daytona_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image is None
    assert fake.env is None


def test_parse_valid_boxlite_cloud_config_builds_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented boxlite YAML shape (cloud: remote ``boxlite serve``)
    parses into a config whose factory constructs boxlite launchers
    carrying the endpoint, image, and env-passthrough names, with the
    boxlite token TTL (no platform lifetime cap; 7-day policy bound).
    """
    cfg = parse_sandbox_config(
        {
            "provider": "boxlite",
            "server_url": "https://srv.example.com/",
            "boxlite": {
                "image": "docker.io/me/omnigent-host:latest",
                "env": ["OPENAI_API_KEY", "GIT_TOKEN"],
                "cloud": {"endpoint": "https://boxlite.example.com:8100"},
            },
        }
    )
    assert cfg is not None
    assert cfg.server_url == "https://srv.example.com"
    assert cfg.token_ttl_s == BOXLITE_MANAGED_TOKEN_TTL_S
    assert cfg.managed_launch_supported is True
    assert cfg.provider == "boxlite"
    fake = FakeSandboxLauncher()
    install_fake_boxlite_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.endpoint == "https://boxlite.example.com:8100"
    assert fake.image == "docker.io/me/omnigent-host:latest"
    assert fake.env == ["OPENAI_API_KEY", "GIT_TOKEN"]


def test_parse_boxlite_without_section_defaults_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provider: boxlite` + `server_url` is a complete config: the boxlite
    block is optional, so endpoint/image/env reach the launcher as None
    — LOCAL mode (embedded micro-VMs on the server host, no endpoint).
    """
    cfg = parse_sandbox_config({"provider": "boxlite", "server_url": "https://s.example.com"})
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_boxlite_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.endpoint is None
    assert fake.image is None
    assert fake.env is None


def test_parse_boxlite_local_customization_reaches_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `sandbox.boxlite.home_dir` + `registry` reach the launcher: a custom data
    dir and a private-registry block (credential env NAMES, never values).
    """
    cfg = parse_sandbox_config(
        {
            "provider": "boxlite",
            "server_url": "https://s.example.com",
            "boxlite": {
                "local": {
                    "home_dir": "/data/boxlite",
                    "registry": {
                        "host": "ghcr.io",
                        "username_env": "GHCR_USER",
                        "password_env": "GHCR_PAT",
                    },
                },
            },
        }
    )
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_boxlite_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.home_dir == "/data/boxlite"
    assert fake.registry == {
        "host": "ghcr.io",
        "username_env": "GHCR_USER",
        "password_env": "GHCR_PAT",
    }


def test_parse_valid_islo_config_builds_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented islo YAML shape parses into a config whose factory
    constructs Islo launchers carrying image, env names, API override,
    and optional Islo sandbox sizing/profile fields.
    """
    cfg = parse_sandbox_config(
        {
            "provider": "islo",
            "server_url": "https://srv.example.com/",
            "islo": {
                "image": "docker.io/me/omnigent-host:latest",
                "env": ["OPENAI_API_KEY", "GIT_TOKEN"],
                "base_url": "https://api.islo.dev/",
                "gateway_profile": "default",
                "snapshot_name": "warm-host",
                "workdir": "/root/workspace",
                "vcpus": 4,
                "memory_mb": 8192,
                "disk_gb": 40,
                "idle_pause_after_s": 1200,
            },
        }
    )
    assert cfg is not None
    assert cfg.server_url == "https://srv.example.com"
    assert cfg.token_ttl_s == ISLO_MANAGED_TOKEN_TTL_S
    assert cfg.managed_launch_supported is True
    assert cfg.provider == "islo"
    fake = FakeSandboxLauncher()
    install_fake_islo_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image == "docker.io/me/omnigent-host:latest"
    assert fake.env == ["OPENAI_API_KEY", "GIT_TOKEN"]
    assert fake.base_url == "https://api.islo.dev/"
    assert fake.gateway_profile == "default"
    assert fake.snapshot_name == "warm-host"
    assert fake.workdir == "/root/workspace"
    assert fake.vcpus == 4
    assert fake.memory_mb == 8192
    assert fake.disk_gb == 40
    assert fake.idle_pause_after_s == 1200


def test_parse_islo_without_section_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provider: islo` + `server_url` is a complete config: optional
    constructor fields reach the launcher as None so its env-var
    fallbacks / official-image default apply.
    """
    cfg = parse_sandbox_config({"provider": "islo", "server_url": "https://s.example.com"})
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_islo_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image is None
    assert fake.env is None
    assert fake.base_url is None
    assert fake.gateway_profile is None
    assert fake.snapshot_name is None
    assert fake.workdir is None
    assert fake.vcpus is None
    assert fake.memory_mb is None
    assert fake.disk_gb is None
    assert fake.idle_pause_after_s == 900


def test_parse_islo_config_idle_pause_null_disables_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit null opts out of Islo's default idle pause policy."""
    cfg = parse_sandbox_config(
        {
            "provider": "islo",
            "server_url": "https://s.example.com",
            "islo": {"idle_pause_after_s": None},
        }
    )
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_islo_launcher(monkeypatch, fake)

    assert cfg.launcher_factory() is fake
    assert fake.idle_pause_after_s is None


def test_parse_valid_e2b_config_builds_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented e2b YAML shape parses into a config whose factory
    constructs E2B launchers carrying the configured template name and
    env-passthrough names, with the e2b token TTL (24h cap → mirror
    Modal's 25h token lifetime).
    """
    cfg = parse_sandbox_config(
        {
            "provider": "e2b",
            "server_url": "https://srv.example.com/",
            "e2b": {
                "template": "omnigent-host",
                "env": ["OPENAI_API_KEY", "GIT_TOKEN"],
            },
        }
    )
    assert cfg is not None
    assert cfg.server_url == "https://srv.example.com"
    assert cfg.token_ttl_s == e2b_managed_token_ttl_s()
    assert cfg.managed_launch_supported is True
    assert cfg.provider == "e2b"
    fake = FakeSandboxLauncher()
    install_fake_e2b_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.template == "omnigent-host"
    assert fake.env == ["OPENAI_API_KEY", "GIT_TOKEN"]


def test_parse_e2b_without_section_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provider: e2b` + `server_url` is a complete config: template and
    env are optional and reach the launcher as None (its own env-var
    fallbacks / default-template apply).
    """
    cfg = parse_sandbox_config({"provider": "e2b", "server_url": "https://s.example.com"})
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_e2b_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.template is None
    assert fake.env is None


def test_parse_e2b_template_rejects_non_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present-but-malformed e2b template fails loud at parse time."""
    with pytest.raises(ValueError, match=r"sandbox\.e2b\.template"):
        parse_sandbox_config(
            {
                "provider": "e2b",
                "server_url": "https://s.example.com",
                "e2b": {"template": ""},
            }
        )


def test_parse_valid_openshell_config_builds_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented openshell YAML shape parses into a config whose
    factory constructs OpenShell launchers carrying image, env names,
    and the optional gateway cluster.
    """
    cfg = parse_sandbox_config(
        {
            "provider": "openshell",
            "server_url": "https://srv.example.com/",
            "openshell": {
                "image": "docker.io/me/omnigent-host:latest",
                "env": ["OPENAI_API_KEY", "GIT_TOKEN"],
                "cluster": "my-gateway",
            },
        }
    )
    assert cfg is not None
    assert cfg.server_url == "https://srv.example.com"
    assert cfg.token_ttl_s == OPENSHELL_MANAGED_TOKEN_TTL_S
    assert cfg.managed_launch_supported is True
    assert cfg.provider == "openshell"
    fake = FakeSandboxLauncher()
    install_fake_openshell_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image == "docker.io/me/omnigent-host:latest"
    assert fake.env == ["OPENAI_API_KEY", "GIT_TOKEN"]
    assert fake.cluster == "my-gateway"


def test_parse_openshell_without_section_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `provider: openshell` + `server_url` is a complete config: optional
    constructor fields reach the launcher as None so its env-var
    fallbacks / official-image default / active-gateway apply.
    """
    cfg = parse_sandbox_config({"provider": "openshell", "server_url": "https://s.example.com"})
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_openshell_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image is None
    assert fake.env is None
    assert fake.cluster is None


def test_parse_valid_kubernetes_config_builds_parameterized_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The documented kubernetes YAML shape parses into a config whose factory
    constructs Kubernetes launchers carrying namespace / Secret / SA / node
    selector / in-cluster / resources, with the 7-day token TTL.
    """
    cfg = parse_sandbox_config(
        {
            "provider": "kubernetes",
            "server_url": "http://omnigent.omnigent.svc.cluster.local/",
            "kubernetes": {
                "image": "ghcr.io/me/omnigent-host:latest",
                "env": ["OPENAI_API_KEY", "GIT_TOKEN"],
                "namespace": "omnigent-sandboxes",
                "secret_name": "omnigent-creds",
                "service_account": "omnigent-runner",
                "node_selector": {"omnigent.ai/runner-ready": "true"},
                "in_cluster": True,
                "resources": {"requests": {"cpu": "500m"}, "limits": {"memory": "8Gi"}},
            },
        }
    )
    assert cfg is not None
    assert cfg.server_url == "http://omnigent.omnigent.svc.cluster.local"
    assert cfg.token_ttl_s == KUBERNETES_MANAGED_TOKEN_TTL_S
    assert cfg.managed_launch_supported is True
    assert cfg.provider == "kubernetes"
    fake = FakeSandboxLauncher()
    install_fake_kubernetes_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.image == "ghcr.io/me/omnigent-host:latest"
    assert fake.env == ["OPENAI_API_KEY", "GIT_TOKEN"]
    assert fake.namespace == "omnigent-sandboxes"
    assert fake.secret_name == "omnigent-creds"
    assert fake.service_account == "omnigent-runner"
    assert fake.node_selector == {"omnigent.ai/runner-ready": "true"}
    assert fake.in_cluster is True
    assert fake.resources == {"requests": {"cpu": "500m"}, "limits": {"memory": "8Gi"}}


def test_parse_kubernetes_without_section_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `provider: kubernetes` + `server_url` is a complete config: optional fields
    reach the launcher as None so its env-var fallbacks / defaults apply.
    """
    cfg = parse_sandbox_config(
        {"provider": "kubernetes", "server_url": "http://s.svc.cluster.local"}
    )
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_kubernetes_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.namespace is None
    assert fake.secret_name is None
    assert fake.in_cluster is None
    assert fake.resources is None


def test_parse_host_config_threads_verbatim_without_resolving_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A valid host_config lands on the parsed config verbatim, and its
    ``api_key_ref: env:`` reference is NOT resolved at parse time — the
    variable names sandbox environment, not server environment, so parsing
    must succeed with the variable unset on the server.
    """
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OMNIGENT_LITELLM_API_KEY", raising=False)
    host_config = {
        "providers": {
            "litellm": {
                "kind": "gateway",
                "default": ["pi"],
                "openai": {
                    "base_url": "http://litellm.litellm.svc.cluster.local/v1",
                    "api_key_ref": "env:LITELLM_API_KEY",
                    "wire_api": "chat",
                },
            }
        }
    }

    cfg = parse_sandbox_config(
        {"provider": "modal", "server_url": "https://s.example.com", "host_config": host_config}
    )

    assert cfg is not None
    assert cfg.host_config == host_config


def test_parse_absent_host_config_is_none() -> None:
    """No host_config key → nothing forwarded, existing configs unchanged."""
    cfg = parse_sandbox_config({"provider": "modal", "server_url": "https://s.example.com"})
    assert cfg is not None
    assert cfg.host_config is None


def test_parse_host_config_null_providers_fails_loud() -> None:
    """
    An explicit ``providers: null`` fails parse. Left through, the sandbox
    merge would write ``providers: null`` over any existing block and the
    harness would silently fall back to its own login — the exact
    degradation this parse exists to stop.
    """
    with pytest.raises(ValueError, match=r"sandbox\.host_config\.providers"):
        parse_sandbox_config(
            {
                "provider": "modal",
                "server_url": "https://s.example.com",
                "host_config": {"providers": None},
            }
        )


def test_parse_host_config_duplicate_default_fails_loud() -> None:
    """Duplicate defaults fail at server startup, before sandbox launch."""
    provider = {
        "kind": "gateway",
        "default": ["pi"],
        "openai": {
            "base_url": "https://gateway.example.com/v1",
            "api_key_ref": "env:GATEWAY_API_KEY",
        },
    }

    with pytest.raises(
        ValueError,
        match=r"sandbox\.host_config\.providers.*multiple providers.*'pi' family",
    ):
        parse_sandbox_config(
            {
                "provider": "modal",
                "server_url": "https://s.example.com",
                "host_config": {
                    "providers": {
                        "first": provider,
                        "second": provider,
                    }
                },
            }
        )


def test_parse_host_config_inline_api_key_fails_loud() -> None:
    """Literal provider credentials cannot ride in the managed host config."""
    with pytest.raises(ValueError, match=r"api_key_ref: env:VAR"):
        parse_sandbox_config(
            {
                "provider": "modal",
                "server_url": "https://s.example.com",
                "host_config": {
                    "providers": {
                        "openai": {
                            "kind": "key",
                            "openai": {
                                "base_url": "https://api.openai.com/v1",
                                "api_key": "sk-inline-secret",
                            },
                        }
                    }
                },
            }
        )


def test_parse_host_config_lossy_json_key_collision_fails_loud() -> None:
    """JSON key coercion cannot silently collapse distinct config entries."""
    with pytest.raises(ValueError, match=r"JSON-serializable"):
        parse_sandbox_config(
            {
                "provider": "modal",
                "server_url": "https://s.example.com",
                "host_config": {"metadata": {1: "integer", "1": "string"}},
            }
        )


@pytest.mark.parametrize(
    ("kubernetes_block", "expected_fragment"),
    [
        ({"namespace": "Bad_NS"}, "sandbox.kubernetes.namespace"),
        ({"node_selector": {"omnigent.ai/x": "Bad Value"}}, "node_selector"),
        ({"resources": {"requests": {"cpu": "not a quantity!"}}}, "valid Kubernetes quantity"),
        ({"resources": {"requests": {"disk": "1Gi"}}}, "unknown key"),
        ({"in_cluster": "yes"}, "must be a boolean"),
    ],
)
def test_parse_kubernetes_invalid_block_fails_loud(
    kubernetes_block: dict[str, object], expected_fragment: str
) -> None:
    """An operator typo in the kubernetes block fails parse loud, not at launch."""
    with pytest.raises(ValueError, match=expected_fragment):
        parse_sandbox_config(
            {
                "provider": "kubernetes",
                "server_url": "http://s.svc.cluster.local",
                "kubernetes": kubernetes_block,
            }
        )


@pytest.mark.parametrize(
    ("raw", "expected_fragment"),
    [
        # Non-mapping section.
        ("modal", "must be a mapping"),
        # Unknown / missing provider.
        ({"provider": "bogus", "server_url": "https://s"}, "sandbox.provider"),
        ({"server_url": "https://s"}, "sandbox.provider"),
        # Missing / empty server_url.
        ({"provider": "modal", "modal": {"image": "x"}}, "server_url"),
        ({"provider": "modal", "server_url": "  ", "modal": {"image": "x"}}, "server_url"),
        # modal section present but malformed.
        ({"provider": "modal", "server_url": "https://s", "modal": "x"}, "sandbox.modal"),
        (
            {"provider": "modal", "server_url": "https://s", "modal": {"image": "  "}},
            "sandbox.modal.image",
        ),
        # daytona section present but malformed.
        ({"provider": "daytona", "server_url": "https://s", "daytona": "x"}, "sandbox.daytona"),
        (
            {"provider": "daytona", "server_url": "https://s", "daytona": {"image": "  "}},
            "sandbox.daytona.image",
        ),
        (
            {"provider": "daytona", "server_url": "https://s", "daytona": {"env": "OPENAI"}},
            "sandbox.daytona.env",
        ),
        (
            {"provider": "daytona", "server_url": "https://s", "daytona": {"env": ["", "X"]}},
            "sandbox.daytona.env",
        ),
        # boxlite section present but malformed.
        ({"provider": "boxlite", "server_url": "https://s", "boxlite": "x"}, "sandbox.boxlite"),
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"image": "  "}},
            "sandbox.boxlite.image",
        ),
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"env": "OPENAI"}},
            "sandbox.boxlite.env",
        ),
        # boxlite mode blocks (local / cloud are mutually exclusive).
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"local": {}, "cloud": {"endpoint": "https://b"}},
            },
            "mutually exclusive",
        ),
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"cloud": "x"}},
            "sandbox.boxlite.cloud",
        ),
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"cloud": {"endpoint": "  "}},
            },
            "sandbox.boxlite.cloud.endpoint",
        ),
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"local": "x"}},
            "sandbox.boxlite.local",
        ),
        # A bare `cloud:` / `local:` YAML key (value None) is malformed — it must
        # be rejected, not silently fall through to LOCAL mode (a `cloud:` typo
        # would otherwise run locally with no diagnostic).
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"cloud": None}},
            "sandbox.boxlite.cloud",
        ),
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"local": None}},
            "sandbox.boxlite.local",
        ),
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"local": {"home_dir": "  "}},
            },
            "sandbox.boxlite.local.home_dir",
        ),
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"local": {"registry": "x"}},
            },
            "sandbox.boxlite.local.registry",
        ),
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"local": {"registry": {"transport": "https"}}},
            },
            "sandbox.boxlite.local.registry.host",
        ),
        # M3: bearer token + basic auth both set (boxlite silently drops basic).
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {
                    "local": {
                        "registry": {"host": "ghcr.io", "token_env": "T", "password_env": "P"}
                    }
                },
            },
            "mutually exclusive",
        ),
        # M4: misplaced / unknown keys are rejected, not silently ignored.
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"endpoint": "https://b"},
            },
            "unknown key",
        ),
        (
            {"provider": "boxlite", "server_url": "https://s", "boxlite": {"bogus": 1}},
            "unknown key",
        ),
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"cloud": {"endpoint": "https://b", "bogus": 1}},
            },
            "unknown key",
        ),
        (
            {
                "provider": "boxlite",
                "server_url": "https://s",
                "boxlite": {"local": {"registry": {"host": "ghcr.io", "passwrod_env": "P"}}},
            },
            "unknown key",
        ),
        # islo section present but malformed.
        ({"provider": "islo", "server_url": "https://s", "islo": "x"}, "sandbox.islo"),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"image": "  "}},
            "sandbox.islo.image",
        ),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"env": "OPENAI"}},
            "sandbox.islo.env",
        ),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"env": ["", "X"]}},
            "sandbox.islo.env",
        ),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"base_url": "  "}},
            "sandbox.islo.base_url",
        ),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"vcpus": 0}},
            "sandbox.islo.vcpus",
        ),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"memory_mb": "large"}},
            "sandbox.islo.memory_mb",
        ),
        (
            {"provider": "islo", "server_url": "https://s", "islo": {"idle_pause_after_s": 0}},
            "sandbox.islo.idle_pause_after_s",
        ),
        (
            {
                "provider": "islo",
                "server_url": "https://s",
                "islo": {"idle_pause_after_s": "900"},
            },
            "sandbox.islo.idle_pause_after_s",
        ),
        # openshell section present but malformed.
        (
            {"provider": "openshell", "server_url": "https://s", "openshell": "x"},
            "sandbox.openshell",
        ),
        (
            {"provider": "openshell", "server_url": "https://s", "openshell": {"image": "  "}},
            "sandbox.openshell.image",
        ),
        (
            {"provider": "openshell", "server_url": "https://s", "openshell": {"env": ["", "X"]}},
            "sandbox.openshell.env",
        ),
        (
            {"provider": "openshell", "server_url": "https://s", "openshell": {"cluster": "  "}},
            "sandbox.openshell.cluster",
        ),
        # host_config present but malformed (provider-agnostic top-level key).
        (
            {"provider": "modal", "server_url": "https://s", "host_config": "providers: {}"},
            "sandbox.host_config",
        ),
        (
            {"provider": "modal", "server_url": "https://s", "host_config": {"providers": "x"}},
            "sandbox.host_config.providers",
        ),
        # An invalid provider entry (bad kind) is caught by the same parser
        # omnigent itself uses — inside the sandbox this would degrade
        # silently, so parse time is the only loud failure point.
        (
            {
                "provider": "modal",
                "server_url": "https://s",
                "host_config": {"providers": {"litellm": {"kind": "bogus"}}},
            },
            "sandbox.host_config.providers",
        ),
        # yaml.safe_load turns an unquoted date into datetime.date, which the
        # per-launch json.dumps cannot take — must fail startup, not launches.
        (
            {
                "provider": "modal",
                "server_url": "https://s",
                "host_config": {"last_rotated": datetime.date(2024, 1, 1)},
            },
            "JSON-serializable",
        ),
    ],
)
def test_parse_invalid_config_fails_loud(raw: object, expected_fragment: str) -> None:
    """
    Malformed config raises with the offending key named — this is
    what stops server startup on an operator typo instead of 502-ing
    the first managed session.
    """
    with pytest.raises(ValueError, match="") as exc:
        parse_sandbox_config(raw)
    assert expected_fragment in str(exc.value)


# ── parse_repo_workspace ────────────────────────────────────


@pytest.mark.parametrize(
    ("workspace", "expected"),
    [
        # Plain https URL — default branch, name from the last segment.
        (
            "https://github.com/org/repo",
            RepoWorkspace(url="https://github.com/org/repo", branch=None, repo_name="repo"),
        ),
        # `.git` suffix stripped from the directory name, kept in the URL.
        (
            "https://github.com/org/repo.git#release-1.2",
            RepoWorkspace(
                url="https://github.com/org/repo.git",
                branch="release-1.2",
                repo_name="repo",
            ),
        ),
        # scp-style ssh form.
        (
            "git@github.com:org/repo.git",
            RepoWorkspace(url="git@github.com:org/repo.git", branch=None, repo_name="repo"),
        ),
        # Branches with slashes are legal git refs.
        (
            "https://github.com/org/repo#feature/x",
            RepoWorkspace(url="https://github.com/org/repo", branch="feature/x", repo_name="repo"),
        ),
    ],
)
def test_parse_repo_workspace_accepts_url_forms(workspace: str, expected: RepoWorkspace) -> None:
    """
    The documented ``<repo>[#<branch>]`` grammar parses into the
    validated spec the clone step consumes — URL, pinned branch, and
    the clone directory name all come from here, so a wrong field
    means a wrong `git clone` invocation.
    """
    assert parse_repo_workspace(workspace) == expected


@pytest.mark.parametrize(
    ("workspace", "expected_fragment"),
    [
        # Absolute paths are the EXTERNAL form — a path points at
        # nothing in a sandbox that doesn't exist yet.
        ("/tmp/w", "not a supported repository URL"),
        # Bare org/repo shorthand is UI-side sugar, never API surface.
        ("org/repo", "not a supported repository URL"),
        # No repo path at all.
        ("https://github.com", "not a usable https repository URL"),
        ("git@github.com", "not a usable ssh repository URL"),
        # Commit SHAs would land the agent on a detached HEAD.
        ("https://github.com/org/repo#" + "a" * 40, "not a commit SHA"),
        # Empty / malformed branch fragments.
        ("https://github.com/org/repo#", "must name a branch"),
        ("https://github.com/org/repo#-flag", "not a valid git branch name"),
        ("https://github.com/org/repo#a..b", "not a valid git branch name"),
        # A second '#' means the branch itself contains '#' —
        # unsupported in the fragment form.
        ("https://github.com/org/repo#a#b", "not a valid git branch name"),
        ("https://github.com/org/repo#a b", "must not contain whitespace"),
    ],
)
def test_parse_repo_workspace_rejects_malformed(workspace: str, expected_fragment: str) -> None:
    """
    Malformed workspaces fail loud at parse time with the offense
    named — this is what turns into the create's 422 instead of a
    mid-provision clone error inside a half-launched sandbox.
    """
    with pytest.raises(ValueError, match="") as exc:
        parse_repo_workspace(workspace)
    assert expected_fragment in str(exc.value)


def test_parse_repo_workspace_rejects_userinfo_without_echoing_credentials() -> None:
    """Authenticated clone URLs cannot leak into durable lease metadata or errors."""
    secret = "sensitive-token-value"

    with pytest.raises(ValueError, match="userinfo credentials") as exc:
        parse_repo_workspace(f"https://user:{secret}@github.com/org/repo")

    assert secret not in str(exc.value)


def test_parse_repo_workspace_rejects_query_without_echoing_credentials() -> None:
    """Query-bearing clone URLs cannot persist embedded access material."""
    secret = "sensitive-query-token"

    with pytest.raises(ValueError, match="query string") as exc:
        parse_repo_workspace(f"https://github.com/org/repo?token={secret}")

    assert secret not in str(exc.value)


def test_parse_repo_workspace_rejects_extra_scp_separator_without_echoing_credentials() -> None:
    """Malformed scp-style remotes cannot smuggle userinfo-like material into storage."""
    secret = "sensitive-scp-token"

    with pytest.raises(ValueError, match="exactly one '@' separator") as exc:
        parse_repo_workspace(f"git@{secret}@github.com:org/repo.git")

    assert secret not in str(exc.value)


def test_parse_repo_workspace_never_echoes_invalid_path_material() -> None:
    """Repository-name validation errors do not reflect path-borne access material."""
    secret = "sensitive-path-token"

    with pytest.raises(ValueError, match="could not derive") as exc:
        parse_repo_workspace(f"https://github.com/org/{secret}@repo.git")

    assert secret not in str(exc.value)


# ── GET /v1/info: managed_sandboxes_enabled ─────────────────


def _capability_probe_app(
    db_uri: str,
    tmp_path: Path,
    sandbox_config: ManagedSandboxConfig | None,
) -> FastAPI:
    """
    Build a real app wired with *sandbox_config* to probe ``GET /v1/info``.

    Minimal store wiring — the probe handler reads only the
    ``sandbox_config`` closure, but the app factory needs real stores.

    :param db_uri: SQLite connection URI for the app's stores.
    :param tmp_path: Per-test scratch dir for artifact/cache stores.
    :param sandbox_config: The sandbox config under test, or ``None``
        when managed hosts are not configured.
    :returns: The assembled FastAPI app.
    """
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=artifact_store,
        agent_cache=AgentCache(artifact_store=artifact_store, cache_dir=tmp_path / "cache"),
        host_store=HostStore(db_uri),
        sandbox_config=sandbox_config,
    )


@pytest.mark.parametrize(
    ("sandbox_raw", "expected", "expected_provider"),
    [
        # Launch-capable provider configured → the web UI may offer the
        # sandbox option, labeled with the provider name ("Modal Sandbox").
        ({"provider": "modal", "server_url": "https://s.example.com"}, True, "modal"),
        # No `sandbox:` section → a managed create would 400; the option
        # must not be advertised and no provider is named.
        (None, False, None),
        # advertising it would offer a create path that always fails, so
        # the option is hidden and the provider stays unnamed.
        ({"provider": "lakebox", "server_url": "https://s.example.com"}, False, None),
        # Daytona has managed-launch support like modal → offered and
        # named so the UI can label it ("Daytona Sandbox").
        ({"provider": "daytona", "server_url": "https://s.example.com"}, True, "daytona"),
        # Islo has managed-launch support too → offered and provider-labeled.
        ({"provider": "islo", "server_url": "https://s.example.com"}, True, "islo"),
    ],
)
async def test_info_reports_managed_sandboxes_capability(
    db_uri: str,
    tmp_path: Path,
    sandbox_raw: dict[str, object] | None,
    expected: bool,
    expected_provider: str | None,
) -> None:
    """
    ``GET /v1/info`` advertises managed sandboxes iff the wired config
    can actually serve a managed launch, and names the backing provider
    (``sandbox_provider``) so the web UI can label the option per
    provider — but only when the option is actually offered.
    """
    app = _capability_probe_app(db_uri, tmp_path, parse_sandbox_config(sandbox_raw))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/info")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["managed_sandboxes_enabled"] is expected
    # The provider name is surfaced only when the option is offered; a
    # staged/absent config leaks nothing (provider stays None), so the
    # daytona/none cases never name a backend.
    assert body["sandbox_provider"] == expected_provider


async def test_info_reports_enabled_for_injected_custom_launcher(
    db_uri: str,
    tmp_path: Path,
) -> None:
    """
    The embedding seam: a directly-constructed config (custom launcher
    factory, no YAML) defaults to advertising managed launch — the
    deployment's factory IS the support. With no provider named, the UI
    falls back to the generic "New Sandbox" label (``sandbox_provider``
    is None).
    """
    config = ManagedSandboxConfig(
        server_url="https://s.example.com",
        launcher_factory=lambda: FakeSandboxLauncher(),
        token_ttl_s=3600,
    )
    app = _capability_probe_app(db_uri, tmp_path, config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/info")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["managed_sandboxes_enabled"] is True
    # No provider set on the injected config → the UI keeps the generic
    # label rather than inventing a name.
    assert body["sandbox_provider"] is None


# ── launch_managed_host ─────────────────────────────────────


async def test_launch_success_registers_host_and_returns_workspace(db_uri: str) -> None:
    """
    Golden path: provision → pre-register the host row with its token
    → start host → host online.

    The launcher arrives through the config's factory seam (no
    patching), and the fake's ``on_host_start`` connects exactly as
    the real tunnel would after validating the launch token
    (``upsert_on_connect`` against the pre-registered row), so the
    online poll observes a genuine hosts-table transition.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        """Simulate the sandbox host connecting over the tunnel."""
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register)

    result = await launch_managed_host(
        config=_injected_config(fake),
        owner=_OWNER,
        host_store=host_store,
    )

    assert fake.prepared is True
    # The workspace was created in the sandbox's home and returned.
    assert result.workspace == "/root/workspace"
    assert any("mkdir -p /root/workspace" in cmd for cmd in fake.commands)
    # The start command dials back to the configured server URL.
    start = fake.host_starts[0]
    assert "--server https://srv.example.com" in start.command
    assert result.host_id == start.host_id
    # The hosts row carries the managed binding with full content; the
    # provider comes from the LAUNCHER (not config), so injected custom
    # launchers record their own name.
    host = host_store.get_host(result.host_id)
    assert host is not None
    assert host.user_id == _OWNER
    assert host.name == start.host_name
    assert host.status == "online"
    assert host.sandbox_provider == "modal"
    assert host.sandbox_id == "sb-fake-1"
    # The token injected into the sandbox is the one whose digest was
    # stored: resolving it (the tunnel's auth path) yields this host,
    # which also proves it is unexpired.
    resolved = host_store.resolve_launch_token(start.host_id, start.token)
    assert resolved is not None
    assert resolved.host_id == result.host_id
    # Nothing was torn down on the success path.
    assert fake.terminated == []


async def test_launch_materializes_host_config_before_host_start(db_uri: str) -> None:
    """
    A configured host_config is written into the sandbox strictly BEFORE
    ``omnigent host`` starts — the whole point of the injection is that the
    host boots with its providers already on disk.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register)
    host_config: dict[str, object] = {"providers": {"litellm": {"kind": "gateway"}}}

    await launch_managed_host(
        config=_injected_config(fake, host_config=host_config),
        owner=_OWNER,
        host_store=host_store,
    )

    write_index = fake.commands.index(render_host_config_write_command(host_config))
    host_index = next(i for i, cmd in enumerate(fake.commands) if "omnigent host --server" in cmd)
    assert write_index < host_index


async def test_resume_rematerializes_host_config_before_host_restart(db_uri: str) -> None:
    """
    Waking a dormant sandbox re-runs the config write before re-execing the
    host — resume_managed_host bypasses _arm_and_start_host, so this is a
    distinct wiring point, and re-materializing is what lets an operator's
    host_config change land on the next wake without a new sandbox.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register, can_resume=True)
    host_config: dict[str, object] = {"providers": {"litellm": {"kind": "gateway"}}}
    config = _injected_config(fake, host_config=host_config)

    result = await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)
    host_store.set_offline(result.host_id)
    commands_before = len(fake.commands)

    await resume_managed_host(result.host_id, host_store, config)

    assert fake.resumed == ["sb-fake-1"]
    resumed_commands = fake.commands[commands_before:]
    write_index = resumed_commands.index(render_host_config_write_command(host_config))
    host_index = next(
        i for i, cmd in enumerate(resumed_commands) if "omnigent host --server" in cmd
    )
    assert write_index < host_index


async def test_launch_without_host_config_writes_no_config(db_uri: str) -> None:
    """No host_config → the launch issues no config-write command at all."""
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register)

    await launch_managed_host(config=_injected_config(fake), owner=_OWNER, host_store=host_store)

    assert not any(cmd.startswith("python3 -c") for cmd in fake.commands)


async def test_launch_without_host_config_supports_legacy_start_host_signature(
    db_uri: str,
) -> None:
    """
    A deployment-injected launcher whose ``start_host`` override predates the
    ``host_config`` parameter keeps launching when no host_config is set —
    the kwarg is omitted entirely rather than passed as ``None``.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    class _LegacySignatureLauncher(FakeSandboxLauncher):
        """Overrides start_host with the pre-host_config explicit signature."""

        def start_host(
            self,
            sandbox_id: str,
            *,
            token: str,
            host_id: str,
            host_name: str,
            server_url: str,
            repo_url: str | None = None,
            repo_branch: str | None = None,
            repo_name: str | None = None,
            on_stage: Callable[[str], None] | None = None,
        ) -> str:
            return super().start_host(
                sandbox_id,
                token=token,
                host_id=host_id,
                host_name=host_name,
                server_url=server_url,
                repo_url=repo_url,
                repo_branch=repo_branch,
                repo_name=repo_name,
                on_stage=on_stage,
            )

    fake = _LegacySignatureLauncher(on_host_start=_register)

    result = await launch_managed_host(
        config=_injected_config(fake), owner=_OWNER, host_store=host_store
    )

    [start] = fake.host_starts
    assert result.host_id == start.host_id


async def test_launch_with_injected_custom_launcher(db_uri: str) -> None:
    """
    The embedding seam end to end: a deployment-defined launcher (a
    provider name the YAML path doesn't even know) drives the whole
    managed flow, and its provider is what lands on the host row — so
    teardown later dispatches back to the same custom launcher.
    """
    host_store = HostStore(db_uri)

    class _AcmeLauncher(FakeSandboxLauncher):
        """Custom launcher under a deployment-private provider name."""

        provider: ClassVar[str] = "acme-cloud"

    def _register(invocation: HostStartInvocation) -> None:
        """Simulate the sandbox host connecting over the tunnel."""
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = _AcmeLauncher(on_host_start=_register)
    config = _injected_config(fake)

    result = await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)

    host = host_store.get_host(result.host_id)
    assert host is not None
    assert host.sandbox_provider == "acme-cloud"
    assert host.sandbox_id == "sb-fake-1"

    # Teardown resolves the launcher through the same config factory
    # (provider matches the row) — the custom launcher's terminate runs.
    await terminate_managed_host(host, host_store, config)
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.get_host(result.host_id) is None


async def test_launch_unsupported_yaml_provider_rejects_before_provisioning(
    db_uri: str,
) -> None:
    """
    A staged-but-unimplemented YAML provider (lakebox) fails with a 400
    naming the provider BEFORE any provisioning happens.
    """
    config = parse_sandbox_config({"provider": "lakebox", "server_url": "https://s.example.com"})
    assert config is not None
    host_store = HostStore(db_uri)
    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)
    assert exc.value.status_code == 400
    assert "lakebox" in exc.value.detail
    # No host row was pre-registered.
    assert host_store.list_hosts(_OWNER) == []


async def test_launch_provision_failure_maps_to_502(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A provider failure before anything exists (preflight) maps to a
    502 with the provider's message, and leaves no host row and
    nothing to terminate.
    """
    fake = FakeSandboxLauncher()

    def _fail_prepare() -> None:
        """Simulate missing provider credentials."""
        raise click.ClickException("No Modal credentials found.")

    monkeypatch.setattr(fake, "prepare", _fail_prepare)
    host_store = HostStore(db_uri)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_injected_config(fake), owner=_OWNER, host_store=host_store
        )
    assert exc.value.status_code == 502
    assert "No Modal credentials found." in exc.value.detail
    assert host_store.list_hosts(_OWNER) == []
    assert fake.terminated == []


async def test_launch_host_start_failure_terminates_and_deletes_host(db_uri: str) -> None:
    """
    A failure AFTER provisioning must clean up: terminate the sandbox
    (no orphaned paid compute) and delete the pre-registered host row
    (the minted token must not stay valid, and a never-started host
    must not linger in the picker).
    """
    fake = FakeSandboxLauncher(fail_on_host_start=True)
    host_store = HostStore(db_uri)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_injected_config(fake), owner=_OWNER, host_store=host_store
        )
    assert exc.value.status_code == 502
    assert "simulated in-sandbox host start failure" in exc.value.detail
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []


async def test_launch_non_click_exception_terminates_and_deletes_host(db_uri: str) -> None:
    """
    A raw (non-Click, non-HTTP) exception during host start — a
    provider SDK error or a network failure from the in-sandbox exec —
    must trigger the same cleanup: terminate the sandbox and delete the
    host row. If the cleanup handler only caught ClickException, the
    sandbox would leak running until the provider's lifetime cap and
    the armed token would stay resolvable.
    """

    def _raise_sdk_error(invocation: HostStartInvocation) -> None:
        raise RuntimeError("simulated provider SDK failure")

    fake = FakeSandboxLauncher(on_host_start=_raise_sdk_error)
    host_store = HostStore(db_uri)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_injected_config(fake), owner=_OWNER, host_store=host_store
        )
    assert exc.value.status_code == 502
    assert "simulated provider SDK failure" in exc.value.detail
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []


async def test_launch_online_timeout_terminates_and_deletes_host(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A host that never registers (e.g. bad image, can't reach the
    server) times out with a 502 pointing at the in-sandbox log, and
    cleans up the sandbox + host row (which revokes the token).
    """
    # No on_host_start → the host never registers.
    fake = FakeSandboxLauncher()
    # Shrink the polling budget so the timeout path runs in
    # milliseconds; production values are module constants read at
    # call time.
    monkeypatch.setattr("omnigent.server.managed_hosts.MANAGED_HOST_ONLINE_TIMEOUT_S", 0.05)
    monkeypatch.setattr("omnigent.server.managed_hosts._ONLINE_POLL_INTERVAL_S", 0.01)
    host_store = HostStore(db_uri)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_injected_config(fake), owner=_OWNER, host_store=host_store
        )
    assert exc.value.status_code == 502
    assert "did not come online" in exc.value.detail
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    # The start command DID run (the failure was registration, not
    # startup), so its minted token exists — and must be dead.
    assert (
        host_store.resolve_launch_token(fake.host_starts[0].host_id, fake.host_starts[0].token)
        is None
    )


async def test_launch_with_repo_clones_into_workspace(db_uri: str) -> None:
    """
    A repository-URL workspace is cloned inside the sandbox BEFORE the
    host starts, and the cloned directory (not the bare workspace root)
    is what the session binds as its workspace.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        """Simulate the sandbox host connecting over the tunnel."""
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register)

    result = await launch_managed_host(
        config=_injected_config(fake),
        owner=_OWNER,
        host_store=host_store,
        repo=parse_repo_workspace("https://github.com/org/myrepo.git#release-1.2"),
    )

    # The session workspace is the clone directory, named after the repo.
    assert result.workspace == "/root/workspace/myrepo"
    # The exact clone invocation: branch-pinned, single-branch, `--`
    # separating options from the user-supplied URL. A drift here means
    # the sandbox clones the wrong thing (or interprets the URL as a
    # flag).
    clone_cmd = (
        "git clone --branch release-1.2 --single-branch "
        "-- https://github.com/org/myrepo.git /root/workspace/myrepo"
    )
    assert clone_cmd in fake.commands
    # Clone runs before the host starts — the workspace must be ready
    # by the time the runner can launch on the registered host.
    host_start_index = next(i for i, c in enumerate(fake.commands) if "omnigent host" in c)
    assert fake.commands.index(clone_cmd) < host_start_index
    assert fake.terminated == []


async def test_launch_clone_failure_terminates_and_deletes_host(db_uri: str) -> None:
    """
    A failed clone (bad URL, missing branch, private repo) cleans up
    exactly like a host-start failure — sandbox terminated, host row
    (and its token) deleted — and the 502 names the repository so the
    create error tells the user WHAT didn't clone.
    """
    fake = FakeSandboxLauncher(fail_on_command="git clone")
    host_store = HostStore(db_uri)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_injected_config(fake),
            owner=_OWNER,
            host_store=host_store,
            repo=parse_repo_workspace("https://github.com/org/private#main"),
        )
    assert exc.value.status_code == 502
    assert "failed to clone repository 'https://github.com/org/private'" in exc.value.detail
    assert "'main'" in exc.value.detail
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    # The host never started — the clone failed first.
    assert fake.host_starts == []


class _EntrypointFakeLauncher(FakeSandboxLauncher):
    """
    An entrypoint-as-host fake (like the kubernetes launcher): ``provision``
    only RESERVES the sandbox id (no box created), and the host is started by a
    ``start_host`` override — not the exec-model base default.

    Records the ``start_host`` call and, to prove the token is armed BEFORE the
    host starts, captures whether the token already resolves at call time (then
    simulates the host dialing back).
    """

    provider: ClassVar[str] = "kubernetes"

    def __init__(self, host_store: HostStore) -> None:
        super().__init__()
        self._host_store = host_store
        self.start_calls: list[dict[str, object]] = []
        self.token_resolved_at_start: bool = False

    def provision(self, name: str) -> str:
        """Reserve a sandbox id (no box created); recorded + deterministic."""
        self.provisioned_names.append(name)
        return f"omnigent-pod-{len(self.provisioned_names)}"

    def run(self, sandbox_id: str, command: str, *, check: bool = True):
        """The entrypoint model never execs in — the base default is overridden."""
        raise AssertionError("entrypoint launcher must not exec via run()")

    def start_host(
        self,
        sandbox_id: str,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repo_url: str | None = None,
        repo_branch: str | None = None,
        repo_name: str | None = None,
        host_config: dict[str, object] | None = None,
        on_stage=None,
    ) -> str:
        """Record the call, prove the token already resolves, and connect."""
        self.start_calls.append(
            {
                "sandbox_id": sandbox_id,
                "token": token,
                "host_id": host_id,
                "server_url": server_url,
                "repo_url": repo_url,
                "repo_name": repo_name,
            }
        )
        # The token was registered before start_host, so it resolves now.
        self.token_resolved_at_start = (
            self._host_store.resolve_launch_token(host_id, token) is not None
        )
        # Simulate the host's entrypoint dialing back over the tunnel.
        self._host_store.upsert_on_connect(host_id=host_id, name=host_name, user_id=_OWNER)
        return f"/home/omnigent/workspace/{repo_name}" if repo_name else "/home/omnigent/workspace"


async def test_launch_entrypoint_provider_arms_token_before_launch_host(db_uri: str) -> None:
    """
    Entrypoint-as-host seam: the uniform launch path reserves the sandbox id via
    provision(), registers the token, THEN calls start_host (never run) — so the
    host authenticates the moment its entrypoint dials back, with no race.
    """
    host_store = HostStore(db_uri)
    fake = _EntrypointFakeLauncher(host_store)

    result = await launch_managed_host(
        config=_injected_config(fake),
        owner=_OWNER,
        host_store=host_store,
        repo=parse_repo_workspace("https://github.com/org/repo.git#main"),
    )

    # start_host ran once, with the reserved id and repo info.
    assert len(fake.start_calls) == 1
    call = fake.start_calls[0]
    assert call["sandbox_id"] == "omnigent-pod-1"
    assert call["server_url"] == "https://srv.example.com"
    assert call["repo_url"] == "https://github.com/org/repo.git"
    assert call["repo_name"] == "repo"
    # The token was already resolvable when start_host ran (no dial-back race).
    assert fake.token_resolved_at_start is True
    # The workspace (cloned dir) is returned and the host is online + bound.
    assert result.workspace == "/home/omnigent/workspace/repo"
    host = host_store.get_host(result.host_id)
    assert host is not None
    assert host.status == "online"
    assert host.sandbox_provider == "kubernetes"
    assert host.sandbox_id == "omnigent-pod-1"


async def test_launch_entrypoint_provider_cleans_up_on_launch_failure(db_uri: str) -> None:
    """
    A start_host failure tears the sandbox down (by the reserved id) and deletes
    the host row, exactly like the exec path.
    """
    host_store = HostStore(db_uri)

    class _Failing(_EntrypointFakeLauncher):
        def start_host(self, sandbox_id: str, **kwargs: object) -> str:
            raise click.ClickException("pod could not be scheduled")

    fake = _Failing(host_store)
    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_injected_config(fake), owner=_OWNER, host_store=host_store
        )
    assert exc.value.status_code == 502
    assert "pod could not be scheduled" in exc.value.detail
    # The reserved sandbox was terminated and no host row survives.
    assert fake.terminated == ["omnigent-pod-1"]
    assert host_store.list_hosts(_OWNER) == []


# ── relaunch_managed_host ───────────────────────────────────


async def test_relaunch_rolls_sandbox_generation_under_same_host(db_uri: str) -> None:
    """
    A relaunch terminates the dead generation, provisions a fresh
    sandbox, and re-arms the SAME host row: identity (host_id, name,
    owner) stable, sandbox id rolled, and the NEW token resolving
    while the old one no longer does — a stale token resolving would
    let a dead sandbox's leaked credential impersonate the new host.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        """Simulate the sandbox host connecting over the tunnel."""
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register)
    config = _injected_config(fake)
    first = await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)
    gen1 = host_store.get_host(first.host_id)
    assert gen1 is not None
    gen1_token = fake.host_starts[0].token

    relaunched = await relaunch_managed_host(config=config, host=gen1, host_store=host_store)

    # Same identity, new generation: the session's host binding (which
    # references host_id) survives the roll.
    assert relaunched.host_id == first.host_id
    assert relaunched.workspace == "/root/workspace"
    assert fake.terminated == ["sb-fake-1"]
    host = host_store.get_host(first.host_id)
    assert host is not None
    assert host.sandbox_id == "sb-fake-2"
    assert host.name == gen1.name
    assert host.user_id == _OWNER
    # Generation 2 authenticated with a NEW token; generation 1's is
    # revoked by the re-arm (its digest no longer matches anything).
    gen2_token = fake.host_starts[1].token
    assert gen2_token != gen1_token
    resolved = host_store.resolve_launch_token(fake.host_starts[1].host_id, gen2_token)
    assert resolved is not None and resolved.host_id == first.host_id
    assert host_store.resolve_launch_token(fake.host_starts[0].host_id, gen1_token) is None


async def test_relaunch_failure_keeps_host_row_and_revokes_token(db_uri: str) -> None:
    """
    A FAILED relaunch must not delete the durable host row — deleting
    it would null the session's host binding (FK SET NULL) and make
    the session permanently unrelaunchable. The new sandbox is torn
    down and the armed token revoked, so nothing of the failed
    generation stays live; a later message retries against the kept
    row.
    """
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        """Simulate the sandbox host connecting over the tunnel."""
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = FakeSandboxLauncher(on_host_start=_register)
    config = _injected_config(fake)
    first = await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)
    gen1 = host_store.get_host(first.host_id)
    assert gen1 is not None

    fake.fail_on_host_start = True
    with pytest.raises(HTTPException) as exc:
        await relaunch_managed_host(config=config, host=gen1, host_store=host_store)

    assert exc.value.status_code == 502
    # Both the dead generation 1 and the failed generation 2 sandboxes
    # were terminated — nothing leaks until the provider lifetime cap.
    assert fake.terminated == ["sb-fake-1", "sb-fake-2"]
    # The row SURVIVES the failure (contrast the first-launch failure
    # tests, which delete it), so the session binding stays relaunchable.
    host = host_store.get_host(first.host_id)
    assert host is not None
    # No credential of ANY generation is live: gen 1's was replaced by
    # the re-arm, and the re-armed token was revoked by the failure
    # cleanup (revoke_launch_token — covered directly in the host-store
    # suite). Gen 1's raw token is the only one observable here (the
    # failed start never executed), so assert on it.
    assert (
        host_store.resolve_launch_token(fake.host_starts[0].host_id, fake.host_starts[0].token)
        is None
    )


async def test_relaunch_rejects_unconfigured_provider(db_uri: str) -> None:
    """
    A provider mismatch (the ``sandbox:`` config changed since launch)
    fails the relaunch with a clear 400 instead of aiming another
    provider's terminate/provision at the recorded sandbox id.
    """
    host_store = HostStore(db_uri)
    host = host_store.register_managed_host(
        host_id="8369cb15e751573a1ee641d5fa09c70a",
        name="managed-mismatch",
        user_id=_OWNER,
        token="tok",
        provider="daytona",
        sandbox_id="dt-1",
        token_expires_at=now_epoch() + 3600,
    )

    fake = FakeSandboxLauncher()  # provider "modal" != row's "daytona"
    with pytest.raises(HTTPException) as exc:
        await relaunch_managed_host(
            config=_injected_config(fake), host=host, host_store=host_store
        )

    assert exc.value.status_code == 400
    assert "daytona" in exc.value.detail
    # Nothing was provisioned or terminated against the mismatched row.
    assert fake.provisioned_names == []
    assert fake.terminated == []


# ── resume_managed_host ─────────────────────────────────────


class _IsloFakeLauncher(FakeSandboxLauncher):
    """Fake launcher carrying Islo's provider label for managed resume tests."""

    provider: ClassVar[str] = "islo"


async def test_host_resume_supported_requires_resumable_matching_launcher(db_uri: str) -> None:
    """The wake gate requires matching provider, sandbox id, and ``can_resume``."""
    host_store = HostStore(db_uri)
    host = host_store.register_managed_host(
        host_id="292a6322075a34e482fde44975da10f3",
        name="managed-resume-gate",
        user_id=_OWNER,
        token="tok-resume-gate",
        provider="islo",
        sandbox_id="sb-resume-gate",
        token_expires_at=now_epoch() + 3600,
    )

    resumable = _IsloFakeLauncher(can_resume=True)
    assert host_resume_supported(host, _injected_config(resumable)) is True

    non_resumable = _IsloFakeLauncher(can_resume=False)
    assert host_resume_supported(host, _injected_config(non_resumable)) is False

    mismatched = FakeSandboxLauncher(can_resume=True)  # provider "modal"
    assert host_resume_supported(host, _injected_config(mismatched)) is False

    no_sandbox = host_store.register_managed_host(
        host_id="0c3d744a455047df9a3c0acf432d08dd",
        name="managed-resume-no-sandbox",
        user_id=_OWNER,
        token="tok-resume-no-sandbox",
        provider="islo",
        sandbox_id="sb-temp",
        token_expires_at=now_epoch() + 3600,
    )
    no_sandbox.sandbox_id = None
    assert host_resume_supported(no_sandbox, _injected_config(resumable)) is False


async def test_resume_managed_host_wakes_same_sandbox_and_refreshes_token(db_uri: str) -> None:
    """A resumable managed host wakes in place under the same sandbox id."""
    host_store = HostStore(db_uri)

    def _register(invocation: HostStartInvocation) -> None:
        """Simulate the sandbox host reconnecting over the tunnel."""
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    fake = _IsloFakeLauncher(on_host_start=_register, can_resume=True)
    config = _injected_config(fake)
    first = await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)
    host = host_store.get_host(first.host_id)
    assert host is not None
    assert host.sandbox_provider == "islo"
    assert host.sandbox_id == "sb-fake-1"
    first_token = fake.host_starts[0].token

    host_store.set_offline(first.host_id)
    assert host_resume_supported(host_store.get_host(first.host_id), config) is True

    await resume_managed_host(first.host_id, host_store, config)

    assert fake.resumed == ["sb-fake-1"]
    assert len(fake.provisioned_names) == 1
    woke = host_store.get_host(first.host_id)
    assert woke is not None
    assert woke.status == "online"
    assert woke.sandbox_provider == "islo"
    assert woke.sandbox_id == "sb-fake-1"
    second_token = fake.host_starts[1].token
    assert second_token != first_token
    assert host_store.resolve_launch_token(fake.host_starts[0].host_id, first_token) is None
    resolved = host_store.resolve_launch_token(fake.host_starts[1].host_id, second_token)
    assert resolved is not None and resolved.host_id == first.host_id


async def test_resume_managed_host_force_wakes_fresh_online_row(db_uri: str) -> None:
    """A local missing-tunnel wake can bypass stale cross-replica DB freshness."""
    host_store = HostStore(db_uri)
    host_store.register_managed_host(
        host_id="62d4405ba38711fe34bebfeb5a7adaf2",
        name="managed-resume-force",
        user_id=_OWNER,
        token="tok-resume-force",
        provider="islo",
        sandbox_id="sb-resume-force",
        token_expires_at=now_epoch() + 3600,
    )
    host_store.upsert_on_connect(
        host_id="62d4405ba38711fe34bebfeb5a7adaf2",
        name="managed-resume-force",
        user_id=_OWNER,
    )
    assert host_store.is_online("62d4405ba38711fe34bebfeb5a7adaf2") is True
    fake = _IsloFakeLauncher(can_resume=True)

    await resume_managed_host(
        "62d4405ba38711fe34bebfeb5a7adaf2", host_store, _injected_config(fake), force=True
    )

    assert fake.resumed == ["sb-resume-force"]
    assert len(fake.host_starts) == 1
    assert (
        host_store.resolve_launch_token("62d4405ba38711fe34bebfeb5a7adaf2", "tok-resume-force")
        is None
    )
    resolved = host_store.resolve_launch_token(
        "62d4405ba38711fe34bebfeb5a7adaf2", fake.host_starts[0].token
    )
    assert resolved is not None and resolved.host_id == "62d4405ba38711fe34bebfeb5a7adaf2"


async def test_resume_managed_host_noops_for_non_resumable_provider(db_uri: str) -> None:
    """Non-resumable providers fall through without mutating the host row."""
    host_store = HostStore(db_uri)
    host_store.register_managed_host(
        host_id="249d058fbcde7b2ce941479cdb8c82d7",
        name="managed-resume-noop",
        user_id=_OWNER,
        token="tok-resume-noop",
        provider="modal",
        sandbox_id="sb-resume-noop",
        token_expires_at=now_epoch() + 3600,
    )
    fake = FakeSandboxLauncher(can_resume=False)

    await resume_managed_host(
        "249d058fbcde7b2ce941479cdb8c82d7", host_store, _injected_config(fake)
    )

    assert fake.resumed == []
    assert fake.host_starts == []
    host = host_store.get_host("249d058fbcde7b2ce941479cdb8c82d7")
    assert host is not None
    assert host.status == "offline"
    assert host.sandbox_id == "sb-resume-noop"
    assert (
        host_store.resolve_launch_token("249d058fbcde7b2ce941479cdb8c82d7", "tok-resume-noop")
        is not None
    )


async def test_resume_managed_host_failure_preserves_existing_row_and_token(db_uri: str) -> None:
    """A failed wake leaves the dormant host retryable."""
    host_store = HostStore(db_uri)
    host_store.register_managed_host(
        host_id="efbef7dede7be6577770cbb1287992f2",
        name="managed-resume-fail",
        user_id=_OWNER,
        token="tok-resume-fail",
        provider="islo",
        sandbox_id="sb-resume-fail",
        token_expires_at=now_epoch() + 3600,
    )
    fake = _IsloFakeLauncher(can_resume=True, fail_on_resume=True)

    with pytest.raises(HTTPException) as exc:
        await resume_managed_host(
            "efbef7dede7be6577770cbb1287992f2", host_store, _injected_config(fake)
        )

    assert exc.value.status_code == 502
    assert "managed host wake failed" in exc.value.detail
    assert fake.host_starts == []
    host = host_store.get_host("efbef7dede7be6577770cbb1287992f2")
    assert host is not None
    assert host.status == "offline"
    assert host.sandbox_id == "sb-resume-fail"
    assert (
        host_store.resolve_launch_token("efbef7dede7be6577770cbb1287992f2", "tok-resume-fail")
        is not None
    )


# ── terminate_managed_host ──────────────────────────────────


async def test_terminate_managed_host_terminates_and_deletes_row(db_uri: str) -> None:
    """
    Cleanup terminates the provider sandbox and deletes the host row —
    one operation that removes the host from the picker AND revokes
    its launch token.
    """
    fake = FakeSandboxLauncher()
    host_store = HostStore(db_uri)
    host = host_store.register_managed_host(
        host_id="62a91eb065624754c6a6dfb5869dd7e8",
        name="managed-term1",
        user_id=_OWNER,
        token="tok-term-1",
        provider="modal",
        sandbox_id="sb-term-1",
        token_expires_at=now_epoch() + 3600,
    )

    await terminate_managed_host(host, host_store, _injected_config(fake))

    assert fake.terminated == ["sb-term-1"]
    assert host_store.get_host("62a91eb065624754c6a6dfb5869dd7e8") is None
    assert (
        host_store.resolve_launch_token("62a91eb065624754c6a6dfb5869dd7e8", "tok-term-1") is None
    )


async def test_terminate_managed_host_retains_row_when_terminate_fails(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A provider failure revokes the launch token but retains the durable
    sandbox identity so a later teardown can retry.
    """
    fake = FakeSandboxLauncher()

    def _explode(sandbox_id: str) -> None:
        """Simulate a provider API failure during termination."""
        raise click.ClickException("provider unavailable")

    monkeypatch.setattr(fake, "terminate", _explode)
    host_store = HostStore(db_uri)
    host = host_store.register_managed_host(
        host_id="057e7fa3f1cdb40c0ec393a3d42affc7",
        name="managed-term2",
        user_id=_OWNER,
        token="tok-term-2",
        provider="modal",
        sandbox_id="sb-term-2",
        token_expires_at=now_epoch() + 3600,
    )

    await terminate_managed_host(host, host_store, _injected_config(fake))

    assert host_store.get_host("057e7fa3f1cdb40c0ec393a3d42affc7") is not None
    assert (
        host_store.resolve_launch_token("057e7fa3f1cdb40c0ec393a3d42affc7", "tok-term-2") is None
    )


async def test_terminate_managed_host_skips_mismatched_provider(db_uri: str) -> None:
    """
    A config change between launch and teardown (current launcher's
    provider ≠ the provider recorded on the row) must NOT aim the new
    provider's terminate at a stale sandbox id. The row remains as the
    durable retry identity, but its launch token is revoked.
    """
    fake = FakeSandboxLauncher()  # provider "modal"
    host_store = HostStore(db_uri)
    host = host_store.register_managed_host(
        host_id="487212fd2b157b6ab6a6d6d3ef06ce5b",
        name="managed-term3",
        user_id=_OWNER,
        token="tok-term-3",
        # Row launched under a provider the current config doesn't run.
        provider="acme-cloud",
        sandbox_id="sb-term-3",
        token_expires_at=now_epoch() + 3600,
    )

    await terminate_managed_host(host, host_store, _injected_config(fake))
    # No cross-provider terminate was attempted.
    assert fake.terminated == []
    assert host_store.get_host("487212fd2b157b6ab6a6d6d3ef06ce5b") is not None
    assert (
        host_store.resolve_launch_token("487212fd2b157b6ab6a6d6d3ef06ce5b", "tok-term-3") is None
    )

    # config=None behaves the same: identity retained, token revoked.
    host2 = host_store.register_managed_host(
        host_id="b114bf90a8fd155ce6007c3bb262aa79",
        name="managed-term4",
        user_id=_OWNER,
        token="tok-term-4",
        provider="modal",
        sandbox_id="sb-term-4",
        token_expires_at=now_epoch() + 3600,
    )
    await terminate_managed_host(host2, host_store, None)
    assert host_store.get_host("b114bf90a8fd155ce6007c3bb262aa79") is not None


def test_parse_modal_secrets_thread_to_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ``sandbox.modal.secrets`` names reach the launcher constructor —
    the path that injects the deployment's harness LLM credentials
    into every managed sandbox.
    """
    cfg = parse_sandbox_config(
        {
            "provider": "modal",
            "server_url": "https://s.example.com",
            "modal": {"secrets": ["omnigent-llm", "gateway-extras"]},
        }
    )
    assert cfg is not None
    fake = FakeSandboxLauncher()
    install_fake_modal_launcher(monkeypatch, fake)
    assert cfg.launcher_factory() is fake
    assert fake.secrets == ["omnigent-llm", "gateway-extras"]
    # secrets without image: the official-image default still applies.
    assert fake.image is None


@pytest.mark.parametrize(
    "secrets",
    [
        "omnigent-llm",  # scalar, not a list
        ["omnigent-llm", 7],  # non-string entry
        ["  "],  # empty name
    ],
)
def test_parse_modal_secrets_malformed_fails_loud(secrets: object) -> None:
    """A present-but-malformed secrets value stops startup with the key named."""
    with pytest.raises(ValueError, match=r"sandbox\.modal\.secrets"):
        parse_sandbox_config(
            {
                "provider": "modal",
                "server_url": "https://s.example.com",
                "modal": {"secrets": secrets},
            }
        )


# ── credential hook seam ────────────────────────────────────


class _FakeCredentialLease(ManagedCredentialLease):
    """
    Test lease carrying a secret that must never surface in repr/logs.

    Records how often it is released so cleanup-semantics tests can assert
    the managed-launch layer honours its one release obligation (failure of
    the launch the lease was acquired for).

    :param reference: The non-secret handle exposed to launcher startup.
    :param secret: A stand-in credential value the lease must NOT leak
        through :meth:`__repr__` / ``str`` — the redaction guarantee.
    """

    def __init__(
        self,
        reference: str | None = "managed-cred-abc123",
        secret: str = "s3cr3t-token-value",
    ) -> None:
        self._reference = reference
        self._secret = secret
        self.release_calls = 0

    @property
    def reference(self) -> str | None:
        """The non-secret handle the launcher would resolve."""
        return self._reference

    async def release(self) -> None:
        """Count releases (the layer must call this at most once, on failure)."""
        self.release_calls += 1


class _ChangingReferenceLease(_FakeCredentialLease):
    """Return a different handle on every read to prove launch snapshots once."""

    def __init__(self) -> None:
        super().__init__(reference=None)
        self.reference_reads = 0

    @property
    def reference(self) -> str:
        self.reference_reads += 1
        return f"managed-cred-{self.reference_reads}"


class _FailingReferenceLease(_FakeCredentialLease):
    """Raise credential-bearing provider text when the reference is resolved."""

    @property
    def reference(self) -> str:
        raise RuntimeError("reference lookup exposed TOP-SECRET-VALUE")


class _FakeCredentialHook(ManagedCredentialHook):
    """
    Recording :class:`ManagedCredentialHook` for the seam tests.

    Captures every :class:`ManagedLaunchContext` it is asked to resolve so a
    test can assert the launch identity, and optionally fails the resolution
    to exercise the abort-and-teardown path.

    :param lease: The lease to hand back on success; a default fake when
        ``None``.
    :param fail: When ``True``, :meth:`acquire` raises instead of returning
        a lease (simulates a resolver that cannot mint credentials).
    """

    def __init__(self, lease: _FakeCredentialLease | None = None, *, fail: bool = False) -> None:
        self._lease = lease if lease is not None else _FakeCredentialLease()
        self._fail = fail
        self.contexts: list[ManagedLaunchContext] = []
        self.generations: list[int] = []
        self.release_calls: list[tuple[ManagedCredentialReleaseContext, int, str | None]] = []

    @property
    def lease(self) -> _FakeCredentialLease:
        """The lease this hook resolves to."""
        return self._lease

    async def acquire(
        self,
        context: ManagedLaunchContext,
        generation: int,
    ) -> ManagedCredentialLease:
        """Record the context, then resolve to the lease (or fail)."""
        self.contexts.append(context)
        self.generations.append(generation)
        if self._fail:
            raise RuntimeError("credential resolution failed")
        return self._lease

    async def release(
        self,
        context: ManagedCredentialReleaseContext,
        generation: int,
        reference: str | None,
    ) -> None:
        """Record a durable release that can run without the original lease object."""
        self.release_calls.append((context, generation, reference))


@dataclasses.dataclass(frozen=True, repr=False)
class _FakeKubernetesSecret:
    """Secret-like provider resource whose value is always redacted."""

    name: str
    host_id: str
    generation: int
    _data: dict[str, str] = dataclasses.field(compare=False)

    def __repr__(self) -> str:
        return f"_FakeKubernetesSecret(name={self.name!r})"


class _FakeKubernetesCredentialProvider:
    """Track provider-side Secrets independently from the durable SQL ledger."""

    _SECRET_VALUE = "fake-provider-secret-value"

    def __init__(self) -> None:
        self._secrets: dict[str, _FakeKubernetesSecret] = {}

    @staticmethod
    def resource_name(
        context: ManagedLaunchContext | ManagedCredentialReleaseContext,
        generation: int,
    ) -> str:
        return f"managed-cred-{context.host_id[:8]}-{generation}"

    @property
    def names(self) -> set[str]:
        return set(self._secrets)

    def create(
        self,
        context: ManagedLaunchContext | ManagedCredentialReleaseContext,
        generation: int,
    ) -> str:
        name = self.resource_name(context, generation)
        self._secrets[name] = _FakeKubernetesSecret(
            name=name,
            host_id=context.host_id,
            generation=generation,
            _data={"token": self._SECRET_VALUE},
        )
        return name

    def delete(self, name: str) -> None:
        self._secrets.pop(name, None)

    def __repr__(self) -> str:
        return f"_FakeKubernetesCredentialProvider(names={sorted(self._secrets)!r})"


class _FakeProviderCredentialHook(ManagedCredentialHook):
    """Materialize deterministic Kubernetes-like Secrets for acceptance tests."""

    def __init__(
        self,
        provider: _FakeKubernetesCredentialProvider,
        *,
        fail_after_create: bool = False,
    ) -> None:
        self.provider = provider
        self.fail_after_create = fail_after_create
        self.release_calls: list[tuple[ManagedCredentialReleaseContext, int, str | None]] = []

    @staticmethod
    def resource_name(
        context: ManagedLaunchContext | ManagedCredentialReleaseContext,
        generation: int,
    ) -> str:
        """Return the deterministic provider resource name for a generation."""
        return _FakeKubernetesCredentialProvider.resource_name(context, generation)

    async def acquire(
        self,
        context: ManagedLaunchContext,
        generation: int,
    ) -> ManagedCredentialLease:
        """Create a provider resource, optionally crashing before returning it."""
        reference = self.provider.create(context, generation)
        if self.fail_after_create:
            raise RuntimeError("credential provider failed after creating resource")
        return _FakeCredentialLease(reference=reference)

    async def release(
        self,
        context: ManagedCredentialReleaseContext,
        generation: int,
        reference: str | None,
    ) -> None:
        """Delete by reference or reconstruct the name for an interrupted acquire."""
        self.release_calls.append((context, generation, reference))
        self.provider.delete(reference or self.resource_name(context, generation))


class _CredentialRefRecordingLauncher(FakeSandboxLauncher):
    """
    A fake launcher that records the ``credential_reference`` ``start_host``
    received — the observable proof the resolved lease's non-secret handle
    reached launcher startup through the generic seam.
    """

    def __init__(
        self,
        *,
        on_host_start: Callable[[HostStartInvocation], None] | None = None,
        fail_on_host_start: bool = False,
    ) -> None:
        super().__init__(on_host_start=on_host_start, fail_on_host_start=fail_on_host_start)
        self.credential_references: list[str | None] = []

    def start_host(
        self,
        sandbox_id: str,
        *,
        token: str,
        host_id: str,
        host_name: str,
        server_url: str,
        repo_url: str | None = None,
        repo_branch: str | None = None,
        repo_name: str | None = None,
        host_config: dict[str, object] | None = None,
        credential_reference: str | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> str:
        """Record the credential reference, then run the shared start path."""
        self.credential_references.append(credential_reference)
        return super().start_host(
            sandbox_id,
            token=token,
            host_id=host_id,
            host_name=host_name,
            server_url=server_url,
            repo_url=repo_url,
            repo_branch=repo_branch,
            repo_name=repo_name,
            host_config=host_config,
            credential_reference=credential_reference,
            on_stage=on_stage,
        )


def _hook_config(
    fake: FakeSandboxLauncher,
    hook: ManagedCredentialHook | None,
    *,
    acquisition_timeout_s: float = 120.0,
) -> ManagedSandboxConfig:
    """
    Build an injected config that also wires *hook* onto ``credential_hook``.

    :param fake: The launcher every launch should use.
    :param hook: The credential hook to configure, or ``None`` for the
        default (no-hook) behavior.
    :returns: A ready :class:`ManagedSandboxConfig`.
    """
    return ManagedSandboxConfig(
        server_url="https://srv.example.com",
        launcher_factory=lambda: fake,
        token_ttl_s=3600,
        credential_hook=hook,
        credential_acquisition_timeout_s=acquisition_timeout_s,
    )


def _online_register(host_store: HostStore) -> Callable[[HostStartInvocation], None]:
    """Return an ``on_host_start`` that brings the host online via the store."""

    def _register(invocation: HostStartInvocation) -> None:
        host_store.upsert_on_connect(
            host_id=invocation.host_id,
            name=invocation.host_name,
            user_id=_OWNER,
        )

    return _register


def test_launch_context_is_immutable() -> None:
    """
    The launch identity handed to a hook is frozen — a hook cannot mutate the
    owner/host/session it is resolving credentials for mid-resolution.
    """
    context = ManagedLaunchContext(owner=_OWNER, host_id="h1", host_name="managed-h1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.owner = "mallory@example.com"  # type: ignore[misc]


async def test_credential_lease_repr_redacts_secret() -> None:
    """
    The base lease repr resolves no provider properties and exposes no handles;
    a subclass that stashes raw credential material still reprs safely.
    """
    lease = _FakeCredentialLease(reference="k8s-secret-xyz", secret="TOP-SECRET-VALUE")
    rendered = repr(lease)
    assert "TOP-SECRET-VALUE" not in rendered
    assert "k8s-secret-xyz" not in rendered
    assert rendered == "_FakeCredentialLease(reference=[REDACTED], secret=[REDACTED])"


async def test_default_lease_release_is_noop() -> None:
    """
    A lease that does not override ``release`` inherits a concrete no-op — an
    addon whose credentials need no teardown need not implement cleanup.
    """

    class _MinimalLease(ManagedCredentialLease):
        @property
        def reference(self) -> str | None:
            return None

    # Must be awaitable and must not raise (idempotent no-op contract).
    await _MinimalLease().release()


async def test_launch_without_hook_passes_no_credential_reference(db_uri: str) -> None:
    """
    No-hook compatibility: with ``credential_hook`` unset the launch behaves
    exactly as before — the launcher's ``start_host`` receives no credential
    reference (``None``), and nothing is torn down.
    """
    host_store = HostStore(db_uri)
    fake = _CredentialRefRecordingLauncher(on_host_start=_online_register(host_store))

    result = await launch_managed_host(
        config=_hook_config(fake, None),
        owner=_OWNER,
        host_store=host_store,
    )

    assert fake.credential_references == [None]
    host = host_store.get_host(result.host_id)
    assert host is not None and host.status == "online"
    assert fake.terminated == []
    records = host_store.list_credential_leases(result.host_id)
    assert len(records) == 1
    assert records[0].credential_cleanup_required is False


async def test_launch_invokes_hook_once_with_full_context(db_uri: str) -> None:
    """
    A first launch consults the hook EXACTLY ONCE and hands it the complete
    launch identity: owner, the freshly minted host id/name, the session id
    threaded from the caller, and the repository coordinates.
    """
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()
    repo = parse_repo_workspace("https://github.com/org/repo#release-1.2")

    result = await launch_managed_host(
        config=_hook_config(fake, hook),
        owner=_OWNER,
        host_store=host_store,
        repo=repo,
        session_id="conv_ctx123",
    )

    assert len(hook.contexts) == 1
    context = hook.contexts[0]
    assert context.owner == _OWNER
    assert context.host_id == result.host_id
    assert context.host_name == fake.host_starts[0].host_name
    assert context.session_id == "conv_ctx123"
    assert context.repo_url == "https://github.com/org/repo"
    assert context.repo_branch == "release-1.2"
    assert context.repo_name == "repo"
    assert hook.generations == [1]


async def test_launch_without_session_id_leaves_context_session_none(db_uri: str) -> None:
    """
    When no session id is available at the launch boundary the context's
    ``session_id`` is ``None`` — never fabricated.
    """
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()

    await launch_managed_host(
        config=_hook_config(fake, hook),
        owner=_OWNER,
        host_store=host_store,
    )

    assert hook.contexts[0].session_id is None
    assert hook.contexts[0].repo_url is None


async def test_launch_exposes_lease_reference_to_start_host(db_uri: str) -> None:
    """
    The resolved lease's NON-SECRET reference is exposed to launcher startup —
    the generic seam a future provider (e.g. the Kubernetes envFrom Secret)
    consumes.
    """
    host_store = HostStore(db_uri)
    fake = _CredentialRefRecordingLauncher(on_host_start=_online_register(host_store))
    lease = _FakeCredentialLease(reference="managed-cred-xyz")
    hook = _FakeCredentialHook(lease)

    await launch_managed_host(
        config=_hook_config(fake, hook),
        owner=_OWNER,
        host_store=host_store,
    )

    assert fake.credential_references == ["managed-cred-xyz"]
    records = host_store.list_credential_leases()
    assert len(records) == 1
    assert records[0].host_id == fake.host_starts[0].host_id
    assert records[0].session_id is None
    assert records[0].generation == 1
    assert records[0].reference == "managed-cred-xyz"
    assert records[0].state == "active"
    # Only durable, non-secret metadata is persisted.
    assert "s3cr3t-token-value" not in repr(records[0])
    engine = sa.create_engine(db_uri)
    try:
        with engine.connect() as connection:
            persisted = (
                connection.execute(sa.text("SELECT * FROM managed_credential_leases"))
                .mappings()
                .one()
            )
        assert set(persisted) == {
            "workspace_id",
            "host_id",
            "generation",
            "user_id",
            "host_name",
            "sandbox_provider",
            "sandbox_id",
            "session_id",
            "repo_url",
            "repo_branch",
            "repo_name",
            "reference",
            "credential_cleanup_required",
            "launch_owner_id",
            "owner_expires_at",
            "claim_owner",
            "claim_expires_at",
            "state",
            "created_at",
            "updated_at",
        }
        assert "s3cr3t-token-value" not in repr(dict(persisted))
    finally:
        engine.dispose()
    assert lease.release_calls == 0


async def test_launch_snapshots_credential_reference_once(db_uri: str) -> None:
    """A stateful reference property cannot diverge between SQL and launcher state."""
    host_store = HostStore(db_uri)
    fake = _CredentialRefRecordingLauncher(on_host_start=_online_register(host_store))
    lease = _ChangingReferenceLease()

    await launch_managed_host(
        config=_hook_config(fake, _FakeCredentialHook(lease)),
        owner=_OWNER,
        host_store=host_store,
    )

    assert lease.reference_reads == 1
    assert fake.credential_references == ["managed-cred-1"]
    assert host_store.list_credential_leases()[0].reference == "managed-cred-1"


async def test_reference_resolution_failure_is_redacted_and_recoverable(
    db_uri: str,
) -> None:
    """A raising reference property uses ordinary launch teardown without leaking text."""
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher()
    lease = _FailingReferenceLease()

    with pytest.raises(HTTPException) as exc_info:
        await launch_managed_host(
            config=_hook_config(fake, _FakeCredentialHook(lease)),
            owner=_OWNER,
            host_store=host_store,
        )

    assert exc_info.value.status_code == 502
    assert "TOP-SECRET-VALUE" not in str(exc_info.value.detail)
    assert lease.release_calls == 1
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_credential_leases() == []


async def test_long_credential_reference_round_trips(db_uri: str) -> None:
    """Provider handles are arbitrary non-secret text, not a 256-byte protocol field."""
    host_store = HostStore(db_uri)
    fake = _CredentialRefRecordingLauncher(on_host_start=_online_register(host_store))
    reference = "provider-resource/" + ("x" * 512)

    await launch_managed_host(
        config=_hook_config(fake, _FakeCredentialHook(_FakeCredentialLease(reference))),
        owner=_OWNER,
        host_store=host_store,
    )

    assert fake.credential_references == [reference]
    assert host_store.list_credential_leases()[0].reference == reference


async def test_provider_secret_and_durable_lease_share_one_lifecycle(db_uri: str) -> None:
    """A launched sandbox has one provider Secret and termination leaves no orphan."""
    launch_store = HostStore(db_uri)
    launcher = _CredentialRefRecordingLauncher(on_host_start=_online_register(launch_store))
    provider = _FakeKubernetesCredentialProvider()
    launch_hook = _FakeProviderCredentialHook(provider)

    result = await launch_managed_host(
        config=_hook_config(launcher, launch_hook),
        owner=_OWNER,
        host_store=launch_store,
        session_id="conv_acceptance",
    )

    active = launch_store.list_credential_leases(result.host_id)
    assert len(active) == 1
    assert active[0].state == "active"
    assert active[0].sandbox_id == "sb-fake-1"
    assert active[0].reference is not None
    assert provider.names == {active[0].reference}
    assert launcher.credential_references == [active[0].reference]
    assert provider._SECRET_VALUE not in repr(provider)

    restarted_store = HostStore(db_uri)
    host = restarted_store.get_host(result.host_id)
    assert host is not None
    restarted_hook = _FakeProviderCredentialHook(provider)
    await terminate_managed_host(
        host,
        restarted_store,
        _hook_config(launcher, restarted_hook),
    )

    assert provider.names == set()
    assert restarted_store.list_credential_leases(result.host_id) == []
    assert restarted_store.get_host(result.host_id) is None
    assert len(restarted_hook.release_calls) == 1


async def test_terminate_releases_successful_lease_after_process_restart(db_uri: str) -> None:
    """Teardown reconstructs cleanup from storage, not the launch's in-memory lease."""
    launch_store = HostStore(db_uri)
    launch_launcher = FakeSandboxLauncher(on_host_start=_online_register(launch_store))
    launch_hook = _FakeCredentialHook(
        _FakeCredentialLease(reference="managed-cred-restart", secret="never-persist-me")
    )
    result = await launch_managed_host(
        config=_hook_config(launch_launcher, launch_hook),
        owner=_OWNER,
        host_store=launch_store,
        session_id="conv_restart",
    )

    # New store + hook instances model a fresh server process with no access to
    # the lease object returned during launch.
    restarted_store = HostStore(db_uri)
    restarted_hook = _FakeCredentialHook()
    restarted_launcher = FakeSandboxLauncher()
    host = restarted_store.get_host(result.host_id)
    assert host is not None

    await terminate_managed_host(
        host,
        restarted_store,
        _hook_config(restarted_launcher, restarted_hook),
    )

    assert restarted_launcher.terminated == ["sb-fake-1"]
    assert restarted_store.get_host(result.host_id) is None
    assert restarted_store.list_credential_leases() == []
    assert len(restarted_hook.release_calls) == 1
    context, generation, reference = restarted_hook.release_calls[0]
    assert context.owner == _OWNER
    assert context.host_id == result.host_id
    assert context.sandbox_provider == "modal"
    assert context.sandbox_id == "sb-fake-1"
    assert context.session_id == "conv_restart"
    assert generation == 1
    assert reference == "managed-cred-restart"
    assert launch_hook.lease.release_calls == 0


async def test_failed_teardown_release_remains_recoverable_after_host_deletion(
    db_uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanup failures stay retryable without persisting or logging hook secrets."""

    secret = "provider-error-secret-value"

    class _FailingReleaseHook(_FakeProviderCredentialHook):
        async def release(
            self,
            context: ManagedCredentialReleaseContext,
            generation: int,
            reference: str | None,
        ) -> None:
            raise RuntimeError(f"temporary credential provider outage: {secret}")

    launch_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(launch_store))
    provider = _FakeKubernetesCredentialProvider()
    result = await launch_managed_host(
        config=_hook_config(launcher, _FakeProviderCredentialHook(provider)),
        owner=_OWNER,
        host_store=launch_store,
    )
    host = launch_store.get_host(result.host_id)
    assert host is not None

    await terminate_managed_host(
        host,
        launch_store,
        _hook_config(launcher, _FailingReleaseHook(provider)),
    )

    assert launch_store.get_host(result.host_id) is None
    retryable = launch_store.list_credential_leases(result.host_id)
    assert len(retryable) == 1
    assert retryable[0].state == "retiring"
    assert retryable[0].claim_owner is not None
    assert retryable[0].claim_expires_at is not None
    assert secret not in repr(retryable[0])
    assert secret not in caplog.text
    assert len(provider.names) == 1

    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()

    recovery_hook = _FakeProviderCredentialHook(provider)
    await recover_managed_credential_leases(
        _hook_config(FakeSandboxLauncher(), recovery_hook),
        HostStore(db_uri),
    )
    assert launch_store.list_credential_leases(result.host_id) == []
    assert provider.names == set()
    assert len(recovery_hook.release_calls) == 1


async def test_recovery_does_not_repeat_completed_provider_release(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider success is durable even when the final tombstone CAS fails."""
    store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(store))
    provider = _FakeKubernetesCredentialProvider()
    result = await launch_managed_host(
        config=_hook_config(launcher, _FakeProviderCredentialHook(provider)),
        owner=_OWNER,
        host_store=store,
    )
    host = store.get_host(result.host_id)
    assert host is not None
    first_cleanup_hook = _FakeProviderCredentialHook(provider)
    original_release = store.release_credential_lease
    monkeypatch.setattr(store, "release_credential_lease", lambda *args, **kwargs: False)

    await terminate_managed_host(
        host,
        store,
        _hook_config(launcher, first_cleanup_hook),
    )

    retryable = store.list_credential_leases(result.host_id)
    assert len(retryable) == 1
    assert retryable[0].state == "retiring"
    assert retryable[0].credential_cleanup_required is False
    assert retryable[0].reference is None
    assert provider.names == set()
    assert len(first_cleanup_hook.release_calls) == 1

    monkeypatch.setattr(store, "release_credential_lease", original_release)
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()

    retry_hook = _FakeProviderCredentialHook(provider)
    await recover_managed_credential_leases(
        _hook_config(FakeSandboxLauncher(), retry_hook),
        store,
    )

    assert retry_hook.release_calls == []
    assert store.list_credential_leases(result.host_id) == []


async def test_restart_recovery_sweeps_pending_credential_lease(db_uri: str) -> None:
    """A process crash mid-acquire leaves enough durable state for zero orphans."""
    initial_store = HostStore(db_uri)
    host_id = "a1b2c3d4e5f6478890abcdef12345678"
    initial_store.register_managed_host(
        host_id=host_id,
        name="managed-recovery",
        user_id=_OWNER,
        token="raw-token-never-persisted",
        provider="modal",
        sandbox_id="sb-orphaned",
        token_expires_at=now_epoch() + 3600,
    )
    pending = initial_store.record_credential_lease(
        host_id=host_id,
        user_id=_OWNER,
        host_name="managed-recovery",
        sandbox_provider="modal",
        sandbox_id="sb-orphaned",
        session_id="conv_crashed",
        repo_url="https://github.com/acme/repo.git",
        repo_branch="main",
        repo_name="repo",
        reference=None,
        owner_token="crashed-launch-owner",
    )
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(sa.text("UPDATE managed_credential_leases SET updated_at = 0"))
    finally:
        engine.dispose()
    context = ManagedLaunchContext(
        owner=_OWNER,
        host_id=host_id,
        host_name="managed-recovery",
        session_id="conv_crashed",
    )
    provider = _FakeKubernetesCredentialProvider()
    provider.create(context, pending.generation)

    restarted_store = HostStore(db_uri)
    restarted_hook = _FakeProviderCredentialHook(provider)
    restarted_launcher = FakeSandboxLauncher()
    await recover_managed_credential_leases(
        _hook_config(restarted_launcher, restarted_hook),
        restarted_store,
    )

    assert restarted_launcher.terminated == ["sb-orphaned"]
    assert restarted_store.get_host(host_id) is None
    assert restarted_store.list_credential_leases() == []
    assert provider.names == set()
    assert len(restarted_hook.release_calls) == 1
    assert restarted_hook.release_calls[0][1:] == (1, None)


async def test_real_app_lifespan_recovers_crashed_launch(
    db_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production lifespan starts a reconciler that closes a prior-process crash."""
    crashed_store = HostStore(db_uri)
    host_id = "f1e2d3c4b5a6478890abcdef12345678"
    crashed_store.register_managed_host(
        host_id=host_id,
        name="managed-lifespan-recovery",
        user_id=_OWNER,
        token="ephemeral-launch-token",
        provider="modal",
        sandbox_id="sb-lifespan-orphan",
        token_expires_at=now_epoch() + 3600,
    )
    context = ManagedLaunchContext(
        owner=_OWNER,
        host_id=host_id,
        host_name="managed-lifespan-recovery",
        session_id="conv_lifespan_crash",
        repo_url="https://github.com/acme/repo.git",
        repo_branch="main",
        repo_name="repo",
    )
    pending = crashed_store.record_credential_lease(
        host_id=host_id,
        user_id=context.owner,
        host_name=context.host_name,
        sandbox_provider="modal",
        sandbox_id="sb-lifespan-orphan",
        session_id=context.session_id,
        repo_url=context.repo_url,
        repo_branch=context.repo_branch,
        repo_name=context.repo_name,
        reference=None,
        owner_token="dead-process-owner",
    )
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(sa.text("UPDATE managed_credential_leases SET updated_at = 0"))
    finally:
        engine.dispose()

    provider = _FakeKubernetesCredentialProvider()
    provider.create(context, pending.generation)
    hook = _FakeProviderCredentialHook(provider)
    launcher = FakeSandboxLauncher()
    app = _capability_probe_app(db_uri, tmp_path, _hook_config(launcher, hook))
    monkeypatch.setattr(_globals, "_terminal_registry", TerminalRegistry())

    async with app.router.lifespan_context(app):
        for _ in range(100):
            leases = await asyncio.to_thread(crashed_store.list_credential_leases, host_id)
            if not provider.names and not leases:
                break
            await asyncio.sleep(0.01)

    restarted_store = HostStore(db_uri)
    assert provider.names == set()
    assert launcher.terminated == ["sb-lifespan-orphan"]
    assert restarted_store.get_host(host_id) is None
    assert restarted_store.list_credential_leases() == []
    release_context = hook.release_calls[0][0]
    assert release_context.repo_url == context.repo_url
    assert release_context.repo_branch == context.repo_branch
    assert release_context.repo_name == context.repo_name


async def test_lifespan_stays_available_when_credential_release_blocks(
    db_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup and shutdown stay bounded while recovery remains durably retryable."""

    class _BlockingReleaseHook(_FakeProviderCredentialHook):
        def __init__(self, provider: _FakeKubernetesCredentialProvider) -> None:
            super().__init__(provider)
            self.entered = asyncio.Event()
            self.release_gate = asyncio.Event()

        async def release(
            self,
            context: ManagedCredentialReleaseContext,
            generation: int,
            reference: str | None,
        ) -> None:
            self.entered.set()
            await self.release_gate.wait()
            await super().release(context, generation, reference)

    store = HostStore(db_uri)
    host_id = "f2e3d4c5b6a7488990abcdef12345678"
    store.register_managed_host(
        host_id=host_id,
        name="managed-blocked-recovery",
        user_id=_OWNER,
        token="ephemeral-launch-token",
        provider="modal",
        sandbox_id="sb-blocked-recovery",
        token_expires_at=now_epoch() + 3600,
    )
    context = ManagedLaunchContext(
        owner=_OWNER,
        host_id=host_id,
        host_name="managed-blocked-recovery",
    )
    pending = store.record_credential_lease(
        host_id=host_id,
        user_id=context.owner,
        host_name=context.host_name,
        sandbox_provider="modal",
        sandbox_id="sb-blocked-recovery",
        session_id=None,
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        reference=None,
        owner_token="dead-process-owner",
    )
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text(
                    "UPDATE managed_credential_leases SET owner_expires_at = 0, updated_at = 0"
                )
            )

        provider = _FakeKubernetesCredentialProvider()
        provider.create(context, pending.generation)
        hook = _BlockingReleaseHook(provider)
        launcher = FakeSandboxLauncher()
        config = _hook_config(launcher, hook)
        app = _capability_probe_app(db_uri, tmp_path, config)
        monkeypatch.setattr(_globals, "_terminal_registry", TerminalRegistry())

        async def probe_startup() -> None:
            async with app.router.lifespan_context(app):
                await asyncio.wait_for(hook.entered.wait(), timeout=0.5)
                assert provider.names
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.get("/v1/info")
                assert response.status_code == 200, response.text

        await asyncio.wait_for(probe_startup(), timeout=1)

        retryable = store.list_credential_leases(host_id)
        assert [(row.state, row.claim_owner is not None) for row in retryable] == [
            ("recovering", True)
        ]
        assert launcher.terminated == ["sb-blocked-recovery"]

        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
        hook.release_gate.set()
        retry_launcher = FakeSandboxLauncher()
        await recover_managed_credential_leases(
            _hook_config(retry_launcher, hook),
            HostStore(db_uri),
        )

        assert retry_launcher.terminated == ["sb-blocked-recovery"]
        assert provider.names == set()
        assert len(hook.release_calls) == 1
        assert store.list_credential_leases(host_id) == []
    finally:
        engine.dispose()


async def test_concurrent_generation_reservations_are_unique(db_uri: str) -> None:
    """Concurrent replicas serialize MAX+insert and never reuse a generation."""
    store = HostStore(db_uri)
    host_id = "aabbccddeeff478890abcdef12345678"
    store.register_managed_host(
        host_id=host_id,
        name="managed-concurrent",
        user_id=_OWNER,
        token="launch-token",
        provider="modal",
        sandbox_id="sb-concurrent",
        token_expires_at=now_epoch() + 3600,
    )

    async def reserve(index: int) -> int:
        record = await asyncio.to_thread(
            store.record_credential_lease,
            host_id=host_id,
            user_id=_OWNER,
            host_name="managed-concurrent",
            sandbox_provider="modal",
            sandbox_id=f"sb-concurrent-{index}",
            session_id=f"conv-{index}",
            repo_url=None,
            repo_branch=None,
            repo_name=None,
            reference=None,
            owner_token=f"owner-{index}",
        )
        return record.generation

    generations = await asyncio.gather(*(reserve(index) for index in range(16)))
    assert sorted(generations) == list(range(1, 17))


async def test_recovery_claim_is_single_owner_across_replicas(db_uri: str) -> None:
    """Two reconcilers cannot release or tombstone the same generation."""
    store = HostStore(db_uri)
    host_id = "112233445566478890abcdef12345678"
    context = ManagedLaunchContext(
        owner=_OWNER,
        host_id=host_id,
        host_name="managed-multi-replica",
    )
    pending = store.record_credential_lease(
        host_id=host_id,
        user_id=context.owner,
        host_name=context.host_name,
        sandbox_provider="modal",
        sandbox_id="sb-multi-replica",
        session_id=None,
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        reference=None,
        owner_token="dead-owner",
    )
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(sa.text("UPDATE managed_credential_leases SET updated_at = 0"))
    finally:
        engine.dispose()

    provider = _FakeKubernetesCredentialProvider()
    provider.create(context, pending.generation)
    hooks = [_FakeProviderCredentialHook(provider), _FakeProviderCredentialHook(provider)]
    launchers = [FakeSandboxLauncher(), FakeSandboxLauncher()]
    await asyncio.gather(
        *(
            recover_managed_credential_leases(
                _hook_config(launcher, hook),
                HostStore(db_uri),
            )
            for launcher, hook in zip(launchers, hooks, strict=True)
        )
    )

    assert provider.names == set()
    assert sum(len(hook.release_calls) for hook in hooks) == 1
    assert sum(len(launcher.terminated) for launcher in launchers) == 1
    assert store.list_credential_leases() == []


async def test_expired_cleanup_owner_is_fenced_before_provider_release(db_uri: str) -> None:
    """A worker that lost its claim cannot invoke provider cleanup."""
    store = HostStore(db_uri)
    host_id = "223344556677488990abcdef12345678"
    pending = store.record_credential_lease(
        host_id=host_id,
        user_id=_OWNER,
        host_name="managed-expired-claim",
        sandbox_provider="modal",
        sandbox_id="sb-expired-claim",
        session_id=None,
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        reference="credential-reference",
        owner_token="launch-owner",
        owner_expires_at=now_epoch() + 120,
    )
    old_claim = store.claim_pending_credential_lease(
        host_id,
        pending.generation,
        owner_token="launch-owner",
        claim_owner="old-cleaner",
        claim_expires_at=now_epoch() + 120,
    )
    assert old_claim is not None
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()
    new_claims = store.claim_recoverable_credential_leases(
        claim_owner="new-cleaner",
        stale_before=now_epoch(),
        claim_expires_at=now_epoch() + 120,
    )
    assert len(new_claims) == 1

    old_hook = _FakeCredentialHook()
    await _release_stored_credential_leases(
        _hook_config(FakeSandboxLauncher(), old_hook),
        store,
        [old_claim],
    )
    assert old_hook.release_calls == []

    new_hook = _FakeCredentialHook()
    await _release_stored_credential_leases(
        _hook_config(FakeSandboxLauncher(), new_hook),
        store,
        new_claims,
    )
    assert len(new_hook.release_calls) == 1
    assert store.list_credential_leases() == []


async def test_restart_recovery_does_not_terminate_newer_sandbox_generation(
    db_uri: str,
) -> None:
    """A stale retiring lease cannot tear down a newer sandbox generation."""
    store = HostStore(db_uri)
    old_launcher = FakeSandboxLauncher(on_host_start=_online_register(store))
    result = await launch_managed_host(
        config=_hook_config(old_launcher, _FakeCredentialHook()),
        owner=_OWNER,
        host_store=store,
    )
    claimed = store.claim_active_credential_leases(
        result.host_id,
        claim_owner="crashed-relaunch",
        claim_expires_at=now_epoch() + 120,
    )
    assert len(claimed) == 1
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()
    store.register_managed_host(
        host_id=result.host_id,
        name=old_launcher.host_starts[0].host_name,
        user_id=_OWNER,
        token="new-generation-token",
        provider="modal",
        sandbox_id="sb-new-generation",
        token_expires_at=now_epoch() + 3600,
    )

    recovery_launcher = FakeSandboxLauncher()
    recovery_hook = _FakeCredentialHook()
    await recover_managed_credential_leases(
        _hook_config(recovery_launcher, recovery_hook),
        HostStore(db_uri),
    )

    assert recovery_launcher.terminated == ["sb-fake-1"]
    recovered_host = store.get_host(result.host_id)
    assert recovered_host is not None
    assert recovered_host.sandbox_id == "sb-new-generation"
    assert store.list_credential_leases(result.host_id) == []
    assert len(recovery_hook.release_calls) == 1


async def test_cleanup_rechecks_claim_after_provider_release(db_uri: str) -> None:
    """A claim takeover during a slow release fences the old worker's tombstone CAS."""
    store = HostStore(db_uri)
    host_id = "112233445566478890abcdef12345678"
    pending = store.record_credential_lease(
        host_id=host_id,
        user_id=_OWNER,
        host_name="managed-claim-handoff",
        sandbox_provider="modal",
        sandbox_id="sb-claim-handoff",
        session_id=None,
        repo_url=None,
        repo_branch=None,
        repo_name=None,
        reference="credential-reference",
        owner_token="launch-owner",
        owner_expires_at=now_epoch() + 120,
    )
    old_claim = store.claim_pending_credential_lease(
        host_id,
        pending.generation,
        owner_token="launch-owner",
        claim_owner="old-cleaner",
        claim_expires_at=now_epoch() + 120,
    )
    assert old_claim is not None

    class _TakeoverDuringReleaseHook(_FakeCredentialHook):
        async def release(
            self,
            context: ManagedCredentialReleaseContext,
            generation: int,
            reference: str | None,
        ) -> None:
            await super().release(context, generation, reference)
            engine = sa.create_engine(db_uri)
            try:
                with engine.begin() as connection:
                    connection.execute(
                        sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
                    )
            finally:
                engine.dispose()
            claims = store.claim_recoverable_credential_leases(
                claim_owner="new-cleaner",
                stale_before=now_epoch(),
                claim_expires_at=now_epoch() + 120,
            )
            assert len(claims) == 1

    hook = _TakeoverDuringReleaseHook()
    await _release_stored_credential_leases(
        _hook_config(FakeSandboxLauncher(), hook),
        store,
        [old_claim],
    )

    remaining = store.list_credential_leases(host_id)
    assert len(remaining) == 1
    assert remaining[0].claim_owner == "new-cleaner"
    assert len(hook.release_calls) == 1


async def test_relaunch_invokes_hook_with_durable_identity(db_uri: str) -> None:
    """
    A relaunch resolves credentials for the SAME durable host identity: the
    context carries the unchanged host id/name/owner and the relaunch's
    session id, so a per-user hook resolves the right owner's credentials for
    the new sandbox generation.
    """
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()
    config = _hook_config(fake, hook)
    first = await launch_managed_host(
        config=config, owner=_OWNER, host_store=host_store, session_id="conv_relaunch"
    )
    gen1 = host_store.get_host(first.host_id)
    assert gen1 is not None

    await relaunch_managed_host(
        config=config, host=gen1, host_store=host_store, session_id="conv_relaunch"
    )

    # Once for the first launch, once for the relaunch.
    assert len(hook.contexts) == 2
    assert hook.generations == [1, 2]
    relaunch_context = hook.contexts[1]
    assert relaunch_context.host_id == first.host_id
    assert relaunch_context.host_name == gen1.name
    assert relaunch_context.owner == _OWNER
    assert relaunch_context.session_id == "conv_relaunch"
    # Generation 1 was durably released before generation 2 became active.
    assert [(generation, reference) for _, generation, reference in hook.release_calls] == [
        (1, "managed-cred-abc123")
    ]
    active = host_store.list_credential_leases(first.host_id)
    assert len(active) == 1
    assert active[0].generation == 2
    assert active[0].state == "active"


async def test_relaunch_defers_until_retiring_predecessor_release_succeeds(
    db_uri: str,
) -> None:
    """A failed predecessor release cannot overlap a successor credential."""

    class _FailOnceReleaseHook(_FakeCredentialHook):
        def __init__(self) -> None:
            super().__init__()
            self.release_attempts = 0

        async def release(
            self,
            context: ManagedCredentialReleaseContext,
            generation: int,
            reference: str | None,
        ) -> None:
            self.release_attempts += 1
            if self.release_attempts == 1:
                raise RuntimeError("temporary credential provider outage")
            await super().release(context, generation, reference)

    store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(store))
    hook = _FailOnceReleaseHook()
    config = _hook_config(launcher, hook)
    result = await launch_managed_host(config=config, owner=_OWNER, host_store=store)
    predecessor = store.get_host(result.host_id)
    assert predecessor is not None

    with pytest.raises(HTTPException) as exc:
        await relaunch_managed_host(config=config, host=predecessor, host_store=store)

    assert exc.value.status_code == 502
    assert "credential cleanup" in exc.value.detail
    assert hook.generations == [1]
    assert launcher.provisioned_names == [predecessor.name]
    remaining = store.list_credential_leases(result.host_id)
    assert [(row.generation, row.state) for row in remaining] == [(1, "retiring")]

    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()

    relaunched = await relaunch_managed_host(
        config=config,
        host=predecessor,
        host_store=HostStore(db_uri),
    )

    assert relaunched.host_id == result.host_id
    assert hook.generations == [1, 2]
    assert [(generation, reference) for _, generation, reference in hook.release_calls] == [
        (1, "managed-cred-abc123")
    ]
    active = store.list_credential_leases(result.host_id)
    assert [(row.generation, row.state) for row in active] == [(2, "active")]


async def test_stale_terminate_does_not_release_newer_generation(db_uri: str) -> None:
    """A delayed teardown is fenced from replacement-generation credentials."""
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()
    config = _hook_config(launcher, hook)
    result = await launch_managed_host(
        config=config,
        owner=_OWNER,
        host_store=host_store,
    )
    stale_host = host_store.get_host(result.host_id)
    assert stale_host is not None

    await relaunch_managed_host(
        config=config,
        host=stale_host,
        host_store=host_store,
    )
    replacement = host_store.get_host(result.host_id)
    assert replacement is not None
    assert replacement.sandbox_id != stale_host.sandbox_id
    hook.release_calls.clear()

    await terminate_managed_host(stale_host, host_store, config)

    assert hook.release_calls == []
    assert host_store.get_host(result.host_id) == replacement
    active = host_store.list_credential_leases(result.host_id)
    assert [(row.state, row.sandbox_id) for row in active] == [("active", replacement.sandbox_id)]


async def test_relaunch_cannot_resurrect_host_deleted_during_provision(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predecessor CAS tears down a new sandbox after concurrent host deletion."""
    entered = threading.Event()
    gate = threading.Event()
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()
    config = _hook_config(launcher, hook)
    result = await launch_managed_host(
        config=config,
        owner=_OWNER,
        host_store=host_store,
    )
    predecessor = host_store.get_host(result.host_id)
    assert predecessor is not None
    original_provision = launcher.provision

    def blocking_provision(name: str) -> str:
        entered.set()
        assert gate.wait(timeout=5), "test never released the provision gate"
        return original_provision(name)

    monkeypatch.setattr(launcher, "provision", blocking_provision)
    relaunch = asyncio.create_task(
        relaunch_managed_host(
            config=config,
            host=predecessor,
            host_store=host_store,
        )
    )
    assert await asyncio.to_thread(entered.wait, 2)

    await terminate_managed_host(predecessor, host_store, config)
    assert host_store.get_host(result.host_id) is None
    gate.set()

    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(relaunch, timeout=2)
    assert exc.value.status_code == 502
    assert "changed during relaunch" in exc.value.detail
    assert launcher.terminated[-1] == "sb-fake-2"
    assert host_store.get_host(result.host_id) is None
    assert host_store.list_credential_leases(result.host_id) == []


async def test_launch_owner_renews_while_credential_acquire_is_blocked(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live producer remains fenced in during a long credential acquisition."""
    import omnigent.server.managed_hosts as managed_hosts_mod

    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingHook(_FakeCredentialHook):
        async def acquire(
            self,
            context: ManagedLaunchContext,
            generation: int,
        ) -> ManagedCredentialLease:
            self.contexts.append(context)
            self.generations.append(generation)
            entered.set()
            await release.wait()
            return self.lease

    monkeypatch.setattr(managed_hosts_mod, "_CREDENTIAL_PENDING_STALE_S", 3)
    monkeypatch.setattr(managed_hosts_mod, "_CREDENTIAL_OWNER_RENEW_INTERVAL_S", 0.1)
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _BlockingHook()
    launch = asyncio.create_task(
        launch_managed_host(
            config=_hook_config(launcher, hook),
            owner=_OWNER,
            host_store=host_store,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    initial = host_store.list_credential_leases()[0]

    # A second replica starting recovery while this launch owns the pending row
    # must not terminate the in-flight sandbox.
    await recover_managed_credential_leases(_hook_config(launcher, hook), HostStore(db_uri))
    pending = host_store.list_credential_leases()[0]
    assert pending.state == "pending"
    assert launcher.terminated == []

    for _ in range(100):
        renewed = host_store.list_credential_leases()[0]
        if renewed.owner_expires_at > initial.owner_expires_at:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("launch owner heartbeat did not renew its lease")
    assert renewed.owner_expires_at > initial.owner_expires_at
    await recover_managed_credential_leases(_hook_config(launcher, hook), host_store)
    assert launcher.terminated == []
    release.set()
    await asyncio.wait_for(launch, timeout=2)


async def test_cancelled_provision_is_armed_before_cleanup(db_uri: str) -> None:
    """Cancellation waits for provision to return, then durably tears it down."""
    gate = threading.Event()
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(provision_gate=gate)
    launch = asyncio.create_task(
        launch_managed_host(
            config=_hook_config(launcher, None),
            owner=_OWNER,
            host_store=host_store,
        )
    )
    await asyncio.sleep(0.05)
    launch.cancel()
    await asyncio.sleep(0.05)
    assert not launch.done()
    assert launcher.terminated == []

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(launch, timeout=2)

    assert launcher.provisioned_names
    assert launcher.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []


async def test_no_hook_launch_timeout_retains_retryable_sandbox_identity(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out first-launch teardown remains durably recoverable without a hook."""
    import omnigent.server.managed_hosts as managed_hosts_mod

    gate = threading.Event()
    entered = threading.Event()
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(fail_on_host_start=True)

    def blocking_terminate(sandbox_id: str) -> None:
        entered.set()
        assert gate.wait(timeout=5), "test never released the terminate gate"
        launcher.terminated.append(sandbox_id)

    monkeypatch.setattr(managed_hosts_mod, "_CREDENTIAL_CLEANUP_TIMEOUT_S", 0.01)
    monkeypatch.setattr(launcher, "terminate", blocking_terminate)

    with pytest.raises(HTTPException):
        await launch_managed_host(
            config=_hook_config(launcher, None),
            owner=_OWNER,
            host_store=host_store,
        )

    assert await asyncio.to_thread(entered.wait, 1)
    records = host_store.list_credential_leases()
    assert len(records) == 1
    assert records[0].state == "retiring"
    assert records[0].sandbox_id == "sb-fake-1"
    assert records[0].credential_cleanup_required is False

    gate.set()
    await asyncio.sleep(0.05)
    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()

    recovery_launcher = FakeSandboxLauncher()
    await recover_managed_credential_leases(
        _hook_config(recovery_launcher, None),
        HostStore(db_uri),
    )

    assert recovery_launcher.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []


async def test_hung_cleanup_workers_do_not_exhaust_startup_or_db_executor(
    db_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed-out daemon workers stay hard-bounded and never block startup."""
    import omnigent.server.managed_hosts as managed_hosts_mod

    executor = managed_hosts_mod._BoundedProviderCleanupExecutor(max_workers=2)
    monkeypatch.setattr(managed_hosts_mod, "_PROVIDER_CLEANUP_EXECUTOR", executor)
    monkeypatch.setattr(managed_hosts_mod, "_CREDENTIAL_CLEANUP_TIMEOUT_S", 0.05)
    gates = [threading.Event(), threading.Event(), threading.Event()]
    entered = [threading.Event(), threading.Event(), threading.Event()]
    launcher = FakeSandboxLauncher()

    def blocking_terminate(sandbox_id: str) -> None:
        index = int(sandbox_id.rsplit("-", 1)[1])
        entered[index].set()
        assert gates[index].wait(timeout=5), "test never released the terminate gate"

    monkeypatch.setattr(launcher, "terminate", blocking_terminate)
    try:
        results = await asyncio.gather(
            *(
                managed_hosts_mod._terminate_sandbox_id_best_effort(
                    launcher,
                    sandbox_id=f"sb-hung-{index}",
                    provider=launcher.provider,
                    host_id=f"host-{index}",
                )
                for index in range(2)
            )
        )
        assert results == [False, False]
        assert all(event.is_set() for event in entered[:2])
        assert not await managed_hosts_mod._terminate_sandbox_id_best_effort(
            launcher,
            sandbox_id="sb-hung-2",
            provider=launcher.provider,
            host_id="host-2",
        )
        assert not entered[2].is_set()
        # A timed-out call keeps its slot until the provider returns; otherwise
        # repeated recovery ticks could create unbounded abandoned threads.
        gates[0].set()
        for _ in range(100):
            await managed_hosts_mod._terminate_sandbox_id_best_effort(
                launcher,
                sandbox_id="sb-hung-2",
                provider=launcher.provider,
                host_id="host-2",
            )
            if entered[2].is_set():
                break
            await asyncio.sleep(0.01)
        assert entered[2].is_set()
        cleanup_threads = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("managed-host-cleanup-")
        ]
        assert cleanup_threads
        assert all(thread.daemon for thread in cleanup_threads)
        await asyncio.wait_for(asyncio.to_thread(executor.shutdown), timeout=0.5)

        store = HostStore(db_uri)
        app = _capability_probe_app(db_uri, tmp_path, _hook_config(launcher, None))
        monkeypatch.setattr(_globals, "_terminal_registry", TerminalRegistry())

        async def probe_startup() -> None:
            async with app.router.lifespan_context(app):
                assert (
                    await asyncio.wait_for(
                        asyncio.to_thread(store.list_hosts, _OWNER),
                        timeout=0.5,
                    )
                    == []
                )

        await asyncio.wait_for(probe_startup(), timeout=2)
    finally:
        for gate in gates:
            gate.set()
        await asyncio.to_thread(executor.shutdown)


async def test_hung_cleanup_worker_cannot_block_interpreter_shutdown() -> None:
    """A provider call that never returns cannot keep the server process alive."""
    script = """
import threading
from omnigent.server.managed_hosts import _BoundedProviderCleanupExecutor

executor = _BoundedProviderCleanupExecutor(max_workers=1)
entered = threading.Event()

def hang_forever():
    entered.set()
    threading.Event().wait()

attempt = executor.submit(hang_forever)
assert attempt is not None
assert entered.wait(timeout=1)
"""
    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )


async def test_cancelled_start_waits_for_provider_worker_before_cleanup(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late start_host worker cannot recreate resources after teardown."""
    entered = threading.Event()
    gate = threading.Event()
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    original_start = launcher.start_host

    def blocking_start(*args: Any, **kwargs: Any) -> str:
        entered.set()
        assert gate.wait(timeout=5), "test never released the start gate"
        return original_start(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(launcher, "start_host", blocking_start)
    launch = asyncio.create_task(
        launch_managed_host(
            config=_hook_config(launcher, _FakeCredentialHook()),
            owner=_OWNER,
            host_store=host_store,
        )
    )
    assert await asyncio.to_thread(entered.wait, 2)
    launch.cancel()
    await asyncio.sleep(0.05)
    assert launcher.terminated == []

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(launch, timeout=2)

    assert launcher.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []


async def test_failed_sandbox_termination_remains_recoverable(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential release does not erase the retry marker for a live sandbox."""
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()
    config = _hook_config(launcher, hook)
    result = await launch_managed_host(
        config=config,
        owner=_OWNER,
        host_store=host_store,
    )
    host = host_store.get_host(result.host_id)
    assert host is not None

    def fail_termination(_sandbox_id: str) -> None:
        raise RuntimeError("provider control plane unavailable")

    monkeypatch.setattr(launcher, "terminate", fail_termination)
    await terminate_managed_host(host, host_store, config)

    assert host_store.get_host(result.host_id) is not None
    assert hook.release_calls == []
    assert [row.state for row in host_store.list_credential_leases(result.host_id)] == ["retiring"]

    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()
    recovery_launcher = FakeSandboxLauncher()
    await recover_managed_credential_leases(
        _hook_config(recovery_launcher, hook),
        HostStore(db_uri),
    )

    assert recovery_launcher.terminated == [host.sandbox_id]
    assert len(hook.release_calls) == 1
    assert host_store.get_host(result.host_id) is None
    assert host_store.list_credential_leases(result.host_id) == []


async def test_hook_failure_aborts_launch_without_exposing_secret(
    db_uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A hook that cannot resolve credentials aborts the launch through the same
    cleanup any post-provision failure takes: the sandbox is terminated, the
    pre-registered host row (and its armed token) is deleted, and a 502 is
    surfaced. The host never starts — the credentials were never in place.
    """
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    secret = "credential-provider-error-secret"

    class _SecretFailingHook(_FakeCredentialHook):
        async def acquire(
            self,
            context: ManagedLaunchContext,
            generation: int,
        ) -> ManagedCredentialLease:
            raise RuntimeError(f"credential resolution failed: {secret}")

    hook = _SecretFailingHook()

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_hook_config(fake, hook),
            owner=_OWNER,
            host_store=host_store,
        )

    assert exc.value.status_code == 502
    assert "managed credential acquisition failed" in exc.value.detail
    assert secret not in exc.value.detail
    assert secret not in caplog.text
    # Provisioned, then torn down: no paid compute leaks past the failed acquire.
    assert fake.terminated == ["sb-fake-1"]
    # The host never started (acquire runs before start_host).
    assert fake.host_starts == []
    assert host_store.list_hosts(_OWNER) == []


async def test_hook_failure_after_provider_create_leaves_zero_orphans(db_uri: str) -> None:
    """Durable cleanup closes the provider-create-before-reference crash window."""
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    provider = _FakeKubernetesCredentialProvider()
    hook = _FakeProviderCredentialHook(provider, fail_after_create=True)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_hook_config(fake, hook),
            owner=_OWNER,
            host_store=host_store,
        )

    assert exc.value.status_code == 502
    assert provider.names == set()
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_credential_leases() == []
    assert len(hook.release_calls) == 1
    assert hook.release_calls[0][1:] == (1, None)


async def test_hung_credential_acquisition_times_out_and_cleans_launch(db_uri: str) -> None:
    """A cancellation-safe hung hook cannot heartbeat a pending lease forever."""

    class _HangingHook(_FakeCredentialHook):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = False

        async def acquire(
            self,
            context: ManagedLaunchContext,
            generation: int,
        ) -> ManagedCredentialLease:
            self.contexts.append(context)
            self.generations.append(generation)
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            raise AssertionError("unreachable")

    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _HangingHook()

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_hook_config(
                launcher,
                hook,
                acquisition_timeout_s=0.05,
            ),
            owner=_OWNER,
            host_store=host_store,
        )

    assert hook.started.is_set()
    assert hook.cancelled
    assert exc.value.status_code == 502
    assert (
        exc.value.detail
        == "managed sandbox host startup failed: managed credential acquisition failed"
    )
    assert launcher.host_starts == []
    assert launcher.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []
    assert len(hook.release_calls) == 1
    assert hook.release_calls[0][2] is None


async def test_start_host_failure_after_acquire_releases_lease(db_uri: str) -> None:
    """
    When the launch fails AFTER the lease was acquired (the in-sandbox host
    start errors), this layer releases the live lease exactly once, then tears
    the sandbox down, deletes the row, and leaves no recoverable lease behind.
    """
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(fail_on_host_start=True)
    lease = _FakeCredentialLease()
    hook = _FakeCredentialHook(lease)

    with pytest.raises(HTTPException) as exc:
        await launch_managed_host(
            config=_hook_config(fake, hook),
            owner=_OWNER,
            host_store=host_store,
        )

    assert exc.value.status_code == 502
    assert lease.release_calls == 1
    assert fake.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []


async def test_launch_failure_defers_credential_release_until_sandbox_termination(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed provider teardown leaves credentials live and retryable."""
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(fail_on_host_start=True)
    terminate = launcher.terminate
    lease = _FakeCredentialLease()
    hook = _FakeCredentialHook(lease)
    config = _hook_config(launcher, hook)

    provider_secret = "provider-control-plane-secret"

    def fail_termination(_sandbox_id: str) -> None:
        raise RuntimeError(f"provider control plane unavailable: {provider_secret}")

    monkeypatch.setattr(launcher, "terminate", fail_termination)
    with pytest.raises(HTTPException):
        await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)

    assert lease.release_calls == 0
    assert hook.release_calls == []
    assert provider_secret not in caplog.text
    records = host_store.list_credential_leases()
    assert [(record.state, record.reference) for record in records] == [
        ("retiring", "managed-cred-abc123")
    ]

    engine = sa.create_engine(db_uri)
    try:
        with engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE managed_credential_leases SET claim_expires_at = 0")
            )
    finally:
        engine.dispose()
    monkeypatch.setattr(launcher, "terminate", terminate)

    await recover_managed_credential_leases(config, HostStore(db_uri))

    assert launcher.terminated == ["sb-fake-1"]
    assert len(hook.release_calls) == 1
    assert host_store.list_credential_leases() == []


async def test_relaunch_start_host_failure_releases_lease_keeps_row(db_uri: str) -> None:
    """
    A relaunch whose host start fails after acquiring the new generation's
    lease releases that lease, but keeps the durable host row (only the new
    sandbox is torn down and its token revoked) — the session stays
    relaunchable.
    """
    host_store = HostStore(db_uri)
    fake = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    hook = _FakeCredentialHook()
    config = _hook_config(fake, hook)
    first = await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)
    gen1 = host_store.get_host(first.host_id)
    assert gen1 is not None

    # Swap in a fresh lease for the relaunch so its release is unambiguous.
    relaunch_lease = _FakeCredentialLease(reference="managed-cred-gen2")
    hook_gen2 = _FakeCredentialHook(relaunch_lease)
    config_gen2 = _hook_config(fake, hook_gen2)
    fake.fail_on_host_start = True

    with pytest.raises(HTTPException) as exc:
        await relaunch_managed_host(config=config_gen2, host=gen1, host_store=host_store)

    assert exc.value.status_code == 502
    assert relaunch_lease.release_calls == 1
    # Durable row survives the failed relaunch.
    assert host_store.get_host(first.host_id) is not None
    assert host_store.list_credential_leases(first.host_id) == []


async def test_cancellation_settles_late_registration_commit(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during registration waits for its commit before teardown."""
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher()
    hook = _FakeCredentialHook()
    entered = threading.Event()
    unblock = threading.Event()
    original = host_store.register_managed_host

    def blocking_register(*args: Any, **kwargs: Any) -> object:
        entered.set()
        assert unblock.wait(timeout=2.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(host_store, "register_managed_host", blocking_register)
    task = asyncio.create_task(
        launch_managed_host(
            config=_hook_config(launcher, hook),
            owner=_OWNER,
            host_store=host_store,
        )
    )
    assert await asyncio.to_thread(entered.wait, 2.0)
    task.cancel()
    unblock.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert launcher.terminated == ["sb-fake-1"]
    assert len(hook.release_calls) == 1
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []


async def test_cancellation_settles_late_activation_commit(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during activation cleans the generation after its late commit."""
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher(on_host_start=_online_register(host_store))
    lease = _FakeCredentialLease()
    entered = threading.Event()
    unblock = threading.Event()
    original = host_store.activate_credential_lease

    def blocking_activate(*args: Any, **kwargs: Any) -> object:
        entered.set()
        assert unblock.wait(timeout=2.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(host_store, "activate_credential_lease", blocking_activate)
    task = asyncio.create_task(
        launch_managed_host(
            config=_hook_config(launcher, _FakeCredentialHook(lease)),
            owner=_OWNER,
            host_store=host_store,
        )
    )
    assert await asyncio.to_thread(entered.wait, 2.0)
    task.cancel()
    unblock.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert launcher.terminated == ["sb-fake-1"]
    assert lease.release_calls == 1
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []


async def test_credential_acquisition_timeout_releases_deterministic_resource(
    db_uri: str,
) -> None:
    """A cancelled acquire is bounded and cleanup reconstructs its provider identity."""
    host_store = HostStore(db_uri)
    launcher = FakeSandboxLauncher()
    provider = _FakeKubernetesCredentialProvider()

    class _SlowAcquireHook(_FakeProviderCredentialHook):
        async def acquire(
            self,
            context: ManagedLaunchContext,
            generation: int,
        ) -> ManagedCredentialLease:
            self.provider.create(context, generation)
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    hook = _SlowAcquireHook(provider)
    config = dataclasses.replace(
        _hook_config(launcher, hook),
        credential_acquisition_timeout_s=0.01,
    )

    with pytest.raises(HTTPException) as exc_info:
        await launch_managed_host(config=config, owner=_OWNER, host_store=host_store)

    assert exc_info.value.status_code == 502
    assert provider.names == set()
    assert len(hook.release_calls) == 1
    assert launcher.terminated == ["sb-fake-1"]
    assert host_store.list_hosts(_OWNER) == []
    assert host_store.list_credential_leases() == []
