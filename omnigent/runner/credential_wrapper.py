"""Executables used by the runner's process-local Git/GitHub wrappers."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from omnigent.runner.credential_broker import (
    BROKER_CAPABILITY_ENV,
    BROKER_ENDPOINT_ENV,
    REAL_GH_ENV,
    REAL_GIT_ENV,
)


def _broker_request(payload: Mapping[str, object]) -> dict[str, object]:
    endpoint = os.environ.get(BROKER_ENDPOINT_ENV, "")
    capability = os.environ.get(BROKER_CAPABILITY_ENV, "")
    try:
        host, raw_port = endpoint.rsplit(":", 1)
        port = int(raw_port)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("credential broker endpoint is unavailable") from exc
    body = dict(payload)
    body["capability"] = capability
    encoded = json.dumps(body, separators=(",", ":")).encode() + b"\n"
    if len(encoded) > 64 * 1024:
        raise RuntimeError("credential broker request is too large")

    with socket.create_connection((host, port), timeout=10.0) as connection:
        connection.sendall(encoded)
        stream = connection.makefile("rb")
        raw = stream.readline(64 * 1024 + 1)
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise RuntimeError("credential broker returned an invalid response")
    response = json.loads(raw)
    if not isinstance(response, dict) or response.get("ok") is not True:
        detail = response.get("error") if isinstance(response, dict) else None
        raise RuntimeError(str(detail or "credential broker denied the request"))
    grant = response.get("grant")
    if not isinstance(grant, dict):
        raise RuntimeError("credential broker returned no grant")
    return grant


def _gh_action(argv: Sequence[str]) -> str:
    """Return the gh subcommand without mistaking a global option value for it."""

    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--":
            index += 1
            break
        if not item.startswith("-"):
            return item[:128]
        index += 1
        if item in ("-R", "--repo") and index < len(argv):
            index += 1
    return argv[index][:128] if index < len(argv) else "unknown"


def _split_git_global_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split options accepted before Git's subcommand from the remaining argv."""

    options_with_value = {
        "-C",
        "-c",
        "--config-env",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--super-prefix",
        "--work-tree",
    }
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--":
            index += 1
            break
        if not item.startswith("-"):
            break
        index += 1
        if item in options_with_value and index < len(argv):
            index += 1
    return list(argv[:index]), list(argv[index:])


def _git_action(argv: Sequence[str]) -> str:
    _, command_args = _split_git_global_args(argv)
    return command_args[0][:128] if command_args else "unknown"


def run_git(argv: Sequence[str]) -> int:
    """Execute real Git with brokered identity and a process-local helper."""

    real_git = os.environ.get(REAL_GIT_ENV)
    if not real_git:
        raise RuntimeError("real git executable is unavailable")
    grant = _broker_request(
        {
            "tool": "git",
            "operation": "identity",
            "action": _git_action(argv),
        }
    )
    name = grant.get("git_user_name")
    email = grant.get("git_user_email")
    if not isinstance(name, str) or not name or not isinstance(email, str) or not email:
        raise RuntimeError("credential broker returned no Git commit identity")

    helper = f"!{shlex.quote(sys.executable)} -m omnigent.runner.credential_wrapper git-credential"
    deny_ssh = f"{shlex.quote(sys.executable)} -m omnigent.runner.credential_wrapper deny-ssh"
    with tempfile.TemporaryDirectory(prefix="omnigent-git-config-") as config_dir:
        global_config = Path(config_dir) / "config"
        global_config.touch(mode=0o600)
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = str(global_config)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.pop("GIT_ASKPASS", None)
        env.pop("SSH_ASKPASS", None)
        global_args, command_args = _split_git_global_args(argv)
        command = [
            real_git,
            *global_args,
            "-c",
            f"user.name={name}",
            "-c",
            f"user.email={email}",
            "-c",
            "credential.helper=",
            "-c",
            f"credential.helper={helper}",
            "-c",
            "credential.useHttpPath=true",
            "-c",
            f"core.sshCommand={deny_ssh}",
            *command_args,
        ]
        return subprocess.run(command, env=env, check=False).returncode


def _read_git_credential_input() -> dict[str, str]:
    fields: dict[str, str] = {}
    total = 0
    for line in sys.stdin:
        total += len(line)
        if total > 64 * 1024:
            raise RuntimeError("git credential input is too large")
        stripped = line.rstrip("\r\n")
        if not stripped:
            break
        key, separator, value = stripped.partition("=")
        if separator and key != "password":
            fields[key] = value
    return fields


def run_git_credential(operation: str) -> int:
    """Implement Git's credential-helper protocol without persisting secrets."""

    if operation not in ("get", "store", "erase"):
        raise RuntimeError(f"unsupported git credential operation: {operation}")
    fields = _read_git_credential_input()
    grant = _broker_request(
        {
            "tool": "git",
            "operation": "credential",
            "action": operation,
            "protocol": fields.get("protocol"),
            "host": fields.get("host"),
            "path": fields.get("path"),
        }
    )
    if operation != "get":
        return 0
    username = grant.get("username")
    secret = grant.get("secret")
    if not isinstance(username, str) or not username or not isinstance(secret, str) or not secret:
        raise RuntimeError("credential broker returned an incomplete Git credential")
    sys.stdout.write(f"username={username}\npassword={secret}\n\n")
    return 0


def run_gh(argv: Sequence[str]) -> int:
    """Execute real gh with a per-invocation token and isolated config dir."""

    real_gh = os.environ.get(REAL_GH_ENV)
    if not real_gh:
        raise RuntimeError("real gh executable is unavailable")
    host = os.environ.get("GH_HOST", "github.com")
    grant = _broker_request(
        {
            "tool": "gh",
            "operation": "credential",
            "action": _gh_action(argv),
            "host": host,
        }
    )
    secret = grant.get("secret")
    if not isinstance(secret, str) or not secret:
        raise RuntimeError("credential broker returned no GitHub token")

    with tempfile.TemporaryDirectory(prefix="omnigent-gh-config-") as config_dir:
        env = dict(os.environ)
        env.pop("GITHUB_TOKEN", None)
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_ENTERPRISE_TOKEN", None)
        env.pop("GH_ENTERPRISE_TOKEN", None)
        token_env = "GH_TOKEN" if host == "github.com" else "GH_ENTERPRISE_TOKEN"
        env[token_env] = secret
        env["GH_CONFIG_DIR"] = config_dir
        return subprocess.run([real_gh, *argv], env=env, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise RuntimeError("credential wrapper mode is required")
    mode = args.pop(0)
    if mode == "git":
        return run_git(args)
    if mode == "gh":
        return run_gh(args)
    if mode == "git-credential":
        if len(args) != 1:
            raise RuntimeError("git credential operation is required")
        return run_git_credential(args[0])
    if mode == "deny-ssh":
        raise RuntimeError("brokered Git credentials require an HTTPS remote")
    raise RuntimeError(f"unknown credential wrapper mode: {mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"omnigent credential wrapper: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
