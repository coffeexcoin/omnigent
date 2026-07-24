"""Executables used by the runner's process-local Git/GitHub wrappers."""

from __future__ import annotations

import base64
import json
import os
import socket
import sys
from collections.abc import Mapping, Sequence

from omnigent.runner.credential_broker import (
    BROKER_CAPABILITY_ENV,
    BROKER_ENDPOINT_ENV,
)

_MAX_BROKER_MESSAGE = 3 * 1024 * 1024


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
    if len(encoded) > _MAX_BROKER_MESSAGE:
        raise RuntimeError("credential broker request is too large")

    with socket.create_connection((host, port), timeout=10.0) as connection:
        connection.sendall(encoded)
        stream = connection.makefile("rb")
        raw = stream.readline(_MAX_BROKER_MESSAGE + 1)
    if not raw or len(raw) > _MAX_BROKER_MESSAGE or not raw.endswith(b"\n"):
        raise RuntimeError("credential broker returned an invalid response")
    response = json.loads(raw)
    if not isinstance(response, dict) or response.get("ok") is not True:
        detail = response.get("error") if isinstance(response, dict) else None
        raise RuntimeError(f"credential broker request failed: {detail or 'denied'}")
    return response


def _read_stdin() -> bytes:
    """Read bounded piped input without blocking interactive invocations."""

    if sys.stdin.isatty():
        return b""
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    try:
        value = stream.read(1024 * 1024 + 1)
    except OSError:
        return b""
    data = value.encode() if isinstance(value, str) else value
    if len(data) > 1024 * 1024:
        raise RuntimeError("credential broker stdin is too large")
    return data


def _write_stream(name: str, encoded: object) -> None:
    if not isinstance(encoded, str):
        raise RuntimeError(f"credential broker returned invalid {name}")
    data = base64.b64decode(encoded, validate=True)
    stream = getattr(getattr(sys, name), "buffer", getattr(sys, name))
    try:
        stream.write(data)
    except TypeError:
        stream.write(data.decode(errors="replace"))
    stream.flush()


def _execute(tool: str, argv: Sequence[str]) -> int:
    payload: dict[str, object] = {
        "tool": tool,
        "operation": "execute",
        "argv": list(argv),
        "cwd": os.getcwd(),
        "stdin": base64.b64encode(_read_stdin()).decode("ascii"),
    }
    if tool == "gh":
        payload["host"] = "github.com"
    response = _broker_request(payload)
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("credential broker returned no execution result")
    _write_stream("stdout", result.get("stdout"))
    _write_stream("stderr", result.get("stderr"))
    returncode = result.get("returncode")
    if not isinstance(returncode, int):
        raise RuntimeError("credential broker returned invalid exit status")
    return returncode


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
    """Ask the broker to execute bounded Git without returning credentials."""

    return _execute("git", argv)


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
    """Reject the former raw-credential helper boundary."""

    del operation
    raise PermissionError("raw Git credential requests are unavailable")


def run_gh(argv: Sequence[str]) -> int:
    """Ask the broker to execute bounded gh without returning its token."""

    return _execute("gh", argv)


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
