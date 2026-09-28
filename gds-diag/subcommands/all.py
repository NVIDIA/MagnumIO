# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""all meta subcommand."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from checks import container
from checks.output import bold
from checks.version import tool_metadata
from ._base import Subcommand


_DESCRIPTION = """\
Run the recommended general GDS diagnostic sequence.

This is a good starting point when the problem is broad or unclear. It detects
whether gds-diag is running on the host or inside a container, runs the
appropriate subcommands in order, and stops at the first unsuccessful return
code. Use the narrower subcommands directly when you have a more specific
diagnostic goal.
"""


@dataclass(frozen=True)
class Step:
    command: str
    args: argparse.Namespace
    display: str


def _command_by_name(name: str) -> Subcommand:
    module = import_module(f"subcommands.{name.replace('-', '_')}")
    return module.COMMAND


def _path(args: argparse.Namespace) -> str:
    return args.path or "."


def _detect_environment() -> container.ContainerContext:
    return container.detect_container_context()


def _step(command: str, display: str, **kwargs: Any) -> Step:
    defaults: dict[str, Any] = {"json": False, "verbose": False}
    defaults.update(kwargs)
    return Step(command, argparse.Namespace(**defaults), display)


def _planned_steps(args: argparse.Namespace, context: container.ContainerContext) -> list[Step]:
    path = _path(args)
    common = {"json": args.json, "verbose": args.verbose}
    if context.in_container:
        return [
            _step("container-check", "container-check", runtime="auto", **common),
            _step("mount-check", f"mount-check {path}", path=path, **common),
        ]
    return [
        _step("pre-install", "pre-install", **common),
        _step("post-install", "post-install", **common),
        _step(
            "config-audit",
            "config-audit",
            profile=None,
            config=None,
            ignore_env=False,
            **common,
        ),
        _step("mount-check", f"mount-check {path}", path=path, **common),
    ]


def _environment_name(context: container.ContainerContext) -> str:
    return context.runtime if context.in_container else "host"


def _print_plan(context: container.ContainerContext, path: str, steps: list[Step]) -> None:
    print()
    print(bold("═" * 70))
    print(bold("  GDS General Diagnostic"))
    print(bold("═" * 70))
    print(f"  Environment : {_environment_name(context)}")
    if context.evidence:
        print(f"  Detection   : {context.evidence.splitlines()[0]}")
    print(f"  Path        : {path}")
    print("  Sequence    :")
    for idx, step in enumerate(steps, start=1):
        print(f"    {idx}. {step.display}")
    print("  Stops at first unsuccessful return code.")
    print()


def _run_human(args: argparse.Namespace, context: container.ContainerContext, steps: list[Step]) -> int:
    _print_plan(context, _path(args), steps)
    for step in steps:
        fill = "─" * max(0, 62 - len(step.display))
        print(bold(f"  ── {step.display} {fill}"))
        rc = int(_command_by_name(step.command).run(step.args))
        if rc != 0:
            print()
            print(f"Stopped after {step.command} returned {rc}.")
            return rc
    print()
    print("All selected diagnostics completed successfully.")
    return 0


def _parse_child_json(stdout: str) -> tuple[Any | None, str | None]:
    text = stdout.strip()
    if not text:
        return None, "child command produced no JSON output"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _run_json(args: argparse.Namespace, context: container.ContainerContext, steps: list[Step]) -> int:
    records: list[dict[str, Any]] = []
    exit_code = 0
    stopped_at: str | None = None

    for step in steps:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            rc = int(_command_by_name(step.command).run(step.args))
        stdout = stream.getvalue()
        result, parse_error = _parse_child_json(stdout)
        record: dict[str, Any] = {
            "command": step.command,
            "display": step.display,
            "exit_code": rc,
            "result": result,
        }
        if parse_error:
            record["parse_error"] = parse_error
            record["raw_stdout"] = stdout
        records.append(record)
        if rc != 0:
            exit_code = rc
            stopped_at = step.command
            break

    payload = {
        "tool": tool_metadata(),
        "mode": "all",
        "environment": {
            "kind": _environment_name(context),
            "in_container": context.in_container,
            "runtime": context.runtime,
            "detection": context.evidence,
        },
        "path": _path(args),
        "planned_sequence": [step.command for step in steps],
        "stopped_at": stopped_at,
        "exit_code": exit_code,
        "steps": records,
    }
    print(json.dumps(payload, indent=2))
    return exit_code


class AllCommand(Subcommand):
    name = "all"
    help = "run the recommended diagnostic sequence for a mount/directory"
    description = _DESCRIPTION
    order = 0

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "path",
            nargs="?",
            default=".",
            help="filesystem path for the final mount-check; defaults to the current directory",
        )

    def run(self, args: argparse.Namespace) -> int:
        context = _detect_environment()
        steps = _planned_steps(args, context)
        return _run_json(args, context, steps) if args.json else _run_human(args, context, steps)


COMMAND = AllCommand()
