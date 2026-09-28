#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
gds-diag: filesystem-aware GPUDirect Storage diagnostic toolkit.

Each operating mode is a subcommand. Top-level CLI uses only the Python
standard library so `--help` works in environments where third-party
packages cannot be installed.

Run `gds-diag.py --help` for the list of subcommands, or
`gds-diag.py <command> --help` for details on a specific subcommand.
"""
from __future__ import annotations

import argparse
import sys

from checks.version import version_string
from subcommands import discover
from subcommands._base import add_common_flags


_TOP_DESCRIPTION_TEMPLATE = """\
gds-diag — GPUDirect Storage diagnostic toolkit.

Subcommands:
{subcommands}

Run `gds-diag.py <command> --help` for details on a specific subcommand.

Common flags (available on every subcommand):
  --json           machine-readable JSON output
  -v / --verbose   show passing checks, not just failures

Exit codes:
  0  all checks pass
  1  at least one FAIL
  2  environment not ready (e.g. CUDA missing when required)
  3  bad arguments / usage error
"""


def _build_parser() -> argparse.ArgumentParser:
    commands = discover()

    # Build a fixed-column subcommand table. We own the formatting so names
    # always appear inline regardless of their length.
    col = max(len(cmd.name) for cmd in commands) + 2
    table = "\n".join(f"  {cmd.name:<{col}}{cmd.help}" for cmd in commands)
    description = _TOP_DESCRIPTION_TEMPLATE.format(subcommands=table)

    parser = argparse.ArgumentParser(
        prog="gds-diag.py",
        usage="%(prog)s <command> [options]",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=version_string(),
        help="show gds-diag version and Git commit, then exit",
    )
    subparsers = parser.add_subparsers(
        dest="_command",
        metavar="<command>",
        required=True,
    )
    subparsers.help = argparse.SUPPRESS

    for cmd in commands:
        sub = subparsers.add_parser(
            cmd.name,
            # Omit help= so argparse creates no pseudo-action for this entry;
            # our manually formatted table in the description is canonical.
            description=cmd.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_common_flags(sub)
        cmd.add_arguments(sub)
        sub.set_defaults(_cmd=cmd)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse uses exit code 2 for usage errors; remap to 3 per doc/DESIGN.md
        if e.code == 2:
            return 3
        return int(e.code) if e.code is not None else 0
    return int(args._cmd.run(args))


if __name__ == "__main__":
    sys.exit(main())
