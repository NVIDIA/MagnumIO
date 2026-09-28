# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subcommand contract for gds-diag.

Each subcommand lives in its own module under `subcommands/` and exposes a
module-level `COMMAND` attribute pointing at a `Subcommand` instance. The
top-level dispatcher discovers these automatically — adding a new subcommand
is a matter of dropping in one new file.
"""
from __future__ import annotations

import argparse


class Subcommand:
    """Base class for a single gds-diag subcommand.

    Subclasses must set `name`, `help`, and `description`, and implement
    `add_arguments` and `run`.
    """

    name: str = ""
    help: str = ""
    description: str = ""
    order: int = 100  # lower sorts earlier in --help; ties broken by name

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add subcommand-specific arguments. Common flags are added by the
        dispatcher; do not redefine `--json` or `-v/--verbose` here."""
        return None

    def run(self, args: argparse.Namespace) -> int:
        """Execute the subcommand and return a process exit code.

        Exit codes follow doc/DESIGN.md:
            0 - all checks pass
            1 - at least one FAIL
            2 - environment not ready (e.g. CUDA missing when required)
            3 - bad arguments / usage error
        """
        raise NotImplementedError


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand."""
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of coloured text",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show passing checks in addition to failures",
    )


def stub(name: str) -> int:
    """Placeholder body for not-yet-implemented subcommands."""
    print(f"{name}: not implemented yet")
    return 0
