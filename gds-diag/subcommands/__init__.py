# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subcommand discovery for gds-diag.

Modules in this package whose names do not start with `_` are imported and
expected to expose a module-level `COMMAND` attribute (a `Subcommand`
instance). `discover()` returns the list of those instances, sorted by
their CLI `name`.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from typing import List

from ._base import Subcommand


def discover(verbose: bool = False) -> List[Subcommand]:
    """Import every non-private module in this package and collect their
    `COMMAND` attributes.

    A subcommand module that fails to import is reported on stderr but does
    not abort discovery — the rest of the CLI remains usable.
    """
    commands: List[Subcommand] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        modname = f"{__name__}.{info.name}"
        try:
            mod = importlib.import_module(modname)
        except Exception:
            print(f"warning: failed to load subcommand module {modname}",
                  file=sys.stderr)
            if verbose:
                traceback.print_exc()
            continue
        cmd = getattr(mod, "COMMAND", None)
        if not isinstance(cmd, Subcommand):
            print(f"warning: {modname} does not export a COMMAND "
                  f"Subcommand instance", file=sys.stderr)
            continue
        commands.append(cmd)
    commands.sort(key=lambda c: (c.order, c.name))
    return commands
