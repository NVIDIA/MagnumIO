# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
config-audit subcommand.

Audit the effective cuFile configuration from gdscheck or cufile.json plus
environment overrides. This is intentionally non-mutating: it reports risks and
suggested actions without editing /etc/cufile.json.
"""
from __future__ import annotations

import argparse
import json

from ._base import Subcommand


_DESCRIPTION = """\
Audit cuFile configuration.

Reads --config when provided. Otherwise it prefers the CUFILE CONFIGURATION
section from `gdscheck -p`, because that is the installed library/runtime view.
If gdscheck is missing or cannot expose that section, it falls back to
CUFILE_ENV_PATH_JSON when set, then /etc/cufile.json, then the CUDA template
fallback. Environment overrides such as CUFILE_FORCE_COMPAT_MODE,
CUFILE_ALLOW_COMPAT_MODE, CUFILE_USE_PCIP2PDMA, CUFILE_LOGGING_LEVEL, and
CUFILE_LOGFILE_PATH are applied to show the effective value unless --ignore-env
is used.

Profiles narrow the output to the values most relevant for a deployment type.
"""

_PROFILES = [
    "local-nvme",
    "nvmeof",
    "lustre",
    "beegfs",
    "scatefs",
    "wekafs",
    "gpfs",
    "nfs-rdma",
    "virtiofs",
    "raid0",
    "compat-safe",
]


def _run_text(args: argparse.Namespace) -> int:
    from checks.cufile_config import audit_config
    import textwrap as _tw
    from checks.output import bold, cyan, dim, green, red, yellow

    def _wrap_field(label: str, value: str) -> None:
        prefix = f"      {label} : "
        print(_tw.fill(
            value,
            width=80,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
            break_on_hyphens=False,
            break_long_words=False,
        ))
    from checks.version import version_string

    audit = audit_config(args.profile, config_path=args.config, apply_env=not args.ignore_env)
    entries = audit["entries"]
    rdma_policy_recs: list[str] = []
    if args.profile in ("gpfs", "wekafs"):
        try:
            from checks import rdma
            rdma_policy_recs = rdma.rdma_policy_recommendations(args.profile)
        except Exception:
            rdma_policy_recs = []

    print()
    print(bold("═" * 70))
    print(bold("  GDS cuFile Configuration Audit"))
    print(bold("═" * 70))
    if args.verbose:
        print(f"  Tool          : {version_string()}")
    print(f"  Config source : {audit.get('config_source_detail') or audit.get('config_source')}")
    if audit.get("gdscheck_path"):
        print(f"  gdscheck      : {audit['gdscheck_path']}")
    if audit.get("file_fallback_path"):
        print(f"  File fallback : {audit['file_fallback_path']}")
    if audit.get("config_path"):
        print(f"  Config path   : {audit['config_path']}")
    elif audit.get("config_source") != "gdscheck":
        print("  Config path   : (not found; defaults apply)")
    if audit.get("gdscheck_fallback"):
        fallback_note = audit.get("gdscheck_error")
        if not fallback_note:
            fallback_note = (
                "unavailable; used file fallback"
                if audit.get("file_fallback_path") or audit.get("config_path")
                else "unavailable; used built-in defaults"
            )
        print(f"  gdscheck note : {fallback_note}")
    if audit.get("requested_config_path"):
        print(f"  Requested   : {audit['requested_config_path']}")
    if audit.get("env_config_path"):
        print(f"  Env path    : {audit['env_config_path']}")
    if not audit.get("env_overrides_applied", True):
        print("  Env values  : ignored")
    if audit.get("profile"):
        print(f"  Profile     : {audit['profile']}")
    print()

    if audit.get("parse_error"):
        print(f"  {red('FAIL')} cufile.json parse error: {audit['parse_error']}")
        print()
        return 1

    to_show = entries if args.verbose else [e for e in entries if e["status"] != "OK"]
    if not to_show:
        print(f"  {green('PASS')} No config risks found for this audit scope.")
        print()

    for entry in to_show:
        status = entry["status"]
        icon = green("OK") if status == "OK" else yellow(status) if status == "INFO" else red(status)
        print(f"  {icon}  {bold(entry['path'])}")
        print(f"      value  : {entry['value']!r}  ({entry['source']})")
        if entry["resolved_key"] != entry["path"]:
            print(f"      alias  : {entry['resolved_key']}")
        print(f"      default: {entry['default']!r}")
        if entry.get("risk"):
            _wrap_field("risk  ", entry["risk"])
        if entry.get("recommendation"):
            _wrap_field("action", entry["recommendation"])
        if entry.get("detail"):
            _wrap_field("detail", entry["detail"])
        if args.verbose and entry.get("description"):
            print(f"      {dim(entry['description'])}")
        print()

    if rdma_policy_recs:
        print(bold("  Live RDMA Policy Recommendation"))
        print("  " + "─" * 66)
        for rec in rdma_policy_recs:
            for i, line in enumerate(rec.splitlines()):
                prefix = "  • " if i == 0 else "    "
                print(f"{prefix}{line}")
        print()

    if not audit.get("parse_error"):
        n_ok   = sum(1 for e in entries if e["status"] == "OK")
        n_info = sum(1 for e in entries if e["status"] == "INFO")
        n_warn = sum(1 for e in entries if e["status"] == "WARN")
        _parts: list[str] = []
        if n_ok:
            _parts.append(green(f"{n_ok} ok"))
        if n_info:
            _parts.append(cyan(f"{n_info} info"))
        if n_warn:
            _parts.append(yellow(f"{n_warn} warning{'s' if n_warn != 1 else ''}"))
        print("  " + (", ".join(_parts) if _parts else "No entries checked."))
        print()

    return 1 if any(e["status"] == "WARN" for e in entries) else 0


def _run_json(args: argparse.Namespace) -> int:
    from checks.cufile_config import audit_config
    from checks.version import tool_metadata

    audit = audit_config(args.profile, config_path=args.config, apply_env=not args.ignore_env)
    audit["tool"] = tool_metadata()
    if args.profile in ("gpfs", "wekafs"):
        try:
            from checks import rdma
            audit["rdma_policy_recommendations"] = rdma.rdma_policy_recommendations(args.profile)
        except Exception:
            audit["rdma_policy_recommendations"] = []
    print(json.dumps(audit, indent=2))
    if audit.get("parse_error"):
        return 1
    return 1 if any(e["status"] == "WARN" for e in audit["entries"]) else 0


class ConfigAuditCommand(Subcommand):
    name = "config-audit"
    help = "audit effective cuFile settings and environment overrides"
    description = _DESCRIPTION
    order = 35

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--profile",
            choices=_PROFILES,
            help="limit recommendations to a deployment profile",
        )
        parser.add_argument(
            "--config",
            metavar="PATH",
            help=(
                "scan this cufile.json instead of live gdscheck or the default file search path"
            ),
        )
        parser.add_argument(
            "--ignore-env",
            action="store_true",
            help="do not apply CUFILE_* environment overrides on top of the selected source",
        )

    def run(self, args: argparse.Namespace) -> int:
        return _run_json(args) if args.json else _run_text(args)


COMMAND = ConfigAuditCommand()
