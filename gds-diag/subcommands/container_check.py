# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
container-check subcommand.

Validate whether GDS is usable from the current container and explain missing
container launch/configuration pieces.
"""
from __future__ import annotations

import argparse
import json

from checks import container
from checks.output import (
    bold,
    overall_exit_code,
    render_mitigation_plan,
    render_sections,
    render_summary,
    results_to_json,
)
from ._base import Subcommand


_DESCRIPTION = """\
Validate GDS installation and configuration inside a container.

This command checks whether the current container has the devices, tools,
configuration files, sysfs, and udev state needed for GDS diagnostics and GDS
workloads. It distinguishes container launch/configuration gaps from host GDS
installation problems.
"""


def _docker_guidance() -> list[str]:
    return [
        bold("  Docker Guidance"),
        "  For a diagnostic container, start from the smallest needed set and add visibility deliberately:",
        "    --gpus=all",
        "    --ipc=host",
        "    --cap-add=IPC_LOCK",
        "    -v /run/udev:/run/udev:ro",
        "    -v /sys:/sys:ro",
        "    -v /etc/cufile.json:/etc/cufile.json:ro",
        "    -v /usr/local/cuda-<version>:/usr/local/cuda-<version>:ro",
        "    --device=/dev/nvidia-fs<N>  # repeat for each host /dev/nvidia-fs* node needed",
        "    --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0  # RDMA routes",
        "    -v /host/gds/mount:/mnt/gds:rw",
        "",
        "  Docker --privileged is useful for diagnostics, but production containers should prefer",
        "  the narrow device and mount set required by the workload.",
        "",
    ]


def _enroot_guidance() -> list[str]:
    return [
        bold("  Enroot Guidance"),
        "  Verify host prerequisites before importing or starting .sqsh images:",
        "    command -v enroot",
        "    command -v squashfuse",
        "    ls -l /dev/fuse",
        "",
        "  For NVIDIA/RDMA visibility, check site hooks such as:",
        "    /etc/enroot/hooks.d/98-nvidia.sh",
        "    /etc/enroot/hooks.d/99-mellanox.sh",
        "",
        "  Bind the repo, CUDA/GDS tools, cufile.json, /run/udev, and the target mount with",
        "  Enroot fstab-style -m entries, for example:",
        "    -m /host/gds:/mnt/gds:none:x-create=dir,rbind,rw:0:0",
        "",
    ]


def _runtime_from_sections(sections: dict[str, list[container.CheckResult]]) -> str:
    context_rows = sections.get("Container Context", [])
    if not context_rows:
        return "unknown"
    why = context_rows[0].why.lower()
    if "docker" in why or "containerd" in why or "kubepods" in why:
        return "docker"
    if "enroot" in why:
        return "enroot"
    if "no common container markers" in why:
        return "host"
    return "unknown"


def _runtime_guidance(sections: dict[str, list[container.CheckResult]]) -> list[str]:
    runtime = _runtime_from_sections(sections)
    if runtime == "docker":
        return _docker_guidance()
    if runtime == "enroot":
        return _enroot_guidance()
    if runtime == "host":
        return [
            bold("  Container Guidance"),
            "  No common container markers were detected. Run container-check inside the target",
            "  Docker or Enroot container to validate its GDS visibility.",
            "",
        ]
    return (
        [bold("  Container Guidance"), "  Runtime is unknown; Docker and Enroot examples are shown for reference.", ""]
        + _docker_guidance()
        + _enroot_guidance()
    )


def _select_context(args: argparse.Namespace) -> container.ContainerContext:
    runtime = getattr(args, "runtime", "auto")
    detected = container.detect_container_context()
    if runtime == "auto":
        return detected
    evidence = [f"Runtime forced by --runtime {runtime}."]
    if detected.in_container:
        evidence.append(f"Auto-detected runtime: {detected.runtime}.")
        if detected.evidence:
            evidence.append(detected.evidence)
    else:
        evidence.append("Auto-detection found no Docker or Enroot container markers.")
    return container.ContainerContext(True, runtime, "\n".join(evidence))


def _run_host_text(verbose: bool = False) -> int:
    from checks.version import version_string

    print()
    print("═" * 70)
    print("  GDS Container Check")
    print("═" * 70)
    if verbose:
        print(f"  Tool: {version_string()}")
    print()
    print("  This command is meant to be run inside a Docker or Enroot container.")
    print("  No Docker or Enroot container markers were detected in this process.")
    print()
    print("  To validate a container, run for example:")
    print("    docker run --rm --gpus=all -v \"$PWD:/work:ro\" -w /work \\")
    print("      --entrypoint python3 \"$IMG\" gds-diag.py container-check -v")
    print()
    print("  If this process is inside a container but auto-detection failed, rerun with:")
    print("    gds-diag.py container-check --runtime docker")
    print("    gds-diag.py container-check --runtime enroot")
    print()
    print("  For host-level validation, use:")
    print("    ./gds-diag.py pre-install")
    print("    ./gds-diag.py post-install -v")
    print()
    return 2


def _run_text(args: argparse.Namespace) -> int:
    from checks.version import version_string

    context = _select_context(args)
    if not context.in_container:
        return _run_host_text(args.verbose)
    sections = container.collect_sections(context)

    print()
    print("═" * 70)
    print("  GDS Container Check")
    print("═" * 70)
    if args.verbose:
        print(f"  Tool: {version_string()}")
    print()

    for line in render_sections(sections, verbose=args.verbose):
        print(line)
    print(render_summary(sections))
    print()
    for line in render_mitigation_plan(sections, title="Container Launch Recommendations"):
        print(line)
    if args.verbose:
        for line in _runtime_guidance(sections):
            print(line)
    return overall_exit_code(sections)


def _run_json(args: argparse.Namespace) -> int:
    from checks.version import tool_metadata

    context = _select_context(args)
    if not context.in_container:
        print(json.dumps({
            "tool": tool_metadata(),
            "command": "container-check",
            "in_container": False,
            "runtime": "host",
            "error": "container_not_detected",
            "message": "container-check is meant to be run inside a Docker or Enroot container.",
            "override_hint": [
                "container-check --runtime docker",
                "container-check --runtime enroot",
            ],
            "host_commands": ["pre-install", "post-install"],
        }, indent=2))
        return 2
    sections = container.collect_sections(context)
    print(json.dumps({
        "tool": tool_metadata(),
        "command": "container-check",
        "runtime": context.runtime,
        "runtime_detection": context.evidence,
        "sections": {title: results_to_json(results) for title, results in sections.items()},
    }, indent=2))
    return overall_exit_code(sections)


class ContainerCheck(Subcommand):
    name = "container-check"
    help = "validate GDS installation and configuration inside a container"
    description = _DESCRIPTION
    order = 35

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--runtime",
            choices=("auto", "docker", "enroot"),
            default="auto",
            help=(
                "container runtime to assume for guidance; default auto-detects. "
                "Use docker or enroot when running in a container that auto-detection misses."
            ),
        )

    def run(self, args: argparse.Namespace) -> int:
        return _run_json(args) if args.json else _run_text(args)


COMMAND = ContainerCheck()
