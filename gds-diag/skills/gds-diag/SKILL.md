---
SPDX-FileCopyrightText: "Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
name: gds-diag
description: Use when diagnosing NVIDIA GPUDirect Storage with this repository: choose and run the right gds-diag.py subcommand, interpret its output, and explain operator next steps without duplicating the deterministic Python checks.
---

# GDS Diag

Use this skill to help an operator diagnose NVIDIA GPUDirect Storage (GDS) with
the deterministic CLI in this repository.

The Python code is the source of truth for probing, parsing, support-matrix
semantics, cufile.json validation, topology parsing, and output formatting. Do
not reimplement those checks in this skill. Use the skill to understand the
operator's goal, select the right subcommand, run it, and interpret its output.

## Workflow

1. Identify the operator's intent and target.
2. Select the narrowest `gds-diag.py` subcommand that answers the question.
3. Run the deterministic CLI from the repository root, or use
   `scripts/gds-diag-wrapper` from this skill directory when that is more
   reliable.
4. If a runtime validation command is likely limited by the agent sandbox, ask
   for permission to rerun the same command outside the sandbox before treating
   the sandboxed output as host truth. See "Sandbox-limited NVIDIA runtime
   checks" below.
5. Prefer verbose human output for operator-facing investigation. Prefer JSON
   output when you need stable structured data for follow-up reasoning.
6. Summarize the relevant WARN and FAIL findings first, then explain what they
   mean for direct GDS, P2PDMA, RDMA, or compat mode.
7. Give concrete next commands or documentation links only when they follow from
   the script output, local evidence, or current NVIDIA documentation.

## Command Routing

Read `references/command-routing.md` when choosing a subcommand.

Common routes:

- Broad or unclear GDS diagnosis, general bug-report collection, or "check this
  system/container/path":
  `python3 gds-diag.py all PATH -v`
  Use this as the default starting point when the operator has not already
  narrowed the problem. It detects host vs container context, runs the
  appropriate deterministic subcommands, and stops at the first unsuccessful
  return code. Use `--json` when collecting a structured report for automation
  or a bug.
- Container GDS visibility, launch options, or comparing good/bad container
  runtime exposure:
  `python3 gds-diag.py container-check -v`
  Run this first from inside the target Docker or Enroot container before
  using `post-install`, `mount-check`, or `support-matrix --live` to diagnose
  that container. It distinguishes missing container devices, mounts, tools,
  and metadata from host-level GDS installation problems.
- Host readiness before CUDA/GDS is fully installed:
  `python3 gds-diag.py pre-install`
- Installed runtime validation:
  `python3 gds-diag.py post-install -v`
- Path-specific compat-mode or storage-route diagnosis:
  `python3 gds-diag.py mount-check PATH -v`
- Effective cuFile configuration audit from `gdscheck -p`, with file fallback
  and `CUFILE_*` environment overlays:
  `python3 gds-diag.py config-audit --profile PROFILE`
- Proposed or copied cufile.json/JSONC audit:
  `python3 gds-diag.py config-audit --config PATH --ignore-env -v`
- Filesystem support reference:
  `python3 gds-diag.py support-matrix`
- Installed/runtime support view:
  `python3 gds-diag.py support-matrix --live`

## Interpretation Rules

Read `references/result-interpretation.md` before explaining non-trivial output.

Important boundaries:

- Do not treat compat mode as direct GDS. It is a CPU bounce-buffer fallback.
- Do not claim a filesystem supports P2PDMA/C2C just because a global JSON
  setting is enabled. Route support is limited by the Python support matrix.
- NFS's direct GDS path is NFSoRDMA, not the NVMe-style `nvfs` path — but
  nvidia-fs (`nvfs`) still has to be loaded to activate it, so NFS is
  Native-applicable, same as Lustre and BeeGFS. Don't describe NFS as
  supporting native GDS "via NVMe" — it's a different mechanism — but also
  don't claim NFS has no relationship to nvidia-fs/`nvfs` at all.
- For GPFS and WekaFS, focus on userspace RDMA via DmaBuf or
  `nvidia_peermem`, not NVMe-style `nvfs`.
- For release-specific behavior, rely on the tool's version-aware output and
  current NVIDIA documentation.

## Escalation

Some commands may need `sudo` or host-specific access to gather complete
evidence. Ask before using privileged commands unless the user already asked for
that level of probing.

### Sandbox-limited NVIDIA runtime checks

Agent execution sandboxes may hide `/dev/nvidia*` device nodes even when the
real host has a working NVIDIA driver. This can make `nvidia-smi`, `gdscheck`,
GPU topology checks, and post-install runtime validation fail inside the agent
while succeeding in the operator's normal terminal.

For commands that validate installed runtime state, especially:

- `python3 gds-diag.py post-install -v`
- `python3 gds-diag.py mount-check PATH -v`
- `python3 gds-diag.py all PATH -v`
- `python3 gds-diag.py support-matrix --live`

use this flow:

1. Run the selected command normally first.
2. If it fails because `nvidia-smi` cannot communicate with the NVIDIA driver,
   the post-install prerequisite gate says runtime GPU validation is
   unavailable, `gdscheck` cannot access the runtime, or `/dev/nvidia*` appears
   missing from the agent environment while `lspci`, `modinfo nvidia`,
   `/proc/driver/nvidia/version`, or `/proc/devices` indicate the driver/GPU
   exists, do not conclude that the host driver is broken.
3. Request permission to rerun the exact same `gds-diag.py` command outside
   the sandbox using the agent's escalation mechanism. In Codex, run the command
   with `sandbox_permissions="require_escalated"` and a justification such as:

   ```text
   Allow running gds-diag post-install outside the sandbox so it can access /dev/nvidia* and validate the live NVIDIA/GDS runtime?
   ```

4. Treat the outside-sandbox result as authoritative for host status. If the
   user declines escalation, say that the result is limited by agent sandbox
   visibility and ask the operator to run the same command in a normal host
   terminal.

Do not change the deterministic CLI result text in the skill. The skill's job is
to choose the right execution environment and explain when sandbox visibility
limits the evidence.
