<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->

# Result Interpretation

Start from the CLI's status rows and mitigation plan. Do not invent additional
root causes unless the evidence supports them.

For operator summaries:

1. Lead with FAIL rows, then WARN rows.
2. Group repeated symptoms under the most likely root prerequisite when the CLI
   already identifies one.
3. Explain the mode impact: native GDS, P2PDMA/C2C, RDMA, or compat fallback.
4. Preserve the distinction between "unsupported", "not configured", and
   "not enough evidence".
5. Give the next command or package query that would reduce uncertainty.

Runtime validation caveat:

- If `post-install`, `mount-check`, or `support-matrix --live` reports that
  `nvidia-smi` cannot communicate with the NVIDIA driver, runtime GPU validation
  is unavailable, or `gdscheck` cannot access the live runtime, check whether the
  agent process is sandboxed away from `/dev/nvidia*`. If driver/GPU evidence is
  otherwise present through `lspci`, `modinfo nvidia`,
  `/proc/driver/nvidia/version`, or `/proc/devices`, ask permission to rerun the
  same `gds-diag.py` command outside the sandbox before concluding that the
  host driver or GDS runtime is broken. If escalation is declined, describe the
  result as limited by the agent execution environment and ask the operator to
  run the same command in a normal host terminal.

Use these phrasing rules:

- "Compat mode" means CPU bounce-buffer fallback, not a direct GDS route.
- "Config-gated" means the route may require cufile.json settings and runtime
  confirmation.
- "P2PDMA" means upstream PCI P2PDMA on x86. "C2C" means the coherent direct
  path on supported NVIDIA Grace Hopper or Grace Blackwell ARM platforms; it
  uses the same `use_pci_p2pdma` configuration keys and may appear as `c2c` in
  `gdscheck`.
- "Client not loaded" in live support output means live tokens were unavailable
  for that filesystem client; it is not the same as a static support denial.
- "Need X.Y" in compat cells means the installed libcufile/GDS version is older
  than the documented version gate.
- `support-matrix`'s RDMA column names a mechanism (`dmabuf/peermem` or
  `FS/kernel RDMA`), not a plain Yes/No — each is a prerequisite for that
  filesystem's Native GDS path, not a standalone alternative to it. In
  `--live` mode the two mechanisms are gated differently: `dmabuf/peermem`
  (GPFS, WekaFS) has its own gdscheck token, so it directly reflects live
  state; `FS/kernel RDMA` (Lustre, BeeGFS, NFS, NVMe-oF, ScaTeFS) has no
  separate live token — it instead mirrors that same row's live-confirmed
  Native verdict. So `--live` showing `Native: No` and `RDMA: No` together
  for one of those filesystems means the client isn't loaded right now, not
  that the filesystem lacks RDMA capability — check `--static` for the
  architectural answer.

Route interpretation boundaries:

- Direct P2P support is route-limited. Local NVMe, NVMe-oF, virtiofs, and
  Grace-configured RAID0 can be P2PDMA/C2C candidates; NFS, GPFS, WekaFS,
  Lustre, BeeGFS, and ScaTeFS are not P2PDMA/C2C routes even if a global JSON
  key is enabled — NFS's direct path is RDMA (NFSoRDMA), not P2PDMA; its
  `fs.nfs.use_pci_p2pdma` JSON key is a legacy no-op, not a real toggle.
- For local NVMe, both `properties.use_pci_p2pdma=true` and
  `block.nvme.use_pci_p2pdma=true` are required before P2PDMA/C2C is attempted.
  On x86, NVMe multipath remains a separate blocker unless a specialized patch
  supports that topology; this is not required on NVIDIA CPUs (e.g. Grace).
- In `gdscheck` DRIVER CONFIGURATION, `p2pdma` means upstream PCI P2PDMA is
  active and `c2c` means the coherent C2C path is active on a supported ARM
  platform. Treat either as direct P2P evidence only for supported routes.
- For NVMe/NVMe-oF, `nvfs` and P2PDMA/C2C are separate direct-GDS routes. Do
  not call a host broken when one route is inactive but the other is active.
- For NVMe-backed ext4/XFS and NVMe-oF, `gdscheck` showing `NVMe: compat`
  without an `nvfs` token means the native nvidia-fs route is not active even if
  local nvidia-fs prerequisites passed. Explain that NVMe/NVMe-oF nvfs also
  needs a GDS-enabled storage stack from MLNX_OFED or DOCA/DOCA-OFED. If
  `p2pdma` or `c2c` is active for the same route, treat inactive nvfs as
  informational rather than a direct-GDS failure.

Config-audit interpretation:

- By default, `config-audit` should be read as the effective runtime view from
  `gdscheck -p` when available, with per-key file fallback for omitted known
  values and environment overlays unless `--ignore-env` was used.
- `--config PATH` findings apply to that proposed or copied file, not
  necessarily to the installed runtime.
- For GPFS RDMA addresses, cuFile precedence is `fs.gpfs.mount_table`, then
  `fs.gpfs.rdma_dev_addr_list`, then global
  `properties.rdma_dev_addr_list`; do not require the GPFS-specific list when a
  usable earlier source exists.
- Empty WekaFS/GPFS RDMA address lists such as `[]`, `[""]`, or blank strings
  should be treated as INFO unless the selected workflow needs explicit RDMA
  addresses and no fallback source exists. For Lustre and NFS, do not flag an
  empty global RDMA list by itself.
- `fs.gpfs.gds_write_support=false` and `fs.gpfs.gds_async_support=false` are
  INFO findings for GPFS profile audits.
- Static routing uses the canonical GDS 1.17+ keys
  `miscellaneous.enable_static_routing` and
  `miscellaneous.static_routing_filepath`; older `sparse.*` names are aliases.
- Low `properties.max_direct_io_size_kb`, GPU memory larger than
  `properties.max_device_pinned_mem_size_kb`, disabled execution parallelism,
  and malformed GPU bounce-buffer slab arrays are performance or validity
  findings, not filesystem support claims.

When the user asks for a fix, keep recommendations tied to the CLI evidence and
the repository's documented command boundaries.
