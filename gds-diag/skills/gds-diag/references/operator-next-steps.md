<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->

# Operator Next Steps

Use this reference when translating a finding into a concrete next action.

Common next steps:

- Missing CUDA Toolkit: install CUDA Toolkit from the current NVIDIA CUDA
  package-manager flow, then re-run `pre-install` or `post-install`.
- Missing GDS tools: install or repair the GDS tooling package so `gdscheck -p`
  is available, then re-run `support-matrix --live` or `post-install -v`.
- Missing `nvidia_fs`: install `nvidia-gds` for the full GDS stack, or query the
  configured CUDA repository for `nvidia-fs` packages when only the kernel
  module package is missing.
- cufile.json route mismatch: run `config-audit -v` on the effective
  configuration, or `config-audit --config PATH --ignore-env -v` for a proposed
  file, and apply only route-supported P2PDMA/C2C settings.
- P2PDMA/C2C not active: use `post-install -v` for host-wide direct-route
  prerequisites, or `mount-check PATH -v` for path-specific topology,
  ACS/IOMMU, NVMe multipath, and `gdscheck` token evidence.
- NVMe/NVMe-oF native route not active: if `mount-check` reports
  `NVMe: compat` with no `nvfs` token, verify the NVMe/NVMe-oF stack has the
  NVIDIA GDS storage-stack patches from MLNX_OFED or DOCA/DOCA-OFED, then
  re-run `gdscheck -p` and `mount-check PATH -v`.
- Path-specific fallback: run `mount-check PATH -v` and inspect filesystem,
  mount options, topology, ACS/IOMMU, and `gdscheck` evidence together.
- Log-only report: log parsing is deferred in the current CLI. Ask for the
  affected mount path or cufile.json and use `mount-check` or `config-audit`
  for implemented diagnostics.

For current NVIDIA behavior, use the live host and current NVIDIA docs rather
than memory when a recommendation depends on CUDA, driver, kernel, DOCA,
MLNX_OFED, filesystem, or release-note behavior.
