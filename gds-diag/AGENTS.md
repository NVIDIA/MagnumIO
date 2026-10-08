# AGENTS.md

Guidance for AI coding agents, assistants, and copilots working on this
repository.

This project is a GPUDirect Storage (GDS) diagnostic toolkit. It helps users
decide whether a host is ready for GDS installation, whether GDS was installed
correctly, which filesystem/storage routes are supported, and what to fix when
GDS falls back to compat mode.

## Read These First

Use these project files as the local source of truth before changing behavior:

- `README.md` - user-facing commands and quick start.
- `doc/DESIGN.md` - command behavior, output contract, support-matrix semantics,
  and GDS-specific design rules.
- `skills/gds-diag/SKILL.md` - shared Claude/Codex diagnostic workflow for
  routing user intent to deterministic CLI commands and interpreting results.
- `doc/GDS_IMPROVEMENT_PLAN.md` - backlog and improvement rationale, when present.
- `checks/` and `subcommands/` - implementation.
- `tests/` - expected behavior and regression coverage.

For current NVIDIA behavior, also check the live host and current NVIDIA docs:

- GDS troubleshooting guide:
  https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
- CUDA downloads and installer flow:
  https://developer.nvidia.com/cuda-downloads
- CUDA Linux package-manager installation guide:
  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- DOCA host/storage installation:
  https://docs.nvidia.com/doca/sdk/doca-host-installation-and-upgrade/index.html#storage-installation

Do not rely on memory alone for release-specific GDS support. If a claim depends
on current driver, CUDA, kernel, DOCA, MLNX_OFED, or filesystem release behavior,
verify against NVIDIA docs/release notes and local command output.

## Local Commands

Primary CLI:

```bash
python3 gds-diag.py --help
python3 gds-diag.py support-matrix
python3 gds-diag.py support-matrix --live
python3 gds-diag.py pre-install
python3 gds-diag.py post-install -v
python3 gds-diag.py mount-check /path/to/mount -v
python3 gds-diag.py config-audit --profile local-nvme
python3 gds-diag.py config-audit --config ./cufile.json --ignore-env -v
```

Development verification:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile gds-diag.py checks/*.py subcommands/*.py tests/*.py
git diff --check
```

This project intentionally uses only the Python standard library for the main
CLI path. Do not add package dependencies unless the user explicitly accepts the
tradeoff and the dependency is isolated from basic command dispatch.

## Python Version Compatibility

The documented minimum is Python 3.8 (see `README.md`), and CI runs the test
suite on 3.8 as well as newer versions. Keep both the CLI and the tests
working on 3.8 unless doing so would require an extraordinary compromise. If
it would, stop and ask the user for a decision before raising the floor,
dropping a CI version, or contorting the code; do not decide this
unilaterally.

Common pitfalls when writing 3.9+ code that breaks 3.8:

- Start every Python file with `from __future__ import annotations`, including
  tests, so annotations are not evaluated at import time.
- Even with that import, annotations are only safe: do not use builtin
  generics (`list[str]`, `dict[str, int]`, `tuple[...]`) or `X | Y` unions in
  runtime expressions such as type aliases, `isinstance`, `cast`, or
  dataclass/`NamedTuple` field defaults. Use `typing.List`, `typing.Optional`,
  and similar there.
- Avoid APIs newer than 3.8, such as `str.removeprefix`/`removesuffix`,
  `functools.cache`, `zip(strict=)`, `math.lcm`, and `match` statements.

## Licensing

This directory is part of the Magnum IO repository and is distributed under
the repository's root Apache License, Version 2.0 (`../LICENSE`), with one
local exception for `skills/` (see below). Preserve the local notice files:

- `NOTICE`
- `LICENSE-CC-BY-4.0`
- `THIRD_PARTY_NOTICES.md`

Contribution and DCO sign-off guidance lives in the repository's root
`../CONTRIBUTING.md`; there is no separate `CONTRIBUTING.md` here.

For NVIDIA-authored source files, preserve or add concise SPDX headers using
file-appropriate comment syntax:

```text
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Do not remove existing copyright, license, attribution, NOTICE, AUTHORS, or
third-party provenance information. If adding, vendoring, copying, or modifying
third-party code, document the source, license, and relevant notes in
`THIRD_PARTY_NOTICES.md` or the project's equivalent third-party notice file.

When auditing licensing, consider Git-tracked files in this repository. Do not
follow Git submodules, symlink targets outside the repository, or untracked
files unless the user explicitly expands the scope.

Preserve DCO sign-off guidance in the repository's root `CONTRIBUTING.md`. New
external contributions should use `Signed-off-by:` lines consistent with the
Developer Certificate of Origin.

## Command Boundaries

`pre-install` checks whether a host is ready before CUDA/GDS installation is
complete. It must not depend on GDS packages, `gdscheck`, `nvidia_fs`,
`/etc/cufile.json`, or `nvidia-smi`.

In `pre-install`:

- Use `lspci` for NVIDIA GPU presence.
- Treat missing CUDA Toolkit as FAIL.
- Treat missing NVIDIA Open Driver installation as FAIL.
- Print a `Runtime Versions` block, but do not call `nvidia-smi`; use
  `modinfo nvidia` for the driver module version. Report CUDA Toolkit,
  NVIDIA driver, nvidia-fs, and libcufile as version values only.
- Include a warning-only `DOCA / MLNX_OFED` advisory. Missing MLNX_OFED/DOCA is
  not a universal GDS blocker, but warn because RDMA paths and NVMe/NVMe-oF
  nvfs deployments that rely on NVIDIA storage-stack patches may need it.
- Keep driver version, compute capability, CoherentGPUMemoryMode, GPU BDFs, and
  GPU/NVMe topology out of pre-install. Those are post-install/runtime checks.
- Do not include an `nvidia-fs` section.
- Use the box-style renderer (`render_sections` from `checks/output.py`) for
  the main report. Keep long documentation URLs out of box cells; the final
  Mitigation Plan prints full URLs as standalone lines for terminals, SSH
  sessions, and logs.

`post-install` validates the installed runtime. It may use `nvidia-smi`,
`gdscheck`, `nvidia_fs`, `libcufile`, `/etc/cufile.json`, and runtime topology.
Gate post-install on base prerequisites first: CUDA Toolkit, NVIDIA driver, and
NVIDIA Open Kernel Driver, with a visible NVIDIA GPU required for GPU-memory GDS
validation. If those are missing, stop after a prerequisite table instead of
cascading into misleading GDS package/module failures. Always print a `Runtime
Versions` block with CUDA Toolkit, NVIDIA driver, nvidia-fs, and libcufile
versions. Print only the version value there, not paths or probe sources. If
CUDA Toolkit is installed but no GPU and/or working NVIDIA driver is present,
add the explicit warning that only system-memory buffers are supported on that
host and GPU-memory buffers are not supported with libcufile until an NVIDIA GPU
and working NVIDIA driver are present.

For nvidia-fs package guidance, do not split package names by distro as if
Ubuntu uses one name and RHEL uses another. NVIDIA's CUDA package-manager docs
use `nvidia-gds` for the full GDS install after the driver and CUDA Toolkit are
installed. Treat `nvidia-fs-dkms` as the kernel-module package that may be
pulled in by `nvidia-gds` or installed directly when only the `nvidia_fs` module
package is missing. When exact names are uncertain, recommend querying the
configured CUDA repository with `apt-cache search nvidia-gds`,
`apt-cache search nvidia-fs`, or
`dnf list --available 'nvidia-gds*' 'nvidia-fs*'`.

`mount-check` is path-specific. It should combine mount information,
filesystem type, block/NVMe mapping, cufile settings, topology, ACS/IOMMU, and
`gdscheck` evidence when available.

For NVMe-backed ext4/XFS and NVMe-oF, distinguish "nvidia-fs prerequisites
passed" from "nvidia-fs/nvfs route is active." If `gdscheck -p` has the driver
line but no `nvfs`/native token, report the native route as WARN/Not active and
recommend MLNX_OFED or DOCA/DOCA-OFED GDS storage-stack patches with links. If
P2PDMA/C2C is active instead, make the inactive nvfs route informational.

`config-audit` validates effective cuFile configuration plus `CUFILE_*`
environment overrides by default. With no `--config`, it should prefer the
`CUFILE CONFIGURATION` section from `gdscheck -p` because that is the installed
library/runtime source of truth. If gdscheck is selected but omits a known key,
fill that key from `CUFILE_ENV_PATH_JSON`, `/etc/cufile.json`, or the CUDA
template and mark the row as file fallback. If gdscheck is missing or cannot
expose that section, fall back fully to the same file search path. It must also
support `--config PATH` for proposed or copied JSON/JSONC files and
`--ignore-env` when the user wants to suppress additional `CUFILE_*`
environment overlays; when `--ignore-env` is set, invoke `gdscheck` without
inheriting `CUFILE_*` variables as well. Use `--config PATH --ignore-env` for
strict file-only validation.

## GDS Rules To Preserve

Direct P2P library support is route-limited. On x86 this is reported as PCI
P2PDMA. On supported NVIDIA Grace Hopper and Grace Blackwell ARM
platforms with coherent C2C, the same `use_pci_p2pdma` cufile.json keys enable
the C2C direct path and `gdscheck` may report the active token as `c2c`.
Supported routes are:

- local NVMe
- NVMe-oF
- NFS
- virtiofs
- RAID0

A JSON setting can enable preference for a supported P2PDMA/C2C route, but it
does not create library support for GPFS, WekaFS, Lustre, BeeGFS, ScaTeFS, or
other unsupported routes.

Local NVMe P2PDMA/C2C requires both:

- `properties.use_pci_p2pdma=true`
- `block.nvme.use_pci_p2pdma=true`

On x86 it also requires NVMe multipath to be disabled unless the host has a specialized
patch that supports that topology. This is not required on NVIDIA CPUs (e.g. Grace),
where multipath works fine alongside the C2C direct path.

NVMe-oF P2PDMA/C2C is controlled separately from local NVMe and requires the
matching NVMe-oF block setting.

RAID0 over NVMe should be presented as Grace-configured support, not generic
opt-in support. Only Grace-based supported RAID0 paths should use
`block.raid.use_pci_p2pdma=true`, and active support still needs confirmation
from `gdscheck` (`p2pdma` or `c2c`) or a workload.

For NVMe and NVMe-oF native `nvidia-fs`/`nvfs` mode, NVIDIA documents patched
storage stacks through MLNX_OFED or DOCA/DOCA-OFED packages. Mention DOCA when
the recommendation is to use `nvidia-fs` mode for NVMe/NVMe-oF paths that need
those patches.

NFS is not the same as NVMe-style `nvfs`. GDS support for NFS is through the
NFS/RDMA path. Verify mount options, RDMA devices, `gdscheck`, and stats.

GPFS and WekaFS native GDS use cuFile userspace RDMA via DmaBuf or
`nvidia_peermem`, not NVMe-style `nvfs`. Their topology recommendations should
consider GPU/NIC distance and RDMA load-balancing policy. P2PDMA should not be
reported as architecturally supported for GPFS or WekaFS even if a global JSON
setting is enabled or `gdscheck` prints a misleading token.

Lustre and DDN EXAScaler should be treated as the same Lustre-based route in the
support matrix.

Compat mode is a CPU bounce-buffer fallback. It can be useful as a safety net,
but recommendations should clearly distinguish compat from direct GDS paths.
For support-matrix work, keep release-note-gated compat support version-aware:
squashfs/tmpfs/ramfs/overlayfs require GDS/libcufile 1.16+, while ZFS and BTRFS
require GDS/libcufile 1.17 for the documented compatibility path across all I/O
APIs. Prefer `gdscheck -p`'s `GDS release version`, then `cuFileGetVersion()`
from libcufile, then the `libcufile.so.X.Y.Z` symlink for patch/build context.
If `support-matrix --live` cannot find or run gdscheck but CUDA Toolkit and
libcufile are present, fall back to the version-aware documentation matrix and
clearly say live driver/client tokens are unavailable until gds-tools is fixed.

## Topology Rules

For NVMe-backed mount points, prefer:

```bash
nvidia-smi topo -m -nvme
```

Parse both the main topology table and the second NVMe table that may appear
after `NIC Legend`. Use `NVMe Legend` entries to map `NVMe0`, `NVMe1`, and
similar labels back to Linux namespace names such as `nvme0n1`.

Do not report "no GPUs found" if `nvidia-smi --query-gpu=pci.bus_id` is empty
but `nvidia-smi topo -m -nvme` contains GPU/NVMe relationships. In that case,
use the topology matrix for placement recommendations.

Normalize PCI domains when comparing BDFs. NVIDIA tools may print
`00000000:65:00.0`, while Linux sysfs often uses `0000:65:00.0`.

Do not report cross-root-port GPU/NVMe placement as a hard GDS/P2PDMA blocker.
Treat it as a performance warning: GDS can operate across root ports, but closer
PIX/PXB/PHB paths are preferred over NODE/SYS paths. Hard P2PDMA blockers should
come from ACS redirect, IOMMU mode, missing kernel/library support, cufile.json
settings, or observed `gdscheck`/workload route failures.

For GPFS and WekaFS, RDMA policy recommendations are valid only when the tool can
observe enough GPUs, NICs, and GPU/NIC distance data to make a useful
recommendation.

## cufile.json Audit Rules

Flag each problematic variable with a specific finding and mitigation. Validate
the JSON shape before making route recommendations.

Important audit areas:

- invalid JSON/JSONC syntax
- unknown keys that may be typos
- bad scalar types
- invalid enum values such as `logging.level`, `io_priority`,
  `rdma_peer_type`, `rdma_transport_type`, and
  `rdma_load_balancing_policy`
- out-of-range profiling/stat values
- negative, unaligned, or nonsensical sizes
- performance-sensitive sizing: INFO when `properties.max_direct_io_size_kb`
  is below 16384 KB, and INFO when observed GPU memory is larger than
  `properties.max_device_pinned_mem_size_kb`
- Newer template/runtime keys such as `properties.vanilla_posix_io_mode`,
  `properties.gds_fallback_io`, `properties.rdma_transport_type`,
  `properties.allow_rdma_token_reset`,
  `properties.compat_odirect_unaligned_read_split`,
  `properties.compat_odirect_unaligned_read_split_min_size_kb`,
  `block.raid1.use_pci_p2pdma`, `block.raid10.use_pci_p2pdma`, and
  `miscellaneous.rdma_token_reset_timeout_secs` are known schema entries, not
  unknown-key findings.
- GPU bounce-buffer slab config: treat
  `properties.gpu_bounce_buffer_slab_config.slab_size_kb` and
  `properties.gpu_bounce_buffer_slab_config.slab_count` as valid nested keys,
  not unknown fields. Also treat flattened
  `properties.gpu_bounce_buffer_slab_size_kb` and
  `properties.gpu_bounce_buffer_slab_count` entries from `gdscheck -p` as valid.
  Validate that both arrays are present, the same length, positive, and that
  slab sizes are 4 KB aligned and ascending.
- execution/threadpool settings that disable parallelism, including
  `execution.parallel_io=false` or `execution.max_request_parallelism=0`
- P2PDMA/C2C global setting mismatched with route-level settings
- P2PDMA/C2C-capable profiles where both global and route-level keys are disabled;
  report this as INFO, not WARN, because nvfs or compat may still be a valid
  non-P2PDMA/C2C path.
- route-level P2PDMA/C2C settings on unsupported filesystem routes
- `rdma_dev_addr_list` values that are malformed or not local client IP
  addresses. Empty global `properties.rdma_dev_addr_list` should only produce
  the auto-discovery INFO for WekaFS/GPFS profiles, not Lustre or NFS.
- GPFS RDMA address precedence from cuFile code: `fs.gpfs.mount_table` first,
  then `fs.gpfs.rdma_dev_addr_list`, then global
  `properties.rdma_dev_addr_list`. Do not require the GPFS-specific list when a
  usable mount table or global list is present.
- Empty WekaFS/GPFS `rdma_dev_addr_list` values include `[]`, `[""]`, and blank
  strings. Explicitly empty global lists should produce INFO, and empty
  profile-specific lists should produce INFO when no global list or mount table
  provides an explicit RDMA address source.
- GPFS capability toggles: report `fs.gpfs.gds_write_support=false` and
  `fs.gpfs.gds_async_support=false` as INFO in the GPFS profile so users can
  see that GPFS GDS writes or GPFS async GDS support are disabled.
- incompatible or duplicate RDMA dynamic-routing policies
- dynamic routing with no usable RDMA address source
- malformed `mount_table` entries
- static topology routing: canonical GDS 1.17+ keys are
  `miscellaneous.enable_static_routing` and
  `miscellaneous.static_routing_filepath`; accept older `sparse.*` names as
  aliases and warn when static routing is enabled but the topology file is
  missing, empty, unreadable, or not a regular file on the host

When auditing a file supplied with `--config PATH`, do not assume it is installed
or currently active. Say whether the finding applies to the file, the current
environment overrides, or the effective merged configuration.

## Output Style

Human output uses the `┌─ Section  ⚠ WARN / │ / └─` box style throughout all
subcommands. Do not introduce a new table renderer or per-string manual
wrapping. Use `render_sections()` from `checks/output.py` for section groups
and `render_section()` for individual sections. All finding and mitigation text
wraps automatically via `_format_block_text()`; adding `textwrap.fill()` or
`.splitlines()` loops directly in rendering code is a sign that the output
should instead be routed through the canonical renderer.

Normal mode shows only sections with warnings or failures; all-pass sections
are suppressed. Verbose mode shows every check including PASS rows and
evidence. All output targets 80 columns. Both normal and verbose output end
with a one-line summary tally and a numbered Mitigation Plan for all warnings
and failures.

Each issue should answer:

- what was checked
- what was found
- why it matters for GDS
- what command or documentation should be used next

Avoid duplicating the same root cause across multiple rows. For example, if the
NVIDIA driver is not installed, emit one clear driver prerequisite failure
instead of repeating the same mitigation in every `nvidia-smi`-dependent check.

## Data Model and Architecture

The canonical data types are `CheckResult` and `ModeReport` in
`checks/result.py`. Read `doc/DESIGN.md` (Data model section) for the full
explanation. Key rules:

**Check functions return `CheckResult` objects, not strings or dicts.**
Every check function must return `CheckResult` or `list[CheckResult]` with
`check`, `mode`, `status`, `why`, and optionally `mitigation` and `evidence`
populated. Do not return raw strings from check functions and format them
later; text wrapping and layout are the renderer's job.

**Static checks belong in `build_mode_reports()` or the `run_all()` helpers.**
Gdscheck-derived diagnosis (findings that depend on running `gdscheck -p` and
parsing its output) belongs in `_enrich_with_gdscheck()`, which is called at
the end of `build_mode_reports()` after all static results are in place. Do
not call `_run_gdscheck_raw()` a second time from a rendering function; the
output is already available from the first call and is passed through to the
enrichment step.

**`render_section()` is the single box renderer.**
Do not add a second box-drawing renderer. If a new subcommand or section needs
the `┌─│└─` style, call `render_section()` from `checks/output.py` and pass
the `list[CheckResult]` directly. Use the optional `status=` parameter when
the caller already knows the aggregate status (e.g. from `ModeReport.status`)
rather than letting the renderer recompute it.

**gdscheck utilities live in `checks/gdscheck.py`.**
`_find_gdscheck()`, `_run_gdscheck_raw()`, `_gdscheck_section()`, and all
DRIVER CONFIGURATION token helpers (`driver_config_has_native()` etc.) are
canonical in that module. Do not add local copies in subcommands or other
check modules; import from `checks.gdscheck` instead.

**JSON output is derived from `ModeReport.results` automatically.**
`render_json_report()` serialises `report.results` via `results_to_json()`.
Any `CheckResult` appended to a `ModeReport` — whether from static checks or
from enrichment — appears in the JSON output without additional wiring. The
JSON field for the finding text is `"finding"` (not `"why"`).

## Coding Notes

Keep changes scoped to the requested diagnostic behavior. Preserve existing CLI
flags and JSON fields unless the user explicitly asks for a breaking change.

Use structured parsers/helpers when available. Avoid brittle string matching for
JSON, mount tables, topology matrices, or `gdscheck` sections when a local helper
already exists.

Add or update tests for parser changes, support-matrix semantics, config-audit
rules, and output regressions. For docs-only changes, at least run
`git diff --check`.

When a change adds a new check, subcommand, flag, or other user-visible
behavior, add a bullet under `[Unreleased]` in `CHANGELOG.md`. Do not bump
`__version__` in `checks/version.py` or add a new version heading — the user
tags and bumps releases separately.

Do not put host-specific credentials, private IPs, SSH keys, or lab-only paths in
this file. Keep this memory portable across GDS users and environments.
