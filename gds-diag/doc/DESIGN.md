# GDS Diag — Design Document

Each mode is a **subcommand** of `gds-diag.py`. `gds-diag.py --version` prints
the tool version plus the current Git commit when available. All subcommands support
`--json` for machine-readable output and `-v/--verbose` for extra evidence
where the subcommand exposes it. Human output uses ANSI colour; JSON is always
plain.

Human output uses a consistent `┌─ Section  ⚠ WARN / │ / └─` box style
throughout. Normal mode shows only sections that have INFO, WARN, or FAIL
findings; all-pass sections are suppressed unless `-v/--verbose` is given.
Each non-pass result shows its check name, status icon, finding text, and —
when present — a `→ Mitigation:` block. All finding and mitigation text wraps
at 80 columns. The report ends with a numbered Mitigation Plan aggregating
every WARN/FAIL action, and a one-line summary tally (`N passed, M warnings`).
Full documentation URLs are kept out of box cells and appear verbatim only in
the Mitigation Plan so SSH sessions and copied logs keep links usable.

The top-level CLI uses only the Python standard library, so `--help` and
subcommand dispatch work even on hosts where `pip install` is awkward.
Heavier dependencies (if any) are imported lazily inside individual
subcommand modules.

---

## Data model (`checks/result.py`)

All diagnostic data flows through three types defined in `checks/result.py`.
Check functions return these types; renderers consume them. Nothing else is
shared between the check layer and the display layer.

### `Status`

Five-valued enum used on both individual checks and aggregated mode verdicts:

| Value | Meaning |
|-------|---------|
| `PASS` | Check passed; no operator action needed |
| `INFO` | Notable but not a blocker; context for the operator |
| `WARN` | Should be reviewed; may block GDS depending on deployment |
| `FAIL` | Hard blocker; must be resolved for this mode to work |
| `N/A`  | Mode is not applicable for this filesystem type |

When aggregating a list of results, the worst status wins: FAIL > WARN > INFO >
PASS. `N/A` is separate and set at the `ModeReport` level, not derived from
individual results.

### `GDSMode`

Four-valued enum representing the GDS data paths:

| Value | Description |
|-------|-------------|
| `NATIVE` | Direct GPU↔storage DMA via the nvidia-fs kernel module (`nvfs` token) or cuFile userspace RDMA (`dmabuf`/`nvidia_peermem` for GPFS/WekaFS) |
| `P2PDMA` | Direct PCIe peer-to-peer between NVMe and GPU without the nvidia-fs kernel module. On x86 this is Linux upstream PCI P2PDMA; on supported Grace/Blackwell ARM platforms it is C2C. |
| `RDMA`   | Network storage path via RDMA (NFSoRDMA, Lustre LNet, BeeGFS/NVMe-oF/ScaTeFS kernel RDMA, GPFS/WekaFS userspace RDMA) |
| `COMPAT` | CPU bounce buffer fallback — always available when `allow_compat_mode` is enabled in cufile.json |

### `CheckResult`

One finding from one check function. Every check function returns either a
single `CheckResult` or a `list[CheckResult]`.

```text
CheckResult
  check      str           human label, e.g. "IOMMU mode" or "nvidia-fs module"
  mode       GDSMode       which data path this finding belongs to
  status     Status        PASS / INFO / WARN / FAIL / N/A
  why        str           root-cause sentence explaining the verdict
  mitigation str | None    action to take; shown in the finding box and Mitigation Plan
  evidence   str | None    raw snippet (log line, file content); shown only in verbose mode
```

`mode` is carried on every result so that check functions (e.g.
`nvidia_fs.run_all()`) can return a flat list without knowing which
`ModeReport` will hold them.

### `ModeReport`

Aggregated picture for one GDS mode after all checks for that mode have run.

```text
ModeReport
  mode        GDSMode
  applicable  bool                False = this mode cannot work for this filesystem type
  results     list[CheckResult]   all check findings for this mode (static + gdscheck-derived)
  status      Status              computed property — worst status in results, or N/A if not applicable
```

`status` is never stored; it is recomputed from `results` on every access.
`blockers()` returns only FAIL results; `failing()` returns FAIL and WARN.

### How data flows through the system

```text
check functions (kernel, iommu, pcie, nvidia_fs, rdma, cufile_config, …)
    └── return list[CheckResult]
            └── collected into ModeReport.results  by  build_mode_reports()
                    │
                    ├── _enrich_with_gdscheck()  appends gdscheck-derived
                    │   CheckResults to each ModeReport after static checks,
                    │   so the complete picture is in ModeReport.results
                    │   before any rendering or serialisation.
                    │
                    ├── render_section()  (checks/output.py)
                    │       ┌─ Section title  ⚠ WARN
                    │       │  ⚠ WARN  check name
                    │       │  finding text, wrapped at 80 cols
                    │       │  → Mitigation:
                    │       │    mitigation text, wrapped at 80 cols
                    │       └─────────────────────────────────────────
                    │
                    └── results_to_json()  (checks/output.py)
                            { "check": "...", "status": "...",
                              "finding": "...", "mitigation": "...",
                              "evidence": "..." }
```

**Static checks** (`kernel.run_all`, `iommu.run_all`, `pcie.run_all`, etc.)
read system files and kernel state to verify prerequisites. They run first
and are always included in `ModeReport.results`.

**gdscheck enrichment** (`_enrich_with_gdscheck`) calls `gdscheck -p` once
and appends `CheckResult` objects for any mode that is "not active" or
"cannot verify" per the gdscheck DRIVER CONFIGURATION output. This runs at
the end of `build_mode_reports()`, reusing the same gdscheck output already
fetched for the driver-version check. After this step the `ModeReport` list
is complete: it contains both the static prerequisite picture and the runtime
gdscheck verdict.

### Rendering

`render_section()` in `checks/output.py` is the single canonical renderer.
It takes a title string, a `list[CheckResult]`, a `verbose` flag, and an
optional `status` override (used when the caller — e.g. `render_text_report`
for mount-check — already knows the aggregate status from `ModeReport.status`).

`render_sections()` wraps it for the common case of a
`dict[section_title, list[CheckResult]]`.

All text in `why` and `mitigation` passes through `_format_block_text()`,
which wraps prose at 80 columns, preserves indented command examples verbatim,
and keeps URLs on their own lines. No per-string manual wrapping is needed
anywhere else.

### Adding a new check

1. Write a function that returns `CheckResult` or `list[CheckResult]`.
2. Set `mode` to the `GDSMode` the check belongs to.
3. Append the result(s) to the appropriate `ModeReport.results` in
   `build_mode_reports()` (or the relevant `run_all()` function).

No changes to rendering, JSON serialisation, or the `all` subcommand are
required — the new data flows through automatically.

---

## subcommand: `support-matrix`

**Purpose:** Reference table of filesystem GDS support. Use `--static` (default)
for the documentation-sourced reference table, or `--live` to see what `gdscheck`
reports on the current system.

**Source of truth:** `checks/fs_matrix.py` → `FS_CAPABILITIES` dict, built from
the [NVIDIA GDS Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).
Release-note-gated behavior is also encoded in that matrix. `support-matrix`
detects the local libcufile/GDS version when it can and gates documented
version-dependent rows such as squashfs/tmpfs/ramfs/overlayfs and ZFS/BTRFS
compatibility-path support.

**Flags:** `--static` (default, no system access needed) | `--live` (runs `gdscheck -p`, requires `gds-tools-*` installed). Mutually exclusive. If `gdscheck` is missing and both CUDA Toolkit plus libcufile are present, `--live` falls back to the documentation matrix with libcufile version gating and clearly says live driver/client tokens are unavailable. If CUDA Toolkit or libcufile are missing, first point to https://developer.nvidia.com/cuda-downloads and the matching GDS packages before recommending gdscheck repair.

### Columns

| Column | Meaning |
|---|---|
| Native (nvidia-fs) | Direct GPU↔storage DMA. For NVMe-backed FSes (ext4, xfs) and Lustre/BeeGFS/NFS: via nvidia-fs kernel module (`nvfs` token) — for the latter three, `nvfs` is activated via each filesystem's own kernel-level RDMA transport, not the NVMe path. For GPFS/WekaFS: via cuFile userspace RDMA (`dmabuf`/`nvidia_peermem` tokens). |
| RDMA | Names the specific RDMA mechanism the filesystem's GDS route uses, if any — not a plain Yes/No, since these are distinct, non-interchangeable mechanisms and each is a prerequisite for Native GDS, not a standalone alternative to it. `dmabuf/peermem` is cuFile userspace RDMA (GPFS, WekaFS). `FS/kernel RDMA` covers Lustre (LNet), BeeGFS, NFS (NFSoRDMA), NVMe-oF (MLNX_OFED/DOCA), and ScaTeFS — all still require nvidia-fs (nvfs) to be loaded. In `--live` mode, `FS/kernel RDMA` only shows when the same row's Native column is also live-confirmed active, since gdscheck has no separate live token for kernel RDMA. |
| P2PDMA/C2C | Direct storage/NIC↔GPU P2P path when the GDS library supports that route. On x86 this is Linux PCI P2PDMA; on NVIDIA CPUs (e.g. Grace) with coherent C2C, the same `use_pci_p2pdma` keys enable the C2C direct path instead — and on those platforms, NVMe multipath is not a blocker for it, unlike on x86. Supported routes are NVMe, NVMe-oF, virtiofs, and RAID0. JSON settings can enable preference for those routes, but they do not create direct-P2P library support for NFS, GPFS, WekaFS, Lustre, BeeGFS, or compat-only filesystems. |
| Compat | CPU bounce buffer fallback. Available when `allow_compat_mode: true` is effective from cufile.json or `CUFILE_ALLOW_COMPAT_MODE`. |
| Compat Since | Earliest libcufile/GDS release required for the compat-only row. `1.16+` applies to squashfs/tmpfs/ramfs/overlayfs compatibility-path support. `1.17+` applies to ZFS/BTRFS compatibility-path support across all I/O APIs. |

### Filesystems shown

Grouped by whether the entry is a real filesystem namespace (a mounted,
POSIX-ish directory tree) vs. a device/transport/block path that something
else (ext4, xfs, ...) mounts on top of — not by physical medium or deployment
context.

| Group | Filesystems |
|---|---|
| Local file systems | ext4 on NVMe, xfs on NVMe |
| Remote file systems | lustre / DDN EXAScaler, gpfs, wekafs, beegfs, nfs, virtiofs, scatefs |
| Storage paths | NVMe-oF, RAID0 over NVMe, nvmesh, scsi, scaleflux |
| Compat-only | squashfs, tmpfs, ramfs, overlayfs, zfs, btrfs |

Notes:
- mmfs is the legacy mount type for GPFS; only `gpfs` is shown
- nfs4 is an alias for nfs (`FS_ALIASES` in `checks/fs_matrix.py` — no separate capability entry); the row displays as `nfs / nfs4`
- virtiofs is in Remote file systems because it's a real mounted namespace (inside a VM guest, backed by the host) — despite that, it has no native GDS, only P2PDMA (config-gated)
- scatefs (NEC's Scalable Technology File System — unrelated to ScaleFlux despite the similar name) has native GDS support via nvidia-fs, activated via kernel-level RDMA — same shape as Lustre and BeeGFS
- WekaFS native GDS uses cuFile userspace RDMA (dmabuf/nvidia_peermem) and shows ✓ Yes unconditionally. Writes default to the POSIX fallback path unless `fs.weka.rdma_write_support` is enabled in cufile.json (default false) — that's a config-gated capability, not a fixed WekaFS architectural limit, so it's checked by `mount-check`'s deeper per-path diagnostic rather than surfaced in this reference table.
- DDN EXAScaler is Lustre-based — identical GDS capabilities (nvidia-fs + LNet kernel RDMA, no P2PDMA)
- BeeGFS native GDS also depends on kernel-level RDMA — nvidia-fs (nvfs) has to be loaded to activate it, same as Lustre and NFS
- NVMe-oF: same native GDS and P2PDMA support as local NVMe; requires `block.nvmeof.use_pci_p2pdma=true` in cufile.json for P2PDMA (separate from `block.nvme`). Its native path also depends on kernel-level RDMA (MLNX_OFED/DOCA) — same shape as Lustre/BeeGFS/NFS/ScaTeFS, shown as `FS/kernel RDMA` in the RDMA column
- NVMe/NVMe-oF `nvidia-fs`/`nvfs` mode requires a GDS-patched NVMe/NVMe-oF stack. NVIDIA documents two ways to get those patches: MLNX_OFED, or DOCA/DOCA-OFED storage packages such as `mlnx-nvme-dkms` / `kmod-mlnx-nvme`. See the GDS troubleshooting guide's DOCA requirements and the DOCA-Host installation guide:
  - https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#doca-requirements-and-installation
  - https://docs.nvidia.com/doca/sdk/doca-host-installation-and-upgrade/index.html#storage-installation
- NFS's direct path is NFSoRDMA (`proto=rdma,port=20049`), a different mechanism than the NVMe `nvfs` path — but `nvidia-fs`/`nvfs` still has to be loaded to activate it, so NFS is Native-applicable, same as Lustre. Also requires MLNX_OFED/DOCA NFS-RDMA support, such as `mlnx-nfsrdma-dkms` where applicable. Verify the NFS/RDMA path, mount options, RDMA devices, and `gdscheck`/stats — don't assume the NVMe-style verdict logic applies unmodified.

### Live mode (--live)

`--live` runs `gdscheck -p` and parses the DRIVER CONFIGURATION section. **Live values replace the static column values in-place** — there is no separate "Live Status" column. If a filesystem's client is not loaded, all its cells show `?` with `(client not loaded)`.

gdscheck driver key → fs_type mapping:

| gdscheck key | fs_type(s) |
|---|---|
| NVMe | ext4, xfs |
| NVMeOF | nvme-of |
| SCSI | scsi |
| ScaleFlux CSD | scaleflux |
| NVMesh | nvmesh |
| DDN EXAScaler | lustre |
| NFS | nfs |
| Lustre | lustre |
| BeeGFS | beegfs |
| ScaTeFS | scatefs |
| WekaFS | wekafs |
| IBM Spectrum Scale | gpfs |
| VIRTIOFS | virtiofs |

Live detection logic per column:

| Column | Token(s) checked | Gating rule |
|---|---|---|
| Native | `nvfs` for NVMe-backed/lustre/beegfs/nfs; `dmabuf` or `nvidia_peermem` for gpfs/wekafs | AND with static matrix — token only counts if FS architecturally supports native GDS |
| RDMA | `dmabuf`/`nvidia_peermem` for `rdma_type=userspace` (gpfs/wekafs) | For `rdma_type=kernel` (lustre/beegfs/nfs), there's no separate live token — the RDMA cell instead mirrors that row's own live-confirmed Native verdict, since kernel RDMA isn't meaningful without nvidia-fs active |
| P2PDMA | `p2pdma` | AND with static matrix. If the live token is absent but the route is documented as config-gated, keep the config label instead of reporting a hard unsupported `No`. Suppresses false positives for gpfs/wekafs when `use_pci_p2pdma=true` globally in cufile.json |
| Compat | `compat` | Raw from gdscheck — reflects actual `allow_compat_mode` state |

compat-only FSes (squashfs, tmpfs, etc.) are not tracked by gdscheck — they always display static values.

`support-matrix` also reports the local libcufile version. Preferred sources:
1. `gdscheck -p` banner (`GDS release version: ...`)
2. `cuFileGetVersion()` from the selected `libcufile.so` (`1000 * major + 10 * minor`)
3. `libcufile.so.X.Y.Z` symlink/realpath for patch-level context

When the installed libcufile/GDS version is older than a row's documented
minimum, the compat cell displays `Need X.Y` instead of `✓ Yes`. Currently,
squashfs/tmpfs/ramfs/overlayfs require GDS/libcufile 1.16+, while ZFS and BTRFS
require GDS/libcufile 1.17 for the documented compatibility path across all I/O
APIs.

**GPFS P2PDMA false positive:** When `properties.use_pci_p2pdma=true` is set globally in cufile.json, gdscheck shows `IBM Spectrum Scale : p2pdma, compat` — but GPFS does not architecturally support P2PDMA. The static matrix gates this out. `cufile_config.run_all()` also emits a WARN when this condition is detected.

**Example output (--live, captured from a real GH200/Grace host with NVMe C2C active, an active NVMe-oF/RDMA target from an earlier test session, and no other clients loaded):**

```text
GDS Filesystem Support Matrix (aarch64)  [live — /usr/local/cuda/gds/tools/gdscheck]
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Storage route            Native (nvidia-fs)   RDMA             P2PDMA/C2C     Compat       Compat Since 
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Local file systems
  ext4 on NVMe             ✓ Yes                ✗ No             ✓ Yes          ✓ Yes
  xfs on NVMe              ✓ Yes                ✗ No             ✓ Yes          ✓ Yes

  Remote file systems
  lustre / DDN EXAScaler   ✗ No                 ✗ No             ✗ No           ✓ Yes
  gpfs                     ✗ No                 ✗ No             ✗ No           ✓ Yes
  wekafs                   ✗ No                 ✗ No             ✗ No           ✓ Yes
  beegfs                   ✗ No                 ✗ No             ✗ No           ✓ Yes
  nfs / nfs4               ✗ No                 ✗ No             ✗ No           ✓ Yes
  virtiofs                 ✗ No                 ✗ No             Config         ✓ Yes
  scatefs                  ✗ No                 ✗ No             ✗ No           ✓ Yes

  Storage paths
  NVMe-oF                  ✓ Yes                FS/kernel RDMA   Config         ✓ Yes
  RAID0 over NVMe          ✓ Yes                ✗ No             Config         ✓ Yes
  nvmesh                   ✗ No                 ✗ No             ✗ No           ✓ Yes
  scsi                     ✗ No                 ✗ No             ✗ No           ✓ Yes
  scaleflux                ✗ No                 ✗ No             ✗ No           ✓ Yes
  ...
```

Title includes `(aarch64)` because `--live`/`--static` now run one lightweight architecture check (`checks.iommu.arch()`) to drop irrelevant x86-only caveats — `ext4`/`xfs`/`raid0`/`nvme-of` all show plain `Config` here instead of `NoMP Config`/`Kernel Config`, since NVMe multipath and RAID0's kernel-version gate are both non-issues on this NVIDIA CPU. ext4/xfs also show `✓ Yes` for P2PDMA/C2C rather than a config-gated label, because gdscheck's live token for this host is literally `c2c, nvfs, compat` — the C2C route is confirmed active. `NVMe-oF` shows `✓ Yes` / `FS/kernel RDMA` — not `✗ No` like its Remote file systems neighbors — because gdscheck's live token for this host is literally `NVMeOF : nvfs, compat`: an NVMe-oF/RDMA target is genuinely active here (left over from an earlier live test on this same host), so both Native and RDMA are live-confirmed, not just architecturally possible. Every Remote file systems row and `nvmesh` show `✗ No` for RDMA/Native here because none of those clients are loaded on this host (gdscheck reports `compat` only for each) — not because they lack a route architecturally; see the `--static` counterpart for the capability question instead of the live one.

**JSON output** (same host, `ext4` and `gpfs` entries):
```json
{
  "live_status_available": true,
  "gdscheck_path": "/usr/local/cuda/gds/tools/gdscheck",
  "filesystems": [
    {
      "fs_type": "ext4",
      "display_name": "ext4 on NVMe",
      "category": "Local file systems",
      "alias_fs_types": [],
      "native": true,
      "p2pdma": true,
      "p2pdma_display": "NoMP Config",
      "p2pdma_note": "NVMe P2PDMA/C2C requires properties.use_pci_p2pdma=true, block.nvme.use_pci_p2pdma=true, and (on x86) NVMe multipath disabled unless the system has a specialized multipath patch; not required on NVIDIA CPUs, where multipath works fine alongside C2C.",
      "compat": true,
      "notes": "...",
      "live_status": "supported",
      "live_modes": { "native": true, "p2pdma": true, "c2c": true, "dmabuf_peermem": false, "compat": true },
      "live_raw": "c2c, nvfs, compat"
    },
    {
      "fs_type": "gpfs",
      "display_name": "gpfs",
      "category": "Remote file systems",
      "alias_fs_types": [],
      "native": true,
      "p2pdma": false,
      "compat": true,
      "notes": "...",
      "live_status": "compat_only",
      "live_modes": { "native": false, "p2pdma": false, "c2c": false, "dmabuf_peermem": false, "compat": true },
      "live_raw": "compat"
    }
  ],
  "legend": {
    "true": "Supported",
    "false": "Not supported",
    "config": "Requires matching cufile.json key and runtime confirmation",
    "NoMP Config": "NVMe P2PDMA/C2C requires matching cufile.json keys and NVMe multipath disabled on x86 — not required on NVIDIA CPUs, where multipath works fine alongside C2C",
    "Arch Config": "RAID0 P2PDMA/C2C requires NVIDIA Grace or Linux kernel >= 7.1 plus runtime confirmation"
  }
}
```

---

## subcommand: `pre-install`

**Purpose:** Validate that the system is GDS-capable *before* installing CUDA/GDS.
Checks hardware, kernel configuration, and existing filesystem mounts. Flags anything
that would block GDS from working after install.

**Does NOT require:** GDS, `gdscheck`, `nvidia_fs`, or `/etc/cufile.json`.
CUDA Toolkit and the NVIDIA driver are base prerequisites for a GDS-capable host,
so missing CUDA or missing driver state is reported as FAIL, not WARN.

**Human output:** Prints a `Runtime Versions` context block (Linux kernel,
CUDA Toolkit, NVIDIA driver from `modinfo`, nvidia-fs, libcufile), then
box-style sections for each check group. Normal mode shows only sections with
findings; `-v` shows every check including passing ones with evidence. The
report ends with a summary tally and a Mitigation Plan for every warning and
failure.

**What it checks:**

### System
- **OS**: `platform.system()` — FAIL if not Linux (GDS is Linux-only)
- **Architecture**: `platform.machine()` — x86_64 and aarch64 both PASS; unknown → WARN

### GPU
- **GPU presence**: `lspci` — detects NVIDIA 3D/VGA/Display/Processing controllers. Works without driver installed. WARN if lspci unavailable.
- **NVIDIA Open Driver install**: `modinfo nvidia` only. If absent or proprietary, FAIL with CUDA Downloads guidance and tell the user to select the Open Kernel Module / NVIDIA Open Driver option.
- **Deferred to post-install**: `nvidia-smi` driver version, GPU compute capability, CoherentGPUMemoryMode, and GPU/NVMe topology are intentionally not run in `pre-install`.

### CUDA Toolkit
- Check `nvcc` in PATH or `/usr/local/cuda*/bin/nvcc`
- FAIL if not found or only partially installed; mitigation links to the CUDA Downloads guide. Package-specific GDS checks are deferred to `post-install`.

### DOCA / MLNX_OFED
- Run an advisory `MLNX_OFED / DOCA` check using `ofed_info -s` and
  `doca_version`.
- PASS if MLNX_OFED or DOCA is detected.
- WARN, not FAIL, when neither is detected. This is not a universal GDS blocker
  because supported upstream PCI P2PDMA or compat routes may still be valid.
- The warning should explain that MLNX_OFED/DOCA is required or recommended for
  GPFS/WekaFS userspace RDMA, Lustre/NFS-RDMA, and NVMe/NVMe-oF
  nvidia-fs/nvfs deployments that need NVIDIA-provided storage-stack patches.
- Mitigation should link to the GDS DOCA requirements and DOCA storage
  installation docs, then tell the user to verify with
  `ofed_info -s || doca_version`.

### Kernel
- `CONFIG_PCI_P2PDMA` compiled in — checked via `/proc/kallsyms` (symbol
  `pci_p2pdma_add_resource` present, even at address `0000...`), not kernel version
- Missing PCI P2PDMA support is a route limitation, not a universal GDS
  installation blocker. In `pre-install`, consolidate the underlying P2PDMA
  kernel failures into one WARN that recommends the nvidia-fs/nvfs route for
  NVMe/NVMe-oF when the required MLNX_OFED/DOCA storage-stack patches are
  available. Recommend changing kernels only when the deployment specifically
  requires upstream PCI P2PDMA. Keep the underlying checks as hard failures
  when evaluating the P2PDMA route directly.
- Kernel version (informational)

### IOMMU
- Read `/proc/cmdline` for `intel_iommu=`, `amd_iommu=`, `iommu=`
- PASS: passthrough (`iommu=pt`) or disabled
- WARN: strict mode for P2PDMA only when local NVMe PCI devices are present
  (strict IOMMU may restrict GPU/NVMe P2PDMA depending on hardware)
- Recommend the CPU-vendor-specific passthrough setting from `/proc/cpuinfo`:
  `intel_iommu=on iommu=pt` on Intel or `amd_iommu=on iommu=pt` on AMD.
  Show both only when the x86 CPU vendor cannot be detected.

### PCIe / ACS
- Run `lspci -vvv` and check for ACS P2P Request Redirect enabled on PCIe switches
- Flag any switch with `ACSCtl: SrcValid+ ... ReqRedir+`
- Show BDF and fix options (pci=noacs, BIOS, or per-device setpci)
- Requires `lspci` (pciutils package)
- Do not run GPU/NVMe topology here; `nvidia-smi topo -m -nvme` belongs to
  `post-install` and path-specific `mount-check`.

### Mounted filesystems
- Read `/proc/mounts`, skip kernel pseudo-FSes
- Suppress runtime/system mounts such as `/dev/shm`, `/run/*`, tmpfs, and
  ramfs from normal host-readiness output because they are not GDS workload
  storage targets
- For each plausible workload storage mount: show filesystem type and GDS mode
  support from the static matrix (same data as `--support-matrix` but filtered
  to what's mounted)
- Flag ext4 mounts that are not explicitly mounted with `data=ordered`.
  The implicit ext4 default is not sufficient for GDS readiness reporting.
  For non-root ext4 filesystems, recommend `mount -o remount,data=ordered`
  and persistent `/etc/fstab` options. For the root filesystem, recommend
  `rootflags=data=ordered` in `GRUB_CMDLINE_LINUX`, followed by GRUB
  regeneration and reboot. Treat this as a mount-specific WARN in
  `pre-install`; use `mount-check PATH` after GDS installation for an
  authoritative path diagnosis.
- Probe O_DIRECT support by attempting `open(..., O_DIRECT)` on a temp file
  for candidate storage mounts. Treat failures as mount-specific WARN findings
  because arbitrary mounted filesystems are not necessarily intended for GDS
  workloads.

**JSON output:**
```json
{
  "mode": "pre-install",
  "version_context": [
    { "component": "CUDA Toolkit", "version": "13.1" },
    { "component": "NVIDIA driver", "version": "580.65.06" },
    { "component": "nvidia-fs", "version": "2.26.6" },
    { "component": "libcufile", "version": "1.16.1" }
  ],
  "summary": { "pass": 4, "info": 1, "warn": 1, "fail": 1 },
  "checks": {
    "kernel": [
      { "check": "CONFIG_PCI_P2PDMA", "status": "PASS", "why": "...", "evidence": "..." }
    ],
    "iommu": [ ... ],
    "pcie_acs": [ ... ],
    "gpu": [ ... ],
    "mounts": [
      {
        "mountpoint": "/mnt/nvme0",
        "fstype": "ext4",
        "device": "/dev/nvme0n1p1",
        "gds_support": { "native": true, "p2pdma": "config", "rdma": false, "compat": true },
        "checks": [ ... ]
      }
    ]
  }
}
```

---

## subcommand: `post-install`

**Purpose:** Confirm that GDS was installed correctly and is configured properly.
System-wide, no path needed. The bridge between "hardware is capable" (pre-install)
and "this workload can use GDS" (mount-check).

**Requires:** CUDA Toolkit plus matching GDS packages. `gdscheck` comes from
`gds-tools-*`; the nvfs kernel route is installed through `nvidia-gds`, with
`nvidia-fs-dkms` providing the `nvidia_fs` kernel module.

**Human output:** Prints a `Runtime Versions` context block (Linux kernel,
CUDA Toolkit, NVIDIA driver from `nvidia-smi` with `modinfo` fallback,
nvidia-fs, libcufile), then box-style sections for each check group. Normal
mode shows only sections with findings; `-v` shows every check with evidence.
When base prerequisites are missing the command renders a remediation table
and stops before checking GDS packages. The report ends with a summary tally
and a Mitigation Plan for all warnings and failures.

**What it checks:**

### Prerequisites
- CUDA Toolkit is present. If missing, fail immediately with CUDA install
  guidance and do not continue into GDS package/module checks.
- NVIDIA GPU is visible through PCI discovery. If no GPU is visible, fail the
  post-install prerequisite gate because GPU-memory GDS cannot be validated.
- NVIDIA driver is present and healthy enough for `nvidia-smi` to report a
  driver version. If missing, fail once as `NVIDIA driver installed` with CUDA
  Downloads guidance and Open Kernel Module / NVIDIA Open Driver selection.
- NVIDIA Open Kernel Driver is installed. If the proprietary driver is detected,
  fail immediately and ask the user to install/switch to the Open Kernel Driver,
  reboot if needed, then rerun `post-install`.
- If CUDA Toolkit is installed but no NVIDIA GPU and/or working NVIDIA driver is
  present, add an explicit warning that only system-memory buffers are supported
  on that host. GPU-memory buffers are not supported with libcufile until an
  NVIDIA GPU and working NVIDIA driver are present.
- When this gate fails, render a remediation table before any detailed findings:
  `Component`, `Status`, `Missing / Issue`, `Mitigation`, `Docs`, and
  `Recommendation`. Use doc references for CUDA downloads, Open Kernel Module /
  NVIDIA Open Driver selection, and the GDS troubleshooting guide. In JSON mode,
  include the same rows under `prerequisite_remediation`.

### Installation
- `gdscheck` binary present at `/usr/local/cuda/gds/tools/gdscheck`
- `nvidia_fs` module: loaded (`lsmod`) or at least installed (`modinfo`).
  Missing or unloaded `nvidia_fs` is WARN, not FAIL, because it blocks the nvfs
  route but alternate GDS routes such as upstream PCI P2PDMA, userspace RDMA,
  or compat may still be valid for some filesystems/block devices.
- NVIDIA Open Driver installed — from `gdscheck -p` PLATFORM INFO:
  `Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)`

### gdscheck -p summary
- Run `gdscheck -p` and parse DRIVER CONFIGURATION section
- Show which filesystem types have which modes active on this system:
  `NVMe : nvfs, compat`, `Lustre : nvfs, compat`, etc.
- Show `p2pdma` tokens as active upstream PCI P2PDMA routes, separate from
  `nvfs`/nvidia-fs routes.
- Surface any ERROR or WARNING lines from gdscheck output

### P2PDMA/C2C Direct Routes
- Report whether the running kernel exposes PCI P2PDMA support using
  `/proc/kallsyms` (`p2pdma_pgmap_ops` or `pci_p2pdma_add_resource`) and
  `CONFIG_PCI_P2PDMA`.
- Report route-level `cufile.json` P2PDMA preferences for local NVMe, NVMe-oF,
  virtiofs, and RAID0. A P2PDMA/C2C route is enabled only when both
  `properties.use_pci_p2pdma=true` and the matching `block.*` / `fs.*` key are
  true, but that is still only a preference; active routes are confirmed from
  `gdscheck` DRIVER CONFIGURATION.
- On GH/GB ARM platforms with coherent C2C, the historical
  `use_pci_p2pdma` key names still gate the C2C path; confirm active runtime
  routing through `gdscheck` tokens (`p2pdma` on x86, `c2c` on C2C platforms).
- For ext4/XFS mounts on `/dev/nvme*`, inspect `/sys/class/nvme/nvmeX/transport`
  before choosing the cufile key: local PCIe NVMe uses `block.nvme.*`, while
  NVMe-oF transports such as RDMA/TCP/FC use `block.nvmeof.*`.
- Report local GPU/NVMe topology candidates and ACS redirect status when the
  devices are visible. Prefer `nvidia-smi topo -m -nvme` for GPU/NVMe
  placement recommendations; use PCI BDF sysfs analysis as a fallback or
  secondary check. Normalize `nvidia-smi --query-gpu=pci.bus_id` output from
  the common 8-hex-digit domain form (`00000000:bb:dd.f`) to Linux sysfs BDF
  form (`0000:bb:dd.f`) before deciding that no GPUs were discovered.
- Keep missing P2PDMA as a warning in post-install because it is an alternate
  direct path; `nvidia-fs`/`nvfs`, RDMA, and compat can still be valid routes.

### cufile.json
- File present at `/etc/cufile.json` (or `CUFILE_ENV_PATH_JSON`)
- Parseable (no JSON syntax errors — file is JSONC, strip comments first)
- Key settings validated:
  - `properties.force_compat_mode` — WARN if true (disables all GDS acceleration)
  - `properties.allow_compat_mode` — WARN if false (no fallback path)
  - `properties.use_pci_p2pdma` — summarized in the P2PDMA/C2C Direct Routes section
  - `logging.level` — suggest DEBUG if diagnosing issues

**JSON output:**
```json
{
  "mode": "post-install",
  "version_context": [
    { "component": "CUDA Toolkit", "version": "13.1" },
    { "component": "NVIDIA driver", "version": "580.65.06" },
    { "component": "nvidia-fs", "version": "2.26.6" },
    { "component": "libcufile", "version": "1.16.1" }
  ],
  "summary": { "pass": 3, "info": 1, "warn": 1, "fail": 0 },
  "checks": {
    "installation": [
      { "check": "gdscheck binary", "status": "PASS", "evidence": "/usr/local/cuda/gds/tools/gdscheck" },
      { "check": "nvidia_fs module", "status": "PASS", "why": "loaded, version 2.17.5" },
      { "check": "NVIDIA Open Driver", "status": "PASS", "evidence": "Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)" }
    ],
    "gdscheck_summary": [
      { "driver_key": "NVMe", "modes": "nvfs, compat" },
      { "driver_key": "Lustre", "modes": "nvfs, compat" }
    ],
    "cufile_config": [
      { "check": "cufile.json present", "status": "PASS", "evidence": "/etc/cufile.json" },
      { "check": "force_compat_mode", "status": "PASS", "why": "false — GDS acceleration enabled" },
      { "check": "allow_compat_mode", "status": "WARN", "why": "false — CPU bounce fallback disabled" }
    ]
  }
}
```

---

## subcommand: `config-audit`

**Purpose:** Audit effective cuFile runtime configuration without changing the
host. This reads an explicit `--config PATH` when provided; otherwise it prefers
the `CUFILE CONFIGURATION` section from `gdscheck -p`, because that is the
installed library/runtime view and includes all effective properties exposed by
the local GDS stack. Some GDS releases omit file-only keys such as
`logging.level` from `gdscheck -p`; when the gdscheck source is selected, fill
missing known keys from the installed cufile.json search path and label those
rows as file fallback. If gdscheck is missing or cannot expose that section, it
falls back fully to `CUFILE_ENV_PATH_JSON` when set, then `/etc/cufile.json`,
then the CUDA template fallback. It applies relevant environment overrides by
default and reports the effective value, source, default, risk, and recommended
action for each known setting. `--ignore-env` suppresses additional `CUFILE_*`
environment overlays in the report.

**Why:** Many GDS failures are caused by a small number of runtime toggles:
compat mode, P2PDMA preference, per-filesystem P2PDMA keys, RDMA device lists,
dynamic routing, logging/profiling, and buffer sizing. Operators need to know
which value is effective, not only what appears in `/etc/cufile.json`.

**Flags:**

`--profile {local-nvme,nvmeof,lustre,beegfs,scatefs,wekafs,gpfs,nfs-rdma,virtiofs,raid0,compat-safe}`
filters the output to keys relevant to that deployment type.

`--config PATH` scans a specific `cufile.json`/JSONC file instead of live
`gdscheck` or the default search path. This is useful for proposed configs,
container configs, and remote copies before replacing `/etc/cufile.json`.

`--ignore-env` suppresses additional `CUFILE_*` environment overlays. Use
`--config PATH --ignore-env` for strict file-only validation. When `gdscheck`
is used as the source, this also runs `gdscheck -p` without inherited
`CUFILE_*` variables so the runtime dump is not environment-tainted.

**Checked setting families:**

- `logging.*` and `profile.*`
- `properties.force_compat_mode`, `properties.allow_compat_mode` /
  `properties.use_compat_mode`, `properties.use_pci_p2pdma`,
  `properties.gds_rdma_write_support`, `properties.force_odirect_mode`,
  `properties.prefer_iouring`, `properties.io_batchsize`, and
  `properties.io_priority`
- GPU/CPU bounce buffer and batch sizing, including slab-array shape and the
  documented `max_device_cache_size_kb / per_buffer_cache_size_kb >= io_batchsize`
  relationship. Treat `properties.gpu_bounce_buffer_slab_config.slab_size_kb`
  and `properties.gpu_bounce_buffer_slab_config.slab_count` as valid nested
  slab keys. Also treat the flattened `properties.gpu_bounce_buffer_slab_size_kb`
  and `properties.gpu_bounce_buffer_slab_count` keys printed by `gdscheck -p`
  as valid. Validate that slab arrays have the same length, use positive values,
  and keep slab sizes 4 KB aligned and ascending. Emit INFO when
  `properties.max_direct_io_size_kb` is below 16384 KB because it can limit
  large-IO throughput. When GPU memory is observable, emit INFO if
  `properties.max_device_pinned_mem_size_kb` is below the observed GPU memory
  size because large buffer registrations can hit the configured cap.
- `sparse.*` P2P/sparse IO thresholds
- static topology routing for GDS 1.17+: canonical keys are
  `miscellaneous.enable_static_routing` and
  `miscellaneous.static_routing_filepath`; older `sparse.*` names are accepted
  as aliases for compatibility. Warn when static routing is enabled but the
  topology file path is missing, the path is not a regular file, or the file is
  empty or unreadable.
- RDMA device lists, peer type, load balancing, and dynamic routing. For
  `rdma_dev_addr_list`, validate that configured entries are client-side IPv4
  addresses assigned to the current host; when interface mapping is available,
  prefer addresses on netdevs associated with `/sys/class/infiniband/*/device/net`.
  Do not emit an INFO finding for an empty global
  `properties.rdma_dev_addr_list` outside WekaFS/GPFS profiles; Lustre and NFS
  should only be flagged when configured values are malformed, non-local, or
  there is a concrete dynamic-routing issue.
  For GPFS, `fs.gpfs.rdma_dev_addr_list` is optional when either
  `fs.gpfs.mount_table` or global `properties.rdma_dev_addr_list` provides the
  client RDMA addresses; cuFile resolves GPFS mount addresses in that order.
  For WekaFS/GPFS, treat `[]`, `[""]`, and blank strings as empty. Emit INFO
  when the global RDMA list is explicitly empty, and emit INFO when the
  profile-specific RDMA list is empty and no global list or mount table provides
  a usable explicit RDMA address source.
- GPFS capability toggles. In the GPFS profile, emit INFO when
  `fs.gpfs.gds_write_support=false` because GPFS writes will not use the GDS
  write path, and emit INFO when `fs.gpfs.gds_async_support=false` because
  cuFile async APIs will not use GPFS async GDS support.
- `properties.rdma_topN_ranks`, which controls how many distinct nearest
  GPU/NIC distance ranks are eligible for the RDMA load-balancing policy
- `fs.*` overrides for Lustre, NFS, WekaFS, GPFS, BeeGFS, ScaTeFS, and virtiofs
- `block.nvme.*`, `block.nvmeof.*`, and `block.raid.*`
- cross-field P2PDMA/C2C consistency between `properties.use_pci_p2pdma` and the
  relevant `block.*` / `fs.*` config key, with a hard GDS-library allowlist of
  NVMe, NVMe-oF, virtiofs, and RAID0. For P2PDMA-capable profiles, emit
  INFO when both the global and route-level keys are off so users can see that
  direct P2P is not configured even though nvfs or compat may still be valid.
  On x86 this direct route is upstream PCI P2PDMA; on GH/GB ARM platforms
  the same keys enable C2C when the route is supported.
- schema validation for type, known enum values, integer ranges/alignment,
  duplicate dynamic-routing policies, mount-table shape, dynamic routing without
  RDMA address configuration, and unknown keys that may be typos or newer
  release/vendor properties
- Newer template/runtime keys including `properties.vanilla_posix_io_mode`,
  `properties.gds_fallback_io`, `properties.rdma_transport_type`,
  `properties.allow_rdma_token_reset`,
  `properties.compat_odirect_unaligned_read_split`,
  `properties.compat_odirect_unaligned_read_split_min_size_kb`,
  `block.raid1.use_pci_p2pdma`, `block.raid10.use_pci_p2pdma`, and
  `miscellaneous.rdma_token_reset_timeout_secs`
- `denylist.*` / legacy `blacklist.*`
- `miscellaneous.*` and `execution.*`
- execution/threadpool settings: emit WARN when `execution.parallel_io=false`
  or `execution.max_request_parallelism=0`, because those settings disable
  parallel request processing.

**JSON output:**

```json
{
  "config_path": "/etc/cufile.json",
  "config_source": "file",
  "config_source_detail": "cufile.json fallback after gdscheck",
  "requested_config_path": null,
  "env_config_path": null,
  "env_overrides_applied": true,
  "gdscheck_path": "/usr/local/cuda/gds/tools/gdscheck",
  "gdscheck_error": "gdscheck -p did not include a CUFILE CONFIGURATION section",
  "gdscheck_fallback": true,
  "parse_error": null,
  "profile": "local-nvme",
  "entries": [
    {
      "path": "properties.use_pci_p2pdma",
      "resolved_key": "properties.use_pci_p2pdma",
      "value": true,
      "default": false,
      "source": "file",
      "status": "OK",
      "risk": "",
      "recommendation": "",
      "scope": "data-path",
      "description": "Prefer direct P2P mode when supported by the GDS library route; this is PCI P2PDMA on x86 and C2C on supported Grace/Blackwell ARM platforms."
    }
  ]
}
```

---

## subcommand: `mount-check <path>`

**Purpose:** Deep, path-specific GDS diagnostic. Given a real directory, determine
exactly which GDS modes are available for I/O at that path, explain any blockers,
and provide performance recommendations based on hardware topology.

**Requires:** CUDA toolkit + GDS installed. A real, accessible path.

**What it checks:**

### Filesystem detection
- Resolve real path, match against `/proc/mounts` (longest-prefix)
- Detect filesystem type, backing device, mount options
- Look up GDS capability from `FS_CAPABILITIES`

### Per-mode checks (from `checks/gds_report.py`)
All checks from `build_mode_reports()` apply here:

- **Native GDS:** nvidia_fs loaded, driver version, GPU compute cap, IOMMU,
  kernel-log nvidia_fs messages, Open Driver, ext4 data mode, O_DIRECT probe
- **P2PDMA:** kernel P2PDMA support, IOMMU, ACS, cufile.json settings
  (route keys: `block.nvme`, `block.nvmeof`, `fs.virtiofs`, `block.raid`),
  Open Driver
- **RDMA:** MLNX_OFED/DOCA, nvidia_peermem or DmaBuf, rdma_dev_addr_list,
  IB/RoCE link state, NFS rdma mount option
- **Compat:** allow_compat_mode in cufile.json

### Performance recommendations (new)
- PCIe topology: GPU ↔ NVMe distance (same root complex = optimal; cross-root-port paths can still work but are expected to be less performant)
- GPU/NVMe placement from `nvidia-smi topo -m -nvme`: recommend the closest
  NVMe device for each GPU, and when the mount's NVMe label is identifiable,
  recommend the closest GPU(s) for that mount. Match the mount's `/dev/nvme*`
  device to topo labels through the NVMe PCI BDF or any device names printed in
  the topology legend; if no reliable mapping is exposed, show best NVMe per
  GPU instead of guessing. Support both NVIDIA topo layouts: a single combined
  GPU/NVMe matrix, and the newer split output where the first matrix contains
  GPU/NIC topology and a second NVMe-only table appears after `NIC Legend`.
  Treat relation tokens such as `X`, `PHB`, `PIX`, `PXB`, `NODE`, `SYS`, and
  `NV#` as data cells even when rows are space-indented for alignment. Some
  releases print a second `Legend:` and an `NVMe Legend:` with `NVMe0: nvme0n1`
  after the NVMe matrix; the parser must still treat the preceding `GPU0 PHB`
  row as the authoritative placement table and use the legend only for label
  to device-name mapping. Strip ANSI terminal styling first because
  `nvidia-smi` may underline table headers even when output is captured.
- NUMA: GPU and NVMe NUMA node alignment
- NVMe queue depth (for P2PDMA): `cat /sys/block/nvmeXnY/queue/nr_requests`
- For network FS: MTU, RDMA link speed, number of active IB links
- For GPFS/WekaFS only: parse `nvidia-smi topo -m` GPU/NIC distance and
  recommend the cuFile RDMA policy from actual topology. The cuFile source
  treats `properties.rdma_load_balancing_policy` as the RDMA device selection
  policy over the K-nearest GPU/NIC rank table:
  - `FirstFit`: always picks the first K-nearest RDMA device. This is only
    appropriate for deliberate static pinning or a one-GPU/one-NIC host.
  - `MaxMinFit`: builds a least-shared one-to-one assignment. This is useful
    when GPU and NIC counts are comparable and workload placement is uniform.
  - `RoundRobin`: rotates within each GPU's K-nearest NIC set. Keep this for
    one GPU, one NIC, or simple equivalent-NIC topologies.
  - `RoundRobinMaxMin`: builds the least-shared table and round-robins within
    it. Prefer this for multi-GPU/multi-NIC GPFS/WekaFS hosts, especially when
    GPUs share nearest NICs or GPU:NIC count is uneven.
  - `Randomized`: experiment/debug only because run-to-run NIC selection varies.
  The checker also evaluates `properties.rdma_dynamic_routing` for GPFS/WekaFS.
  Dynamic routing is recommended when a NIC has only a subset of GPUs at the
  best distance; cuFile then uses `rdma_dynamic_routing_order`
  (`GPU_MEM_NVLINKS`, `GPU_MEM`, `SYS_MEM`, `P2P` by default) to select a route
  for IOs issued from non-local GPUs. It is not useful on a single-GPU host.

**JSON output:** Same structure as existing `mount-check --json` output:
```json
{
  "mode": "mount-check",
  "path": "/mnt/nvme0",
  "filesystem": "ext4",
  "device": "/dev/nvme0n1",
  "modes": [
    {
      "mode": "Native GDS (nvidia-fs)",
      "applicable": true,
      "status": "PASS",
      "checks": [
        { "check": "nvidia-fs module", "status": "PASS", "why": "...", "evidence": "..." }
      ]
    },
    ...
  ],
  "performance": {
    "pcie_topology": "GPU (0000:01:00.0) and NVMe (0000:02:00.0) share root complex — optimal",
    "numa_alignment": "GPU NUMA 0, NVMe NUMA 0 — aligned",
    "nvme_queue_depth": 1024
  }
}
```

### NVMe / NVMe-oF direct-path rule

For ext4, XFS, and NVMe-oF paths, direct GDS has two valid routes:

- upstream Linux PCI P2PDMA (`p2pdma` token in `gdscheck`)
- `nvidia-fs` / `nvfs`, when the NVMe or NVMe-oF stack has the necessary GDS
  patches supplied by MLNX_OFED or DOCA/DOCA-OFED storage packages

P2PDMA takes precedence when both are present. A failed P2PDMA check is not a
blocking mount-check failure if the nvfs route is active, and a failed nvfs
check is not blocking if P2PDMA is active.

`mount-check` must distinguish local prerequisites from active runtime routes.
If nvidia-fs prerequisites pass but `gdscheck -p` shows the NVMe/NVMe-oF driver
line without an `nvfs`/native token, report native mode as WARN/Not active and
explain that NVMe/NVMe-oF nvidia-fs mode needs the GDS-patched storage stack
from MLNX_OFED or DOCA/DOCA-OFED. Include the DOCA/GDS documentation links in
the mitigation plan. If P2PDMA/C2C is active, the inactive nvfs route should be
INFO rather than WARN.

For NFS, the direct path is NFSoRDMA — a different mechanism than the NVMe
`nvfs` path, but `nvidia-fs`/`nvfs` still has to be loaded to activate it, so
NFS is treated as Native-applicable the same way Lustre is. The server and
client need MLNX_OFED/DOCA RDMA support, the RDMA kernel modules, and an RDMA
mount such as `proto=rdma,port=20049`. DOCA storage packages include
`mlnx-nfsrdma-*` packages on supported distributions. If `gdscheck` only shows
`NFS : compat`, the NFS direct/RDMA path is not active.

---

## Deferred subcommands

These ideas are intentionally out of the initial supported CLI surface. Keep
the first testing pass focused on the robust host, path, support-matrix, and
configuration flows.

### Log explanation

Goal: parse selected `cufile.log` snippets after a workload run and correlate
fallback or error lines with `mount-check`, `config-audit`, and runtime support
evidence. A future implementation should be more than a loose regex scanner:
it should produce findings that line up with the same mode and mitigation model
used by the rest of the toolkit.

### Application validators

Goal: allow application-specific validators to compose `post-install`,
`mount-check`, and config checks for known deployment layouts. Dynamo KVBM was
the first proposed example, but application validators should wait until the
core host/path/config flows and their JSON contracts are stable.

---

## Common CLI

```text
usage: gds-diag.py <command> [--json] [-v] [command-specific args...]
       gds-diag.py --version

commands:
    all [path]
    support-matrix
    pre-install
    post-install
    config-audit
    mount-check <path>
```

- `--json` — machine-readable JSON output (every subcommand)
- `-v` / `--verbose` — show passing checks alongside failures (every subcommand)
- Exactly one subcommand must be given, except for the global `--version` flag,
  which prints the tool version and Git commit without a subcommand

---

## subcommand: `all [path]`

**Purpose:** General diagnostic starting point. It detects whether `gds-diag`
is running on the host or inside a container, prints the planned sequence, runs
the appropriate existing subcommands in order, and stops at the first
unsuccessful return code. If `path` is omitted, it uses the current directory
for the final `mount-check`.

Host sequence:

1. `pre-install`
2. `post-install`
3. `config-audit`
4. `mount-check PATH`

Container sequence:

1. `container-check`
2. `mount-check PATH`

`all` relies on deterministic child return codes for control flow; it does not
parse child human output to decide whether to continue.

**JSON output:** `all --json [path]` runs each child command in JSON mode,
captures and parses each child JSON document, and emits one combined JSON
object with tool metadata, environment detection, planned sequence, child
results, the first failed command in `stopped_at`, and the returned `exit_code`.

---

## Adding a new filesystem

`checks/fs_matrix.py` → `FS_CAPABILITIES` is the entry point, but four
locations must be kept in sync:

### 1. `checks/fs_matrix.py` — capability declaration (required)

Add an entry to `FS_CAPABILITIES` keyed by the kernel fs_type string (e.g.
`"myfs"`).  Required fields:

| Field | Type | Meaning |
|---|---|---|
| `native` | bool / None | nvidia-fs nvfs path supported |
| `p2pdma` | bool / `"config"` | P2PDMA/C2C path supported (or config-gated) |
| `rdma` | bool | RDMA path supported |
| `rdma_type` | str | `"userspace"` or `"kernel"` — only when `rdma=True` |
| `compat` | bool | cuFile compat (CPU bounce) path available |
| `notes` | str | Human-readable summary for `support-matrix` and AI skill |

If the FS is a known alternate name for an existing type (e.g. a legacy mount
type string), add it to `FS_ALIASES` instead of creating a duplicate entry.

### 2. `checks/gds_report.py` — `_FS_DRIVER_KEYS` dict (required)

Map the fs_type string to the label(s) that appear under `DRIVER
CONFIGURATION` in `gdscheck -p` output.  Example:

```python
"myfs": ("MyFS",),
```

Without this entry, `gdscheck` verdicts are silently ignored for the new FS
and live mode in `support-matrix` will show `?` for all columns.

If the FS requires a unique runtime check (e.g. a required mount option, a
write-path warning), add it in `build_mode_reports()` with an explicit
`if fs_type == "myfs":` branch alongside the existing ext4 and NFS blocks.

### 3. `checks/cufile_config.py` — `CONFIG_SCHEMA` list (if the FS has `fs.*` keys)

Add one schema dict per `fs.<name>.*` cufile.json key the FS supports.
Tag each entry with `"profiles": ["myfs"]` so `config-audit --profile myfs`
surfaces only the relevant keys.  Follow the pattern of the existing
`fs.gpfs.*` or `fs.lustre.*` entries.

### 4. `subcommands/config_audit.py` — `_PROFILES` list (if step 3 was done)

Add `"myfs"` to `_PROFILES` so it is selectable via
`gds-diag config-audit --profile myfs`.

If the FS uses cuFile userspace RDMA (`rdma_type="userspace"` — same path as
GPFS and WekaFS), also add it to `RDMA_POLICY_FS_TYPES` in `checks/rdma.py`
so RDMA policy recommendations are generated for it.

---

## Adding a new subcommand

The dispatcher discovers subcommands by importing every non-underscore
module in the `subcommands/` package and reading its module-level
`COMMAND` attribute. To add a new subcommand:

1. Create `subcommands/<your_name>.py`.
2. Subclass `subcommands._base.Subcommand`, set `name`, `help`,
   `description`, and `order` (lower sorts earlier in the top-level
   `--help` list; in-tree subcommands are spaced 10 apart so plugins can
   slot between them), and implement `add_arguments` and `run`.
3. At module bottom, expose `COMMAND = YourCommand()`.

No edits to `gds-diag.py` or any central registry are required — the
new subcommand appears in `--help` automatically. The CLI name is
whatever you put in `name` (use hyphens; the underscore-named module
file is fine). Common flags (`--json`, `-v/--verbose`) are added by the
dispatcher; do not redefine them in `add_arguments`.

This keeps the generic install/diagnose commands small and independently
testable.

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All checks pass |
| 1 | At least one FAIL |
| 2 | Environment not ready (e.g. CUDA not installed when required) |
| 3 | Bad arguments / usage error |
