# GDS Diag

gds-diag is a filesystem-aware GPUDirect Storage (GDS) diagnostic toolkit. It can be used to assess bare metal hosts, virtual machines, or containers to identify any GDS blockers or sub-optimal configurations and suggest exact mitigations for them.

gds-diag is structured as a single script (`gds-diag.py`) with subcommands for the following tasks:

| Subcommand | Purpose |
|---|---|
| `all` | Run the entire recommended diagnostic sequence (good starting point) |
| `support-matrix` | Show the GDS filesystem support matrix |
| `pre-install` | Check host readiness before CUDA/GDS is installed |
| `post-install` | Validate the installed GDS runtime and version stack |
| `mount-check` | Diagnose GDS mode availability for a specific filesystem path |
| `config-audit` | Audit the cuFile configuration |
| `container-check` | Validate GDS visibility from inside a container |

If you are working inside Claude Code or Codex, the built-in AI skill can
route your question to the right subcommand and interpret the results for
you. See [Agent Skill](#agent-skill) below.

## Requirements

- Linux with Python 3.8+ — no third-party packages required
- CUDA Toolkit and GDS packages for most subcommands; `sudo` for deeper diagnostics
- Validated for use on CUDA 12.2 (GDS 1.7.2) and newer — see `tests/container_matrix/README.md` for the versions tested

`pre-install` has no GDS dependency and is designed to run before GDS is installed.
`support-matrix` and `config-audit` work without a GPU present.

## Quick Start

```bash
# Run the recommended general diagnostic sequence for this host or container
python3 gds-diag.py all

# Optionally pass a path for the mount-check step (defaults to current directory)
python3 gds-diag.py all /mnt/nvme0

# Check a specific path
python3 gds-diag.py mount-check /mnt/nvme0

# Check host readiness before CUDA/GDS is installed
python3 gds-diag.py pre-install

# Validate the installed GDS runtime
python3 gds-diag.py post-install

# Audit effective cuFile config
python3 gds-diag.py config-audit --profile local-nvme

# Validate a proposed cufile.json without applying local env overrides
python3 gds-diag.py config-audit --config ./cufile.json --ignore-env -v

# Show filesystem GDS support matrix (auto-detects gdscheck when available)
python3 gds-diag.py support-matrix

# JSON output for automation
python3 gds-diag.py mount-check /mnt/nvme0 --json
```

Get help on any subcommand:

```bash
python3 gds-diag.py <subcommand> --help
```

## GDS Modes Checked

| Mode | Description |
|------|-------------|
| **Native GDS (nvfs)** | Direct GPU↔storage DMA via `nvidia-fs` kernel module |
| **P2P DMA (P2PDMA/C2C)** | Direct P2P path between a supported storage/NIC path and GPU. On x86 this is PCIe P2PDMA; on supported Grace Hopper and Grace Blackwell ARM platforms it may be C2C. |
| **RDMA** | GPU↔network-storage via InfiniBand/RoCE |
| **Compat** | CPU bounce buffer fallback |

## Container Validation

Use `container-check` from inside a Docker or Enroot container to validate
whether GDS runtime files, device nodes, `/run/udev`, and diagnostic tools are
visible from that container. It is a container-focused companion to
`post-install`, `support-matrix --live`, and `mount-check`.

Docker example:

```bash
docker run --rm --gpus=all \
  -v "$PWD:/work:ro" -w /work \
  --entrypoint python3 \
  <image> \
  gds-diag.py container-check -v
```

Enroot example:

```bash
enroot start --root \
  -m "$PWD:/work:none:x-create=dir,rbind,ro:0:0" \
  <container-or-image> \
  python3 /work/gds-diag.py container-check -v
```

If container auto-detection is ambiguous, pass `--runtime docker` or
`--runtime enroot`.

## Agent Skill

This repository includes a shared GDS diagnostic skill for Claude Code and
Codex. The skill does not duplicate the deterministic checks; it helps an agent
choose the right `gds-diag.py` subcommand and interpret the results.

The canonical skill lives at:

```text
skills/gds-diag/SKILL.md
```

Repo-local discovery paths are provided for both agents:

- Codex: `.agents/skills/gds-diag`
- Claude Code: `.claude/skills/gds-diag`

From the repository root, start either agent:

```bash
cd /path/to/gds-diag
codex
# or
claude
```

No separate skill install is needed when working from this checkout. The
repo-local discovery paths let the agent find the shared skill, and the skill
routes requests to the deterministic Python CLI.

Ask naturally, or name the skill explicitly on the first request:

```text
Use the gds-diag skill to collect a general diagnostic report for this host.
Use the gds-diag skill to check whether this host is ready for GDS installation.
Use the gds-diag skill to diagnose /mnt/nvme0.
Use the gds-diag skill to audit ./cufile.json as a proposed config.
```

## Contents

| File / Directory | Description |
|-----------------|-------------|
| `gds-diag.py` | Main diagnostic CLI — checks GDS mode availability, configuration, and install readiness |
| `checks/` | Check modules: kernel, IOMMU, ACS, PCIe, nvidia-fs, cufile.json, RDMA, filesystem matrix |
| `subcommands/` | CLI subcommands for host readiness, runtime validation, mount checks, config audits, and support matrices |
| `skills/gds-diag/` | Shared Claude/Codex skill for routing operator intent to the deterministic CLI and interpreting results |
| `doc/` | Human design notes and project planning documents |

## Development Tests

```bash
pytest
```

### Manual Container Matrix

Launches real Docker, Enroot, and/or Kubernetes containers to compare GDS
diagnostic visibility across launch shapes, and to sweep CUDA/GDS versions
via Kubernetes. See `tests/container_matrix/README.md` for setup and usage.

## Reference

[NVIDIA GDS Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)

## License

Skill documentation and assets (SKILL.md, reference files, and non-script content under `skills/`) are licensed under CC-BY-4.0 AND Apache-2.0. All source code — including Python, shell scripts, and tests, as well as scripts under `skills/gds-diag/scripts/` — is licensed under Apache-2.0.
