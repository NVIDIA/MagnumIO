# Changelog

All notable changes to `gds-diag` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `config-audit`: warn when `miscellaneous.enable_static_config` is true but
  `miscellaneous.static_config_filepath` is missing or not a non-empty string.
- `mount-check`: detect device-mapper-backed mounts (LVM, dm-crypt,
  dm-multipath, dm-raid, or generic device-mapper) and flag them as
  unsupported for native GDS (nvidia-fs) and P2PDMA/C2C, since GDS resolves a
  mount to a raw NVMe (or supported RAID0) block device and cannot see
  through device-mapper's block remapping.

### Fixed
- An unreadable `cufile.json` (e.g. a root-only `/etc/cufile.json` read as a
  normal user) is reported as "could not be read" instead of crashing
  `post-install` with a `PermissionError`, and is no longer silently replaced
  by the next candidate path.
- `support-matrix`: when `gdscheck` is installed but fails (e.g. permission
  denied), report `[static — gdscheck failed]` with the reason instead of
  "gdscheck not found". JSON output gains `gdscheck_error`.

### Removed
- `mount-check`: redundant gdscheck-derived P2PDMA/C2C cufile.json and RAID
  findings that repeated the static checks.

### Fixed
- `config-audit`: import `subprocess` in `checks/cufile_config.py` so the
  `nvidia-smi` GPU memory lookup works and the pinned-memory cap INFO can be
  reported.

## [1.0.0] - Initial release

Initial version of gds-diag with the following core capabilities implemented as subcommands:
- `all`: run the recommended general GDS diagnostic sequence.
- `support-matrix`: print the GDS filesystem support matrix.
- `pre-install`: pre-install GDS readiness check.
- `post-install`: post-install GDS validation.
- `config-audit`: audit cuFile configuration.
- `mount-check`: path-specific GDS diagnostic.
- `container-check`: validate GDS installation and configuration inside a container.
