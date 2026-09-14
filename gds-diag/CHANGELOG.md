# Changelog

All notable changes to `gds-diag` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - Initial release

Initial version of gds-diag with the following core capabilities implemented as subcommands:
- `all`: run the recommended general GDS diagnostic sequence.
- `support-matrix`: print the GDS filesystem support matrix.
- `pre-install`: pre-install GDS readiness check.
- `post-install`: post-install GDS validation.
- `config-audit`: audit cuFile configuration.
- `mount-check`: path-specific GDS diagnostic.
- `container-check`: validate GDS installation and configuration inside a container.
