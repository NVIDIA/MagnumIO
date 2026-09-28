<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->

# Command Routing

Use this file to map the user's goal to the deterministic CLI. Keep the routing
logic here small; the Python subcommands own the actual checks.

| User goal | Preferred command |
|---|---|
| "Check GDS", "diagnose this system", "collect a general report", or unclear/broad GDS problem | `python3 gds-diag.py all [PATH] -v` |
| "Is this host ready for GDS before install?" | `python3 gds-diag.py pre-install` |
| "Was CUDA/GDS installed correctly?" | `python3 gds-diag.py post-install -v` |
| "Why is this mount/path in compat mode?" | `python3 gds-diag.py mount-check PATH -v` |
| "Can this specific path use direct GDS?" | `python3 gds-diag.py mount-check PATH -v` |
| "What filesystems/routes are supported?" | `python3 gds-diag.py support-matrix` |
| "What does this installed system report live?" | `python3 gds-diag.py support-matrix --live` |
| "Is this active cufile config sane?" | `python3 gds-diag.py config-audit --profile PROFILE` |
| "Is this proposed cufile.json valid?" | `python3 gds-diag.py config-audit --config PATH --ignore-env -v` |

Use `all` for broad or ambiguous diagnosis. If the user gives a path and asks a
specific path question, prefer `mount-check`. If the user gives only a
filesystem type or asks a general support question, prefer `support-matrix`
unless they are asking about a real installed host.

For active configuration questions, prefer plain `config-audit --profile
PROFILE`; it uses `gdscheck -p` CUFILE CONFIGURATION when available, falls back
to `CUFILE_ENV_PATH_JSON`, `/etc/cufile.json`, then the CUDA template, and
applies `CUFILE_*` environment overlays. Use `--config PATH --ignore-env` only
for strict proposed-file validation or when the user wants local environment
overrides suppressed.

If the user provides only a cuFile log, explain that log parsing is deferred in
the current CLI and ask for the affected mount path or cufile.json so the
implemented `mount-check` or `config-audit` flow can be used.

Use `all --json [PATH]` when collecting one structured general diagnostic
artifact. Use `--json` on narrower subcommands when you need to make structured
follow-up decisions from specific results. Use normal or verbose human output
when the answer is primarily for an operator reading the terminal.
