# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from subcommands import config_audit


def _run(config_text: str, as_json: bool = False) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cufile.json"
        path.write_text(config_text, encoding="utf-8")
        args = argparse.Namespace(
            profile=None, config=str(path), ignore_env=True, json=as_json, verbose=False,
        )
        with redirect_stdout(io.StringIO()):
            return config_audit.COMMAND.run(args)


class ConfigAuditExitCodeTests(unittest.TestCase):
    # allow_compat_mode=false is reported as WARN; it must not fail the command.
    WARN_ONLY = '{"properties": {"allow_compat_mode": false}}'

    def test_warn_only_exits_zero_text(self):
        self.assertEqual(_run(self.WARN_ONLY), 0)

    def test_warn_only_exits_zero_json(self):
        self.assertEqual(_run(self.WARN_ONLY, as_json=True), 0)

    def test_parse_error_exits_one(self):
        self.assertEqual(_run("{not json"), 1)
        self.assertEqual(_run("{not json", as_json=True), 1)


if __name__ == "__main__":
    unittest.main()
