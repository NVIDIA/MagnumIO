# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from checks import version


REPO = Path(__file__).resolve().parents[1]


class VersionMetadataTests(unittest.TestCase):
    def test_tool_metadata_contains_reportable_identity(self):
        with mock.patch("checks.version.git_commit", return_value="abc123def456"):
            self.assertEqual(
                version.tool_metadata(),
                {
                    "name": "gds-diag",
                    "version": version.__version__,
                    "git_commit": "abc123def456",
                },
            )

    def test_version_string_reports_unknown_commit_without_git_metadata(self):
        with mock.patch("checks.version.git_commit", return_value=None):
            self.assertEqual(
                version.version_string(),
                f"gds-diag {version.__version__} (git unknown)",
            )

    def test_git_commit_resolves_git_executable_before_running(self):
        completed = subprocess.CompletedProcess(
            args=["/usr/bin/git", "rev-parse", "--short=12", "HEAD"],
            returncode=0,
            stdout="abc123def456\n",
            stderr="",
        )
        with mock.patch("checks.version.shutil.which", return_value="/usr/bin/git"), mock.patch(
            "checks.version.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(version.git_commit(), "abc123def456")

        self.assertEqual(run.call_args.args[0][0], "/usr/bin/git")

    def test_cli_version_does_not_require_subcommand(self):
        completed = subprocess.run(
            [sys.executable, "gds-diag.py", "--version"],
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(
            completed.stdout.strip(),
            rf"^gds-diag {version.__version__} \(git ([0-9a-f]{{1,12}}|unknown)\)$",
        )


if __name__ == "__main__":
    unittest.main()
