# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import unittest

from checks import gdscheck, gds_report

GDSCHECK_OUTPUT = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : c2c, nvfs, compat
=====================
CUFILE CONFIGURATION:
=====================
  logging.level : INFO
=====================
"""


class GdscheckInvocationTests(unittest.TestCase):
    def test_post_install_runs_gdscheck_without_sudo(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=GDSCHECK_OUTPUT, stderr="")

        old_run = gdscheck.subprocess.run
        try:
            gdscheck.subprocess.run = fake_run

            output, _ = gdscheck._run_gdscheck_raw("/usr/local/cuda/gds/tools/gdscheck")
        finally:
            gdscheck.subprocess.run = old_run

        self.assertIn("DRIVER CONFIGURATION", output)
        self.assertEqual(calls, [["/usr/local/cuda/gds/tools/gdscheck", "-p"]])

    def test_support_matrix_runs_gdscheck_without_sudo(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=GDSCHECK_OUTPUT, stderr="")

        old_run = gdscheck.subprocess.run
        try:
            gdscheck.subprocess.run = fake_run

            output, _ = gdscheck._run_gdscheck_raw("/usr/local/cuda/gds/tools/gdscheck")
        finally:
            gdscheck.subprocess.run = old_run

        self.assertIn("DRIVER CONFIGURATION", output)
        self.assertEqual(calls, [["/usr/local/cuda/gds/tools/gdscheck", "-p"]])

    def test_mount_check_runs_gdscheck_without_sudo(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=GDSCHECK_OUTPUT, stderr="")

        old_run = gds_report.subprocess.run
        old_glob = gds_report._glob.glob
        try:
            gds_report.subprocess.run = fake_run
            gds_report._glob.glob = lambda pattern: ["/usr/local/cuda/gds/tools/gdscheck"]

            output = gds_report._run_gdscheck_raw()
        finally:
            gds_report.subprocess.run = old_run
            gds_report._glob.glob = old_glob

        self.assertIn("DRIVER CONFIGURATION", output)
        self.assertEqual(calls, [["/usr/local/cuda/gds/tools/gdscheck", "-p"]])


if __name__ == "__main__":
    unittest.main()
