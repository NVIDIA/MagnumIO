# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import unittest

from checks import kmsg


class KernelLogAccessTests(unittest.TestCase):
    def setUp(self):
        kmsg.read_kmsg_result.cache_clear()

    def tearDown(self):
        kmsg.read_kmsg_result.cache_clear()

    def test_read_kmsg_uses_noninteractive_sudo_journalctl(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="nvidia_fs loaded\n", stderr="")

        old_run = kmsg.subprocess.run
        try:
            kmsg.subprocess.run = fake_run

            lines = kmsg.read_kmsg()
        finally:
            kmsg.subprocess.run = old_run

        self.assertEqual(lines, ("nvidia_fs loaded",))
        self.assertEqual(
            calls,
            [["sudo", "-n", "journalctl", "-k", "-b", "-o", "cat", "--no-pager"]],
        )

    def test_read_kmsg_falls_back_to_sudo_dmesg(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd == ["sudo", "-n", "dmesg"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="nvidia_fs p2pdma enabled\n", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        old_run = kmsg.subprocess.run
        try:
            kmsg.subprocess.run = fake_run

            lines = kmsg.read_kmsg()
        finally:
            kmsg.subprocess.run = old_run

        self.assertEqual(lines, ("nvidia_fs p2pdma enabled",))
        self.assertEqual(
            calls,
            [
                ["sudo", "-n", "journalctl", "-k", "-b", "-o", "cat", "--no-pager"],
                ["sudo", "-n", "dmesg"],
            ],
        )

    def test_dmesg_fallback_preserves_journalctl_failure(self):
        def fake_run(cmd, **kwargs):
            if cmd == ["sudo", "-n", "dmesg"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="sudo: a password is required",
            )

        old_run = kmsg.subprocess.run
        try:
            kmsg.subprocess.run = fake_run

            result = kmsg.read_kmsg_result()
        finally:
            kmsg.subprocess.run = old_run

        self.assertTrue(result.available)
        self.assertEqual(result.source, "sudo -n dmesg")
        self.assertTrue(result.permission_denied)
        self.assertIn("sudo -n journalctl -k -b", result.error)
        self.assertIn("password is required", result.error)

    def test_read_kmsg_forces_c_locale_without_mutating_parent_env(self):
        seen_envs = []
        parent_values = {}

        def fake_run(cmd, **kwargs):
            seen_envs.append(kwargs.get("env"))
            return subprocess.CompletedProcess(cmd, 0, stdout="nvidia_fs loaded\n", stderr="")

        old_run = kmsg.subprocess.run
        old_lc_all = kmsg.os.environ.get("LC_ALL")
        old_lang = kmsg.os.environ.get("LANG")
        try:
            kmsg.os.environ["LC_ALL"] = "zz_TEST"
            kmsg.os.environ["LANG"] = "zz_TEST"
            kmsg.subprocess.run = fake_run

            lines = kmsg.read_kmsg()
            parent_values["LC_ALL"] = kmsg.os.environ.get("LC_ALL")
            parent_values["LANG"] = kmsg.os.environ.get("LANG")
        finally:
            kmsg.subprocess.run = old_run
            if old_lc_all is None:
                kmsg.os.environ.pop("LC_ALL", None)
            else:
                kmsg.os.environ["LC_ALL"] = old_lc_all
            if old_lang is None:
                kmsg.os.environ.pop("LANG", None)
            else:
                kmsg.os.environ["LANG"] = old_lang

        self.assertEqual(lines, ("nvidia_fs loaded",))
        self.assertEqual(seen_envs[0]["LC_ALL"], "C")
        self.assertEqual(seen_envs[0]["LANG"], "C")
        self.assertEqual(parent_values, {"LC_ALL": "zz_TEST", "LANG": "zz_TEST"})

    def test_read_kmsg_result_reports_permission_failure(self):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="sudo: a password is required",
            )

        old_run = kmsg.subprocess.run
        try:
            kmsg.subprocess.run = fake_run

            result = kmsg.read_kmsg_result()
        finally:
            kmsg.subprocess.run = old_run

        self.assertFalse(result.available)
        self.assertTrue(result.permission_denied)
        self.assertIn("password is required", result.error)

    def test_read_kmsg_result_reports_no_new_privileges_as_permission_failure(self):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr='sudo: The "no new privileges" flag is set',
            )

        old_run = kmsg.subprocess.run
        try:
            kmsg.subprocess.run = fake_run

            result = kmsg.read_kmsg_result()
        finally:
            kmsg.subprocess.run = old_run

        self.assertFalse(result.available)
        self.assertTrue(result.permission_denied)
        self.assertIn("no new privileges", result.error)

    def test_grep_kmsg_result_distinguishes_no_match_from_no_access(self):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="iommu enabled\n", stderr="")

        old_run = kmsg.subprocess.run
        try:
            kmsg.subprocess.run = fake_run

            result = kmsg.grep_kmsg_result("nvidia_fs")
        finally:
            kmsg.subprocess.run = old_run

        self.assertTrue(result.available)
        self.assertEqual(result.lines, ())
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
