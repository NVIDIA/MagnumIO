# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from checks import container
from subcommands import all as all_cmd


REPO = Path(__file__).resolve().parents[1]


class _FakeCommand:
    def __init__(self, name: str, rc: int, calls: list[tuple[str, argparse.Namespace]]):
        self.name = name
        self.rc = rc
        self.calls = calls

    def run(self, args: argparse.Namespace) -> int:
        self.calls.append((self.name, args))
        if args.json:
            print(json.dumps({"child": self.name}))
        else:
            print(f"{self.name} ran")
        return self.rc


class AllCommandTests(unittest.TestCase):
    def _run_with_fakes(
        self,
        *,
        context: container.ContainerContext,
        args: argparse.Namespace,
        rc_by_name: dict[str, int] | None = None,
    ):
        calls: list[tuple[str, argparse.Namespace]] = []
        rc_by_name = rc_by_name or {}

        def fake_command(name: str):
            return _FakeCommand(name, rc_by_name.get(name, 0), calls)

        with mock.patch.object(all_cmd, "_detect_environment", return_value=context), mock.patch.object(
            all_cmd, "_command_by_name", side_effect=fake_command
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                rc = all_cmd.COMMAND.run(args)
        return rc, out.getvalue(), calls

    def test_host_sequence_stops_at_first_failure(self):
        rc, output, calls = self._run_with_fakes(
            context=container.ContainerContext(False, "host", "No common container markers found."),
            args=argparse.Namespace(json=False, verbose=False, path="/mnt/data"),
            rc_by_name={"post-install": 1},
        )

        self.assertEqual(rc, 1)
        self.assertEqual([name for name, _ in calls], ["pre-install", "post-install"])
        self.assertIn("pre-install", output)
        self.assertIn("post-install", output)
        self.assertIn("mount-check /mnt/data", output)

    def test_container_sequence_uses_container_check_then_mount_check(self):
        rc, _, calls = self._run_with_fakes(
            context=container.ContainerContext(True, "docker", "/.dockerenv exists"),
            args=argparse.Namespace(json=False, verbose=True, path="/work/data"),
        )

        self.assertEqual(rc, 0)
        self.assertEqual([name for name, _ in calls], ["container-check", "mount-check"])
        self.assertEqual(calls[0][1].runtime, "auto")
        self.assertTrue(calls[0][1].verbose)
        self.assertEqual(calls[1][1].path, "/work/data")

    def test_default_path_is_current_directory(self):
        rc, _, calls = self._run_with_fakes(
            context=container.ContainerContext(True, "enroot", "enroot marker"),
            args=argparse.Namespace(json=False, verbose=False, path="."),
        )

        self.assertEqual(rc, 0)
        self.assertEqual(calls[-1][1].path, ".")

    def test_json_aggregates_child_json_and_stops_on_failure(self):
        rc, output, calls = self._run_with_fakes(
            context=container.ContainerContext(False, "host", "No common container markers found."),
            args=argparse.Namespace(json=True, verbose=False, path="/mnt/data"),
            rc_by_name={"config-audit": 1},
        )

        payload = json.loads(output)
        self.assertEqual(rc, 1)
        self.assertEqual(payload["mode"], "all")
        self.assertEqual(payload["path"], "/mnt/data")
        self.assertEqual(payload["stopped_at"], "config-audit")
        self.assertEqual(
            payload["planned_sequence"],
            ["pre-install", "post-install", "config-audit", "mount-check"],
        )
        self.assertEqual([name for name, _ in calls], ["pre-install", "post-install", "config-audit"])
        self.assertEqual(payload["steps"][0]["result"], {"child": "pre-install"})

    def test_help_lists_all_first(self):
        completed = subprocess.run(
            [sys.executable, "gds-diag.py", "--help"],
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        subcommands = completed.stdout.split("Subcommands:", 1)[1].split("Run `gds-diag.py", 1)[0]
        first = next(line.strip() for line in subcommands.splitlines() if line.strip())
        self.assertTrue(first.startswith("all "), first)


if __name__ == "__main__":
    unittest.main()
