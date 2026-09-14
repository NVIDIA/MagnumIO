# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from checks import cufile_version


class CufileVersionTests(unittest.TestCase):
    def test_cufile_version_int_decodes_major_minor(self):
        self.assertEqual(cufile_version.decode_cufile_version_int(1160), (1, 16))
        self.assertEqual(cufile_version.decode_cufile_version_int(1070), (1, 7))

    def test_parse_gds_release_version(self):
        output = "warn: ignored\n GDS release version: 1.16.1.26\n Platform: x86_64\n"
        self.assertEqual(cufile_version.parse_gds_release_version(output), "1.16.1.26")

    def test_version_at_least_pads_components(self):
        self.assertTrue(cufile_version.version_at_least((1, 17), (1, 17, 0)))
        self.assertFalse(cufile_version.version_at_least((1, 16, 1), (1, 17)))

    def test_probe_script_is_checkout_local_and_syntax_valid(self):
        path = cufile_version._probe_script_path()
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(os.path.isfile(path))
        completed = subprocess.run(
            [sys.executable, "-m", "py_compile", path], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_probe_invocation_uses_the_checkout_local_script(self):
        completed = subprocess.CompletedProcess([], 0, stdout='{"stage": "call", "rc": 0, "version": 1160}\n', stderr="")
        with mock.patch.object(cufile_version.subprocess, "run", return_value=completed) as run:
            result = cufile_version._query_cufile_get_version("/opt/cuda/lib64/libcufile.so")

        self.assertEqual(result, {"loaded": True, "version_int": 1160, "error": None})
        self.assertEqual(
            run.call_args.args[0],
            [sys.executable, cufile_version._probe_script_path(), "/opt/cuda/lib64/libcufile.so"],
        )

    def test_probe_runs_from_checkout_and_reports_load_failure(self):
        completed = subprocess.run(
            [sys.executable, cufile_version._probe_script_path(), "/definitely/not/libcufile.so"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["stage"], "load")
        self.assertTrue(payload["error"])


class DetectLibcufileTests(unittest.TestCase):
    """
    detect_libcufile()'s three real states: full version known, library
    found but the version probe didn't work (e.g. GDS 1.7.x predates
    cuFileGetVersion), and genuinely not found. "found" must track whether
    the library itself loaded, not whether cuFileGetVersion succeeded --
    that's the whole point of the found/api_version split.
    """

    def test_full_success_reports_api_and_file_version(self):
        with mock.patch.object(
            cufile_version, "_candidate_paths", return_value=["/usr/local/cuda/lib64/libcufile.so.1.16.1"]
        ), mock.patch.object(
            cufile_version, "_query_cufile_get_version",
            return_value={"loaded": True, "version_int": 1160, "error": None},
        ):
            info = cufile_version.detect_libcufile()

        self.assertTrue(info["found"])
        self.assertEqual(info["api_version"], "1.16")
        self.assertEqual(info["file_version"], "1.16.1")
        self.assertEqual(info["version_source"], "cuFileGetVersion")

    def test_loaded_but_version_probe_fails_is_found_with_unknown_api_version(self):
        # This is the GDS 1.7.2 / CUDA 12.2 case: the library loads (ctypes.CDLL
        # succeeds) but cuFileGetVersion isn't an exported symbol yet.
        with mock.patch.object(
            cufile_version, "_candidate_paths", return_value=["/usr/local/cuda/lib64/libcufile.so.1.7.2"]
        ), mock.patch.object(
            cufile_version, "_query_cufile_get_version",
            return_value={"loaded": True, "version_int": None, "error": "undefined symbol: cuFileGetVersion"},
        ):
            info = cufile_version.detect_libcufile("GDS release version: 1.7.2.10")

        self.assertTrue(info["found"], "library that loaded successfully must count as found")
        self.assertIsNone(info["api_version"])
        self.assertIsNone(info["api_version_tuple"])
        self.assertIsNone(info["version_source"])
        self.assertEqual(info["file_version"], "1.7.2")
        self.assertEqual(info["gds_release_version"], "1.7.2.10")
        self.assertIn("undefined symbol: cuFileGetVersion", info["probe_errors"][0])

    def test_loaded_candidates_own_error_is_promoted_to_front(self):
        # An earlier candidate that never loaded (e.g. a stale path that
        # doesn't exist on disk) must not bury the loaded candidate's own
        # probe error behind it -- callers read probe_errors[0] to explain
        # why the found library's API version is unknown.
        probes = {
            "/usr/local/cuda-12.1/lib64/libcufile.so": {
                "loaded": False, "version_int": None, "error": "cannot open shared object file",
            },
            "/usr/local/cuda/lib64/libcufile.so.1.7.2": {
                "loaded": True, "version_int": None, "error": "undefined symbol: cuFileGetVersion",
            },
        }
        with mock.patch.object(
            cufile_version, "_candidate_paths", return_value=list(probes.keys())
        ), mock.patch.object(
            cufile_version, "_query_cufile_get_version", side_effect=lambda path: probes[path]
        ):
            info = cufile_version.detect_libcufile()

        self.assertTrue(info["found"])
        self.assertEqual(info["path"], "/usr/local/cuda/lib64/libcufile.so.1.7.2")
        self.assertIn("undefined symbol: cuFileGetVersion", info["probe_errors"][0])
        # the unrelated earlier failure is still reported, just not first
        self.assertTrue(any("cannot open shared object file" in e for e in info["probe_errors"]))

    def test_library_never_loads_is_genuinely_not_found(self):
        with mock.patch.object(
            cufile_version, "_candidate_paths", return_value=["/usr/local/cuda/lib64/libcufile.so"]
        ), mock.patch.object(
            cufile_version, "_query_cufile_get_version",
            return_value={"loaded": False, "version_int": None, "error": "cannot open shared object file"},
        ):
            info = cufile_version.detect_libcufile()

        self.assertFalse(info["found"])
        self.assertIsNone(info["path"])
        self.assertIsNone(info["file_version"])
        self.assertIn("cannot open shared object file", info["probe_errors"][0])

    def test_no_candidates_at_all_is_not_found(self):
        with mock.patch.object(cufile_version, "_candidate_paths", return_value=[]):
            info = cufile_version.detect_libcufile()

        self.assertFalse(info["found"])
        self.assertEqual(info["probe_errors"], [])


if __name__ == "__main__":
    unittest.main()
