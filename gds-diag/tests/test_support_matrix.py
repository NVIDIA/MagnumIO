# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import argparse
import io
import json
from contextlib import redirect_stdout

from checks.fs_matrix import FS_CAPABILITIES
from subcommands import support_matrix


GDSCHECK_OLD_SUPPORTED = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe P2PDMA        : Unsupported
  NVMe               : Supported
  DDN EXAScaler      : Supported
  IBM Spectrum Scale : Unsupported
=====================
"""


class SupportMatrixDisplayTests(unittest.TestCase):
    def test_local_nvme_keeps_no_multipath_config_label_when_live_p2pdma_absent(self):
        value = support_matrix._p2pdma_ref_value("ext4", FS_CAPABILITIES["ext4"], live_p2pdma=False)

        self.assertEqual(value, "nomp-config")
        self.assertEqual(support_matrix._fmt(value, col=support_matrix.P2PDMA_COL).strip(), "NoMP Config")

    def test_live_p2pdma_token_overrides_config_label(self):
        value = support_matrix._p2pdma_ref_value("ext4", FS_CAPABILITIES["ext4"], live_p2pdma=True)

        self.assertIs(value, True)

    def test_live_c2c_token_is_direct_p2p_support(self):
        modes = support_matrix._parse_driver_config("""\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : c2c, nvfs, compat
=====================
""")
        live_map = support_matrix._live_status_map(modes)
        modes_str = live_map["ext4"].lower()

        self.assertIn("c2c", modes_str)
        self.assertTrue("p2pdma" in modes_str or "c2c" in modes_str)

    def test_c2c_token_counts_as_live_p2pdma_route(self):
        self.assertTrue(support_matrix._has_direct_p2pdma_token("c2c, nvfs, compat"))

    def test_old_supported_status_counts_as_live_native_route(self):
        self.assertTrue(support_matrix._live_native_from_modes("Supported", False))
        self.assertFalse(support_matrix._live_native_from_modes("Unsupported", False))

    def test_old_nvme_p2pdma_status_is_separate_from_nvme_native_status(self):
        driver_config = support_matrix._parse_driver_config(GDSCHECK_OLD_SUPPORTED)

        self.assertFalse(
            support_matrix._live_p2pdma_from_modes("ext4", driver_config["nvme"], driver_config)
        )
        self.assertTrue(support_matrix._live_native_from_modes(driver_config["nvme"], False))

    def test_live_p2pdma_token_does_not_override_unsupported_filesystem(self):
        value = support_matrix._p2pdma_ref_value("gpfs", FS_CAPABILITIES["gpfs"], live_p2pdma=True)

        self.assertIs(value, False)

    def test_raid0_uses_kernel_config_label_on_x86(self):
        value = support_matrix._p2pdma_ref_value(
            "raid0", FS_CAPABILITIES["raid0"], live_p2pdma=False, host_arch=support_matrix.Arch.X86,
        )

        self.assertEqual(value, "Kernel Config")
        self.assertEqual(support_matrix._fmt(value, col=support_matrix.P2PDMA_COL).strip(), "Kernel Config")

    def test_raid0_collapses_to_plain_config_on_arm(self):
        value = support_matrix._p2pdma_ref_value(
            "raid0", FS_CAPABILITIES["raid0"], live_p2pdma=False, host_arch=support_matrix.Arch.ARM,
        )

        self.assertEqual(value, "config")

    def test_local_nvme_collapses_to_plain_config_on_arm(self):
        value = support_matrix._p2pdma_ref_value(
            "ext4", FS_CAPABILITIES["ext4"], live_p2pdma=False, host_arch=support_matrix.Arch.ARM,
        )

        self.assertEqual(value, "config")

    def test_local_nvme_keeps_nomp_config_label_on_x86(self):
        value = support_matrix._p2pdma_ref_value(
            "ext4", FS_CAPABILITIES["ext4"], live_p2pdma=False, host_arch=support_matrix.Arch.X86,
        )

        self.assertEqual(value, "nomp-config")

    def test_live_missing_gdscheck_reports_cuda_first_when_toolkit_missing(self):
        old_find = support_matrix._find_gdscheck
        old_cuda = support_matrix._detect_cuda_toolkit
        old_gds = support_matrix._detect_gds_packages
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix._detect_cuda_toolkit = lambda: {"found": False, "evidence": None}
            support_matrix._detect_gds_packages = lambda: []
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {"found": False}
            args = argparse.Namespace(live=True, static=False, json=False)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_text(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._detect_cuda_toolkit = old_cuda
            support_matrix._detect_gds_packages = old_gds
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 3)
        text = output.getvalue()
        self.assertIn("CUDA Toolkit check : not found", text)
        self.assertIn("Install CUDA Toolkit first", text)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", text)
        self.assertIn("gds-tools", text)
        self.assertNotIn("cuda-gds", text)

    def test_live_missing_gdscheck_recommends_gds_tools_when_cuda_exists(self):
        old_find = support_matrix._find_gdscheck
        old_cuda = support_matrix._detect_cuda_toolkit
        old_gds = support_matrix._detect_gds_packages
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix._detect_cuda_toolkit = lambda: {"found": True, "evidence": "/usr/local/cuda-13.3"}
            support_matrix._detect_gds_packages = lambda: []
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {"found": False}
            args = argparse.Namespace(live=True, static=False, json=False)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_text(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._detect_cuda_toolkit = old_cuda
            support_matrix._detect_gds_packages = old_gds
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 3)
        text = output.getvalue()
        self.assertIn("CUDA Toolkit check : found (/usr/local/cuda-13.3)", text)
        self.assertIn("Install the matching GDS tools package", text)
        self.assertIn("gds-tools", text)
        self.assertNotIn("cuda-gds", text)

    def test_live_missing_gdscheck_falls_back_when_cuda_and_libcufile_exist(self):
        old_find = support_matrix._find_gdscheck
        old_cuda = support_matrix._detect_cuda_toolkit
        old_gds = support_matrix._detect_gds_packages
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix._detect_cuda_toolkit = lambda: {"found": True, "evidence": "/usr/local/cuda-13.3"}
            support_matrix._detect_gds_packages = lambda: []
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "path": "/usr/local/cuda/lib64/libcufile.so",
                "realpath": "/usr/local/cuda/lib64/libcufile.so.1.16.1",
                "api_version": "1.16",
                "api_version_tuple": [1, 16],
                "file_version": "1.16.1",
                "gds_release_version": None,
            }
            args = argparse.Namespace(live=True, static=False, json=False)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_text(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._detect_cuda_toolkit = old_cuda
            support_matrix._detect_gds_packages = old_gds
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("fallback", text)
        self.assertIn("libcufile: API 1.16", text)
        self.assertIn("Live driver/client mode tokens are unavailable", text)
        self.assertIn("GDS Filesystem Support Matrix", text)

    def test_live_missing_gdscheck_json_includes_install_guidance(self):
        old_find = support_matrix._find_gdscheck
        old_cuda = support_matrix._detect_cuda_toolkit
        old_gds = support_matrix._detect_gds_packages
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix._detect_cuda_toolkit = lambda: {"found": True, "evidence": "/usr/local/cuda-13.3"}
            support_matrix._detect_gds_packages = lambda: ["gds-tools-13-3-1.18.0.66-1.x86_64"]
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {"found": False}
            args = argparse.Namespace(live=True, static=False, json=True)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_json(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._detect_cuda_toolkit = old_cuda
            support_matrix._detect_gds_packages = old_gds
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 3)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["error"], "gdscheck_not_found")
        self.assertTrue(payload["cuda_toolkit_found"])
        self.assertIn("gds-tools-13-3", payload["gds_packages"][0])
        self.assertEqual(payload["install_cuda_url"], "https://developer.nvidia.com/cuda-downloads")

    def test_live_missing_gdscheck_json_falls_back_when_libcufile_exists(self):
        old_find = support_matrix._find_gdscheck
        old_cuda = support_matrix._detect_cuda_toolkit
        old_gds = support_matrix._detect_gds_packages
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix._detect_cuda_toolkit = lambda: {"found": True, "evidence": "/usr/local/cuda-13.3"}
            support_matrix._detect_gds_packages = lambda: []
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "api_version": "1.16",
                "api_version_tuple": [1, 16],
                "file_version": "1.16.1",
            }
            args = argparse.Namespace(live=True, static=False, json=True)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_json(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._detect_cuda_toolkit = old_cuda
            support_matrix._detect_gds_packages = old_gds
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["live_requested"])
        self.assertFalse(payload["live_status_available"])
        self.assertTrue(payload["gdscheck_fallback"])
        self.assertEqual(payload["libcufile"]["api_version"], "1.16")

    def test_live_json_accepts_old_supported_driver_config_format(self):
        old_find = support_matrix._find_gdscheck
        old_run = support_matrix._run_gdscheck_raw
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: "/usr/local/cuda/gds/tools/gdscheck"
            support_matrix._run_gdscheck_raw = lambda path, apply_env=True, required_section="CUFILE CONFIGURATION": (GDSCHECK_OLD_SUPPORTED, None)
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "api_version": "1.15",
                "api_version_tuple": [1, 15],
            }
            args = argparse.Namespace(live=True, static=False, json=True)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_json(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._run_gdscheck_raw = old_run
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        lustre = next(item for item in payload["filesystems"] if item["fs_type"] == "lustre")
        ext4 = next(item for item in payload["filesystems"] if item["fs_type"] == "ext4")

        self.assertEqual(lustre["live_status"], "supported")
        self.assertTrue(lustre["live_modes"]["native"])
        self.assertEqual(lustre["live_raw"], "Supported")
        self.assertEqual(ext4["live_status"], "supported")
        self.assertTrue(ext4["live_modes"]["native"])
        self.assertFalse(ext4["live_modes"]["p2pdma"])

    def test_live_json_c2c_token_does_not_crash_and_sets_c2c_flag(self):
        # Regression test: entry["live_modes"]["c2c"] previously referenced an
        # undefined `raw_c2c` name, crashing --live --json with NameError for
        # any filesystem whose active token included p2pdma/c2c.
        old_find = support_matrix._find_gdscheck
        old_run = support_matrix._run_gdscheck_raw
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: "/usr/local/cuda/gds/tools/gdscheck"
            support_matrix._run_gdscheck_raw = lambda path, apply_env=True, required_section="CUFILE CONFIGURATION": (
                """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : c2c, nvfs, compat
=====================
""",
                None,
            )
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "api_version": "1.17",
                "api_version_tuple": [1, 17],
            }
            args = argparse.Namespace(live=True, static=False, json=True)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_json(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix._run_gdscheck_raw = old_run
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        ext4 = next(item for item in payload["filesystems"] if item["fs_type"] == "ext4")
        self.assertTrue(ext4["live_modes"]["p2pdma"])
        self.assertTrue(ext4["live_modes"]["c2c"])

    def test_zfs_btrfs_compat_are_gated_by_libcufile_version(self):
        old_find = support_matrix._find_gdscheck
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "path": "/usr/local/cuda/lib64/libcufile.so",
                "realpath": "/usr/local/cuda/lib64/libcufile.so.1.16.1",
                "api_version": "1.16",
                "api_version_tuple": [1, 16],
                "file_version": "1.16.1",
                "gds_release_version": None,
            }
            args = argparse.Namespace(live=False, static=False, json=False)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_text(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("zfs", text)
        self.assertIn("btrfs", text)
        self.assertIn("Need 1.17", text)
        self.assertIn(">=1.16", text)
        self.assertIn("libcufile/GDS 1.16+", text)
        self.assertIn(">=1.17", text)
        self.assertIn("libcufile/GDS 1.17+", text)

    def test_support_matrix_json_includes_libcufile_and_compat_since(self):
        old_find = support_matrix._find_gdscheck
        old_detect = support_matrix.cufile_version.detect_libcufile
        try:
            support_matrix._find_gdscheck = lambda: None
            support_matrix.cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "path": "/usr/local/cuda/lib64/libcufile.so",
                "realpath": "/usr/local/cuda/lib64/libcufile.so.1.16.1",
                "api_version": "1.16",
                "api_version_tuple": [1, 16],
                "file_version": "1.16.1",
                "gds_release_version": None,
            }
            args = argparse.Namespace(live=False, static=False, json=True)
            output = io.StringIO()
            with redirect_stdout(output):
                code = support_matrix._run_json(args)
        finally:
            support_matrix._find_gdscheck = old_find
            support_matrix.cufile_version.detect_libcufile = old_detect

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["libcufile"]["api_version"], "1.16")
        zfs = next(item for item in payload["filesystems"] if item["fs_type"] == "zfs")
        self.assertEqual(zfs["compat_since"], "1.17+")
        self.assertEqual(zfs["compat_effective"], "needs-1.17")
        self.assertIn("Requires libcufile/GDS >= 1.17", zfs["compat_warning"])
        tmpfs = next(item for item in payload["filesystems"] if item["fs_type"] == "tmpfs")
        self.assertEqual(tmpfs["compat_since"], "1.16+")
        self.assertEqual(tmpfs["compat_min_version"], [1, 16])
        self.assertIs(tmpfs["compat_effective"], True)

    def test_libcufile_summary_explains_unavailable_api_version_when_found(self):
        # Older GDS releases (e.g. 1.7.x, shipped with CUDA 12.2) don't export
        # cuFileGetVersion, so the version probe fails even though the library
        # loaded fine and is genuinely installed -- detect_libcufile() reports
        # found=True for this case (see checks/cufile_version.py). The summary
        # should say why the API version is missing, not just silently omit it.
        info = {
            "found": True,
            "api_version": None,
            "file_version": "1.7.2",
            "gds_release_version": "1.7.2.10",
            "path": "/usr/local/cuda/lib64/libcufile.so",
            "probe_errors": [
                "/usr/local/cuda/lib64/libcufile.so: undefined symbol: cuFileGetVersion",
                "libcufile.so.0: undefined symbol: cuFileGetVersion",
            ],
        }

        summary = support_matrix._libcufile_summary(info)

        self.assertIn("API version unavailable (/usr/local/cuda/lib64/libcufile.so: undefined symbol: cuFileGetVersion)", summary)
        self.assertIn("file 1.7.2", summary)
        self.assertIn("gdscheck 1.7.2.10", summary)
        self.assertNotIn("libcufile.so.0", summary)  # only the first probe error is surfaced, to stay concise
        self.assertNotIn("not found", summary)

    def test_libcufile_summary_surfaces_probe_error_when_genuinely_not_found(self):
        info = {
            "found": False,
            "gds_release_version": None,
            "probe_errors": ["/usr/local/cuda/lib64/libcufile.so: cannot open shared object file"],
        }

        summary = support_matrix._libcufile_summary(info)

        self.assertIn("not found", summary)
        self.assertIn("probe failed: /usr/local/cuda/lib64/libcufile.so: cannot open shared object file", summary)

    def test_libcufile_summary_not_found_without_probe_errors_is_unchanged(self):
        info = {"found": False, "gds_release_version": None, "probe_errors": []}

        self.assertEqual(support_matrix._libcufile_summary(info), "not found")

    def test_libcufile_summary_found_is_unaffected(self):
        info = {
            "found": True,
            "api_version": "1.16",
            "file_version": "1.16.1",
            "gds_release_version": None,
            "path": "/usr/local/cuda/lib64/libcufile.so",
            "probe_errors": [],
        }

        summary = support_matrix._libcufile_summary(info)

        self.assertNotIn("probe failed", summary)
        self.assertIn("API 1.16", summary)


if __name__ == "__main__":
    unittest.main()
