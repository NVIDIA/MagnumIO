# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import tempfile
from pathlib import Path

from checks import cufile_config, gdscheck


class CufileConfigAuditTests(unittest.TestCase):
    def setUp(self):
        self._old_gdscheck_loader = cufile_config._load_cufile_config_from_gdscheck
        cufile_config._load_cufile_config_from_gdscheck = lambda: (None, None, "test: gdscheck disabled")

    def tearDown(self):
        cufile_config._load_cufile_config_from_gdscheck = self._old_gdscheck_loader

    def test_default_audit_prefers_gdscheck_configuration(self):
        old_loader = cufile_config._load_cufile_json_with_path

        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda: (
                {
                    "properties": {"allow_compat_mode": True},
                },
                "/usr/local/cuda/gds/tools/gdscheck",
                None,
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {"logging": {"level": "TRACE"}},
                "/etc/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        self.assertEqual(audit["config_source"], "gdscheck")
        self.assertEqual(audit["file_fallback_path"], "/etc/cufile.json")
        self.assertEqual(audit["gdscheck_path"], "/usr/local/cuda/gds/tools/gdscheck")
        self.assertIsNone(audit["config_path"])
        level = next(entry for entry in audit["entries"] if entry["path"] == "logging.level")
        self.assertEqual(level["value"], "TRACE")
        self.assertEqual(level["source"], "file fallback")

    def test_gdscheck_value_wins_over_file_fallback(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda: (
                {
                    "logging": {"level": "DEBUG"},
                    "properties": {"allow_compat_mode": True},
                },
                "/usr/local/cuda/gds/tools/gdscheck",
                None,
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {"logging": {"level": "TRACE"}},
                "/etc/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        level = next(entry for entry in audit["entries"] if entry["path"] == "logging.level")
        self.assertEqual(level["value"], "DEBUG")
        self.assertEqual(level["source"], "gdscheck")

    def test_env_precedence_and_ignore_env(self):
        old_loader = cufile_config._load_cufile_json_with_path
        old_env = os.environ.get("CUFILE_LOGGING_LEVEL")
        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda *args, **kwargs: (
                {"logging": {"level": "DEBUG"}},
                "/usr/local/cuda/gds/tools/gdscheck",
                None,
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {"logging": {"level": "ERROR"}},
                "/etc/cufile.json",
            )
            os.environ["CUFILE_LOGGING_LEVEL"] = "TRACE"

            with_env = cufile_config.audit_config(apply_env=True)
            without_env = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader
            if old_env is None:
                os.environ.pop("CUFILE_LOGGING_LEVEL", None)
            else:
                os.environ["CUFILE_LOGGING_LEVEL"] = old_env

        with_level = next(entry for entry in with_env["entries"] if entry["path"] == "logging.level")
        without_level = next(entry for entry in without_env["entries"] if entry["path"] == "logging.level")
        self.assertEqual(with_level["value"], "TRACE")
        self.assertEqual(with_level["source"], "env:CUFILE_LOGGING_LEVEL")
        self.assertEqual(without_level["value"], "DEBUG")
        self.assertEqual(without_level["source"], "gdscheck")

    def test_default_audit_falls_back_to_file_when_gdscheck_unavailable(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda: (
                None,
                "/usr/local/cuda/gds/tools/gdscheck",
                "gdscheck -p exited with 1",
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {"logging": {"level": "INFO"}},
                "/etc/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        self.assertEqual(audit["config_source"], "file")
        self.assertTrue(audit["gdscheck_fallback"])
        self.assertEqual(audit["gdscheck_error"], "gdscheck -p exited with 1")
        self.assertEqual(audit["config_path"], "/etc/cufile.json")

    def test_explicit_config_path_bypasses_gdscheck(self):
        def fail_gdscheck_loader():
            raise AssertionError("gdscheck should not be used for explicit --config audits")

        cufile_config._load_cufile_config_from_gdscheck = fail_gdscheck_loader
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "custom-cufile.json"
            config_path.write_text('{"logging": {"level": "DEBUG"}}\n', encoding="utf-8")
            audit = cufile_config.audit_config(config_path=str(config_path), apply_env=False)

        self.assertEqual(audit["config_source"], "file")
        self.assertEqual(audit["config_path"], str(config_path))
        self.assertEqual(audit["requested_config_path"], str(config_path))

    def test_parse_gdscheck_cufile_configuration(self):
        output = """
========================
CUFILE CONFIGURATION:
========================
 properties.allow_compat_mode : true
 properties.rdma_dev_addr_list : [10.1.2.3, 10.1.2.4]
 properties.posix_pool_slab_size_kb : 4 1024 16384
 properties.posix_pool_slab_count : 128 64 64
 block.nvme.use_pci_p2pdma : false
 ignored heading : value
========================
"""
        parsed = cufile_config._parse_gdscheck_cufile_config(output)

        self.assertTrue(parsed["properties"]["allow_compat_mode"])
        self.assertEqual(parsed["properties"]["rdma_dev_addr_list"], ["10.1.2.3", "10.1.2.4"])
        self.assertEqual(parsed["properties"]["posix_pool_slab_size_kb"], [4, 1024, 16384])
        self.assertEqual(parsed["properties"]["posix_pool_slab_count"], [128, 64, 64])
        self.assertFalse(parsed["block"]["nvme"]["use_pci_p2pdma"])

    def test_newer_template_keys_are_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "cufile.json"
            config_path.write_text(
                """
{
  "properties": {
    "vanilla_posix_io_mode": false,
    "gds_fallback_io": false,
    "rdma_transport_type": "DC_V1",
    "allow_rdma_token_reset": false,
    "compat_odirect_unaligned_read_split": false,
    "compat_odirect_unaligned_read_split_min_size_kb": 128
  },
  "block": {
    "raid1": {"use_pci_p2pdma": false},
    "raid10": {"use_pci_p2pdma": false}
  },
  "miscellaneous": {
    "rdma_token_reset_timeout_secs": 30
  }
}
""",
                encoding="utf-8",
            )
            audit = cufile_config.audit_config(
                config_path=str(config_path),
                apply_env=False,
                prefer_gdscheck=False,
            )

        entries = {entry["path"]: entry for entry in audit["entries"]}
        for path in (
            "properties.vanilla_posix_io_mode",
            "properties.gds_fallback_io",
            "properties.rdma_transport_type",
            "properties.allow_rdma_token_reset",
            "properties.compat_odirect_unaligned_read_split",
            "properties.compat_odirect_unaligned_read_split_min_size_kb",
            "block.raid1.use_pci_p2pdma",
            "block.raid10.use_pci_p2pdma",
            "miscellaneous.rdma_token_reset_timeout_secs",
        ):
            self.assertIn(path, entries)
            self.assertNotEqual(entries[path]["risk"], "Unknown cufile.json key for this tool's documented schema")
            self.assertEqual(entries[path]["status"], "OK")

    def test_find_gdscheck_prefers_newer_versioned_cuda_path(self):
        old_glob = gdscheck._glob.glob
        old_which = gdscheck.shutil.which
        old_isfile = gdscheck.os.path.isfile
        old_access = gdscheck.os.access
        try:
            gdscheck._glob.glob = lambda pattern: [
                "/usr/local/cuda-9.2/gds/tools/gdscheck",
                "/usr/local/cuda-12.4/gds/tools/gdscheck",
            ]
            gdscheck.shutil.which = lambda candidate: None
            gdscheck.os.path.isfile = lambda path: path.startswith("/usr/local/cuda-")
            gdscheck.os.access = lambda path, mode: True

            path = gdscheck._find_gdscheck()
        finally:
            gdscheck._glob.glob = old_glob
            gdscheck.shutil.which = old_which
            gdscheck.os.path.isfile = old_isfile
            gdscheck.os.access = old_access

        self.assertEqual(path, "/usr/local/cuda-12.4/gds/tools/gdscheck")

    def test_run_gdscheck_raw_can_ignore_cufile_env(self):
        old_env = os.environ.get("CUFILE_LOGGING_LEVEL")
        try:
            os.environ["CUFILE_LOGGING_LEVEL"] = "TRACE"
            with tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "gdscheck"
                script.write_text(
                    "#!/bin/sh\n"
                    "echo '========================'\n"
                    "echo 'CUFILE CONFIGURATION:'\n"
                    "echo '========================'\n"
                    "echo \" logging.level : ${CUFILE_LOGGING_LEVEL:-unset}\"\n"
                    "echo '========================'\n",
                    encoding="utf-8",
                )
                script.chmod(0o755)
                with_env, with_error = gdscheck._run_gdscheck_raw(str(script), apply_env=True)
                without_env, without_error = gdscheck._run_gdscheck_raw(str(script), apply_env=False)
        finally:
            if old_env is None:
                os.environ.pop("CUFILE_LOGGING_LEVEL", None)
            else:
                os.environ["CUFILE_LOGGING_LEVEL"] = old_env

        self.assertIsNone(with_error)
        self.assertIsNone(without_error)
        self.assertIn("logging.level : TRACE", with_env)
        self.assertIn("logging.level : unset", without_env)

    def test_gdscheck_posix_pool_slabs_are_not_warned(self):
        cufile_config._load_cufile_config_from_gdscheck = lambda: (
            {
                "properties": {
                    "posix_pool_slab_size_kb": [4, 1024, 16384],
                    "posix_pool_slab_count": [128, 64, 64],
                },
            },
            "/usr/local/cuda/gds/tools/gdscheck",
            None,
        )
        audit = cufile_config.audit_config("compat-safe", apply_env=False)

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["properties.posix_pool_slab_size_kb"]["status"], "OK")
        self.assertEqual(by_path["properties.posix_pool_slab_count"]["status"], "OK")

    def test_gpu_bounce_buffer_slab_nested_keys_are_valid(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {
                        "gpu_bounce_buffer_slab_config": {
                            "slab_size_kb": [1024, 8192, 16384, 65536],
                            "slab_count": [128, 8, 4, 2],
                        },
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["properties.gpu_bounce_buffer_slab_config"]["status"], "OK")
        unknown_paths = {
            entry["path"] for entry in audit["entries"]
            if entry["scope"] == "unknown"
        }
        self.assertNotIn("properties.gpu_bounce_buffer_slab_config.slab_size_kb", unknown_paths)
        self.assertNotIn("properties.gpu_bounce_buffer_slab_config.slab_count", unknown_paths)

    def test_flat_gpu_bounce_buffer_slab_keys_are_valid(self):
        cufile_config._load_cufile_config_from_gdscheck = lambda: (
            {
                "properties": {
                    "gpu_bounce_buffer_slab_size_kb": [1024, 8192, 16384, 65536],
                    "gpu_bounce_buffer_slab_count": [128, 8, 4, 2],
                },
            },
            "/usr/local/cuda/gds/tools/gdscheck",
            None,
        )
        audit = cufile_config.audit_config("local-nvme", apply_env=False)

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["properties.gpu_bounce_buffer_slab_size_kb"]["status"], "OK")
        self.assertEqual(by_path["properties.gpu_bounce_buffer_slab_count"]["status"], "OK")
        unknown_paths = {
            entry["path"] for entry in audit["entries"]
            if entry["scope"] == "unknown"
        }
        self.assertNotIn("properties.gpu_bounce_buffer_slab_size_kb", unknown_paths)
        self.assertNotIn("properties.gpu_bounce_buffer_slab_count", unknown_paths)

    def test_flat_gpu_bounce_buffer_slab_partial_config_warns(self):
        cufile_config._load_cufile_config_from_gdscheck = lambda *args, **kwargs: (
            {
                "properties": {
                    "gpu_bounce_buffer_slab_size_kb": [1024, 8192],
                },
            },
            "/usr/local/cuda/gds/tools/gdscheck",
            None,
        )
        audit = cufile_config.audit_config("local-nvme", apply_env=False)

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.gpu_bounce_buffer_slab_size_kb/properties.gpu_bounce_buffer_slab_count"
        )
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("incomplete", entry["risk"])

    def test_gpu_bounce_buffer_slab_config_suppresses_legacy_cache_batch_warning(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda *args, **kwargs: (
                {
                    "properties": {
                        "max_device_cache_size_kb": 393216,
                        "per_buffer_cache_size_kb": 16384,
                        "io_batchsize": 128,
                    },
                },
                "/usr/local/cuda/gds/tools/gdscheck",
                None,
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {
                        "gpu_bounce_buffer_slab_config": {
                            "slab_size_kb": [1024, 8192, 16384, 65536],
                            "slab_count": [128, 8, 4, 2],
                        },
                    },
                },
                "/etc/cufile.json",
            )
            audit = cufile_config.audit_config("local-nvme", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        paths = {entry["path"] for entry in audit["entries"]}
        self.assertNotIn(
            "properties.max_device_cache_size_kb/properties.per_buffer_cache_size_kb/properties.io_batchsize",
            paths,
        )

    def test_flat_gpu_bounce_buffer_slab_config_suppresses_legacy_cache_batch_warning(self):
        cufile_config._load_cufile_config_from_gdscheck = lambda *args, **kwargs: (
            {
                "properties": {
                    "max_device_cache_size_kb": 393216,
                    "per_buffer_cache_size_kb": 16384,
                    "io_batchsize": 128,
                    "gpu_bounce_buffer_slab_size_kb": [1024, 8192, 16384, 65536],
                    "gpu_bounce_buffer_slab_count": [128, 8, 4, 2],
                },
            },
            "/usr/local/cuda/gds/tools/gdscheck",
            None,
        )
        audit = cufile_config.audit_config("local-nvme", apply_env=False)

        paths = {entry["path"] for entry in audit["entries"]}
        self.assertNotIn(
            "properties.max_device_cache_size_kb/properties.per_buffer_cache_size_kb/properties.io_batchsize",
            paths,
        )

    def test_gpu_bounce_buffer_slab_sizes_must_be_ascending(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {
                        "gpu_bounce_buffer_slab_config": {
                            "slab_size_kb": [16384, 1024],
                            "slab_count": [4, 128],
                        },
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.gpu_bounce_buffer_slab_config"
        )
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("ascending order", entry["risk"])

    def test_local_nvme_profile_reports_p2pdma_disabled(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: ({}, "/tmp/cufile.json")
            audit = cufile_config.audit_config("local-nvme", apply_env=False, prefer_gdscheck=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["properties.use_pci_p2pdma"]["status"], "INFO")
        self.assertIn("not globally enabled", by_path["properties.use_pci_p2pdma"]["risk"])
        self.assertIn("C2C", by_path["properties.use_pci_p2pdma"]["recommendation"])
        self.assertEqual(by_path["block.nvme.use_pci_p2pdma"]["status"], "INFO")
        self.assertIn("not enabled", by_path["block.nvme.use_pci_p2pdma"]["risk"])

    def test_p2pdma_reconciliation_preserves_type_warning(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {"use_pci_p2pdma": "yes"},
                    "block": {"nvme": {"use_pci_p2pdma": False}},
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("local-nvme", apply_env=False, prefer_gdscheck=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.use_pci_p2pdma"
        )
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("expected a JSON boolean", entry["risk"])
        self.assertIn("not globally enabled", entry["risk"])

    def test_nfs_rdma_warns_when_global_p2pdma_is_enabled(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"properties": {"use_pci_p2pdma": True}},
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("nfs-rdma", apply_env=False, prefer_gdscheck=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.use_pci_p2pdma"
        )
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("not a GDS library-supported direct P2P route", entry["risk"])

    def test_gdscheck_numeric_bool_and_scalar_rdma_list_are_normalized(self):
        cufile_config._load_cufile_config_from_gdscheck = lambda: (
            {
                "properties": {"rdma_dynamic_routing": 0},
                "fs": {"gpfs": {"rdma_dev_addr_list": "192.168.4.201"}},
            },
            "/usr/local/cuda/gds/tools/gdscheck",
            None,
        )
        audit = cufile_config.audit_config("gpfs", apply_env=False)

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertIs(by_path["properties.rdma_dynamic_routing"]["value"], False)
        self.assertEqual(by_path["properties.rdma_dynamic_routing"]["status"], "OK")
        self.assertEqual(by_path["fs.gpfs.rdma_dev_addr_list"]["value"], ["192.168.4.201"])
        self.assertNotIn("expected a list", by_path["fs.gpfs.rdma_dev_addr_list"]["risk"])

    def test_rdma_multipath_knobs_are_ignored_for_now(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {
                        "rdma_multipath_enabled": True,
                        "rdma_max_backup_devices": 2,
                        "rdma_io_retry_count": 5,
                        "rdma_io_retry_delay_ms": 10,
                        "rdma_failback_enabled": True,
                        "rdma_failback_delay_ms": 1000,
                        "rdma_health_check_interval_ms": 1000,
                        "rdma_async_event_monitoring": True,
                        "rdma_unhealthy_threshold": 3,
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("gpfs", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        unknown_paths = {
            entry["path"] for entry in audit["entries"]
            if entry["scope"] == "unknown"
        }
        self.assertFalse(unknown_paths & {
            "properties.rdma_multipath_enabled",
            "properties.rdma_max_backup_devices",
            "properties.rdma_io_retry_count",
            "properties.rdma_io_retry_delay_ms",
            "properties.rdma_failback_enabled",
            "properties.rdma_failback_delay_ms",
            "properties.rdma_health_check_interval_ms",
            "properties.rdma_async_event_monitoring",
            "properties.rdma_unhealthy_threshold",
        })

    def test_performance_infos_for_direct_io_and_pinned_memory_caps(self):
        old_loader = cufile_config._load_cufile_json_with_path
        old_gpu_mem = cufile_config._gpu_memory_totals_kb
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {
                        "max_direct_io_size_kb": 8192,
                        "max_device_pinned_mem_size_kb": 33554432,
                    },
                },
                "/tmp/cufile.json",
            )
            cufile_config._gpu_memory_totals_kb = lambda: [80 * 1024 * 1024]
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader
            cufile_config._gpu_memory_totals_kb = old_gpu_mem

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["properties.max_direct_io_size_kb"]["status"], "INFO")
        self.assertIn("below 16 MiB", by_path["properties.max_direct_io_size_kb"]["risk"])
        self.assertEqual(by_path["properties.max_device_pinned_mem_size_kb"]["status"], "INFO")
        self.assertIn("below available GPU memory", by_path["properties.max_device_pinned_mem_size_kb"]["risk"])

    def test_threadpool_disabled_settings_warn(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "execution": {
                        "parallel_io": False,
                        "max_request_parallelism": 0,
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["execution.parallel_io"]["status"], "WARN")
        self.assertIn("threadpool parallel IO is disabled", by_path["execution.parallel_io"]["risk"])
        self.assertEqual(by_path["execution.max_request_parallelism"]["status"], "WARN")
        self.assertIn("request parallelism is disabled", by_path["execution.max_request_parallelism"]["risk"])

    def test_gpfs_mount_table_satisfies_rdma_address_source(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {"rdma_dev_addr_list": []},
                    "fs": {
                        "gpfs": {
                            "mount_table": {
                                "/gpfs/fs1": {"rdma_dev_addr_list": ["10.1.2.3"]}
                            }
                        }
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("gpfs", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        global_entry = next(
            entry for entry in audit["entries"]
            if entry["path"] == "properties.rdma_dev_addr_list"
        )
        self.assertEqual(global_entry["status"], "INFO")
        self.assertIn("relying on a per-filesystem or per-mount", global_entry["risk"])

    def test_gpfs_write_and_async_disabled_are_reported(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "fs": {
                        "gpfs": {
                            "gds_write_support": False,
                            "gds_async_support": False,
                        },
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("gpfs", apply_env=False, prefer_gdscheck=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["fs.gpfs.gds_write_support"]["status"], "INFO")
        self.assertIn("GPFS GDS writes are disabled", by_path["fs.gpfs.gds_write_support"]["risk"])
        self.assertEqual(by_path["fs.gpfs.gds_async_support"]["status"], "INFO")
        self.assertIn("GPFS async GDS support is disabled", by_path["fs.gpfs.gds_async_support"]["risk"])

    def test_empty_global_rdma_list_info_is_limited_to_weka_gpfs(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"properties": {"rdma_dev_addr_list": []}},
                "/tmp/cufile.json",
            )
            lustre_audit = cufile_config.audit_config("lustre", apply_env=False)
            nfs_audit = cufile_config.audit_config("nfs-rdma", apply_env=False)
            gpfs_audit = cufile_config.audit_config("gpfs", apply_env=False)
            weka_audit = cufile_config.audit_config("wekafs", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        def global_entry(audit):
            return next(
                entry for entry in audit["entries"]
                if entry["path"] == "properties.rdma_dev_addr_list"
            )

        self.assertEqual(global_entry(lustre_audit)["status"], "OK")
        self.assertNotIn("auto-discovery", global_entry(lustre_audit)["risk"])
        self.assertEqual(global_entry(nfs_audit)["status"], "OK")
        self.assertNotIn("auto-discovery", global_entry(nfs_audit)["risk"])
        self.assertEqual(global_entry(gpfs_audit)["status"], "INFO")
        self.assertIn("no global RDMA client address source", global_entry(gpfs_audit)["risk"])
        self.assertEqual(global_entry(weka_audit)["status"], "INFO")
        self.assertIn("no global RDMA client address source", global_entry(weka_audit)["risk"])

    def test_empty_profile_rdma_list_is_reported_for_weka_gpfs(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: ({}, "/tmp/cufile.json")
            gpfs_audit = cufile_config.audit_config("gpfs", apply_env=False, prefer_gdscheck=False)
            weka_audit = cufile_config.audit_config("wekafs", apply_env=False, prefer_gdscheck=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        gpfs_entry = next(
            entry for entry in gpfs_audit["entries"]
            if entry["path"] == "fs.gpfs.rdma_dev_addr_list"
        )
        weka_entry = next(
            entry for entry in weka_audit["entries"]
            if entry["path"] == "fs.weka.rdma_dev_addr_list"
        )
        self.assertEqual(gpfs_entry["status"], "INFO")
        self.assertIn("no explicit RDMA client address source", gpfs_entry["risk"])
        self.assertEqual(weka_entry["status"], "INFO")
        self.assertIn("no explicit RDMA client address source", weka_entry["risk"])

    def test_blank_global_rdma_list_is_reported_even_with_gpfs_specific_list(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {"rdma_dev_addr_list": [""]},
                    "fs": {"gpfs": {"rdma_dev_addr_list": ["10.1.2.3"]}},
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("gpfs", apply_env=False, prefer_gdscheck=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["properties.rdma_dev_addr_list"]["status"], "INFO")
        self.assertIn("relying on a per-filesystem", by_path["properties.rdma_dev_addr_list"]["risk"])

    def test_file_fallback_rdma_source_is_used_for_gpfs_effective_config(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda *args, **kwargs: (
                {"properties": {"rdma_dev_addr_list": []}},
                "/usr/local/cuda/gds/tools/gdscheck",
                None,
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {"fs": {"gpfs": {"rdma_dev_addr_list": ["10.1.2.3"]}}},
                "/etc/cufile.json",
            )
            audit = cufile_config.audit_config("gpfs", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.rdma_dev_addr_list"
        )
        self.assertEqual(entry["status"], "INFO")
        self.assertIn("relying on a per-filesystem", entry["risk"])

    def test_static_routing_miscellaneous_keys_are_known(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "miscellaneous": {
                        "enable_static_routing": True,
                        "static_routing_filepath": "/tmp/does-not-exist-gds-topology.json",
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("local-nvme", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        enable = next(
            entry for entry in audit["entries"]
            if entry["path"] == "miscellaneous.enable_static_routing"
            and entry["source"] != "derived"
        )
        filepath = next(
            entry for entry in audit["entries"]
            if entry["path"] == "miscellaneous.static_routing_filepath"
            and entry["source"] != "derived"
        )
        self.assertEqual(enable["value"], True)
        self.assertEqual(filepath["value"], "/tmp/does-not-exist-gds-topology.json")
        self.assertFalse([
            entry for entry in audit["entries"]
            if entry["path"].startswith("miscellaneous.") and entry["scope"] == "unknown"
        ])
        derived = [
            entry for entry in audit["entries"]
            if entry["path"] == "miscellaneous.static_routing_filepath"
            and entry["source"] == "derived"
        ]
        self.assertEqual(derived[0]["status"], "WARN")
        self.assertIn("topology file is missing", derived[0]["risk"])

    def test_static_routing_empty_file_warns(self):
        old_loader = cufile_config._load_cufile_json_with_path
        with tempfile.TemporaryDirectory() as tmp:
            topology = Path(tmp) / "topology.json"
            topology.write_text("", encoding="utf-8")
            try:
                cufile_config._load_cufile_json_with_path = lambda: (
                    {
                        "miscellaneous": {
                            "enable_static_routing": True,
                            "static_routing_filepath": str(topology),
                        },
                    },
                    str(Path(tmp) / "cufile.json"),
                )
                audit = cufile_config.audit_config("local-nvme", apply_env=False)
            finally:
                cufile_config._load_cufile_json_with_path = old_loader

        derived = [
            entry for entry in audit["entries"]
            if entry["path"] == "miscellaneous.static_routing_filepath"
            and entry["source"] == "derived"
        ]
        self.assertEqual(derived[0]["status"], "WARN")
        self.assertIn("empty", derived[0]["risk"])

    def test_static_routing_sparse_keys_remain_aliases(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "sparse": {
                        "enable_static_routing": True,
                        "static_routing_filepath": "/tmp/does-not-exist-gds-topology.json",
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("local-nvme", apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        enable = next(
            entry for entry in audit["entries"]
            if entry["path"] == "miscellaneous.enable_static_routing"
        )
        filepath = next(
            entry for entry in audit["entries"]
            if entry["path"] == "miscellaneous.static_routing_filepath"
            and entry["source"] != "derived"
        )
        self.assertEqual(enable["resolved_key"], "sparse.enable_static_routing")
        self.assertEqual(filepath["resolved_key"], "sparse.static_routing_filepath")
        self.assertFalse([
            entry for entry in audit["entries"]
            if entry["path"].startswith("sparse.") and entry["scope"] == "unknown"
        ])

    def test_raid0_p2pdma_reports_release_note_limitation(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "properties": {"use_pci_p2pdma": True},
                    "block": {"raid": {"use_pci_p2pdma": True}},
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("raid0")
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        raid_entry = next(
            entry for entry in audit["entries"]
            if entry["path"] == "block.raid.use_pci_p2pdma"
        )
        self.assertEqual(raid_entry["status"], "WARN")
        self.assertIn("Release-note limitation", raid_entry["risk"])
        self.assertIn("gdscheck/runtime", raid_entry["recommendation"])

    def test_gdscheck_file_fallback_parse_error_is_reported(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_config_from_gdscheck = lambda *args, **kwargs: (
                {"properties": {"allow_compat_mode": True}},
                "/usr/local/cuda/gds/tools/gdscheck",
                None,
            )
            cufile_config._load_cufile_json_with_path = lambda: (
                {"_parse_error": "bad json", "_path": "/etc/cufile.json"},
                "/etc/cufile.json",
            )
            audit = cufile_config.audit_config(apply_env=False)
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "file_fallback.cufile_json"
        )
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("could not be parsed", entry["risk"])

    def test_rdma_dev_addr_list_must_be_client_local(self):
        from checks import rdma

        old_loader = cufile_config._load_cufile_json_with_path
        old_local = rdma._local_ipv4_map
        old_rdma_netdevs = rdma._rdma_netdevs
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"properties": {"rdma_dev_addr_list": ["192.0.2.10"]}},
                "/tmp/cufile.json",
            )
            rdma._local_ipv4_map = lambda: {"10.1.2.3": "ib0"}
            rdma._rdma_netdevs = lambda: {"ib0"}
            audit = cufile_config.audit_config("gpfs")
        finally:
            cufile_config._load_cufile_json_with_path = old_loader
            rdma._local_ipv4_map = old_local
            rdma._rdma_netdevs = old_rdma_netdevs

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.rdma_dev_addr_list"
        )
        self.assertEqual(entry["status"], "WARN")
        self.assertIn("client-side RDMA NIC", entry["risk"])
        self.assertIn("not local to this client", entry["recommendation"])

    def test_rdma_dev_addr_list_reports_verified_client_local_detail(self):
        from checks import rdma

        old_loader = cufile_config._load_cufile_json_with_path
        old_local = rdma._local_ipv4_map
        old_rdma_netdevs = rdma._rdma_netdevs
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"properties": {"rdma_dev_addr_list": ["10.1.2.3"]}},
                "/tmp/cufile.json",
            )
            rdma._local_ipv4_map = lambda: {"10.1.2.3": "ib0"}
            rdma._rdma_netdevs = lambda: {"ib0"}
            audit = cufile_config.audit_config("gpfs")
        finally:
            cufile_config._load_cufile_json_with_path = old_loader
            rdma._local_ipv4_map = old_local
            rdma._rdma_netdevs = old_rdma_netdevs

        entry = next(
            item for item in audit["entries"]
            if item["path"] == "properties.rdma_dev_addr_list"
        )
        self.assertEqual(entry["status"], "OK")
        self.assertIn("Verified client-local RDMA IP", entry["detail"])

    def test_audit_accepts_explicit_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "custom-cufile.json"
            config_path.write_text(
                '{\n'
                '  // JSONC comments are allowed in NVIDIA cufile.json\n'
                '  "logging": { "level": "DEBUG" },\n'
                '  "properties": { "allow_compat_mode": true }\n'
                '}\n',
                encoding="utf-8",
            )

            audit = cufile_config.audit_config(config_path=str(config_path), apply_env=False)

        self.assertEqual(audit["config_path"], str(config_path))
        self.assertEqual(audit["requested_config_path"], str(config_path))
        level = next(entry for entry in audit["entries"] if entry["path"] == "logging.level")
        self.assertEqual(level["value"], "DEBUG")
        self.assertEqual(level["status"], "OK")

    def test_audit_flags_bad_known_values(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {
                    "logging": {"level": "LOUD"},
                    "profile": {"cufile_stats": 9},
                    "properties": {
                        "allow_compat_mode": True,
                        "io_priority": "urgent",
                        "rdma_dynamic_routing_order": ["SYS_MEM", "SYS_MEM", "BOGUS"],
                    },
                },
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config()
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        self.assertEqual(by_path["logging.level"]["status"], "WARN")
        self.assertIn("unsupported logging level", by_path["logging.level"]["risk"])
        self.assertEqual(by_path["profile.cufile_stats"]["status"], "WARN")
        self.assertEqual(by_path["properties.io_priority"]["status"], "WARN")
        self.assertEqual(by_path["properties.rdma_dynamic_routing_order"]["status"], "WARN")
        self.assertIn("duplicates", by_path["properties.rdma_dynamic_routing_order"]["risk"])

    def test_audit_reports_unknown_json_keys(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"properties": {"allow_compat_mode": True, "allow_compat": True}},
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config()
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        unknown = next(entry for entry in audit["entries"] if entry["path"] == "properties.allow_compat")
        self.assertEqual(unknown["status"], "INFO")
        self.assertIn("Unknown cufile.json key", unknown["risk"])

    def test_dynamic_routing_requires_address_configuration(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"properties": {"allow_compat_mode": True, "rdma_dynamic_routing": True}},
                "/tmp/cufile.json",
            )
            audit = cufile_config.audit_config("gpfs")
        finally:
            cufile_config._load_cufile_json_with_path = old_loader

        derived = [
            entry for entry in audit["entries"]
            if entry["path"] == "properties.rdma_dynamic_routing"
            and entry["source"] == "derived"
        ]
        self.assertEqual(derived[0]["status"], "WARN")
        self.assertIn("without RDMA address", derived[0]["risk"])


class WekaWriteSupportTests(unittest.TestCase):
    def test_defaults_to_warn_when_unset(self):
        result = cufile_config.check_weka_write_support({})
        self.assertEqual(result.status, "WARN")
        self.assertIn("POSIX fallback path", result.why)
        self.assertIn("not a fixed WekaFS architectural limit", result.why)

    def test_warn_when_explicitly_false(self):
        result = cufile_config.check_weka_write_support(
            {"fs": {"weka": {"rdma_write_support": False}}}
        )
        self.assertEqual(result.status, "WARN")

    def test_info_when_enabled(self):
        result = cufile_config.check_weka_write_support(
            {"fs": {"weka": {"rdma_write_support": True}}}
        )
        self.assertEqual(result.status, "INFO")
        self.assertIn("may use GDS/RDMA", result.why)

    def test_loads_config_when_not_provided(self):
        old_loader = cufile_config._load_cufile_json_with_path
        try:
            cufile_config._load_cufile_json_with_path = lambda: (
                {"fs": {"weka": {"rdma_write_support": True}}},
                "/tmp/cufile.json",
            )
            result = cufile_config.check_weka_write_support()
        finally:
            cufile_config._load_cufile_json_with_path = old_loader
        self.assertEqual(result.status, "INFO")


if __name__ == "__main__":
    unittest.main()
