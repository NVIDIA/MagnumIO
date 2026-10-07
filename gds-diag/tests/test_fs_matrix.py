# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from unittest import mock

from checks import fs_matrix


class FsMatrixBackingDeviceTests(unittest.TestCase):
    def test_stacked_md_device_with_nvme_slave_is_nvme_backed(self):
        old_listdir = fs_matrix.os.listdir
        try:
            slaves = {
                "/sys/class/block/md0/slaves": ["nvme0n1", "nvme1n1"],
            }
            fs_matrix.os.listdir = lambda path: slaves[path]

            self.assertTrue(fs_matrix._block_device_has_nvme_leaf("md0"))
        finally:
            fs_matrix.os.listdir = old_listdir

    def test_nested_stacked_device_with_nvme_leaf_is_nvme_backed(self):
        old_listdir = fs_matrix.os.listdir
        try:
            slaves = {
                "/sys/class/block/dm-0/slaves": ["md0"],
                "/sys/class/block/md0/slaves": ["nvme0n1p1"],
            }
            fs_matrix.os.listdir = lambda path: slaves[path]

            self.assertTrue(fs_matrix._block_device_has_nvme_leaf("dm-0"))
        finally:
            fs_matrix.os.listdir = old_listdir

    def test_stacked_device_without_nvme_leaf_is_not_nvme_backed(self):
        old_listdir = fs_matrix.os.listdir
        try:
            slaves = {
                "/sys/class/block/md0/slaves": ["sda", "sdb"],
                "/sys/class/block/sda/slaves": [],
                "/sys/class/block/sdb/slaves": [],
            }
            fs_matrix.os.listdir = lambda path: slaves[path]

            self.assertFalse(fs_matrix._block_device_has_nvme_leaf("md0"))
        finally:
            fs_matrix.os.listdir = old_listdir

    def test_get_raid_level_reads_backing_md_device(self):
        old_backing = fs_matrix.get_backing_device
        try:
            fs_matrix.get_backing_device = lambda path: "/dev/md0"

            def fake_open(path, *args, **kwargs):
                if path == "/sys/class/block/md0/md/level":
                    return mock.mock_open(read_data="raid0\n").return_value
                raise FileNotFoundError(path)

            with mock.patch("builtins.open", fake_open):
                self.assertEqual(fs_matrix.get_raid_level("/raid"), "raid0")
        finally:
            fs_matrix.get_backing_device = old_backing

    def test_block_device_dm_info_classifies_lvm_uuid(self):
        old_listdir = fs_matrix.os.listdir
        old_isdir = fs_matrix.os.path.isdir
        try:
            slaves = {"/sys/class/block/dm-0/slaves": []}
            fs_matrix.os.listdir = lambda path: slaves[path]
            fs_matrix.os.path.isdir = lambda path: path == "/sys/class/block/dm-0/dm"

            def fake_open(path, *args, **kwargs):
                if path == "/sys/class/block/dm-0/dm/uuid":
                    return mock.mock_open(read_data="LVM-abcdef0123456789\n").return_value
                raise FileNotFoundError(path)

            with mock.patch("builtins.open", fake_open):
                self.assertEqual(fs_matrix._block_device_dm_info("dm-0"), ["LVM"])
        finally:
            fs_matrix.os.listdir = old_listdir
            fs_matrix.os.path.isdir = old_isdir

    def test_block_device_dm_info_walks_through_nested_slave(self):
        old_listdir = fs_matrix.os.listdir
        old_isdir = fs_matrix.os.path.isdir
        try:
            slaves = {
                "/sys/class/block/dm-1/slaves": ["nvme0n1p1"],
                "/sys/class/block/nvme0n1p1/slaves": [],
            }
            fs_matrix.os.listdir = lambda path: slaves[path]
            fs_matrix.os.path.isdir = lambda path: path == "/sys/class/block/dm-1/dm"

            def fake_open(path, *args, **kwargs):
                if path == "/sys/class/block/dm-1/dm/uuid":
                    return mock.mock_open(read_data="CRYPT-LUKS2-abcdef\n").return_value
                raise FileNotFoundError(path)

            with mock.patch("builtins.open", fake_open):
                self.assertEqual(fs_matrix._block_device_dm_info("dm-1"), ["dm-crypt"])
        finally:
            fs_matrix.os.listdir = old_listdir
            fs_matrix.os.path.isdir = old_isdir

    def test_block_device_without_dm_dir_has_no_dm_info(self):
        old_listdir = fs_matrix.os.listdir
        old_isdir = fs_matrix.os.path.isdir
        try:
            slaves = {"/sys/class/block/nvme0n1/slaves": []}
            fs_matrix.os.listdir = lambda path: slaves[path]
            fs_matrix.os.path.isdir = lambda path: False

            with mock.patch("builtins.open", side_effect=FileNotFoundError):
                self.assertEqual(fs_matrix._block_device_dm_info("nvme0n1"), [])
        finally:
            fs_matrix.os.listdir = old_listdir
            fs_matrix.os.path.isdir = old_isdir

    def test_block_device_dm_dir_with_empty_uuid_is_still_detected(self):
        # A bare `dmsetup create` with no --uuid leaves DM_UUID empty, but the
        # dm/ sysfs directory itself is kernel-created and always present —
        # detection must not depend on a recognized/non-empty uuid string.
        old_listdir = fs_matrix.os.listdir
        old_isdir = fs_matrix.os.path.isdir
        try:
            slaves = {"/sys/class/block/dm-2/slaves": []}
            fs_matrix.os.listdir = lambda path: slaves[path]
            fs_matrix.os.path.isdir = lambda path: path == "/sys/class/block/dm-2/dm"

            def fake_open(path, *args, **kwargs):
                if path == "/sys/class/block/dm-2/dm/uuid":
                    return mock.mock_open(read_data="\n").return_value
                raise FileNotFoundError(path)

            with mock.patch("builtins.open", fake_open):
                self.assertEqual(fs_matrix._block_device_dm_info("dm-2"), ["device-mapper"])
        finally:
            fs_matrix.os.listdir = old_listdir
            fs_matrix.os.path.isdir = old_isdir

    def test_sysfs_block_name_resolves_mapper_symlink(self):
        # /dev/mapper/<vg>-<lv> is a symlink to /dev/dm-N; only dm-N exists in sysfs.
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "dm-0")
            link = os.path.join(tmp, "vg0-lv0")
            open(target, "w").close()
            os.symlink(target, link)
            self.assertEqual(fs_matrix._sysfs_block_name(link), "dm-0")

    def test_get_dm_info_reads_backing_dm_device(self):
        old_backing = fs_matrix.get_backing_device
        try:
            fs_matrix.get_backing_device = lambda path: "/dev/mapper/vg0-lv0"

            with mock.patch.object(fs_matrix, "_sysfs_block_name", return_value="dm-0") as normalize, \
                    mock.patch.object(fs_matrix, "_block_device_dm_info", return_value=["LVM"]) as mocked:
                self.assertEqual(fs_matrix.get_dm_info("/mnt/lv0"), "LVM")
                normalize.assert_called_once_with("/dev/mapper/vg0-lv0")
                mocked.assert_called_once_with("dm-0")
        finally:
            fs_matrix.get_backing_device = old_backing


class Ext4DataModeTests(unittest.TestCase):
    def _check_with_mounts(self, mounts: str, path: str = "/") -> tuple[str, str]:
        old_realpath = fs_matrix.os.path.realpath
        try:
            fs_matrix.os.path.realpath = lambda value: value

            def fake_open(file_path, *args, **kwargs):
                if file_path == "/proc/mounts":
                    return mock.mock_open(read_data=mounts).return_value
                raise FileNotFoundError(file_path)

            with mock.patch("builtins.open", fake_open):
                result = fs_matrix.check_ext4_data_mode(path)
        finally:
            fs_matrix.os.path.realpath = old_realpath

        self.assertIsNotNone(result)
        return result

    def test_ext4_explicit_data_ordered_is_distinguished(self):
        mode, evidence = self._check_with_mounts(
            "/dev/nvme0n1p2 / ext4 rw,relatime,data=ordered 0 0\n"
        )

        self.assertEqual(mode, "ordered")
        self.assertIn("data=ordered", evidence)

    def test_ext4_implicit_default_is_not_reported_as_ordered(self):
        mode, evidence = self._check_with_mounts(
            "/dev/nvme0n1p2 / ext4 rw,relatime 0 0\n"
        )

        self.assertEqual(mode, "default")
        self.assertIn("implicit ext4 default", evidence)


if __name__ == "__main__":
    unittest.main()
