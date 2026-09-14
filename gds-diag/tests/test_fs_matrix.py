# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
