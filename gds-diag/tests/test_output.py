# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks.output import (
    render_mitigation_plan, render_sections_table, render_summary,
    render_version_context,
)
from checks.result import CheckResult, GDSMode, Status


class OutputRenderingTests(unittest.TestCase):
    def test_render_sections_table_normal_mode_shows_only_issues(self):
        sections = {
            "System": [
                CheckResult(
                    check="CUDA Toolkit",
                    mode=GDSMode.NATIVE,
                    status=Status.FAIL,
                    why="CUDA Toolkit not found.",
                    mitigation="Install CUDA Toolkit.",
                ),
                CheckResult(
                    check="Operating system",
                    mode=GDSMode.NATIVE,
                    status=Status.PASS,
                    why="Linux detected.",
                ),
            ]
        }

        text = "\n".join(render_sections_table(sections))

        self.assertIn("System", text)
        self.assertIn("Check", text)
        self.assertIn("Status", text)
        self.assertIn("Finding", text)
        self.assertNotIn("Mitigation", text)
        self.assertIn("CUDA Toolkit", text)
        self.assertIn("FAIL", text)
        self.assertNotIn("Operating system", text)

    def test_render_version_context_prints_versions_only(self):
        rows = [
            {
                "component": "libcufile",
                "version": "1.17.1",
                "source": "/usr/local/cuda/lib64/libcufile.so",
            },
            {
                "component": "nvidia-fs",
                "version": "2.26.6",
                "source": "modinfo nvidia_fs",
            },
        ]

        text = "\n".join(render_version_context(rows))

        self.assertIn("libcufile", text)
        self.assertIn("1.17.1", text)
        self.assertIn("nvidia-fs", text)
        self.assertIn("2.26.6", text)
        self.assertNotIn("/usr/local/cuda", text)
        self.assertNotIn("modinfo nvidia_fs", text)

    def test_render_sections_table_verbose_includes_pass_rows(self):
        sections = {
            "System": [
                CheckResult(
                    check="Operating system",
                    mode=GDSMode.NATIVE,
                    status=Status.PASS,
                    why="Linux detected.",
                ),
            ]
        }

        text = "\n".join(render_sections_table(sections, verbose=True))

        self.assertIn("Operating system", text)
        self.assertIn("PASS", text)
        self.assertIn("Linux detected.", text)

    def test_render_sections_table_normal_mode_reports_no_issues(self):
        sections = {
            "System": [
                CheckResult(
                    check="Operating system",
                    mode=GDSMode.NATIVE,
                    status=Status.PASS,
                    why="Linux detected.",
                ),
            ]
        }

        text = "\n".join(render_sections_table(sections))

        self.assertIn("No warnings or errors.", text)
        self.assertNotIn("Operating system", text)

    def test_render_sections_table_normal_mode_shows_info(self):
        sections = {
            "P2PDMA/C2C Direct Routes": [
                CheckResult(
                    check="P2PDMA topology candidates",
                    mode=GDSMode.P2PDMA,
                    status=Status.INFO,
                    why="Cross-root topology is active but suboptimal.",
                    mitigation="Prefer closer placement for peak performance.",
                ),
                CheckResult(
                    check="Active P2PDMA/C2C routes",
                    mode=GDSMode.P2PDMA,
                    status=Status.PASS,
                    why="NVMe: c2c, nvfs, compat.",
                ),
            ]
        }

        text = "\n".join(render_sections_table(sections))

        self.assertIn("P2PDMA/C2C Direct Routes", text)
        self.assertIn("INFO", text)
        self.assertIn("P2PDMA topology", text)
        self.assertIn("Cross-root topology is active", text)
        self.assertNotIn("Active P2PDMA/C2C routes", text)

    def test_info_is_summarized_but_not_in_mitigation_plan(self):
        sections = {
            "P2PDMA/C2C Direct Routes": [
                CheckResult(
                    check="P2PDMA topology candidates",
                    mode=GDSMode.P2PDMA,
                    status=Status.INFO,
                    why="Cross-root topology is active but suboptimal.",
                    mitigation="Prefer closer placement for peak performance.",
                ),
            ]
        }

        summary = render_summary(sections)
        mitigation = "\n".join(render_mitigation_plan(sections))

        self.assertIn("1 info", summary)
        self.assertNotIn("warning", summary)
        self.assertIn("No warnings or errors require mitigation.", mitigation)
        self.assertNotIn("P2PDMA topology candidates", mitigation)

    def test_render_mitigation_plan_preserves_multiline_mitigation(self):
        sections = {
            "Installation": [
                CheckResult(
                    check="nvidia-fs module",
                    mode=GDSMode.NATIVE,
                    status=Status.FAIL,
                    why="nvidia_fs is installed but not loaded.",
                    mitigation="Run:\n  sudo modprobe nvidia_fs\nThen rerun post-install.",
                ),
            ]
        }

        plan = "\n".join(render_mitigation_plan(sections))

        self.assertIn("sudo modprobe nvidia_fs", plan)
        self.assertIn("Then rerun post-install.", plan)

    def test_p2pdma_registry_table_defers_exact_command_to_mitigation_plan(self):
        long_command = (
            'options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;'
            'ForceP2P=0;RmForceDisableIomapWC=1;"'
        )
        sections = {
            "P2PDMA Direct Routes": [
                CheckResult(
                    check="NVIDIA P2PDMA driver registries",
                    mode=GDSMode.P2PDMA,
                    status=Status.WARN,
                    why="The loaded NVIDIA driver is missing required x86 PCI P2PDMA registry dwords.",
                    mitigation=(
                        "Configure /etc/modprobe.d/nvidia-p2pdma.conf with:\n"
                        f"  {long_command}\n"
                        "Then rebuild initramfs, reboot, and verify:\n"
                        "  cat /proc/driver/nvidia/params | grep -i static"
                    ),
                ),
            ]
        }

        table = "\n".join(render_sections_table(sections))
        plan = "\n".join(render_mitigation_plan(sections))

        self.assertNotIn(long_command, table)
        self.assertIn(long_command, plan)
        self.assertIn("cat /proc/driver/nvidia/params", plan)

    def test_render_sections_table_redacts_urls_from_finding(self):
        url = (
            "https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/"
            "index.html#doca-requirements-and-installation"
        )
        sections = {
            "RDMA": [
                CheckResult(
                    check="MLNX_OFED / DOCA",
                    mode=GDSMode.RDMA,
                    status=Status.WARN,
                    why=f"DOCA not found. See: {url}",
                    mitigation=f"Read the GDS DOCA requirements: {url}",
                ),
            ]
        }

        table = "\n".join(render_sections_table(sections))
        plan = "\n".join(render_mitigation_plan(sections))

        self.assertNotIn(url, table)
        self.assertIn("Mitigation Plan for link.", table)
        self.assertIn(url, plan)
        self.assertIn("GDS DOCA requirements", plan)

    def test_render_mitigation_plan_summarizes_warnings_and_failures(self):
        sections = {
            "GPU": [
                CheckResult(
                    check="NVIDIA Open Driver install",
                    mode=GDSMode.NATIVE,
                    status=Status.FAIL,
                    why="Driver missing.",
                    mitigation="Install the NVIDIA Open Driver.",
                ),
                CheckResult(
                    check="NVIDIA GPU (lspci)",
                    mode=GDSMode.NATIVE,
                    status=Status.PASS,
                    why="GPU found.",
                ),
            ],
            "IOMMU": [
                CheckResult(
                    check="IOMMU mode",
                    mode=GDSMode.P2PDMA,
                    status=Status.WARN,
                    why="Strict IOMMU.",
                ),
            ],
        }

        text = "\n".join(render_mitigation_plan(sections))

        self.assertIn("Mitigation Plan", text)
        self.assertIn("NVIDIA Open Driver install", text)
        self.assertIn("Install the NVIDIA Open Driver.", text)
        self.assertIn("IOMMU mode", text)
        self.assertIn("Review before workload testing", text)
        self.assertNotIn("NVIDIA GPU (lspci)", text)

    def test_render_mitigation_plan_keeps_long_urls_copyable(self):
        url = (
            "https://docs.nvidia.com/doca/sdk/"
            "doca-host-installation-and-upgrade/index.html#storage-installation"
        )
        sections = {
            "RDMA": [
                CheckResult(
                    check="MLNX_OFED / DOCA",
                    mode=GDSMode.RDMA,
                    status=Status.WARN,
                    why="DOCA not found.",
                    mitigation=f"Install DOCA from {url}",
                ),
            ],
        }

        lines = render_mitigation_plan(sections)
        text = "\n".join(lines)

        self.assertIn(url, text)
        self.assertNotIn("[link 1]", text)
        self.assertNotIn("Copyable Links", text)
        self.assertTrue(any(line.strip() == url for line in lines))


if __name__ == "__main__":
    unittest.main()
