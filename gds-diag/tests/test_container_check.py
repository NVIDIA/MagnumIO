# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from checks import container
from checks.result import Status
from subcommands import container_check


class ContainerCheckLogicTests(unittest.TestCase):
    def _collect_with(self, **overrides):
        defaults = {
            "context": container.ContainerContext(True, "docker", "/.dockerenv exists"),
            "gdscheck": None,
            "libcufile": [],
            "which": None,
            "command": (False, "nvidia-smi not found"),
            "exists_any": [],
            "isdir": False,
            "exists": False,
            "listdir": [],
            "listdir_side_effect": None,
        }
        defaults.update(overrides)

        def fake_exists_any(patterns):
            joined = " ".join(patterns)
            if "/dev/nvidia-fs" in joined:
                return defaults.get("nvidia_fs_devices", [])
            if "/dev/nvidia" in joined:
                return defaults.get("nvidia_devices", [])
            if "/dev/infiniband/uverbs" in joined:
                return defaults.get("uverbs", [])
            if "/dev/nvme" in joined:
                return defaults.get("nvme", [])
            return defaults["exists_any"]

        def fake_exists(path):
            present = set(defaults.get("present_paths", []))
            return path in present or defaults["exists"]

        def fake_isdir(path):
            present_dirs = set(defaults.get("present_dirs", []))
            return path in present_dirs or defaults["isdir"]

        listdir_kwargs = (
            {"side_effect": defaults["listdir_side_effect"]}
            if defaults["listdir_side_effect"] is not None
            else {"return_value": defaults["listdir"]}
        )

        with mock.patch.object(container, "detect_container_context", return_value=defaults["context"]), \
             mock.patch.object(container, "find_gdscheck", return_value=defaults["gdscheck"]), \
             mock.patch.object(container, "find_libcufile", return_value=defaults["libcufile"]), \
             mock.patch.object(container, "_exists_any", side_effect=fake_exists_any), \
             mock.patch.object(container.shutil, "which", return_value=defaults["which"]), \
             mock.patch.object(container, "_command_works", return_value=defaults["command"]), \
             mock.patch.object(container.os.path, "exists", side_effect=fake_exists), \
             mock.patch.object(container.os.path, "isdir", side_effect=fake_isdir), \
             mock.patch.object(container.os, "listdir", **listdir_kwargs):
            return container.collect_sections()

    def test_minimal_container_warns_without_gds_observability(self):
        sections = self._collect_with()

        tools = {result.check: result for result in sections["Diagnostic Tools"]}
        runtime_files = {result.check: result for result in sections["GDS Runtime Files"]}
        runtime_devices = {result.check: result for result in sections["Runtime Devices"]}
        readiness = {result.check: result for result in sections["Command Readiness"]}

        self.assertEqual(tools["nvidia-smi"].status, Status.INFO)
        self.assertIn("not installed", tools["nvidia-smi"].why)
        self.assertIn("not by itself a GDS runtime blocker", tools["nvidia-smi"].why)
        self.assertIn("--gpus=all", tools["nvidia-smi"].mitigation)
        self.assertIn("NVML libraries", tools["nvidia-smi"].mitigation)
        self.assertEqual(tools["gdscheck"].status, Status.INFO)
        self.assertIn("GDS workloads may still work", tools["gdscheck"].why)
        self.assertEqual(runtime_devices["NVIDIA GPU devices"].status, Status.WARN)
        self.assertEqual(runtime_devices["nvidia-fs devices"].status, Status.WARN)
        self.assertIn("P2PDMA/C2C or compat routes may still work", runtime_devices["nvidia-fs devices"].why)
        self.assertIn("Map each host /dev/nvidia-fs* node", runtime_devices["nvidia-fs devices"].mitigation)
        self.assertIn("--privileged", runtime_devices["nvidia-fs devices"].mitigation)
        self.assertEqual(runtime_devices["udev database"].status, Status.WARN)
        self.assertIn("GDS may be unable to resolve block-device metadata", runtime_devices["udev database"].why)
        self.assertNotIn("NVMe devices", runtime_devices)
        self.assertIn("only matters for RDMA-backed GDS routes", runtime_devices["RDMA connection manager"].why)
        self.assertIn("only matters for RDMA-backed GDS routes", runtime_devices["RDMA uverbs devices"].why)
        self.assertIn("libcufile", runtime_files)
        self.assertEqual(readiness["support-matrix --live"].status, Status.INFO)
        self.assertIn("fall back", readiness["support-matrix --live"].why)

    def test_nvidia_smi_present_but_unusable_recommends_device_exposure(self):
        sections = self._collect_with(
            which="/usr/bin/nvidia-smi",
            command=(False, "Failed to initialize NVML"),
        )

        tools = {result.check: result for result in sections["Diagnostic Tools"]}
        self.assertEqual(tools["nvidia-smi"].status, Status.INFO)
        self.assertIn("installed", tools["nvidia-smi"].why)
        self.assertIn("cannot list GPUs", tools["nvidia-smi"].why)
        self.assertIn("not by itself a GDS runtime blocker", tools["nvidia-smi"].why)
        self.assertIn("--gpus=all", tools["nvidia-smi"].mitigation)
        self.assertIn("Failed to initialize NVML", tools["nvidia-smi"].evidence)

    def test_enroot_container_uses_enroot_specific_mitigations(self):
        sections = self._collect_with(
            context=container.ContainerContext(True, "enroot", "ENROOT_ENVIRON set"),
            which="/usr/bin/nvidia-smi",
            command=(False, "Failed to initialize NVML"),
        )

        tools = {result.check: result for result in sections["Diagnostic Tools"]}
        runtime_devices = {result.check: result for result in sections["Runtime Devices"]}

        self.assertIn("Enroot NVIDIA hook", tools["nvidia-smi"].mitigation)
        self.assertIn("98-nvidia.sh", runtime_devices["NVIDIA GPU devices"].mitigation)
        self.assertIn("--mount /dev:/dev:none:rbind,ro", runtime_devices["NVIDIA GPU devices"].mitigation)
        self.assertIn("/dev/nvidia-fs", runtime_devices["nvidia-fs devices"].mitigation)
        self.assertIn("--mount /dev:/dev:none:rbind,ro", runtime_devices["nvidia-fs devices"].mitigation)
        self.assertIn("-m /run/udev:/run/udev:none:x-create=dir,rbind,ro:0:0", runtime_devices["udev database"].mitigation)

    def test_full_container_inputs_make_mount_check_ready(self):
        sections = self._collect_with(
            gdscheck="/usr/local/cuda/gds/tools/gdscheck",
            libcufile=["/usr/local/cuda/lib64/libcufile.so.0"],
            which="/usr/bin/nvidia-smi",
            command=(True, "GPU 0: test"),
            nvidia_devices=["/dev/nvidia0", "/dev/nvidiactl"],
            nvidia_fs_devices=["/dev/nvidia-fs0"],
            nvme=["/dev/nvme0n1"],
            uverbs=["/dev/infiniband/uverbs0"],
            present_paths=["/etc/cufile.json", "/proc/self/mountinfo", "/dev/infiniband/rdma_cm"],
            present_dirs=["/sys/bus/pci/devices", "/run/udev"],
            listdir=["data"],
        )

        readiness = {result.check: result for result in sections["Command Readiness"]}
        runtime_devices = {result.check: result for result in sections["Runtime Devices"]}

        self.assertEqual(runtime_devices["udev database"].status, Status.PASS)
        self.assertEqual(readiness["mount-check"].status, Status.PASS)
        self.assertEqual(readiness["post-install"].status, Status.PASS)

    def test_udev_listdir_error_warns_without_raising(self):
        sections = self._collect_with(
            present_dirs=["/run/udev"],
            listdir_side_effect=OSError("permission denied"),
        )

        runtime_devices = {result.check: result for result in sections["Runtime Devices"]}
        udev = runtime_devices["udev database"]

        self.assertEqual(udev.status, Status.WARN)
        self.assertIn("GDS may be unable to resolve block-device metadata", udev.why)
        self.assertIn("Unable to list /run/udev", udev.evidence)
        self.assertIn("permission denied", udev.evidence)

    def test_json_output_contains_sections(self):
        args = argparse.Namespace(json=True, verbose=False)
        fake_sections = {
            "Container Context": [
                container._result("Container runtime", Status.INFO, "Running inside docker.")
            ]
        }
        with mock.patch.object(
            container,
            "detect_container_context",
            return_value=container.ContainerContext(True, "docker", "/.dockerenv exists"),
        ), mock.patch.object(container, "collect_sections", return_value=fake_sections):
            out = io.StringIO()
            with redirect_stdout(out):
                rc = container_check.COMMAND.run(args)

        payload = json.loads(out.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["command"], "container-check")
        self.assertNotIn("profile", payload)
        self.assertNotIn("target", payload)
        self.assertIn("Container Context", payload["sections"])

    def test_host_text_output_short_circuits(self):
        args = argparse.Namespace(json=False, verbose=False)
        with mock.patch.object(
            container,
            "detect_container_context",
            return_value=container.ContainerContext(False, "host", "No common container markers found."),
        ), mock.patch.object(container, "collect_sections") as collect:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = container_check.COMMAND.run(args)

        self.assertEqual(rc, 2)
        self.assertIn("meant to be run inside a Docker or Enroot container", out.getvalue())
        self.assertIn("--runtime docker", out.getvalue())
        self.assertIn("--runtime enroot", out.getvalue())
        collect.assert_not_called()

    def test_host_verbose_text_includes_tool_version(self):
        args = argparse.Namespace(json=False, verbose=True)
        with mock.patch.object(
            container,
            "detect_container_context",
            return_value=container.ContainerContext(False, "host", "No common container markers found."),
        ), mock.patch.object(container, "collect_sections") as collect:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = container_check.COMMAND.run(args)

        self.assertEqual(rc, 2)
        self.assertIn("Tool: gds-diag", out.getvalue())
        collect.assert_not_called()

    def test_host_json_output_short_circuits(self):
        args = argparse.Namespace(json=True, verbose=False)
        with mock.patch.object(
            container,
            "detect_container_context",
            return_value=container.ContainerContext(False, "host", "No common container markers found."),
        ), mock.patch.object(container, "collect_sections") as collect:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = container_check.COMMAND.run(args)

        payload = json.loads(out.getvalue())
        self.assertEqual(rc, 2)
        self.assertFalse(payload["in_container"])
        self.assertEqual(payload["error"], "container_not_detected")
        self.assertIn("container-check --runtime docker", payload["override_hint"])
        collect.assert_not_called()

    def test_forced_runtime_bypasses_host_short_circuit(self):
        args = argparse.Namespace(json=False, verbose=False, runtime="enroot")
        with mock.patch.object(
            container,
            "detect_container_context",
            return_value=container.ContainerContext(False, "host", "No common container markers found."),
        ), mock.patch.object(container, "collect_sections") as collect:
            collect.return_value = {
                "Container Context": [
                    container._result(
                        "Container runtime",
                        Status.INFO,
                        "Running inside a likely enroot container.",
                        evidence="Runtime forced by --runtime enroot.",
                    )
                ]
            }
            out = io.StringIO()
            with redirect_stdout(out):
                rc = container_check.COMMAND.run(args)

        self.assertEqual(rc, 0)
        self.assertIn("Running inside a likely enroot", out.getvalue())
        forced_context = collect.call_args.args[0]
        self.assertEqual(forced_context.runtime, "enroot")
        self.assertTrue(forced_context.in_container)
        self.assertIn("Runtime forced by --runtime enroot", forced_context.evidence)

    def test_forced_runtime_json_reports_override(self):
        args = argparse.Namespace(json=True, verbose=False, runtime="docker")
        fake_sections = {
            "Container Context": [
                container._result("Container runtime", Status.INFO, "Running inside a likely docker container.")
            ]
        }
        with mock.patch.object(
            container,
            "detect_container_context",
            return_value=container.ContainerContext(False, "host", "No common container markers found."),
        ), mock.patch.object(container, "collect_sections", return_value=fake_sections):
            out = io.StringIO()
            with redirect_stdout(out):
                rc = container_check.COMMAND.run(args)

        payload = json.loads(out.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["runtime"], "docker")
        self.assertIn("--runtime docker", payload["runtime_detection"])
        self.assertIn("sections", payload)

    def test_forced_context_is_visible_in_context_row(self):
        sections = self._collect_with(
            context=container.ContainerContext(
                True,
                "enroot",
                "Runtime forced by --runtime enroot.\nAuto-detection found no Docker or Enroot container markers.",
            )
        )

        context = sections["Container Context"][0]
        self.assertIn("Assuming an enroot container", context.why)
        self.assertIn("--runtime enroot", context.why)

    def test_overlay_mountinfo_alone_is_not_container_detection(self):
        with mock.patch.object(container.os.path, "exists", return_value=False), \
             mock.patch.object(container.os, "environ", {}), \
             mock.patch.object(container, "_read_text") as read_text:
            read_text.side_effect = lambda path, limit=65536: (
                "0::/user.slice\n" if path == "/proc/self/cgroup" else "1 2 0:1 / / rw - overlay overlay rw\n"
            )
            context = container.detect_container_context()

        self.assertFalse(context.in_container)
        self.assertEqual(context.runtime, "host")

    def test_enroot_root_mountinfo_detects_enroot(self):
        with mock.patch.object(container.os.path, "exists", return_value=False), \
             mock.patch.object(container.os, "environ", {}), \
             mock.patch.object(container, "_read_text") as read_text:
            read_text.side_effect = lambda path, limit=65536: (
                "0::/\n"
                if path == "/proc/self/cgroup"
                else (
                    "1332 1130 259:2 "
                    "/localhome/user/.local/share/enroot/example / "
                    "ro,nosuid,nodev,relatime - ext4 /dev/nvme0n1p2 rw\n"
                )
            )
            context = container.detect_container_context()

        self.assertTrue(context.in_container)
        self.assertEqual(context.runtime, "enroot")
        self.assertIn("mountinfo root mount", context.evidence)

    def test_kubernetes_service_host_env_detects_kubepods_under_cgroup_v2(self):
        # A private cgroup namespace (the current runc/crun default on a
        # cgroup v2 host, regardless of orchestrator) reports
        # "/proc/self/cgroup" as "0::/" from inside the container, so the
        # old cgroup-string check alone can't see it.
        with mock.patch.object(container.os.path, "exists", return_value=False), \
             mock.patch.object(container.os, "environ", {"KUBERNETES_SERVICE_HOST": "10.43.0.1"}), \
             mock.patch.object(container, "_read_text") as read_text:
            read_text.side_effect = lambda path, limit=65536: (
                "0::/\n" if path == "/proc/self/cgroup" else ""
            )
            context = container.detect_container_context()

        self.assertTrue(context.in_container)
        self.assertEqual(context.runtime, "kubepods")
        self.assertIn("KUBERNETES_SERVICE_HOST", context.evidence)

    def test_serviceaccount_mount_detects_kubepods_under_cgroup_v2(self):
        with mock.patch.object(
            container.os.path,
            "exists",
            side_effect=lambda path: path == "/var/run/secrets/kubernetes.io/serviceaccount",
        ), mock.patch.object(container.os, "environ", {}), \
           mock.patch.object(container, "_read_text") as read_text:
            read_text.side_effect = lambda path, limit=65536: (
                "0::/\n" if path == "/proc/self/cgroup" else ""
            )
            context = container.detect_container_context()

        self.assertTrue(context.in_container)
        self.assertEqual(context.runtime, "kubepods")
        self.assertIn("serviceaccount is mounted", context.evidence)

    def test_containerd_root_mountinfo_detects_containerd_under_cgroup_v2(self):
        # Observed on a k3s/containerd node: the pod's root filesystem is an
        # overlay mount whose lower/upper dirs reference containerd's
        # snapshotter, even though cgroup and env markers are both absent.
        with mock.patch.object(container.os.path, "exists", return_value=False), \
             mock.patch.object(container.os, "environ", {}), \
             mock.patch.object(container, "_read_text") as read_text:
            read_text.side_effect = lambda path, limit=65536: (
                "0::/\n"
                if path == "/proc/self/cgroup"
                else (
                    "6604 6177 0:307 / / rw,relatime - overlay overlay "
                    "rw,lowerdir=/var/lib/rancher/k3s/agent/containerd/"
                    "io.containerd.snapshotter.v1.overlayfs/snapshots/255/fs\n"
                )
            )
            context = container.detect_container_context()

        self.assertTrue(context.in_container)
        self.assertEqual(context.runtime, "containerd")
        self.assertIn("containerd snapshot", context.evidence)


if __name__ == "__main__":
    unittest.main()
