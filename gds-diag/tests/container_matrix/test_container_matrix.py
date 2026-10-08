# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import glob
import os
import pathlib
import re
import shutil
import string
import subprocess
import time
import unittest


REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = pathlib.Path(__file__).resolve().parent / "results"
K8S_POD_TEMPLATE = pathlib.Path(__file__).resolve().parent / "k8s" / "pod-template.yaml"


def _enabled() -> bool:
    return os.environ.get("GDS_DIAG_CONTAINER_TESTS") == "1"


def _enroot_mount(source: str | pathlib.Path, destination: str, *, kind: str = "dir", writable: bool = False) -> str:
    access = "rw" if writable else "ro"
    create = "x-create=file" if kind == "file" else "x-create=dir"
    bind = "bind" if kind == "file" else "rbind"
    return f"{source}:{destination}:none:{create},{bind},{access}:0:0"


def _docker_device_args(patterns: tuple[str, ...]) -> list[str]:
    args: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            args.extend(["--device", path])
    return args


def _repo_mount_args(engine: str, runner: pathlib.Path) -> list[str]:
    if engine == "docker":
        return [
            "-v", f"{REPO}:/work:ro",
            "-v", f"{runner}:/runner.py:ro",
            "-w", "/work",
        ]
    return [
        "-m", _enroot_mount(REPO, "/work"),
        "-m", _enroot_mount(runner, "/tmp/container_runner.py", kind="file"),
    ]


def _runner_path(engine: str) -> str:
    return "/runner.py" if engine == "docker" else "/tmp/container_runner.py"


def _k8s_pod_manifest(
    *,
    pod_name: str,
    namespace: str,
    image: str,
    runtime_class: str,
    repo_hostpath: pathlib.Path,
    node_name: str | None,
    host_cuda: str | None,
    gds_mount: str | None,
    gpu_count: str,
) -> str:
    # Every optional piece of the pod (host CUDA/GDS tools, a real GDS mount,
    # node pinning) is filled in as a pre-formatted YAML fragment, or an
    # empty string when unused. Flow-style mappings ({name: ..., ...}) are
    # used throughout so this stays a single find-and-replace per fragment,
    # with no indentation-sensitive block YAML to get wrong from Python.
    extra_env = ""
    extra_volume_mounts = ""
    extra_volumes = ""
    if host_cuda:
        extra_env += (
            f'        - {{name: PATH, value: "{host_cuda}/gds/tools:{host_cuda}/bin:'
            '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}\n'
            f'        - {{name: LD_LIBRARY_PATH, value: "{host_cuda}/lib64"}}\n'
        )
        extra_volume_mounts += f'        - {{name: host-cuda, mountPath: "{host_cuda}", readOnly: true}}\n'
        extra_volumes += f'    - {{name: host-cuda, hostPath: {{path: "{host_cuda}", type: Directory}}}}\n'
    if gds_mount:
        extra_env += '        - {name: GDS_MOUNT_IN_CONTAINER, value: "/mnt/gds"}\n'
        extra_volume_mounts += '        - {name: gds-mount, mountPath: /mnt/gds, readOnly: false}\n'
        # DirectoryOrCreate, not Directory: the harness already mkdir's this
        # on the machine running pytest, but on a multi-node cluster the pod
        # can land on a different node than that -- this covers that case too.
        extra_volumes += f'    - {{name: gds-mount, hostPath: {{path: "{gds_mount}", type: DirectoryOrCreate}}}}\n'

    template = string.Template(K8S_POD_TEMPLATE.read_text(encoding="utf-8"))
    return template.safe_substitute(
        POD_NAME=pod_name,
        NAMESPACE=namespace,
        IMAGE=image,
        RUNTIME_CLASS=runtime_class,
        REPO_HOSTPATH=str(repo_hostpath),
        NODE_NAME_LINE=f"  nodeName: {node_name}\n" if node_name else "",
        EXTRA_ENV=extra_env,
        EXTRA_VOLUME_MOUNTS=extra_volume_mounts,
        EXTRA_VOLUMES=extra_volumes,
        GPU_COUNT=gpu_count,
    )


def _kubectl_apply(log: pathlib.Path, manifest: str, namespace: str, timeout: int) -> int:
    return _run(log, ["kubectl", "apply", "-n", namespace, "-f", "-"], timeout, input_text=manifest)


def _kubectl_wait_ready(log: pathlib.Path, pod_name: str, namespace: str, timeout: int) -> int:
    return _run(
        log,
        ["kubectl", "wait", f"pod/{pod_name}", "-n", namespace, "--for=condition=Ready", f"--timeout={timeout}s"],
        timeout + 10,
    )


def _kubectl_delete(pod_name: str, namespace: str, timeout: int) -> None:
    # Best-effort cleanup: not reported as a matrix row (a delete failure
    # here isn't diagnostically interesting), but never allowed to raise and
    # mask the real test outcome.
    try:
        subprocess.run(
            ["kubectl", "delete", "pod", pod_name, "-n", namespace, "--ignore-not-found", "--wait=false"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def _sanitize_k8s_name(value: str, limit: int) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:limit]


def _runner_source() -> str:
    return """#!/usr/bin/env python3
import os
import subprocess
import sys

os.chdir(os.environ.get("GDS_DIAG_WORKDIR", "/work"))
env = os.environ.copy()
env["PATH"] = "/usr/local/cuda/bin:/usr/local/cuda/gds/tools:/usr/local/cuda-13.2/bin:/usr/local/cuda-13.2/gds/tools:" + env.get("PATH", "")

commands = [
    ["python3", "gds-diag.py", "container-check", "-v"],
    ["python3", "gds-diag.py", "support-matrix", "--live", "-v"],
    ["python3", "gds-diag.py", "config-audit", "-v"],
]
mount_check_path = env.get("GDS_MOUNT_IN_CONTAINER")
if mount_check_path:
    commands.append(["python3", "gds-diag.py", "mount-check", mount_check_path, "-v"])

overall_rc = 0
for command in commands:
    print("\\n$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, env=env, text=True)
    print("[exit:%s] %s" % (completed.returncode, " ".join(command)), flush=True)
    overall_rc = max(overall_rc, completed.returncode)
sys.exit(overall_rc)
"""


def _write_runner(path: pathlib.Path) -> None:
    path.write_text(_runner_source(), encoding="utf-8")
    path.chmod(0o755)


def _run(log: pathlib.Path, command: list[str], timeout: int, input_text: str | None = None) -> int:
    timeout_rc = 124
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(command) + "\n\n")
        fh.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO),
                input=input_text,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            fh.write(f"\n[TIMEOUT] command exceeded {timeout} seconds\n")
            fh.write(f"{type(exc).__name__}: {exc}\n")
            if exc.output:
                fh.write("\n[timeout-output]\n")
                fh.write(exc.output.decode("utf-8", errors="replace") if isinstance(exc.output, bytes) else str(exc.output))
                fh.write("\n")
            if exc.stderr:
                fh.write("\n[timeout-stderr]\n")
                fh.write(exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr))
                fh.write("\n")
            fh.write(f"\n[case-exit:{timeout_rc}]\n")
            fh.flush()
            return timeout_rc
        fh.write("\n[case-exit:%s]\n" % completed.returncode)
        return completed.returncode


def _html_report(markdown: str) -> str:
    body = []
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            body.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            body.append(html.escape(line))
            continue
        escaped = html.escape(line)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
        if line.startswith("# "):
            body.append(f"<h1>{escaped[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{escaped[3:]}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{escaped[4:]}</h3>")
        elif line.strip():
            body.append(f"<p>{escaped}</p>")
        else:
            body.append("")
    return (
        "<!doctype html><meta charset='utf-8'><title>gds-diag container matrix</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}"
        "pre{background:#111827;color:#f9fafb;padding:12px;overflow:auto}code{background:#f3f4f6;padding:2px 4px}</style>"
        + "\n".join(body)
    )


def _shorten(line: str, limit: int = 180) -> str:
    return line if len(line) <= limit else line[: limit - 4] + " ..."


def _is_issue_row(line: str) -> bool:
    return bool(
        re.search(r"\|\s*[^|]+\|\s*(WARN|FAIL|ERROR)\s*\|", line)
        or re.match(r"\s*(WARN|FAIL|ERROR)\s+\S+", line)
        # kubectl/docker/enroot's own CLI errors ("error: ..."), as opposed
        # to gds-diag.py's own WARN/FAIL/ERROR rows above.
        or re.match(r"\s*error:", line)
    )


def _is_summary_line(line: str) -> bool:
    return bool(re.search(r"\b\d+ passed\b", line)) or "No warnings or errors require mitigation." in line


def _extract_highlights(text: str) -> list[str]:
    highlights: list[str] = []
    include_wrapped_row = False
    in_recommendations = False
    recommendation_lines = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if line.startswith("$ "):
            include_wrapped_row = False
            in_recommendations = False
            recommendation_lines = 0
            if line.startswith("$ enroot start ") or line.startswith("$ docker run "):
                highlights.append(_shorten(line))
            else:
                highlights.append(line)
            continue

        if stripped.startswith("[exit:") or stripped.startswith("[case-exit:"):
            include_wrapped_row = False
            highlights.append(line)
            continue

        if stripped in {
            "GDS Container Check",
            "Container Launch Recommendations",
        } or stripped.startswith("GDS Filesystem Support Matrix"):
            include_wrapped_row = False
            in_recommendations = stripped == "Container Launch Recommendations"
            recommendation_lines = 0
            highlights.append(line)
            continue

        if _is_summary_line(line):
            include_wrapped_row = False
            highlights.append(line)
            if "No warnings or errors require mitigation." in line:
                in_recommendations = False
            continue

        if _is_issue_row(line):
            include_wrapped_row = line.lstrip().startswith("|")
            highlights.append(line)
            continue

        if include_wrapped_row and line.lstrip().startswith("|") and "No action needed" not in line:
            highlights.append(line)
            continue
        include_wrapped_row = False

        if in_recommendations:
            if stripped in {"Docker Guidance", "Enroot Guidance"}:
                in_recommendations = False
                continue
            if stripped.startswith("$ ") or stripped.startswith("[exit:"):
                in_recommendations = False
            elif stripped and recommendation_lines < 14:
                highlights.append(line)
                recommendation_lines += 1
            continue

    if not highlights and text.strip():
        highlights.append(_shorten(text.strip().splitlines()[0]))
    return highlights


@unittest.skipUnless(_enabled(), "set GDS_DIAG_CONTAINER_TESTS=1 to run container integration tests")
class ContainerMatrixTests(unittest.TestCase):
    def test_container_matrix(self):
        image = os.environ.get("GDS_DIAG_CONTAINER_IMAGE")
        enroot_container = os.environ.get("GDS_DIAG_ENROOT_CONTAINER")

        engines = os.environ.get("GDS_DIAG_CONTAINER_ENGINE", "docker").split(",")
        if engines == ["all"]:
            engines = ["docker", "enroot", "k8s"]
        if not image and ("docker" in engines or ("enroot" in engines and not enroot_container)):
            self.skipTest("set GDS_DIAG_CONTAINER_IMAGE to a runnable image")
        timeout = int(os.environ.get("GDS_DIAG_CONTAINER_TIMEOUT", "180"))
        host_cuda = os.environ.get("GDS_DIAG_HOST_CUDA", "/usr/local/cuda")
        def _prepare_gds_mount(path: str, env_var: str) -> None:
            # mount-check's O_DIRECT probe writes one small temp file here, so
            # this needs to actually exist and be writable -- create it if
            # it's a fresh scratch dir, and fail clearly (not with a raw
            # traceback mid-sweep) if the path is unusable, before any
            # matrix cases run.
            try:
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.fail(f"{env_var}={path!r} could not be created: {exc}")
            if not os.path.isdir(path) or not os.access(path, os.W_OK):
                self.fail(f"{env_var}={path!r} is not a writable directory")

        gds_mount = os.environ.get("GDS_DIAG_GDS_MOUNT")
        if gds_mount:
            _prepare_gds_mount(gds_mount, "GDS_DIAG_GDS_MOUNT")
        results_root = pathlib.Path(os.environ.get("GDS_DIAG_CONTAINER_RESULTS", DEFAULT_RESULTS))
        out_dir = results_root / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        runner = out_dir / "container_runner.py"
        _write_runner(runner)

        rows = []

        def add_case(name: str, engine: str, command: list[str]) -> None:
            rc = _run(log_dir / f"{name}.log", command, timeout)
            rows.append((name, engine, rc, f"logs/{name}.log"))

        if "docker" in engines:
            if not shutil.which("docker"):
                self.skipTest("docker is not installed")
            base = ["docker", "run", "--rm", "--entrypoint", "python3"]
            mount = _repo_mount_args("docker", runner)
            mount_check_args = []
            gds_device_args = _docker_device_args((
                "/dev/nvidia-fs*",
                "/dev/infiniband/rdma_cm",
                "/dev/infiniband/uverbs*",
            ))
            if gds_mount:
                mount_check_args = ["-v", f"{gds_mount}:/mnt/gds:rw", "-e", "GDS_MOUNT_IN_CONTAINER=/mnt/gds"]
            add_case("docker_minimal", "docker", base + mount + mount_check_args + [image, _runner_path("docker")])
            add_case("docker_nvidia", "docker", base + ["--gpus=all"] + mount + mount_check_args + [image, _runner_path("docker")])
            add_case(
                "docker_host_gds_tools",
                "docker",
                base + ["--gpus=all", "-v", f"{host_cuda}:{host_cuda}:ro", "-v", "/etc/cufile.json:/etc/cufile.json:ro"]
                + mount + mount_check_args + [image, _runner_path("docker")],
            )
            add_case(
                "docker_full_observability",
                "docker",
                base + [
                    "--gpus=all", "--ipc=host", "--cap-add=IPC_LOCK",
                    "-v", f"{host_cuda}:{host_cuda}:ro",
                    "-v", "/etc/cufile.json:/etc/cufile.json:ro",
                    "-v", "/run/udev:/run/udev:ro",
                    "-v", "/sys:/sys:ro",
                ] + gds_device_args + mount + mount_check_args + [image, _runner_path("docker")],
            )

        if "enroot" in engines:
            if not shutil.which("enroot"):
                self.skipTest("enroot is not installed")
            if not enroot_container and not shutil.which("squashfuse"):
                rows.append(("enroot_prereq", "enroot", 1, "squashfuse missing"))
            else:
                import_rc = 0
                import_log = log_dir / "enroot_import.log"
                if enroot_container:
                    enroot_target = enroot_container
                    import_log.write_text(f"reusing unpacked Enroot container {enroot_container}\n[case-exit:0]\n", encoding="utf-8")
                else:
                    sqsh_env = os.environ.get("GDS_DIAG_ENROOT_SQSH")
                    sqsh = pathlib.Path(sqsh_env) if sqsh_env else out_dir / "image.sqsh"
                    enroot_target = str(sqsh)
                    if sqsh_env and sqsh.exists():
                        import_log.write_text(f"reusing {sqsh}\n[case-exit:0]\n", encoding="utf-8")
                    else:
                        import_rc = _run(import_log, ["enroot", "import", "-o", str(sqsh), f"dockerd://{image}"], timeout * 4)
                rows.append(("enroot_import", "enroot", import_rc, "logs/enroot_import.log"))
                if import_rc == 0:
                    mounts = _repo_mount_args("enroot", runner) + ["-m", _enroot_mount(host_cuda, host_cuda)]
                    if pathlib.Path("/etc/cufile.json").exists():
                        mounts += ["-m", _enroot_mount("/etc/cufile.json", "/etc/cufile.json", kind="file")]
                    if gds_mount:
                        mounts += ["-m", _enroot_mount(gds_mount, "/mnt", writable=True), "-e", "GDS_MOUNT_IN_CONTAINER=/mnt"]
                    add_case("enroot_default", "enroot", ["enroot", "start", "--root"] + mounts + [enroot_target, "python3", _runner_path("enroot")])
                    full_mounts = mounts + ["--mount", "/dev:/dev:none:rbind,ro"]
                    if pathlib.Path("/run/udev").exists():
                        full_mounts += ["-m", _enroot_mount("/run/udev", "/run/udev")]
                    if pathlib.Path("/sys").exists():
                        full_mounts += ["-m", _enroot_mount("/sys", "/sys")]
                    add_case(
                        "enroot_full_observability",
                        "enroot",
                        ["enroot", "start", "--root"] + full_mounts + [enroot_target, "python3", _runner_path("enroot")],
                    )

        k8s_images: list[str] = []
        if "k8s" in engines:
            if not shutil.which("kubectl"):
                self.skipTest("kubectl is not installed")
            images_env = os.environ.get("GDS_DIAG_K8S_IMAGES") or os.environ.get("GDS_DIAG_K8S_IMAGE") or image
            if not images_env:
                self.skipTest("set GDS_DIAG_K8S_IMAGE (or a comma list in GDS_DIAG_K8S_IMAGES) to one or more images")
            k8s_images = [item.strip() for item in images_env.split(",") if item.strip()]
            namespace = os.environ.get("GDS_DIAG_K8S_NAMESPACE", "default")
            runtime_class = os.environ.get("GDS_DIAG_K8S_RUNTIME_CLASS", "nvidia")
            # Unset by default: the version matrix is meant to come from
            # each image bundling its own cuda-toolkit/nvidia-gds packages,
            # not from overlaying one shared host toolkit on every pod (that
            # would make every case test the same GDS user-space version
            # regardless of image). Set this only for a quick sanity check
            # against a minimal image that has no CUDA/GDS of its own.
            host_cuda_k8s = os.environ.get("GDS_DIAG_K8S_HOST_CUDA")
            gds_mount_k8s = os.environ.get("GDS_DIAG_K8S_GDS_MOUNT", gds_mount)
            if gds_mount_k8s:
                _prepare_gds_mount(gds_mount_k8s, "GDS_DIAG_K8S_GDS_MOUNT")
            node_name = os.environ.get("GDS_DIAG_K8S_NODE_NAME")
            gpu_count = os.environ.get("GDS_DIAG_K8S_GPU_COUNT", "1")

            for index, k8s_image in enumerate(k8s_images):
                case_name = "k8s_" + (_sanitize_k8s_name(k8s_image, 50) or "case") + f"_{index}"
                pod_name = "gds-diag-" + (_sanitize_k8s_name(k8s_image, 40) or "case") + f"-{index}"
                manifest = _k8s_pod_manifest(
                    pod_name=pod_name,
                    namespace=namespace,
                    image=k8s_image,
                    runtime_class=runtime_class,
                    repo_hostpath=REPO,
                    node_name=node_name,
                    host_cuda=host_cuda_k8s,
                    gds_mount=gds_mount_k8s,
                    gpu_count=gpu_count,
                )
                try:
                    apply_rc = _kubectl_apply(log_dir / f"{case_name}_apply.log", manifest, namespace, timeout)
                    if apply_rc != 0:
                        rows.append((case_name, "k8s", apply_rc, f"logs/{case_name}_apply.log"))
                        continue
                    wait_rc = _kubectl_wait_ready(log_dir / f"{case_name}_wait.log", pod_name, namespace, timeout)
                    if wait_rc != 0:
                        rows.append((case_name, "k8s", wait_rc, f"logs/{case_name}_wait.log"))
                        continue
                    exec_rc = _run(
                        log_dir / f"{case_name}.log",
                        ["kubectl", "exec", "-i", pod_name, "-n", namespace, "--", "python3", "-"],
                        timeout,
                        input_text=_runner_source(),
                    )
                    rows.append((case_name, "k8s", exec_rc, f"logs/{case_name}.log"))
                finally:
                    _kubectl_delete(pod_name, namespace, timeout)

        image_label = image or enroot_container or (", ".join(k8s_images) if k8s_images else None) or "not set"
        report = ["# gds-diag container matrix report", "", f"- Image: `{image_label}`", ""]
        report.append("| Case | Engine | Exit | Log |")
        report.append("| --- | --- | ---: | --- |")
        for name, engine, rc, log in rows:
            if log.startswith("logs/"):
                report.append(f"| `{name}` | `{engine}` | `{rc}` | [`{log}`]({log}) |")
            else:
                report.append(f"| `{name}` | `{engine}` | `{rc}` | {log} |")
        report.append("")
        report.append("## Highlights")
        report.append("Highlights are abbreviated. Open the linked raw log for full command output and details.")
        for name, _engine, _rc, log in rows:
            if not log.startswith("logs/"):
                continue
            text = (out_dir / log).read_text(encoding="utf-8", errors="replace")
            report.append(f"### {name}")
            report.append(f"Full log: [`{log}`]({log})")
            report.append("```text")
            report.extend(_extract_highlights(text))
            report.append("```")
        markdown = "\n".join(report) + "\n"
        (out_dir / "index.md").write_text(markdown, encoding="utf-8")
        (out_dir / "index.html").write_text(_html_report(markdown), encoding="utf-8")
        print(f"\nresults: {out_dir / 'index.md'}")

        self.assertTrue(rows, "no container cases ran")
        self.assertTrue((out_dir / "index.md").exists())
        self.assertIn("container matrix report", markdown)


if __name__ == "__main__":
    unittest.main()
