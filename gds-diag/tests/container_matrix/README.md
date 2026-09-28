# Container Matrix Tests

These tests launch real Docker, Enroot, and/or Kubernetes containers and are
skipped during normal `unittest discover` runs unless explicitly enabled
(`GDS_DIAG_CONTAINER_TESTS=1`). They don't decide pass/fail on their own
(see [Assertions](#assertions)) -- run them, then read the generated report.

The runner script these tests exec inside the container needs `python3` on
its `PATH`. For Docker and Enroot, the image must already have it -- there's
no bootstrap step for those engines. Kubernetes is more forgiving: its pod
installs `python3` via `apt-get` on first start if missing (Debian/Ubuntu
images only; see [Sweep multiple images](#sweep-multiple-images)), so a
plain `nvcr.io/nvidia/cuda` image works there without modification.
`python:3.11-slim` (a small public image with no CUDA/GDS of its own) is
used throughout the examples below because it already satisfies the
requirement on every engine and needs no registry auth, which makes it a
good default for confirming the mechanics work before testing a real
CUDA/GDS image.

## Prerequisites

### Docker
- `docker` installed and the daemon reachable (`docker info` succeeds).
- `nvidia-container-toolkit` installed and registered as a Docker runtime
  (check `docker info | grep -i runtime`; should list `nvidia`). Three of
  the four Docker cases pass `--gpus=all`, which fails at the daemon without
  it -- only `docker_minimal` works without the toolkit.

### Enroot
- `enroot` and `squashfs-tools` installed (`squashfs-tools` provides `mksquashfs`,
  used to build the `.sqsh` file on import).
- `squashfuse` installed too, so `enroot start` can mount that `.sqsh` directly
  instead of falling back to a full `unsquashfs` extract on every run (slower,
  and uses disk space equal to the image's uncompressed size). Enroot itself
  can run without it via that fallback, but this test's Enroot case checks for
  `squashfuse` up front and reports `squashfuse missing` rather than trying it.

### Kubernetes
- `kubectl` installed, with `KUBECONFIG` pointing at a config the user
  running pytest can read (`kubectl get nodes` succeeds).
- A GPU-capable node the pod can schedule onto, and a `runtimeClassName` that
  exists on the cluster (default `nvidia`; check with `kubectl get runtimeclass`).
- **Before running this for the first time on a given cluster**, confirm
  whether GDS is provided by a pre-installed host driver or deployed
  in-cluster by the GPU Operator:
  `kubectl get clusterpolicies.nvidia.com -o yaml | grep -A5 '^\s*gds:'`.
  This materially changes what's needed. With a pre-installed driver (the
  common case, and what these tests assume by default), `/dev/nvidia-fs*`,
  `nvidia-fs.ko`, and the host CUDA/GDS packages already exist outside
  Kubernetes and just need exposing into the pod. With an operator-managed
  driver, GDS must be enabled in the `ClusterPolicy` first
  (`gds.enabled: true`), which deploys an `nvidia-fs` driver pod to every
  node -- a real cluster-wide change this test will not make for you.
- On a **multi-node** cluster: the pod uses `hostPath` mounts for the repo
  and (if set) `GDS_DIAG_K8S_HOST_CUDA`, so it must land on a node where
  those paths actually exist. Pin it with `GDS_DIAG_K8S_NODE_NAME` if the
  default scheduler might place it elsewhere.

## Quickstart

Confirm the mechanics work for whichever engine(s) you care about, using one
image and no version sweep. Run from the repository root:

```bash
# Docker
GDS_DIAG_CONTAINER_TESTS=1 \
GDS_DIAG_CONTAINER_ENGINE=docker \
GDS_DIAG_CONTAINER_IMAGE=python:3.11-slim \
python3 -m unittest tests.container_matrix.test_container_matrix -v
```

```bash
# Enroot (imports python:3.11-slim from the registry -- no local unpacked
# container needed, just squashfuse; see Prerequisites above)
GDS_DIAG_CONTAINER_TESTS=1 \
GDS_DIAG_CONTAINER_ENGINE=enroot \
GDS_DIAG_CONTAINER_IMAGE=python:3.11-slim \
python3 -m unittest tests.container_matrix.test_container_matrix -v
```

```bash
# Kubernetes (GDS_DIAG_K8S_HOST_CUDA is set here because python:3.11-slim has
# no CUDA/GDS of its own -- see the note on that variable further down)
export KUBECONFIG=/path/to/your/kubeconfig
GDS_DIAG_CONTAINER_TESTS=1 \
GDS_DIAG_CONTAINER_ENGINE=k8s \
GDS_DIAG_K8S_IMAGE=python:3.11-slim \
GDS_DIAG_K8S_HOST_CUDA=/usr/local/cuda \
python3 -m unittest tests.container_matrix.test_container_matrix -v
```

Or all three engines in one run:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
GDS_DIAG_CONTAINER_TESTS=1 \
GDS_DIAG_CONTAINER_ENGINE=all \
GDS_DIAG_CONTAINER_IMAGE=python:3.11-slim \
GDS_DIAG_K8S_HOST_CUDA=/usr/local/cuda \
python3 -m unittest tests.container_matrix.test_container_matrix -v
```

Each run writes a report to `tests/container_matrix/results/<timestamp>/` --
see [Output](#output). Open `index.md` and read it. A nonzero exit in the
table isn't necessarily a bug in these tests -- it can mean `gds-diag.py`
itself correctly reported WARN/FAIL for that container shape (e.g. the
`_minimal` Docker case has no GPU or GDS devices exposed on purpose).

The environment assignments above apply to that one `python3` command. If
you set them on separate shell prompts instead, `export` them first.

## Advanced usage

### Sweep multiple images

Each engine has a real "matrix" mode, but they invert what varies:

- **Docker and Enroot fix the image and vary privilege level.** For one
  image, they automatically run several cases (minimal -> GPU -> host GDS
  tools -> full observability) to see how `gds-diag` responds as more of the
  host is exposed -- see [Case shapes](#case-shapes).
- **Kubernetes fixes the privilege level and varies the image.** A Pod spec
  is committed upfront rather than built from incremental CLI flags, so
  there's no equivalent "add one more mount and rerun" step -- every k8s
  case already requests the same full-observability shape (repo,
  `/run/udev`, `/sys`, `/etc/cufile.json`, `/dev`; see
  [Case shapes](#case-shapes)). What's cheap to vary instead is *which
  image* runs in that fixed shape, each with its own pod, run, and teardown.

  That inversion is what makes the k8s engine the fast path for comparing
  `gds-diag` across CUDA/GDS versions: point it at a comma-separated list of
  images and it spins up one pod per image against the same, already-tested
  pod shape. To do that:
  ```bash
  export KUBECONFIG=/path/to/your/kubeconfig
  GDS_DIAG_CONTAINER_TESTS=1 \
  GDS_DIAG_CONTAINER_ENGINE=k8s \
  GDS_DIAG_K8S_IMAGES=nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04,nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04,nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04,nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04,nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04,nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  python3 -m unittest tests.container_matrix.test_container_matrix -v
  ```
  Plain `nvcr.io/nvidia/cuda` images have neither `python3` nor `nvidia-gds`
  -- the pod installs both via `apt-get` on first start if missing (matching
  `nvidia-gds` to the image's own CUDA version; see the `command` in
  [`k8s/pod-template.yaml`](k8s/pod-template.yaml)), so this works without
  needing a hand-picked image, and `support-matrix --live` uses real
  `gdscheck` output rather than falling back to the documentation matrix.

  **CUDA versions older than 12.2 don't have a matching `nvidia-gds` apt
  package.** Those images will still run, but gds-diag will correctly
  report `gdscheck`/live GDS as unavailable rather than exercising a real
  install -- that's a limitation of this apt-based bootstrap method, not a
  `gds-diag` bug. Swap in whichever CUDA versions you actually want to
  compare -- the six above are just what's verified working here.

  Pulling from `nvcr.io` may require NGC authentication for gated/private
  catalog content; see NVIDIA's NGC registry docs:
  https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#accessing-ngc-registry

  Leave `GDS_DIAG_K8S_HOST_CUDA` unset for this mode -- each image now
  installs its own matching `nvidia-gds`; binding one host toolkit over
  every pod would make every case test the same GDS version regardless of
  image, defeating the point of the sweep.

  **One step further: add real GDS I/O to the sweep.** The three commands
  above (`container-check`, `support-matrix --live`, `config-audit`) confirm
  each version launches cleanly and reports itself correctly, but not
  whether it can actually do GDS I/O. Add `GDS_DIAG_K8S_GDS_MOUNT` and every
  image in the sweep also runs `mount-check` against a real filesystem --
  this is the complete picture, not just "does it start."

  `mount-check`'s O_DIRECT probe creates (and deletes) one small temp file
  in that path, so it needs to be a real, writable directory -- **point it
  at a small scratch directory, not a filesystem root.** A repo-relative
  one keeps this self-contained and is created automatically if it doesn't
  exist (gitignored, same as `results/`):

  ```bash
  export KUBECONFIG=/path/to/your/kubeconfig
  GDS_DIAG_CONTAINER_TESTS=1 \
  GDS_DIAG_CONTAINER_ENGINE=k8s \
  GDS_DIAG_K8S_IMAGES=nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04,nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04,nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04,nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04,nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04,nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  GDS_DIAG_K8S_GDS_MOUNT=$PWD/tests/container_matrix/scratch \
  python3 -m unittest tests.container_matrix.test_container_matrix -v
  ```

  Point `GDS_DIAG_K8S_GDS_MOUNT` at a different path to test a specific real
  filesystem instead of the repo-relative scratch dir -- just keep it
  scoped to a directory you're fine having a small file written to.

### Reuse an existing Enroot container or sqsh

```bash
GDS_DIAG_CONTAINER_TESTS=1 \
GDS_DIAG_CONTAINER_ENGINE=enroot \
GDS_DIAG_ENROOT_CONTAINER=nvps_dynamo-trt-llm-kvbm-aiperf \
python3 -m unittest tests.container_matrix.test_container_matrix -v
```

Use `GDS_DIAG_ENROOT_SQSH=/path/to/image.sqsh` instead to reuse an already
imported `.sqsh` file without re-importing.

### The k8s pod template

The pod shape lives in [`k8s/pod-template.yaml`](k8s/pod-template.yaml), not
inline Python, on purpose: it's a plain `$VAR`-style template (the same
substitution style `envsubst` uses), so you can read the exact manifest as
YAML in a PR diff, or apply it by hand outside pytest to debug a
scheduling/mount problem -- see the comment at the top of that file for the
manual `envsubst` invocation.

### mount-check on Docker/Enroot

`container-check` itself does not take a target path. `GDS_DIAG_GDS_MOUNT`
does the same job for Docker/Enroot that `GDS_DIAG_K8S_GDS_MOUNT` does for
Kubernetes in [Sweep multiple images](#sweep-multiple-images) above -- same
mechanism (bind-mounts the path in and additionally runs `mount-check`
against it), same scratch-directory guidance.

## Environment variables

```bash
GDS_DIAG_CONTAINER_ENGINE=all          # docker, enroot, k8s, or all
GDS_DIAG_CONTAINER_IMAGE=...           # image for docker/enroot; also the k8s fallback if no k8s-specific var is set
GDS_DIAG_HOST_CUDA=/usr/local/cuda     # host CUDA/GDS path to bind (docker/enroot)
GDS_DIAG_GDS_MOUNT=/path/to/mount      # optional host path for a separate mount-check run
GDS_DIAG_CONTAINER_TIMEOUT=180
GDS_DIAG_CONTAINER_RESULTS=/tmp/gds-diag-container-results
GDS_DIAG_ENROOT_CONTAINER=name         # optional existing unpacked Enroot container (skips import; see Advanced usage)
GDS_DIAG_ENROOT_SQSH=/path/to/image.sqsh  # optional reuse of an already-imported sqsh image

GDS_DIAG_K8S_IMAGE=...                 # single image (or use the comma list below for a sweep)
GDS_DIAG_K8S_IMAGES=img1,img2          # comma list of images to matrix over; overrides GDS_DIAG_K8S_IMAGE
GDS_DIAG_K8S_NAMESPACE=default         # namespace the test pods are created/deleted in
GDS_DIAG_K8S_RUNTIME_CLASS=nvidia      # runtimeClassName on the pod (check `kubectl get runtimeclass`)
GDS_DIAG_K8S_HOST_CUDA=/usr/local/cuda-13.2  # optional: bind host CUDA/GDS tools into the pod at the same path.
                                        # Leave unset for a version sweep (see Advanced usage); set it for a
                                        # quick sanity check against a minimal image with no CUDA/GDS of its own.
GDS_DIAG_K8S_GDS_MOUNT=/path/to/mount  # optional hostPath for a real k8s mount-check run (falls back to GDS_DIAG_GDS_MOUNT)
GDS_DIAG_K8S_NODE_NAME=node-1          # optional: pin the pod to a specific node (see Prerequisites: multi-node)
GDS_DIAG_K8S_GPU_COUNT=1               # nvidia.com/gpu resource request
```

## Case shapes

The Docker full-observability case prefers explicit `--device` mappings for
host `/dev/nvidia-fs*` and `/dev/infiniband/*` nodes that exist, rather than
using `--privileged` as the default. Use a separate manual run with
`--privileged` only when you want a broad diagnostic shortcut.

The Enroot full-observability case adds a read-only host `/dev` bind with
`--mount /dev:/dev:none:rbind,ro`, plus `/run/udev` and `/sys` when present.
This covers sites where the default NVIDIA or Mellanox Enroot hooks do not
expose all GDS-relevant device nodes.

The Kubernetes case always requests the repo, `/run/udev`, `/sys`,
`/etc/cufile.json`, and `/dev` -- there's no incremental privilege-level
variant like Docker/Enroot, since a Pod spec is committed upfront (see
[Sweep multiple images](#sweep-multiple-images) for what does vary).

## Output

By default, the tests produce `index.md`, `index.html`, and raw logs under:

```text
tests/container_matrix/results/<timestamp>/
```

If `GDS_DIAG_CONTAINER_RESULTS` is set, results are written under that
directory instead.

## Assertions

The assertions intentionally avoid host-specific GDS success criteria. They
check that `container-check` and related commands run without Python
tracebacks, produce a report, and describe missing observability inputs
clearly. They should be inspected manually to confirm expected
recommendations.

NVIDIA's MagnumIO Docker guidance is useful background for these launch shapes:
https://github.com/NVIDIA/MagnumIO/blob/main/gds/docker/README.md
