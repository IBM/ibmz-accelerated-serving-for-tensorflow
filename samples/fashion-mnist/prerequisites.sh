#!/bin/bash
# Prerequisites for the fashion-mnist serving sample.
#
# Usage:
#   ./prerequisites.sh <tf-base-image>
#
# <tf-base-image> must be an IBM Z Accelerated for TensorFlow production image, e.g.:
#   icr.io/ibmz/ibmz-accelerated-for-tensorflow:1.6.0
#
# The script builds a new container image with tensorflow-serving-api pre-installed,
# then starts an interactive shell inside it. Sample scripts are mounted
# read-only at /scripts. A user-owned workspace directory is created alongside
# the sample scripts and mounted at /workspace — this is where output files
# (trained model, saved model, etc.) will be written.
#
# NOTE: This script only sets up the TensorFlow container for running the
# training and inference scripts. The TensorFlow Serving container is started
# separately after training — see README.md for the full workflow.

set -euo pipefail

BASE_IMAGE="${1:-}"
if [[ -z "${BASE_IMAGE}" ]]; then
    echo "Error: base image argument is required." >&2
    echo "Usage: ./prerequisites.sh <tf-base-image>" >&2
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "Error: docker not found. Run this script on the host, not inside a container." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/workspace"
IMAGE_TAG="fashion-mnist-serving-sample:latest"

# Under rootless podman (including when invoked via a docker alias), use the
# cgroupfs cgroup manager to avoid systemd unit errors in SSH sessions, and
# remap workspace ownership into the container's UID namespace.
# Neither applies to real Docker (no podman present).
PODMAN_EXTRA_FLAGS=()
if command -v podman &>/dev/null; then
    PODMAN_EXTRA_FLAGS=(--cgroup-manager=cgroupfs)
fi

echo "Building sample image from ${BASE_IMAGE} ..."
docker build \
    "${PODMAN_EXTRA_FLAGS[@]}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

# Create the workspace directory
mkdir -p "${WORKSPACE_DIR}"

if command -v podman &>/dev/null; then
    # Query ibm-user's numeric UID from the built image so podman unshare chown
    # can use it (podman unshare runs inside the user namespace where usernames
    # are not resolved — only numeric UIDs are valid).
    IBM_USER_UID=$(docker run --rm "${PODMAN_EXTRA_FLAGS[@]}" --entrypoint id "${IMAGE_TAG}" -u)
    if [[ -z "${IBM_USER_UID}" ]]; then
        echo "Error: could not determine ibm-user UID from image ${IMAGE_TAG}" >&2
        exit 1
    fi
    podman unshare chown "${IBM_USER_UID}:${IBM_USER_UID}" "${WORKSPACE_DIR}"
fi

echo ""
echo "Build complete."
echo ""
echo "Inside the container, run the training script from /workspace, e.g.:"
echo "  python /scripts/fashion_mnist_training.py"
echo ""
echo "To run inference against a running serving container, re-run this script"
echo "with 'inference' as the second argument to enable --network=host, e.g.:"
echo "  ./prerequisites.sh <tf-base-image> inference"
echo ""

MODE="${2:-}"

NETWORK_FLAGS=()
if [[ "${MODE}" == "inference" ]]; then
    NETWORK_FLAGS=(--network=host)
fi

docker run -it --rm \
    "${PODMAN_EXTRA_FLAGS[@]}" \
    "${NETWORK_FLAGS[@]}" \
    -v "${SCRIPT_DIR}":/scripts:ro,z \
    -v "${WORKSPACE_DIR}":/workspace:z \
    -w /workspace \
    "${IMAGE_TAG}" \
    bash
