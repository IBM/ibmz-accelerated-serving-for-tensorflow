#!/bin/bash
# Prerequisites for the fashion-mnist serving sample.
#
# Usage:
#   ./prerequisites.sh <tf-base-image>
#
# <tf-base-image> must be an IBM Z Accelerated for TensorFlow production image, e.g.:
#   icr.io/ibmz/ibmz-accelerated-for-tensorflow:1.6.0
#
# The script builds a new container image with tensorflow-serving-api pre-installed.
# A timestamped image tag is generated so re-running the script never
# overwrites a previously built image.
#
# After the build, the image tag and example docker run commands are printed.
# See the sample README for full instructions on running the sample.
#
# NOTE: This script only sets up the TensorFlow container for running the
# training and inference scripts. The TensorFlow Serving container is started
# separately after training — see README.md for the full workflow.
#
# When you are finished with the sample, remove the workspace volume with:
#   docker volume rm fashion-mnist-serving-workspace

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
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
IMAGE_TAG="fashion-mnist-serving-sample:${TIMESTAMP}"
VOLUME_NAME="fashion-mnist-serving-workspace"

echo "Building sample image from ${BASE_IMAGE} ..."
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "Build complete."
echo ""
echo "Image tag: ${IMAGE_TAG}"
echo ""
echo "To start the sample container for training, run:"
echo ""
echo "  docker run -it --rm \\"
echo "      -v ${VOLUME_NAME}:/workspace \\"
echo "      -w /workspace \\"
echo "      ${IMAGE_TAG} \\"
echo "      bash"
echo ""
echo "To start the container for inference (adds --network=host to reach the"
echo "serving container on localhost), run:"
echo ""
echo "  docker run -it --rm \\"
echo "      --network=host \\"
echo "      -v ${VOLUME_NAME}:/workspace \\"
echo "      -w /workspace \\"
echo "      ${IMAGE_TAG} \\"
echo "      bash"
echo ""
echo "See the sample README for instructions on copying scripts into the container."
echo ""
echo "When finished, remove the workspace volume with:"
echo "  docker volume rm ${VOLUME_NAME}"
echo ""
echo "If using rootless Podman, run 'docker container prune -f' after exiting"
echo "the container to ensure networking processes are cleaned up."
