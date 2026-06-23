#!/usr/bin/env bash
# Build aeon-vllm-x86:sm120 on an x86_64 host with RTX 5060 Ti (sm_120)
# Run this on the Unraid host (or any Linux x86_64 machine with Docker + NVIDIA runtime)
#
# Requirements:
#   - Docker >= 27.0 with NVIDIA Container Toolkit installed
#   - CUDA 13.0+ driver (NVIDIA driver >= 575 for Blackwell sm_120)
#   - ~40 GB free disk space (build cache + final image)
#   - 60–90 minutes build time (CUDA kernel compilation)
#   - Internet access (clones lesj0610/vllm, downloads PyTorch/flashinfer wheels)
#
# Usage:
#   chmod +x build-x86_64.sh
#   ./build-x86_64.sh              # build with defaults
#   ./build-x86_64.sh --no-cache   # force fresh build (no Docker layer cache)

set -euo pipefail

IMAGE="aeon-vllm-x86:sm120"
DOCKERFILE="Dockerfile.x86_64"
EXTRA_ARGS=("$@")

echo "========================================"
echo " AEON vLLM x86_64 + sm_120 Builder"
echo " Image: $IMAGE"
echo " Dockerfile: $DOCKERFILE"
echo "========================================"

# Sanity checks
if ! command -v docker &>/dev/null; then
    echo "[ERROR] docker not found in PATH"
    exit 1
fi

if ! docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "[WARN] NVIDIA runtime not detected in docker info. Ensure nvidia-container-toolkit is installed."
    echo "       Build may still succeed (CUDA kernels compile without GPU at build time)."
fi

# Verify build context has required files
for f in "$DOCKERFILE" patches/patch_cuda_optional_import.py patches/patch_kv_cache_utils.py \
          patches/patch_cudagraph_align.py humming-stub/setup.py verify.py; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Missing required file: $f"
        echo "        Make sure you are running from the vllm-x86-sm120 repo root."
        exit 1
    fi
done

echo ""
echo "[INFO] Starting build — this will take 60–90 minutes..."
echo "[INFO] Logs: build output will stream to stdout + /tmp/aeon-vllm-build.log"
echo ""

docker build \
    --platform linux/amd64 \
    --file "$DOCKERFILE" \
    --tag "$IMAGE" \
    --progress plain \
    "${EXTRA_ARGS[@]}" \
    . 2>&1 | tee /tmp/aeon-vllm-build.log

BUILD_EXIT=${PIPESTATUS[0]}

if [[ $BUILD_EXIT -eq 0 ]]; then
    echo ""
    echo "========================================"
    echo " BUILD SUCCESSFUL: $IMAGE"
    echo " Image size: $(docker image inspect $IMAGE --format '{{.Size}}' | numfmt --to=iec 2>/dev/null || echo 'unknown')"
    echo ""
    echo " Next steps:"
    echo "   1. Copy docker-compose.x86_64.yml to Unraid compose manager"
    echo "   2. Update model path / HF token in compose if needed"
    echo "   3. docker compose -f docker-compose.x86_64.yml up -d"
    echo "   4. Monitor: docker logs -f aeon_vllm"
    echo "   5. Test: curl http://localhost:8000/health"
    echo "========================================"
else
    echo ""
    echo "[ERROR] Build failed (exit $BUILD_EXIT). Check /tmp/aeon-vllm-build.log"
    echo "Common issues:"
    echo "  - lesj0610/vllm fork unavailable: see build-x86_64.sh for cherry-pick fallback"
    echo "  - flashinfer source build: check sm_120 Triton support in your CUDA 13.0 install"
    echo "  - modelopt not found: harmless WARN, NVFP4 weights still load via compressed-tensors"
    exit $BUILD_EXIT
fi
