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
#   ./build-x86_64.sh                   # build locally only
#   ./build-x86_64.sh --push-cache      # build + push image + layer cache to GHCR
#                                        # (primes GitHub Actions for fast subsequent builds)
#   ./build-x86_64.sh --no-cache        # force fresh build, ignore local cache
#   ./build-x86_64.sh --no-cache --push-cache  # fresh build + prime GHCR cache

set -euo pipefail

IMAGE_LOCAL="aeon-vllm-x86:sm120"
GHCR_IMAGE="ghcr.io/enricoferrero64/aeon-vllm-x86:sm120"
GHCR_CACHE="ghcr.io/enricoferrero64/aeon-vllm-x86:buildcache"
DOCKERFILE="Dockerfile.x86_64"
PUSH_CACHE=false
NO_CACHE=false

for arg in "$@"; do
    case "$arg" in
        --push-cache) PUSH_CACHE=true ;;
        --no-cache)   NO_CACHE=true ;;
        *) echo "[ERROR] Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "========================================"
echo " AEON vLLM x86_64 + sm_120 Builder"
echo " Local image:  $IMAGE_LOCAL"
if $PUSH_CACHE; then
    echo " GHCR image:   $GHCR_IMAGE"
    echo " GHCR cache:   $GHCR_CACHE"
fi
echo " Push cache:   $PUSH_CACHE"
echo " No cache:     $NO_CACHE"
echo "========================================"

# Sanity checks
if ! command -v docker &>/dev/null; then
    echo "[ERROR] docker not found in PATH"; exit 1
fi

for f in "$DOCKERFILE" patches/patch_cuda_optional_import.py patches/patch_kv_cache_utils.py \
          patches/patch_cudagraph_align.py humming-stub/setup.py verify.py; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Missing required file: $f"
        echo "        Run from the vllm-x86-sm120 repo root."
        exit 1
    fi
done

# Use docker buildx so we can target linux/amd64 explicitly and use registry cache
BUILDER=$(docker buildx ls | grep -E '^aeon-builder' | awk '{print $1}' || true)
if [[ -z "$BUILDER" ]]; then
    docker buildx create --name aeon-builder --driver docker-container --use
else
    docker buildx use aeon-builder
fi

BUILD_ARGS=(
    --platform linux/amd64
    --file "$DOCKERFILE"
    --tag "$IMAGE_LOCAL"
    --build-arg BUILDKIT_INLINE_CACHE=1
    --progress plain
)

if $PUSH_CACHE; then
    BUILD_ARGS+=(
        --tag "$GHCR_IMAGE"
        # Pull warm layers from GHCR if available (speeds up incremental rebuilds)
        --cache-from "type=registry,ref=$GHCR_CACHE"
        # Push all intermediate layers back to GHCR cache (primes GitHub Actions)
        --cache-to  "type=registry,ref=$GHCR_CACHE,mode=max"
        --push
    )
else
    # Load into local Docker daemon (--push and --load are mutually exclusive)
    BUILD_ARGS+=(--load)
fi

if $NO_CACHE; then
    BUILD_ARGS+=(--no-cache)
fi

echo ""
echo "[INFO] Starting build — CUDA compilation takes 60–90 min on RTX 5060 Ti..."
echo "[INFO] Full log: /tmp/aeon-vllm-build.log"
echo ""

docker buildx build "${BUILD_ARGS[@]}" . 2>&1 | tee /tmp/aeon-vllm-build.log

echo ""
echo "========================================"
echo " BUILD SUCCESSFUL"
if $PUSH_CACHE; then
    echo " Image pushed:  $GHCR_IMAGE"
    echo " Cache pushed:  $GHCR_CACHE"
    echo ""
    echo " GitHub Actions will now use the GHCR cache."
    echo " Future CI builds only recompile changed layers."
    echo ""
    echo " Update docker-compose.x86_64.yml image: to:"
    echo "   image: $GHCR_IMAGE"
else
    echo " Image ready:   $IMAGE_LOCAL (local only)"
    echo ""
    echo " To prime GitHub Actions cache, re-run with --push-cache"
    echo " (requires: gh auth login or GITHUB_TOKEN with packages:write)"
fi
echo ""
echo " Next: docker compose -f docker-compose.x86_64.yml up -d"
echo "========================================"
