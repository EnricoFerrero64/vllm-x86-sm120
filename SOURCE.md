# vLLM source pin

## 2026-06-23 — x86_64 sm_120 port + stability/performance pass

Adapted from AEON-7/vllm-ultimate-dgx-spark (ARM64/sm_121a) to x86_64 + sm_120 (RTX 5060 Ti / GB206).

### Patches applied on top of vLLM fork

| Patch | What it does |
|---|---|
| `patch_cuda_optional_import.py` | RTLD_LAZY dlopen for SM100-only MXFP4 symbols absent on sm_120 (ported from AEON) |
| `patch_kv_cache_utils.py` | Fixes TypeError on hybrid-attention models where GDN layers have block_size=None (ported from AEON) |
| `patch_cudagraph_align.py` | Fixes cudaErrorIllegalAddress in PIECEWISE CUDA graph mode for spec-decode (ported from AEON) |
| `patch_sm120_triton_launch.py` | **New (2026-06-23)**: sets `num_warps=4, num_stages=2` explicitly for sm_12x family in `triton_unified_attention.py`, preventing Triton JIT autotuning at first-token time (~100-200 ms cold latency spike) |

### Runtime config changes vs upstream AEON

| Flag | Value | Reason |
|---|---|---|
| `--quantization` | `modelopt` | MTP-XS uses modelopt format, not compressed-tensors |
| `--speculative-config method` | `qwen3_5_mtp` | Model-specific MTP for Qwen3.6 (not generic `mtp`) |
| `--mamba-block-size` | `256` | Required for GatedDeltaNet hybrid layers in Qwen3.6 |
| `--mamba-cache-dtype` | `float16` | SSM cache precision (BF16 recurrence layers preserved) |
| `--mm-encoder-tp-mode` | `data` | Vision encoder runs DP per GPU, LLM uses TP (reduces inter-GPU comm during image encoding) |
| `--disable-custom-all-reduce` | present | **Critical**: vLLM SymmMemCommunicator doesn't support sm_120; hangs at worker init on PCIe Blackwell without this flag |
| `--enable-prefix-caching` | present | Reuse KV cache for repeated system prompts |
| `--enable-chunked-prefill` | present | Reduces peak VRAM during long-context prefill |
| `--block-size` | `32` | Better DRAM utilization for head_dim=128 GQA models |
| `--gpu-memory-utilization` | `0.92` | Safe on dedicated GDDR7 (vs 0.88 limit on Spark's unified LPDDR5X) |
| `shm_size` | `16gb` | Sufficient for NCCL TP=2 buffers with 27B model (2GB was insufficient) |

### NCCL environment variables (Blackwell PCIe, no NVLink)

| Variable | Value | Reason |
|---|---|---|
| `NCCL_DMABUF_ENABLE` | `1` | Linux DMA-BUF for direct GPU-to-GPU mapping (resolves race conditions in early Blackwell drivers) |
| `NCCL_P2P_LEVEL` | `PCI` | Explicit PCIe P2P mode; `NCCL_P2P_LEVEL=NVL` would fail without NVLink |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512,garbage_collection_threshold:0.8` | Prevents VRAM fragmentation under long-running inference |

### Architecture changes vs upstream

- Base image: `nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu22.04` (x86_64 amd64, not ARM64 AEON base)
- `TORCH_CUDA_ARCH_LIST="12.0"` (sm_120, not 12.1a)
- CUDA lib symlink path: `x86_64-linux` (not `sbsa-linux`)
- `MAX_JOBS=4` (tuned for i7-14700K + 62GB RAM to avoid OOM during compile)

---

Build was against:

- **Repo**: `lesj0610/vllm`
- **Branch**: `lesj/triton-nvfp4-kv-fork-20260602`
- **Commit**: `e4a9fbee08b14a49470cfcf6a87dd0b2bddb6345`
- **Previous pin**: `e8c77b85` (branch was rebased/updated 2026-06-23; old commit no longer in history)
- **Upstream PR**: [vllm-project/vllm#44389](https://github.com/vllm-project/vllm/pull/44389) — Triton software NVFP4 KV cache (~3× capacity)

To reproduce the build:

```bash
git clone --filter=blob:none \
  --branch lesj/triton-nvfp4-kv-fork-20260602 \
  https://github.com/lesj0610/vllm.git vllm-src
cd vllm-src
git checkout e4a9fbee08b14a49470cfcf6a87dd0b2bddb6345
# Then: cd .. && ./build-x86_64.sh
```

The full source is not vendored in this repo (~140 MB) — only the patches, Dockerfile, humming-stub, verify script, bench tooling, and bench artifacts.

## 2026-06-11 — PR #40898 + #41703 overlay (`:2026-06-11-pr41703` = `:latest`)

DFlash drafter fixes merged ahead of upstream (both PRs open at merge time; the z-lab
drafter README pins the #41703 revision):
- vLLM tree: `aeon-dflash-fix` branch = `main@2026-06-05 merge (542fe78)` + merge of
  `pull/41703/head` (contains #40898). 5 conflicts resolved; key resolution: kept the PR's
  KV-shape helper structure but re-grafted PR #44389's per-spec KV dtype
  (`get_attn_backend_cache_dtype_str`) at both `_get_attention_kv_cache_shape` call sites,
  and re-established `shape_block_size`/`cache_dtype_str` for the MLA `page_size_padded` branch.
- Both PRs touch only Python (the DFlash kernel is Triton), so the image is a thin overlay:
  see `Dockerfile.pr41703-layer` (copies 11 files into site-packages, re-applies the AEON
  patches — the merge touches `kv_cache_utils.py` — and smoke-asserts the fixes are present).
- ⚠️ Drafter `attention_backend` must be `flash_attn` on this image; `flex_attention` crashes
  on a non-contiguous KV view (upstream's KV-sharing path is only tested with flash_attn).

## Build it yourself (advanced)

Most users should just pull the prebuilt image (`docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest`). To reproduce it from source:

**Prereqs:** a DGX Spark (GB10 / sm_121a) or another Blackwell sm_120/121 box, ~30 GB free disk, ~60–90 min wall clock. The `12.1a` arch tag means the resulting image runs on the sm_121a GPU it was built for.

**Build:** clone the vLLM source per the pin above into `vllm-src/`, then build against the `Dockerfile` in this repo (the `Dockerfile.pr41703-layer` overlay carries the PR #40898/#41703 DFlash fixes — see the dated section above and the README's *Build provenance* for which source/overlay maps to which tag):

```bash
docker build -t aeon-vllm-ultimate:latest .
```

**Build knobs** (defaults are set in the Dockerfile, tuned for a ~20-core / 128 GB Spark):

| Env var | Default | Notes |
|---|---|---|
| `MAX_JOBS` | `12` | Compile parallelism. **Lower to 8/6 if the build OOMs.** |
| `NVCC_THREADS` | `2` | Per-`nvcc` threads. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `8` | CMake parallelism. |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | GB10 / sm_121a target. |
| `ENABLE_NVFP4_SM100` | `0` | Skips SM100-only NVFP4 kernels that fail to compile on SM121. |

The Dockerfile installs the CUDA 13.0 dev headers (`cuda-nvrtc-dev-13-0`, `libcusparse/cublas/cusolver/cufft/curand/nvjitlink-dev-13-0`), builds vLLM from the COPY'd `vllm-src/`, applies the three idempotent AEON sm_121a patches (`patch_cuda_optional_import`, `patch_kv_cache_utils`, `patch_cudagraph_align`), then layers TurboQuant (AEON-7 fork) + transformers HEAD + the `humming-stub`.

**Build troubleshooting:**

- `nvcc fatal: Unsupported gpu architecture` — your CUDA toolkit is too old; this build needs **CUDA ≥ 13.0** (the Dockerfile installs the `*-dev-13-0` headers).
- `RuntimeError: CUDA out of memory` during compile — lower `MAX_JOBS` (e.g. `--build-arg MAX_JOBS=8`).
- First build appears to "hang" generating CUDA stubs — that's normal (nvcc is compiling hundreds of objects); confirm progress with `docker stats` / `htop`.

**Verify** (the build runs `verify.py` automatically; to re-check manually):

```bash
docker run --rm aeon-vllm-ultimate:latest python3 -c "import vllm; print(vllm.__version__)"
```

No registry patch is needed — unlike the old `vllm-spark-omni-q36` image, the unified build loads the Qwen3.5/3.6 and Gemma-4 multimodal classes natively.
