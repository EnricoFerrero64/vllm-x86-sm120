# vLLM source pin

Build was against:

- **Repo**: `lesj0610/vllm`
- **Branch**: `lesj/triton-nvfp4-kv-fork-20260602`
- **Commit**: `e8c77b85`
- **Upstream PR**: [vllm-project/vllm#44389](https://github.com/vllm-project/vllm/pull/44389) — Triton software NVFP4 KV cache (~3× capacity)

To reproduce the build:

```bash
git clone https://github.com/lesj0610/vllm.git vllm-src
cd vllm-src
git checkout lesj/triton-nvfp4-kv-fork-20260602
git checkout e8c77b85
cd ..
# Then docker build -t aeon-vllm-ultimate:latest .
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
