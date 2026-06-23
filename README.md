# aeon-vllm-x86 — sm_120 (RTX 5060 Ti / RTX 5070 / RTX 5080)

[![docker](https://img.shields.io/badge/ghcr.io-enricoferrero64%2Faeon--vllm--x86-blue?logo=docker)](https://ghcr.io/enricoferrero64/aeon-vllm-x86)
[![vLLM](https://img.shields.io/badge/vLLM-0.23.0%2Bsm__120.x86-orange)](https://github.com/lesj0610/vllm)
[![sm_120](https://img.shields.io/badge/sm__120-RTX%2050%20series-green)](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
[![upstream](https://img.shields.io/badge/upstream-AEON--7%2Fvllm--ultimate--dgx--spark-lightgrey)](https://github.com/AEON-7/vllm-ultimate-dgx-spark)

**x86_64 port of [AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark), adapted for consumer Blackwell sm_120 GPUs (RTX 50-series) on standard x86 hardware.**

Primary target: **2× RTX 5060 Ti 16 GB** (32 GB VRAM total, PCIe, no NVLink) running **AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS** with NVFP4 KV cache and MTP speculative decoding.

---

## What this fork adds vs upstream AEON

The upstream image targets the **NVIDIA DGX Spark (GB10, sm_121a, 128 GB unified LPDDR5X)**. That hardware is very different from a dual-GPU consumer PCIe setup:

| | DGX Spark (upstream) | 2× RTX 5060 Ti (this fork) |
|---|---|---|
| Architecture | sm_121a (GB10) | **sm_120** (GB206) |
| Memory type | Unified LPDDR5X | **Dedicated GDDR7** |
| GPU interconnect | NVLink | **PCIe Gen 5** |
| Best speculator | DFlash (non-causal) | **MTP** (`qwen3_5_mtp`) |
| KV dtype | `fp8_e4m3` w/ DFlash | **`nvfp4`** w/ MTP |
| GPU mem utilization | 0.88 max (unified) | **0.92** (dedicated) |

### Key changes in this fork

- **Base image**: `nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu22.04` x86_64 (not ARM64)
- **`TORCH_CUDA_ARCH_LIST=12.0`** — compiles sm_120 kernels (not 12.1a)
- **`--disable-custom-all-reduce`** — prevents hang at startup; vLLM SymmMemCommunicator does not support sm_120 on PCIe
- **`NCCL_DMABUF_ENABLE=1` + `NCCL_P2P_LEVEL=PCI`** — stable P2P on Blackwell PCIe without NVLink
- **`--mm-encoder-tp-mode data`** — vision encoder runs DP per GPU (not TP), reducing inter-GPU comms during image encoding
- **`--mamba-cache-dtype float16`** — SSM cache for GatedDeltaNet hybrid layers
- **`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`** — prevents VRAM fragmentation under long inference
- **Patch 4 (`patch_sm120_triton_launch.py`)** — sets explicit `num_warps=4, num_stages=2` for sm_12x family in `triton_unified_attention.py`, eliminating Triton JIT autotune cold-start latency (~100-200 ms first-token spike)
- **`shm_size: 16gb`** — correct for NCCL TP=2 buffers (was 2 GB, insufficient for 27B model)
- **`--block-size 32`** — better DRAM bandwidth utilization for head_dim=128 GQA attention

---

## Quickstart (Jarvis / Unraid — copy-paste)

```bash
# 1. Clone this repo on the Unraid host
git clone https://github.com/EnricoFerrero64/vllm-x86-sm120.git /mnt/user/appdata/aeon
cd /mnt/user/appdata/aeon

# 2. (First time) Log in to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u EnricoFerrero64 --password-stdin

# 3. Pull the prebuilt image (or build locally — see Build section)
docker pull ghcr.io/enricoferrero64/aeon-vllm-x86:sm120

# 4. Create .env with your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env

# 5. Start the server
docker compose -f docker-compose.x86_64.yml up -d

# 6. Smoke test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS",
       "messages":[{"role":"user","content":"Hello!"}],
       "max_tokens":32}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

---

## Configuration reference

Default config in `docker-compose.x86_64.yml`, tuned for 2× RTX 5060 Ti 16 GB:

| Flag | Value | Notes |
|---|---|---|
| `--model` | `AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS` | NVFP4 modelopt format, ~21 GB |
| `--quantization` | `modelopt` | MTP-XS uses modelopt, not compressed-tensors |
| `--kv-cache-dtype` | `nvfp4` | ~3× capacity vs BF16; requires causal speculator |
| `--speculative-config` | `{"method":"qwen3_5_mtp","num_speculative_tokens":4}` | MTP self-speculation; Qwen3.6-specific method |
| `--tensor-parallel-size` | `2` | TP across both GPUs over PCIe |
| `--max-model-len` | `131072` | 128K context (NVFP4 KV enables this in 32 GB) |
| `--gpu-memory-utilization` | `0.92` | Safe on dedicated GDDR7 (vs 0.88 on Spark unified) |
| `--mamba-block-size` | `256` | Required for GatedDeltaNet hybrid layers |
| `--mamba-cache-dtype` | `float16` | SSM cache precision |
| `--mm-encoder-tp-mode` | `data` | ViT data-parallel; LLM tensor-parallel |
| `--enable-prefix-caching` | yes | Reuse KV for repeated system prompts |
| `--enable-chunked-prefill` | yes | Reduces peak VRAM on long prompts |
| `--block-size` | `32` | Better DRAM utilization for head_dim=128 |
| `--disable-custom-all-reduce` | yes | Required on sm_120 PCIe (prevents init hang) |
| `--max-num-seqs` | `8` | Concurrent sequences (conservative for 32 GB) |

### Benchmark PP=2 vs TP=2

PCIe limits TP=2 throughput (28 all-reduce calls per forward pass). `docker-compose.pp2-bench.yml` provides an identical config with `--pipeline-parallel-size 2` on port 8001 for A/B testing. See the instructions in that file.

---

## Build locally

Required when: you want to prime the GitHub Actions cache, or pull the latest vLLM fork changes.

```bash
# On the Unraid host (or any x86_64 Linux with Docker + NVIDIA runtime)
cd /mnt/user/appdata/aeon
./build-x86_64.sh              # build + load to local Docker daemon

# After successful local build, push image + layer cache to GHCR
# so GitHub Actions uses the cache for subsequent builds:
./build-x86_64.sh --push-cache
```

**Build time**: ~2-3 h on i7-14700K (MAX_JOBS=4). Monitor progress:
```bash
tail -f /tmp/aeon-vllm-build.log
```

**OOM during build**: lower `MAX_JOBS` in `Dockerfile.x86_64` (default 4, each job ~3 GB RAM).

### GitHub Actions (incremental builds)

After the first local `--push-cache` run, `.github/workflows/build-x86.yml` uses the GHCR layer cache. Subsequent CI builds only recompile changed layers — typically 5 minutes if only patches or compose changed, 4-6 hours for a full CUDA recompile.

Trigger manually: GitHub → Actions → "Build aeon-vllm-x86 (sm_120)" → Run workflow.

---

## Applied patches

This fork applies 4 idempotent patches on top of the vLLM source at build time:

| Patch | Target file | What it fixes |
|---|---|---|
| `patch_cuda_optional_import` | `vllm/__init__` or `_C_stable_libtorch` | RTLD_LAZY dlopen for SM100-only MXFP4 symbols absent on sm_120 |
| `patch_kv_cache_utils` | `vllm/worker/cache_engine.py` | TypeError on GDN hybrid-attention models where block_size=None |
| `patch_cudagraph_align` | `vllm/config/compilation.py` | cudaErrorIllegalAddress in PIECEWISE CUDA graph mode under spec-decode |
| `patch_sm120_triton_launch` | `vllm/v1/attention/ops/triton_unified_attention.py` | Explicit num_warps=4/num_stages=2 for sm_12x; prevents Triton JIT cold-start spike |

---

## Expected performance (estimated, 2× RTX 5060 Ti)

Based on AEON upstream benchmarks on DGX Spark (sm_121a, 128 GB LPDDR5X) scaled to sm_120 hardware, with adjustments for PCIe TP overhead:

| Category | Estimated decode tok/s | Notes |
|---|---:|---|
| Coding / JSON | 35–50 | High MTP acceptance (predictable tokens) |
| Math / Reasoning | 30–45 | Good MTP acceptance |
| Prose / Dialogue | 20–30 | Lower MTP acceptance (high entropy) |
| Concurrent ×4 | 60–90 agg | With prefix cache warm |

> Numbers are rough estimates — actual results depend on prompt distribution, concurrency, and driver version. Run `bench_vllm.py` to measure your setup.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Container hangs at startup, no output | Add `--disable-custom-all-reduce` (may be missing from older images) |
| `SymmMemCommunicator: Device capability 12.0 not supported` | Expected warning on sm_120 — safe to ignore, NCCL handles comms instead |
| OOM at startup | Reduce `--max-model-len` (try 65536) or `--max-num-seqs 4` |
| `nvfp4 KV` error at first request | Check `VLLM_ATTENTION_BACKEND=FLASH_ATTN` is set; NVFP4 KV needs Triton attention backend |
| `block_size=None` error | `patch_kv_cache_utils` not applied — rebuild the image |
| `cudaErrorIllegalAddress` during spec-decode | `patch_cudagraph_align` not applied — rebuild |
| `humming` ImportError | `humming-stub` not installed — rebuild (check build log for step 18) |
| VRAM fragmentation OOM after hours | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` missing from env |
| Low MTP acceptance (<20%) on all categories | Normal for prose; for coding/math check model downloaded correctly (`git lfs pull`) |

---

## Upstream & credits

- **Upstream**: [AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark) — original DGX Spark image, benchmarks, and patches
- **vLLM fork**: [lesj0610/vllm@lesj/triton-nvfp4-kv-fork-20260602](https://github.com/lesj0610/vllm/tree/lesj/triton-nvfp4-kv-fork-20260602) — merges PR #44389, #40898, #41703, #43982
- **Model**: [AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS](https://huggingface.co/AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS)
- **TurboQuant**: [AEON-7/turboquant](https://github.com/AEON-7/turboquant) (CUDA-graph-safe QJL fork)

## License

vLLM Apache-2.0 · PyTorch BSD-3-Clause · TurboQuant Apache-2.0 · AEON patches MIT · x86/sm_120 adaptations MIT
