# AGENTS.md — aeon-vllm-x86:sm120

Agent instructions for this fork. This is the **x86_64 + sm_120 (RTX 5060 Ti)** port of
[AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark).
If you are working on the upstream DGX Spark image, use that repo's AGENTS.md instead.

---

## What this image is

**Image**: `ghcr.io/enricoferrero64/aeon-vllm-x86:sm120` (or local `aeon-vllm-x86:sm120`)
**Hardware target**: 2× NVIDIA RTX 5060 Ti 16 GB (PCIe, no NVLink), x86_64, Unraid 7.3
**Primary model**: `AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS`
**Key capability**: NVFP4 KV cache via Triton (PR #44389) + MTP speculative decoding

---

## Serve (OpenAI API)

Use `docker-compose.x86_64.yml` — it contains all stability and performance flags pre-configured:

```bash
docker compose -f docker-compose.x86_64.yml up -d
```

Or run manually (all flags are required):

```bash
docker run --rm --gpus all \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_P2P_LEVEL=PCI \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8 \
  --shm-size 16g \
  --ulimit memlock=-1 \
  -p 8000:8000 \
  aeon-vllm-x86:sm120 -c \
  "python3 -m vllm.entrypoints.openai.api_server \
    --model AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS \
    --tensor-parallel-size 2 \
    --quantization modelopt \
    --kv-cache-dtype nvfp4 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":4}' \
    --mamba-block-size 256 \
    --mamba-cache-dtype float16 \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt 'image=4,video=1' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --block-size 32 \
    --disable-custom-all-reduce \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --host 0.0.0.0 --port 8000"
```

**Critical flags** — do not remove:

| Flag | Why |
|---|---|
| `--quantization modelopt` | MTP-XS uses modelopt format, NOT compressed-tensors |
| `--speculative-config method: qwen3_5_mtp` | Qwen3.6-specific MTP; generic `mtp` does not work |
| `--disable-custom-all-reduce` | vLLM SymmMemCommunicator doesn't support sm_120 — hangs without this |
| `NCCL_DMABUF_ENABLE=1` | DMA-BUF P2P; prevents race conditions in Blackwell PCIe drivers |
| `--mamba-block-size 256` | Required for GatedDeltaNet hybrid layers in Qwen3.6 |
| `--mm-encoder-tp-mode data` | ViT runs DP per GPU; wrong mode causes inter-GPU ViT TP tensor errors |
| `shm_size 16g` | NCCL TP=2 with 27B model needs this; 2 GB causes NCCL OOM |

---

## Verify build

```bash
docker run --rm aeon-vllm-x86:sm120 python3 -c "
import vllm; print('vLLM:', vllm.__version__)
import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
import flashinfer; print('flashinfer:', flashinfer.__version__)
from humming.dtypes import DataType; print('humming-stub: ok')
"
```

Expected output includes:
- `vLLM: 0.23.0+aeon.x86.sm120`
- `torch: 2.11.0+cu130`
- `flashinfer: 0.6.12`
- `humming-stub: ok`

---

## Smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AEON-7/Qwen3.6-27B-Multimodal-NVFP4-MTP-XS",
    "messages": [{"role": "user", "content": "Write hello world in Python."}],
    "max_tokens": 64
  }' | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
```

---

## Benchmark

```bash
# Quick decode throughput (single-stream, 512 input / 256 output tokens)
python3 bench_vllm.py --num-prompts 10 --input-len 512 --output-len 256

# Throughput + concurrency sweep
python3 bench_categories.py

# PP=2 vs TP=2 A/B test (see docker-compose.pp2-bench.yml for setup instructions)
python3 bench_vllm.py --port 8001 --num-prompts 20 --output bench_pp2.json
```

---

## Build

```bash
# Full build on Unraid host (MAX_JOBS=4, ~2-3h on i7-14700K)
./build-x86_64.sh

# After local build, push cache to GHCR (primes GitHub Actions for next CI build)
./build-x86_64.sh --push-cache
```

Monitor build:
```bash
tail -f /tmp/aeon-vllm-build.log
```

---

## Key files

| File | Purpose |
|---|---|
| `Dockerfile.x86_64` | x86_64 sm_120 image definition |
| `docker-compose.x86_64.yml` | Production serve config (all flags tuned) |
| `docker-compose.pp2-bench.yml` | PP=2 A/B benchmark config (port 8001) |
| `build-x86_64.sh` | Build script with optional `--push-cache` |
| `verify.py` | 16-check build verification (runs automatically in Docker build) |
| `patches/patch_cuda_optional_import.py` | RTLD_LAZY dlopen for SM100-only symbols |
| `patches/patch_kv_cache_utils.py` | GDN hybrid-attn block_size=None fix |
| `patches/patch_cudagraph_align.py` | PIECEWISE cuda graph alignment fix |
| `patches/patch_sm120_triton_launch.py` | sm_12x Triton JIT cold-start fix |
| `humming-stub/` | Stub for NVIDIA-internal `humming` quant lib |
| `SOURCE.md` | vLLM fork pin + changelog |

---

## Failure modes

| Error | Cause | Fix |
|---|---|---|
| Container hangs at startup with no logs | `--disable-custom-all-reduce` missing | Add flag; restart |
| `SymmMemCommunicator: Device capability 12.0 not supported` | Expected on sm_120 | Safe to ignore — NCCL handles comms |
| `TypeError: '<' not supported between 'NoneType' and 'int'` | `patch_kv_cache_utils` not applied | Rebuild image |
| `cudaErrorIllegalAddress` in spec-decode | `patch_cudagraph_align` not applied | Rebuild image |
| `ImportError: humming` | humming-stub not installed | Rebuild; check step 18 in build log |
| OOM at startup | `--max-model-len 131072` too large | Try `--max-model-len 65536` first |
| NCCL timeout during TP init | shm too small or `NCCL_DMABUF_ENABLE` missing | Check env vars and `shm_size: 16gb` |
| `nvfp4 KV` error at first request | Wrong attention backend | Ensure `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| `block_size=None` crash in GDN | `patch_kv_cache_utils` not applied | Rebuild |
| Wrong quantization method error | `--quantization compressed-tensors` (old flag) | Change to `--quantization modelopt` |
| MTP speculator not found | `--speculative-config method: mtp` (generic) | Change to `qwen3_5_mtp` |
| VRAM fragmentation OOM after hours | `PYTORCH_CUDA_ALLOC_CONF` missing | Add `max_split_size_mb:512,garbage_collection_threshold:0.8` |

---

## Key differences from upstream AEON DGX Spark

If you are familiar with the upstream AEON image, note these critical differences:

- **Do NOT use DFlash speculator** — this fork uses MTP (`qwen3_5_mtp`), not DFlash
- **`--kv-cache-dtype nvfp4`** (enabled) — DFlash requires `auto` (BF16); MTP works with nvfp4
- **`--quantization modelopt`** (not compressed-tensors) — same in upstream, but worth noting
- **`--gpu-memory-utilization 0.92`** (not 0.88) — dedicated GDDR7 allows higher utilization
- **`NCCL_P2P_LEVEL=PCI`** (not NVL) — no NVLink on RTX 5060 Ti PCIe
- **No DMA-BUF conflicts** — `NCCL_DMABUF_ENABLE=1` is safe on x86 (would differ on DGX Spark)
