"""
patch_sm120_triton_launch.py — Triton NVFP4 launch-config tuning for sm_120
(RTX 5060 Ti / GB206 Blackwell, GDDR7 448 GB/s)

WHAT THIS PATCHES
-----------------
vllm/v1/attention/ops/triton_unified_attention.py contains a `_get_nvfp4_launch_config`-
style block that sets Triton launch parameters (BLOCK_M, num_warps, num_stages) before
dispatching the unified attention kernel.  The upstream code has one explicit tuning
override: `tuned_large_head` for sm_100 (B100/B200) with head_size=256.

For sm_12x (GB10 sm_121a / GB206 sm_120) with head_size=128 — the Qwen3.6 family —
no override exists and Triton picks default num_warps/num_stages via JIT autotuning.
The problem: Triton's JIT autotuning at first-token time adds ~100-200 ms cold latency
AND the auto-selected config can be suboptimal for sm_120's hardware profile:

  sm_120 (RTX 5060 Ti)  :  36 SMs,  GDDR7  448 GB/s,  32 MB L2
  sm_121a (DGX GB10)    :  72 SMs,  LPDDR5X 273 GB/s, 128 MB L2

Key difference: sm_120 has ~3× higher bandwidth-per-SM than sm_121a, so it tolerates
fewer pipeline stages (less prefetch buffering needed) and benefits from 4 warps × 2
stages rather than the heavier configs sometimes auto-selected.

WHAT THE PATCH INSERTS
-----------------------
After the existing `tuned_large_head` block, injects:

    # sm_12x family (sm_120 RTX 5060 Ti / sm_121a DGX Spark) — head_size 128
    # Standard Qwen3.x / Llama-3.x shape.  Explicit config prevents Triton JIT
    # from autotuning at first inference request (avoids cold +100-200 ms spike).
    # Values chosen for GDDR7 bandwidth profile (sm_120) and verified stable on
    # sm_121a (same family, slightly different bandwidth envelope).
    tuned_sm12x_decode = (
        head_size == 128
        and current_platform.is_device_capability_family(120)
    )
    if tuned_sm12x_decode:
        launch_num_warps = 4
        launch_num_stages = 2

ANCHOR
------
The patch anchors on the end of the `tuned_large_head` block, which is unique
in the file.  Idempotent: re-running after the anchor text has already been
replaced is a no-op (the anchor won't match).
"""

import sys
from pathlib import Path


ANCHOR = """\
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2
"""

INSERTION = """\
    # sm_12x family (sm_120 RTX 5060 Ti / sm_121a DGX Spark) — head_size 128.
    # Prevents Triton JIT from autotuning at first-token time (+100-200 ms cold
    # latency spike).  4 warps × 2 stages matches sm_120 GDDR7 bandwidth profile.
    tuned_sm12x_decode = (
        head_size == 128
        and current_platform.is_device_capability_family(120)
    )
    if tuned_sm12x_decode:
        launch_num_warps = 4
        launch_num_stages = 2
"""

TARGET_MARKER = "tuned_sm12x_decode"


def find_triton_attention_file() -> Path:
    candidates = [
        Path("/build/vllm-src/vllm/v1/attention/ops/triton_unified_attention.py"),
        # installed package path (post pip-install)
        *Path("/usr").rglob("triton_unified_attention.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: search site-packages
    import site
    for sp in site.getsitepackages():
        p = Path(sp) / "vllm/v1/attention/ops/triton_unified_attention.py"
        if p.exists():
            return p
    return None


def main() -> int:
    target = find_triton_attention_file()
    if target is None:
        print("[sm120-triton] ERROR: triton_unified_attention.py not found — skipping")
        return 1

    text = target.read_text(encoding="utf-8")

    if TARGET_MARKER in text:
        print(f"[sm120-triton] already patched: {target}")
        return 0

    if ANCHOR not in text:
        print(f"[sm120-triton] WARN: anchor not found in {target} — skipping (upstream may have changed)")
        return 0

    patched = text.replace(ANCHOR, ANCHOR + INSERTION, 1)
    target.write_text(patched, encoding="utf-8")
    print(f"[sm120-triton] patched: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
