"""
verify.py — Build-time environment check for aeon-vllm-x86:sm120.

Runs inside the Docker build (no GPU present).  Exits non-zero on any FAIL.
WARNs are logged but do not abort the build — they surface optional deps.
"""

import importlib
import site
import sys
from pathlib import Path

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results = []
fatal  = False


def check(label, fn):
    global fatal
    try:
        status, detail = fn()
    except Exception as exc:
        status, detail = FAIL, str(exc)
    results.append((status, label, detail))
    if status == FAIL:
        fatal = True


# ── helpers ──────────────────────────────────────────────────────────────────

def _site_packages():
    return [Path(p) for p in site.getsitepackages() if Path(p).exists()]


def _find_file(rel: str) -> Path | None:
    for sp in _site_packages():
        p = sp / rel
        if p.exists():
            return p
    # Also check the source build dir (pre-install path)
    src = Path("/build/vllm-src") / rel
    if src.exists():
        return src
    return None


def _version_tuple(ver_str: str):
    parts = []
    for p in ver_str.split(".")[:3]:
        try:
            parts.append(int("".join(c for c in p if c.isdigit()) or "0"))
        except ValueError:
            parts.append(0)
    return tuple(parts)


# ── package checks ────────────────────────────────────────────────────────────

def chk_torch():
    import torch
    ver = torch.__version__
    cuda = torch.version.cuda or "none"
    avail = torch.cuda.is_available()
    if _version_tuple(ver) < (2, 11, 0):
        return FAIL, f"{ver} (need >=2.11.0)"
    return PASS, f"{ver}  cuda={cuda}  device_available={avail}"


def chk_torch_sm120():
    import torch
    if not torch.cuda.is_available():
        return WARN, "no GPU at build time — sm_120 check skipped"
    cap = torch.cuda.get_device_capability(0)
    cap_int = cap[0] * 10 + cap[1]
    if cap_int < 120:
        return FAIL, f"sm_{cap_int} < sm_120 — this image targets RTX 50-series Blackwell"
    return PASS, f"sm_{cap_int} ({torch.cuda.get_device_name(0)})"


def chk_vllm():
    import vllm
    ver = vllm.__version__
    if "0.23" not in ver:
        return WARN, f"{ver} (expected 0.23.x)"
    return PASS, f"{ver}  ({vllm.__file__})"


def chk_vllm_core():
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig
    return PASS, "LLM, SamplingParams, VllmConfig importable"


def chk_vllm_mamba():
    # mamba-block-size / mamba-cache-dtype support requires this config class
    try:
        from vllm.config.model import MambaConfig  # vLLM ≥0.21
        return PASS, "MambaConfig importable"
    except ImportError:
        pass
    try:
        # Older path
        from vllm.config import ModelConfig
        import inspect
        if "mamba" in inspect.getsource(ModelConfig).lower():
            return PASS, "ModelConfig has mamba references"
        return WARN, "MambaConfig not found — --mamba-block-size may be unsupported"
    except Exception as e:
        return WARN, f"mamba config check failed: {e}"


def chk_vllm_multimodal():
    try:
        from vllm.multimodal import MultiModalRegistry
        return PASS, "MultiModalRegistry importable"
    except ImportError as e:
        return FAIL, f"multimodal registry missing: {e}"


def chk_vllm_triton_backend():
    p = _find_file("vllm/v1/attention/backends/triton_attn.py")
    if p is None:
        return FAIL, "triton_attn.py not found in site-packages or /build/vllm-src"
    return PASS, str(p)


def chk_vllm_nvfp4_kv():
    p = _find_file("vllm/v1/attention/ops/triton_reshape_and_cache_flash.py")
    if p is None:
        return FAIL, "triton_reshape_and_cache_flash.py not found"
    text = p.read_text(encoding="utf-8")
    if "_reshape_cache_nvfp4_kernel" not in text:
        return FAIL, "NVFP4 KV reshape kernel missing — PR #44389 not applied?"
    return PASS, "NVFP4 KV reshape kernel present"


def chk_patch_sm120_triton():
    """Verify patch_sm120_triton_launch.py was applied."""
    p = _find_file("vllm/v1/attention/ops/triton_unified_attention.py")
    if p is None:
        return WARN, "triton_unified_attention.py not found — patch check skipped"
    text = p.read_text(encoding="utf-8")
    if "tuned_sm12x_decode" not in text:
        return WARN, "sm_120 Triton launch patch NOT applied (cold-start latency unoptimized)"
    return PASS, "sm_12x launch config patch applied"


def chk_patch_cuda_optional():
    """Verify RTLD_LAZY patch for SM100-only symbols.

    The patch modifies vllm/platforms/cuda.py, wrapping the
    '_C_stable_libtorch' import with RTLD_LAZY flags.
    Marker inserted by the patch: '# stable_libtorch_lazy_dlopen'
    """
    # Primary target — what the patch actually modifies
    p = _find_file("vllm/platforms/cuda.py")
    if p and p.exists():
        text = p.read_text(encoding="utf-8")
        if "stable_libtorch_lazy_dlopen" in text:
            return PASS, f"RTLD_LAZY patch applied (platforms/cuda.py)"
        if "import vllm._C_stable_libtorch" not in text:
            return WARN, "platforms/cuda.py: _C_stable_libtorch import not present (upstream refactored)"
        return WARN, "cuda_optional_import patch NOT applied — sm_120 may crash on MXFP4 symbol load"
    return WARN, "vllm/platforms/cuda.py not found — patch check skipped"


def chk_flashinfer():
    import flashinfer
    ver = flashinfer.__version__
    if _version_tuple(ver) < (0, 6, 12):
        return WARN, f"{ver} (expected >=0.6.12)"
    return PASS, ver


def chk_transformers():
    import transformers
    ver = transformers.__version__
    # Qwen3.6 multimodal needs transformers 5.10+
    if _version_tuple(ver) < (5, 10, 0):
        return FAIL, f"{ver} (need >=5.10.0 for Qwen3.6 multimodal)"
    # Check Qwen3.6 model class registered
    try:
        from transformers import AutoModelForCausalLM
        return PASS, f"{ver}"
    except ImportError as e:
        return WARN, f"{ver} (AutoModelForCausalLM missing: {e})"


def chk_modelopt():
    try:
        import modelopt
        return PASS, getattr(modelopt, "__version__", "installed")
    except ImportError as e:
        return WARN, f"not installed ({e}) — NVFP4 weight loading may fail at runtime"


def chk_turboquant():
    try:
        import turboquant
        return PASS, getattr(turboquant, "__version__", "installed")
    except ImportError as e:
        return WARN, f"not installed ({e})"


def chk_humming():
    try:
        from humming.dtypes import DataType
        return PASS, "humming-stub importable (DataType OK)"
    except ImportError as e:
        return FAIL, f"humming-stub missing: {e} — vLLM will fail to load on is_cuda() check"


def chk_scipy():
    try:
        import scipy
        return PASS, scipy.__version__
    except ImportError as e:
        return WARN, f"not installed ({e})"


# ── run ──────────────────────────────────────────────────────────────────────

CHECKS = [
    ("torch ≥2.11.0+cu130",          chk_torch),
    ("GPU sm_120 capability",         chk_torch_sm120),
    ("vLLM 0.23.x",                   chk_vllm),
    ("vLLM core (LLM/VllmConfig)",    chk_vllm_core),
    ("vLLM mamba/GDN config",         chk_vllm_mamba),
    ("vLLM multimodal registry",      chk_vllm_multimodal),
    ("vLLM Triton attention backend", chk_vllm_triton_backend),
    ("vLLM NVFP4 KV kernel (PR#44389)", chk_vllm_nvfp4_kv),
    ("patch: sm_120 Triton launch",   chk_patch_sm120_triton),
    ("patch: cuda_optional_import",   chk_patch_cuda_optional),
    ("flashinfer ≥0.6.12",            chk_flashinfer),
    ("transformers ≥5.10.0",          chk_transformers),
    ("modelopt (NVFP4 weights)",      chk_modelopt),
    ("turboquant (optional KV compr)",chk_turboquant),
    ("humming-stub (eager import fix)", chk_humming),
    ("scipy (turboquant dep)",        chk_scipy),
]

W = 42
print("=" * (W + 30))
print("  aeon-vllm-x86:sm120 — build verification")
print("=" * (W + 30))

for label, fn in CHECKS:
    check(label, fn)

for status, label, detail in results:
    icon = "✓" if status == PASS else ("!" if status == WARN else "✗")
    print(f"  [{icon}] {status:<4}  {label:<{W}}  {detail}")

print("=" * (W + 30))
fails  = [r for r in results if r[0] == FAIL]
warns  = [r for r in results if r[0] == WARN]
passes = [r for r in results if r[0] == PASS]
print(f"  {len(passes)} passed  {len(warns)} warnings  {len(fails)} failed")
print("=" * (W + 30))

if fatal:
    print("\nBUILD FAILED — fix the errors above before running.")
    sys.exit(1)

print("\nBUILD OK — aeon-vllm-x86:sm120 ready.")
