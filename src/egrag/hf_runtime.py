"""Lazy Hugging Face text-generation runtime shared by the structured-extraction
and answer-generation adapters.

``transformers``/``torch`` are imported **lazily** (never at module import), so
importing core EG-RAG pulls in no heavy dependency and triggers no download. The
runtime:

* loads the tokenizer + pipeline once (no duplicate loading);
* applies ``tokenizer.apply_chat_template`` when the tokenizer provides one, and
  falls back to a plain role-concatenated prompt otherwise;
* supports ``cpu`` / ``mps`` / ``cuda`` / integer pipeline devices and ``auto``;
* supports an optional dtype (``float32``/``float16``/``bfloat16``/``auto``);
* supports optional 4-bit/8-bit ``bitsandbytes`` quantization (the ``quantization``
  extra), always explicit and recorded, never silent;
* fails loudly (``ensure_device_available``) rather than silently falling back to
  CPU when a CUDA device is explicitly required and unavailable;
* keeps decoding deterministic (``do_sample=False`` unless explicitly sampling);
* uses ``return_full_text=False`` and a safe ``pad_token_id`` fallback;
* records resolved model/tokenizer/device/dtype/quantization for manifests.

Messages are passed as plain ``{"role", "content"}`` dicts so this module needs
no domain import; adapters convert their own message types.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

Message = dict[str, str]


def resolve_device(spec: str | int | None) -> str | int | None:
    """Resolve a device spec deterministically.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU (documented, deterministic).
    Any other value (``"cpu"``/``"mps"``/``"cuda"``/``"cuda:N"``/int/``None``) is
    returned unchanged. torch is imported lazily and only for ``"auto"``.
    """

    if spec != "auto":
        return spec
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:  # pragma: no cover - env guard
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _wants_cuda(device: str | int | None) -> bool:
    return isinstance(device, str) and device.startswith("cuda")


def ensure_device_available(device: str | int | None) -> None:
    """Fail loudly if ``device`` (after ``"auto"`` resolution) requests CUDA that
    is not actually available.

    This is the single enforcement point for "CUDA required, do not silently run
    on CPU": callers that need a hard guarantee resolve ``device`` first (or pass
    ``"cuda"`` directly) and call this before any model load. Non-CUDA devices
    (``cpu``/``mps``/``auto`` that resolved to something else) are always a no-op.
    """

    resolved = resolve_device(device)
    if not _wants_cuda(resolved):
        return
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"device {resolved!r} was requested but torch is not installed; "
            "install the 'local-models' extra"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device {resolved!r} was requested but CUDA is not available on this "
            "machine (torch.cuda.is_available() is False); refusing to silently "
            "fall back to CPU"
        )


def gpu_report(*, device: str | int | None = "auto", dtype: str | None = None) -> dict[str, Any]:
    """Return a GPU/device readiness report (torch, CUDA/MPS, selected device, disk)."""

    report: dict[str, Any] = {"requested_device": device, "requested_dtype": dtype}
    try:
        torch = importlib.import_module("torch")
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        report["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["cuda_device_count"] = torch.cuda.device_count()
            report["cuda_vram_total_gib"] = round(props.total_memory / (1024**3), 1)
        else:
            report["cuda_device_count"] = 0
            report["cuda_vram_total_gib"] = None
        mps = getattr(torch.backends, "mps", None)
        report["mps_available"] = bool(mps is not None and mps.is_available())
    except ModuleNotFoundError:  # pragma: no cover - env guard
        report.update(
            {
                "torch": None,
                "cuda_available": False,
                "cuda_device_name": None,
                "cuda_runtime_version": None,
                "cuda_device_count": 0,
                "cuda_vram_total_gib": None,
                "mps_available": False,
            }
        )
    try:
        report["transformers"] = importlib.import_module("transformers").__version__
    except ModuleNotFoundError:  # pragma: no cover - env guard
        report["transformers"] = None
    report["selected_device"] = resolve_device(device)
    report["model_dtype"] = dtype or "float32 (torch default)"
    hub = Path("~/.cache/huggingface/hub").expanduser()
    report["hf_cache_dir"] = str(hub)
    report["hf_cache_exists"] = hub.is_dir()
    try:
        usage = shutil.disk_usage(Path.home())
        report["free_disk_gib"] = round(usage.free / (1024**3), 1)
    except OSError:  # pragma: no cover - defensive
        report["free_disk_gib"] = None
    return report


_TORCH_DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def _resolve_dtype(dtype: str | None) -> Any:
    if dtype is None:
        return None
    if dtype == "auto":
        return "auto"
    torch = importlib.import_module("torch")
    name = _TORCH_DTYPES.get(dtype.lower())
    if name is None:
        raise ValueError(f"unsupported dtype {dtype!r}; use float32/float16/bfloat16/auto")
    return getattr(torch, name)


_QUANTIZATIONS = ("none", "4bit", "8bit")


def _build_quantization_config(quantization: str, compute_dtype: Any) -> Any:
    """Build a ``transformers.BitsAndBytesConfig`` for ``"4bit"``/``"8bit"``.

    Requires the ``bitsandbytes`` package (the ``quantization`` extra); import
    errors surface as a clear ``RuntimeError`` rather than a silent fallback to
    an unquantized load.
    """

    if quantization not in _QUANTIZATIONS:
        raise ValueError(f"unsupported quantization {quantization!r}; use one of {_QUANTIZATIONS}")
    torch = importlib.import_module("torch")
    try:
        transformers: Any = importlib.import_module("transformers")
    except ModuleNotFoundError as exc:  # pragma: no cover - env guard
        raise RuntimeError(
            "transformers is not installed; install the 'local-models' extra"
        ) from exc
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "quantization was requested but 'bitsandbytes' is not installed; "
            "install it with: uv pip install 'egrag[quantization]'"
        )
    dtype = compute_dtype if isinstance(compute_dtype, torch.dtype) else torch.bfloat16
    if quantization == "4bit":
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    return transformers.BitsAndBytesConfig(load_in_8bit=True)


class HFTextPipeline:
    """A lazily-loaded ``text-generation`` pipeline with chat-template support."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | int | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
        require_cuda: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.device = device
        self.dtype = dtype
        self.quantization = quantization or "none"
        self.require_cuda = require_cuda
        # Passed verbatim to `tokenizer.apply_chat_template(...)`. Generic (not
        # model-specific): some instruction-tuned models expose chat-template
        # flags such as `enable_thinking=False` to request a direct answer
        # instead of an interleaved reasoning trace; this is how a caller
        # requests that, without EG-RAG hardcoding any particular model family.
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._pipeline: Any = None
        self._tokenizer: Any = None

    def _ensure(self) -> tuple[Any, Any]:
        if self._pipeline is None:
            try:
                transformers: Any = importlib.import_module("transformers")
            except ModuleNotFoundError as exc:  # pragma: no cover - env guard
                raise RuntimeError(
                    "transformers is not installed; install the 'local-models' extra"
                ) from exc
            if self.require_cuda:
                ensure_device_available("cuda")
            else:
                ensure_device_available(self.device)
            kwargs: dict[str, Any] = {"model": self.model_name, "return_full_text": False}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            if self.tokenizer_revision is not None:
                kwargs["tokenizer"] = self.model_name
            resolved_dtype = _resolve_dtype(self.dtype)
            if resolved_dtype is not None:
                kwargs["torch_dtype"] = resolved_dtype
            if self.quantization != "none":
                # bitsandbytes quantized weights must be sharded via device_map;
                # a plain `device=` kwarg is not supported with quantization_config.
                kwargs["model_kwargs"] = {
                    "quantization_config": _build_quantization_config(
                        self.quantization, resolved_dtype
                    )
                }
                kwargs["device_map"] = "auto"
            elif self.device == "auto":
                kwargs["device_map"] = "auto"
            elif self.device is not None:
                kwargs["device"] = self.device
            self._pipeline = transformers.pipeline("text-generation", **kwargs)
            self._tokenizer = self._pipeline.tokenizer
            tok = self._tokenizer
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None):
                tok.pad_token_id = tok.eos_token_id
        return self._pipeline, self._tokenizer

    def has_chat_template(self) -> bool:
        """True if the loaded tokenizer defines a chat template."""

        _, tok = self._ensure()
        return bool(getattr(tok, "chat_template", None))

    def resolved_revisions(self) -> dict[str, str]:
        """Best-effort resolved model/tokenizer/runtime identifiers, for manifests."""

        pipe, tok = self._ensure()
        info: dict[str, str] = {
            "model_name": self.model_name,
            "requested_device": str(self.device),
            "requested_dtype": str(self.dtype),
            "quantization": self.quantization,
        }
        name_or_path = getattr(tok, "name_or_path", None)
        if name_or_path:
            info["tokenizer_name_or_path"] = str(name_or_path)
        if self.revision:
            info["model_revision"] = self.revision
        if self.tokenizer_revision:
            info["tokenizer_revision"] = self.tokenizer_revision
        model = getattr(pipe, "model", None)
        resolved_dtype = getattr(model, "dtype", None)
        if resolved_dtype is not None:
            info["resolved_dtype"] = str(resolved_dtype)
        device = getattr(pipe, "device", None)
        if device is not None:
            info["resolved_device"] = str(device)
        return info

    def _seed(self, seed: int | None) -> None:
        if seed is None:
            return
        try:
            transformers: Any = importlib.import_module("transformers")
            transformers.set_seed(seed)
        except (ModuleNotFoundError, AttributeError):  # pragma: no cover - defensive
            pass

    def generate(
        self,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        max_new_tokens: int = 256,
        deterministic: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = 0,
        stop: tuple[str, ...] = (),
    ) -> str:
        """Generate text from either a raw ``prompt`` or chat ``messages``.

        When ``messages`` is given and the tokenizer has a chat template, the
        template is applied (with a generation prompt); otherwise messages are
        concatenated into a plain prompt. Deterministic mode never samples.
        """

        pipe, tok = self._ensure()
        if messages is not None:
            if self.has_chat_template():
                text = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            else:
                joined = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
                text = f"{joined}\n\nASSISTANT:\n"
        elif prompt is not None:
            text = prompt
        else:  # pragma: no cover - programmer error
            raise ValueError("generate requires either prompt or messages")

        self._seed(seed)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "do_sample": not deterministic,
        }
        if not deterministic:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if getattr(tok, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = tok.pad_token_id
        if stop:
            # stop_strings requires the tokenizer; fall back gracefully if unsupported.
            try:
                outputs = pipe(text, stop_strings=list(stop), tokenizer=tok, **gen_kwargs)
            except (TypeError, ValueError):  # pragma: no cover - version guard
                outputs = pipe(text, **gen_kwargs)
        else:
            outputs = pipe(text, **gen_kwargs)
        return str(outputs[0]["generated_text"])


__all__ = [
    "HFTextPipeline",
    "Message",
    "ensure_device_available",
    "gpu_report",
    "resolve_device",
]
