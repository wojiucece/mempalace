"""Embedding function factory with hardware acceleration.

Returns a ChromaDB-compatible embedding function bound to a user-selected
ONNX Runtime execution provider.

Two embedding models are available, selected via ``MEMPALACE_EMBEDDING_MODEL``
or ``embedding_model`` in ``~/.mempalace/config.json``:

* ``minilm`` (default) — ``all-MiniLM-L6-v2``, 384-dim, English-only training.
  ChromaDB's default; what every existing palace was built with.
* ``embeddinggemma`` — ``onnx-community/embeddinggemma-300m-ONNX`` (q8), 384-dim
  via Matryoshka truncation, multilingual (100+ languages). Cross-lingual cos
  ~0.88 on parallel translations vs MiniLM's ~0.35. Recommended for any
  non-English use; onboarding offers it as the default. The ~300 MB ONNX
  model is lazy-downloaded from HuggingFace on first use. Switching models
  on an existing palace requires ``mempalace repair rebuild-index``
  (different vector space).

Supported devices (env ``MEMPALACE_EMBEDDING_DEVICE`` or ``embedding_device``
in ``~/.mempalace/config.json``):

* ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
* ``cpu`` — force CPU (the historical default)
* ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu`` (``pip install mempalace[gpu]``)
* ``coreml`` — Apple Neural Engine (macOS)
* ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

_EF_CACHE: dict = {}
# Check-then-construct on the cache must be atomic: without it, two threads
# resolving the same key each keep their own EF instance, and each instance
# later lazy-loads its own copy of the model.
_EF_CACHE_LOCK = threading.Lock()
_WARNED: set = set()


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _intra_op_session_options(intra_op_num_threads: int):
    """Build ORT ``SessionOptions`` capping the intra-op thread pool (#1068).

    Returns ``None`` when ``intra_op_num_threads <= 0`` so the caller leaves
    ORT at its default (≈ physical core count). ChromaDB's embedder ignores
    ``OMP_NUM_THREADS`` — ORT owns its own intra-op pool, settable only via
    ``SessionOptions`` at session construction — so a cap has to be threaded
    through here rather than via the environment.
    """
    if not intra_op_num_threads or intra_op_num_threads <= 0:
        return None
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_num_threads
    return so


def _resolve_intra_op_threads() -> int:
    """Read the configured ORT intra-op thread cap (``0`` = uncapped, #1068)."""
    try:
        from .config import MempalaceConfig

        return MempalaceConfig().embedding_threads
    except Exception:
        logger.debug("embedding_threads resolution failed; leaving ORT default", exc_info=True)
        return 0


def _build_ef_class():
    """Subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF (``onnx_mini_lm_l6_v2``
    vs ``default``). The vectors and model are identical — only the
    ``name()`` tag differs — so spoofing the name lets one EF class serve
    palaces created with ``DefaultEmbeddingFunction`` *and* palaces we
    create ourselves, with the same GPU-capable ``preferred_providers``.
    """
    from functools import cached_property

    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            super().__init__(preferred_providers=preferred_providers)
            self._intra_op_num_threads = intra_op_num_threads

        @staticmethod
        def name() -> str:
            return "default"

        @cached_property
        def model(self):
            # Upstream builds the InferenceSession with no intra-op thread cap,
            # so ORT defaults its pool to the physical core count and a
            # background mine pins every core (#1068). Rebuild the session the
            # same way upstream does (same SessionOptions, same CoreML pruning,
            # same model path) but with our cap applied. If upstream's
            # internals shift, fall back to its uncapped build so embedding
            # still works.
            cap = getattr(self, "_intra_op_num_threads", 0)
            if not cap or cap <= 0:
                return super().model
            try:
                ort = self.ort
                providers = self._preferred_providers or ort.get_available_providers()
                providers = [p for p in providers if p != "CoreMLExecutionProvider"]
                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = cap
                return ort.InferenceSession(
                    os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
                    providers=providers,
                    sess_options=so,
                )
            except Exception:
                logger.warning(
                    "thread-capped ORT session build failed; using ORT defaults",
                    exc_info=True,
                )
                return super().model

    return _MempalaceONNX


# Embeddinggemma-300m ONNX (q8) — 100+ languages, MRL-truncated to 384 dims so
# it drops into existing ChromaDB collections without a schema change. Lazy:
# the model (~300 MB) downloads on first call and is cached by huggingface_hub.
_EMBEDDINGGEMMA_REPO = "onnx-community/embeddinggemma-300m-ONNX"
_EMBEDDINGGEMMA_ONNX = "model_quantized.onnx"
_EMBEDDINGGEMMA_PREFIX = "task: sentence similarity | query: "
_EMBEDDINGGEMMA_DIM = 384  # Matryoshka truncation — first 384 dims of the 768
_EMBEDDINGGEMMA_MAX_LEN = 2048
# Default docs per session.run. The ONNX graph has no internal batching,
# so one unchunked run over a repair-scale batch (5000 docs, repair.py/
# cli.py) allocates attention buffers that grow with batch size and
# superlinearly with padded length (score tensors are batch x heads x
# len^2 per layer), and the kernel OOM-kills the process (#1770). 32
# matches the internal batch size of chromadb's ONNXMiniLM_L6_V2, whose
# chunked _forward survives the same call sites. embeddinggemma's
# sentence_embedding output is attention-masked, so sub-batch padding
# does not change any row's vector.
_EMBEDDINGGEMMA_BATCH_SIZE = 32


# Ollama integration — delegates embedding to a local Ollama server via ChromaDB's
# native OllamaEmbeddingFunction. Activated by MEMPALACE_EMBEDDING_MODEL=ollama.
# Unlike minilm/embeddinggemma, no model is loaded in-process: vectors come from
# a single HTTP call per batch to Ollama, so onnxruntime providers/devices are
# irrelevant for this path. Switching an existing palace to ollama changes both
# the EF name() to "ollama" and the vector dimension (e.g. qwen3-embedding:0.6b
# → 1024d), so it requires `mempalace repair --yes` like any other embedder
# change. Repair has no resume — see plan doc.
# Base URL — 不要末尾加 `/api/embeddings`。如果用户从旧版 ChromaDB 文档复制带后缀的
# URL，ChromaDB OllamaEmbeddingFunction.__init__ 自身有 `endswith("/api/embeddings")`
# 兼容逻辑（剥后缀），所以这里不必再做 normalize。
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen3-embedding:0.6b"
_OLLAMA_DEFAULT_TIMEOUT = 60  # 秒
# Health-probe text: longer than a single token so models that special-case very
# short inputs (some BERT-family tokenizers emit only [CLS]+[SEP]) still produce
# a meaningful vector and surface real failure modes during probe.
_OLLAMA_HEALTH_PROBE = "mempalace embedding health check"

# llama.cpp embedding backend -- 通过 mempalace 自管的 llama-server 子进程
# 做 embedding。激活方式：MEMPALACE_EMBEDDING_MODEL=llamacpp。
#
# 跟前两种 backend 的根本区别：
# - minilm / embeddinggemma 在 mempalace 进程内做推理（onnxruntime）
# - llamacpp 走外部进程 + HTTP，由 _llamacpp_server.ensure_running()
#   按需 spawn、绑定 Job Object、健康探测、log 抓取
#
# 跟旧 Ollama 路径的根本区别：旧路径 mempalace 只是个 HTTP 客户端，
# Ollama daemon 由用户手动启动；新路径子进程归 mempalace 管。
#
# 切换到 llamacpp 改变 EF name() → "llamacpp"，跟旧 collection 不兼容，
# 必须 mempalace repair --yes 重建索引。
_LLAMACPP_HEALTH_PROBE = "mempalace embedding health check"


class LlamacppEF:
    """ChromaDB-compatible EF，调用本地 llama-server 的 /embeddings 端点（带 s，非 OAI 兼容版）。

    实例化时不立刻发请求；首次调用 __call__ 时通过
    _llamacpp_server.ensure_running() 保证服务在跑（lazy spawn）。

    Batch chunking 边界：本类**不做内部 batch 切分**，假设调用方
    （ChromaDB / mempalace miner / searcher）已按合理大小切好。
    ChromaDB 内部 ONNX EF 用 batch_size=32 的语义在这里同样适用——
    实际 mempalace 调用点的 batch ≤ 几十条 drawer，远低于会让单次
    HTTP POST 出问题的规模。如果未来有人直接传 1000+ 条进来，要么
    在调用方加切分，要么改这里。
    """

    def __init__(self, url: str, timeout: int):
        self._url = url
        self._timeout = timeout

    @staticmethod
    def name() -> str:
        # ChromaDB 把 name() 持久化到 collection 上，读写必须一致。
        # 跟旧 ollama EF 的 "ollama" 不一样 → 旧 collection 必须 repair 重建。
        return "llamacpp"

    def __call__(self, input):  # noqa: A002 -- ChromaDB EF 协议
        """嵌入一组文本，返回 list[list[float]]。

        响应契约（curl 在 llama.cpp b9768 实测 + 对照 README）：
          - endpoint：POST /embeddings（带 s，非 OAI 兼容）
          - 根是 list（README 明确写出这个格式）
          - 每项形如 {"index": 0, "embedding": [[...1024 floats...]]}
          - embedding 是嵌套数组：外层是 token-chunk 维度
          - 启动用 --pooling mean，每个 input 只产生 1 个 chunk → 安全取 [0]
        """
        if isinstance(input, str):
            input = [input]
        if not input:
            return []

        resp = requests.post(
            f"{self._url}/embeddings",  # 带 s，非 OAI 兼容端点
            json={"content": list(input)},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return [item["embedding"][0] for item in resp.json()]

    def embed_query(self, input):  # noqa: A002
        return self(input)

    def embed_documents(self, input):  # noqa: A002
        return self(input)


def _build_llamacpp_ef() -> LlamacppEF:
    """构造 LlamacppEF 并做 fail-fast 健康探测。

    ensure_running() 保证服务在跑（spawn / 复用 / fail-fast）。然后
    立即跑一次 embedding 探测语义正确性——_llamacpp_server 的 /health
    检查只看 HTTP 层，不验证模型真的能输出有效向量。
    """
    from mempalace import _llamacpp_server

    url = _llamacpp_server.ensure_running()
    timeout_raw = os.getenv("MEMPALACE_LLAMACPP_TIMEOUT")
    try:
        timeout = int(timeout_raw) if timeout_raw else 60
    except ValueError as e:
        raise ValueError(f"MEMPALACE_LLAMACPP_TIMEOUT 必须是整数，收到 {timeout_raw!r}") from e
    if timeout <= 0:
        raise ValueError(f"MEMPALACE_LLAMACPP_TIMEOUT 必须是正整数，收到 {timeout}")

    ef = LlamacppEF(url=url, timeout=timeout)
    try:
        vec = ef([_LLAMACPP_HEALTH_PROBE])
    except Exception as e:
        raise RuntimeError(
            f"llamacpp embedding 健康探测失败 (url={url}, timeout={timeout}s): {e}。"
            f"日志：~/.mempalace/llamacpp_server.log"
        ) from e
    if not vec or not vec[0]:
        raise RuntimeError(
            f"llamacpp 返回空向量。url={url}。日志：~/.mempalace/llamacpp_server.log"
        )

    logger.info(
        "llamacpp embedding function initialized (url=%s timeout=%ds dim=%d)",
        url,
        timeout,
        len(vec[0]),
    )
    return ef


class EmbeddinggemmaONNX:
    """ChromaDB-compatible EF using embeddinggemma-300m ONNX (q8, MRL→384d).

    Cross-lingual cosine similarity on parallel-translated text averages 0.88
    across DE/FR/HI/IT/KO/RU vs 0.35 for ``all-MiniLM-L6-v2``. Output dim is
    truncated to 384 via Matryoshka Representation Learning so the model is a
    drop-in replacement for the MiniLM-shaped 384-dim collections ChromaDB
    creates by default — same vector width, no schema change.

    Switching an existing palace from minilm → embeddinggemma still requires
    re-embedding (different vector space) — collections persist the EF name
    and ChromaDB rejects mismatched reads. Run ``mempalace repair rebuild-index``.
    """

    @staticmethod
    def name() -> str:
        # ChromaDB persists this on the collection and refuses reads with a
        # mismatched EF — that's the signal that forces users to rebuild_index
        # when switching models. Keep it stable.
        return "embeddinggemma_300m"

    def __init__(
        self,
        preferred_providers=None,
        batch_size: int = _EMBEDDINGGEMMA_BATCH_SIZE,
        intra_op_num_threads: int = 0,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._providers = (
            list(preferred_providers) if preferred_providers else ["CPUExecutionProvider"]
        )
        self._batch_size = batch_size
        self._intra_op_num_threads = intra_op_num_threads
        self._session = None
        self._tokenizer = None
        self._np = None
        self._output_idx = None
        # Instances are shared across threads via _EF_CACHE; serialize the
        # one-time model load so concurrent cold calls cannot build (and
        # transiently hold) two full model sessions.
        self._load_lock = threading.Lock()

    def _lazy_load(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            try:
                import numpy as np
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
            except ImportError as e:
                raise ImportError(
                    "EmbeddinggemmaONNX requires huggingface_hub, tokenizers, and "
                    "numpy — these ship with mempalace core, so this error usually "
                    "means one was uninstalled or pinned to an incompatible version. "
                    "Reinstall with: pip install --upgrade --force-reinstall mempalace"
                ) from e

            logger.info(
                "Downloading %s/%s (cached after first run)…",
                _EMBEDDINGGEMMA_REPO,
                _EMBEDDINGGEMMA_ONNX,
            )
            model_path = hf_hub_download(
                _EMBEDDINGGEMMA_REPO, subfolder="onnx", filename=_EMBEDDINGGEMMA_ONNX
            )
            hf_hub_download(
                _EMBEDDINGGEMMA_REPO, subfolder="onnx", filename=_EMBEDDINGGEMMA_ONNX + "_data"
            )
            tok_path = hf_hub_download(_EMBEDDINGGEMMA_REPO, filename="tokenizer.json")

            session = ort.InferenceSession(
                model_path,
                sess_options=_intra_op_session_options(self._intra_op_num_threads),
                providers=self._providers,
            )
            out_names = [o.name for o in session.get_outputs()]
            # Model card: sentence_embedding is the pooled output (last_hidden_state
            # is the per-token output we don't want).
            output_idx = (
                out_names.index("sentence_embedding") if "sentence_embedding" in out_names else 1
            )

            tokenizer = Tokenizer.from_file(tok_path)
            tokenizer.enable_padding()
            tokenizer.enable_truncation(max_length=_EMBEDDINGGEMMA_MAX_LEN)
            self._output_idx = output_idx
            self._tokenizer = tokenizer
            self._np = np
            # Session is assigned last: the unlocked fast path above treats a
            # non-None session as "fully loaded", so every other attribute
            # must already be in place when it becomes visible.
            self._session = session

    def __call__(self, input: str | list[str] | None) -> list[list[float]]:  # noqa: A002 — ChromaDB EF protocol
        if isinstance(input, str):
            # A bare string would be iterated character by character below,
            # silently producing one garbage vector per character.
            input = [input]
        if input is None or len(input) == 0:
            # None or zero docs: nothing to embed; skip the lazy model
            # download. len() over truthiness so an array-like documents
            # sequence is not rejected by ambiguous-truth-value semantics.
            return []
        self._lazy_load()
        np = self._np
        embeddings: list[list[float]] = []
        # Tokenize and run per sub-batch, not over the whole input: padding
        # is to the longest sequence in the sub-batch, and the ONNX runtime
        # only ever holds batch_size rows of attention buffers at a time
        # (#1770).
        for start in range(0, len(input), self._batch_size):
            chunk = input[start : start + self._batch_size]
            texts = [_EMBEDDINGGEMMA_PREFIX + t for t in chunk]
            encs = self._tokenizer.encode_batch(texts)
            input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            attention_mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            outputs = self._session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            sent_emb = outputs[self._output_idx][:, :_EMBEDDINGGEMMA_DIM]
            # L2-normalize so cosine similarity == dot product (matches what the
            # MTEB methodology assumes; ChromaDB's distance is configured for it).
            norms = np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12
            embeddings.extend((sent_emb / norms).tolist())
        return embeddings

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002 — ChromaDB EF protocol
        """Embed query documents (ChromaDB EF protocol)."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a batch of documents (ChromaDB EF protocol)."""
        return self(input)


def get_embedding_function(device: Optional[str] = None, model: Optional[str] = None):
    """Return a cached embedding function for the requested device + model.

    ``device=None`` reads :attr:`MempalaceConfig.embedding_device`;
    ``model=None`` reads :attr:`MempalaceConfig.embedding_model`.
    The returned function is shared across calls with the same resolved
    provider list + model so we only pay model-load cost once per process.
    """
    if device is None or model is None:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        if device is None:
            device = cfg.embedding_device
        if model is None:
            model = cfg.embedding_model

    providers, effective = _resolve_providers(device)
    cache_key = (model, tuple(providers))
    cached = _EF_CACHE.get(cache_key)  # lock-free fast path; dict.get is GIL-atomic
    if cached is not None:
        return cached
    with _EF_CACHE_LOCK:
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached

        threads = _resolve_intra_op_threads()
        if model == "llamacpp":
            ef = _build_llamacpp_ef()
        elif model == "embeddinggemma":
            ef = EmbeddinggemmaONNX(preferred_providers=providers, intra_op_num_threads=threads)
        else:
            # Default: minilm (or anything we don't recognize — back-compat win).
            ef_cls = _build_ef_class()
            ef = ef_cls(preferred_providers=providers, intra_op_num_threads=threads)

        _EF_CACHE[cache_key] = ef
    logger.info(
        "Embedding function initialized (model=%s device=%s providers=%s)",
        model,
        effective,
        providers,
    )
    return ef


def _build_ollama_ef():
    """构造 ChromaDB 自带的 OllamaEmbeddingFunction，并做 fail-fast 健康探测。

    Why 健康探测：OllamaEmbeddingFunction.__init__ 只创建 ollama.Client，
    并不发起 HTTP 请求；服务/模型不可达要等到首次 __call__ 才暴露。
    这让 mempalace 各异步调用点（miner、search、mcp）报错时机不一、
    诊断困难。我们在工厂里显式 embed() 一次，确保只要本函数返回成功，
    后续调用就能用——不可达直接 raise，绝不静默回退到 minilm。

    所有参数从环境变量读，没有就用合理默认。避免动 config.py。
    """
    url = os.getenv("MEMPALACE_OLLAMA_URL", _OLLAMA_DEFAULT_URL)
    model_name = os.getenv("MEMPALACE_OLLAMA_MODEL", _OLLAMA_DEFAULT_MODEL)
    timeout_raw = os.getenv("MEMPALACE_OLLAMA_TIMEOUT")
    try:
        timeout = int(timeout_raw) if timeout_raw else _OLLAMA_DEFAULT_TIMEOUT
    except ValueError as e:
        raise ValueError(f"MEMPALACE_OLLAMA_TIMEOUT 必须是整数，收到 {timeout_raw!r}") from e
    if timeout <= 0:
        raise ValueError(f"MEMPALACE_OLLAMA_TIMEOUT 必须是正整数，收到 {timeout}")

    from chromadb.utils.embedding_functions.ollama_embedding_function import (
        OllamaEmbeddingFunction,
    )

    ef = OllamaEmbeddingFunction(url=url, model_name=model_name, timeout=timeout)

    # 健康探测：让 Ollama 服务/模型不可达问题立刻暴露
    try:
        ef([_OLLAMA_HEALTH_PROBE])
    except Exception as e:
        raise RuntimeError(
            f"Ollama embedding 服务健康探测失败 "
            f"(url={url}, model={model_name}, timeout={timeout}s): {e}. "
            f"请检查 Ollama 服务是否在运行、模型是否已 `ollama pull {model_name}`。"
        ) from e

    logger.info(
        "Ollama embedding function initialized (url=%s model=%s timeout=%ds)",
        url,
        model_name,
        timeout,
    )
    return ef


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved device.

    Used by the miner CLI header so users can see at a glance whether GPU
    acceleration actually engaged.
    """
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device
    _, effective = _resolve_providers(device)
    return effective


# Probed vector widths, keyed by resolved model name. Populated once per
# process the first time an identity is resolved for a model.
_DIM_CACHE: dict = {}


def current_model_name(model: Optional[str] = None) -> str:
    """Resolve the canonical embedder model name (cheap, no model load).

    This is the configured ``embedding_model`` (``"minilm"`` /
    ``"embeddinggemma"`` / ...), not the embedding function's internal
    ``name()`` (which is spoofed to ``"default"`` for ChromaDB compatibility).
    """
    if model is not None:
        return str(model).strip().lower()
    from .config import MempalaceConfig

    return MempalaceConfig().embedding_model


def probe_dimension(device: Optional[str] = None, model: Optional[str] = None) -> int:
    """Return the embedder's output dimension by embedding a short probe.

    Model-agnostic — works for any model without a hardcoded table — and
    cached per resolved model name so the probe is paid at most once per
    process. Returns ``0`` if the probe fails (treated as "dimension unknown"
    by the identity check, so a probe failure never blocks normal operation).
    """
    name = current_model_name(model)
    cached = _DIM_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        ef = get_embedding_function(device=device, model=model)
        vectors = ef(input=["probe"])
        dim = len(vectors[0]) if vectors and vectors[0] is not None else 0
    except Exception:
        logger.debug("Embedding dimension probe failed for model=%s", name, exc_info=True)
        dim = 0
    _DIM_CACHE[name] = dim
    return dim


def get_embedder_identity(device: Optional[str] = None, model: Optional[str] = None):
    """Resolve the current embedder identity (RFC 001).

    ``model_name`` from config (cheap); ``dimension`` from a cached one-time
    probe. Returns an :class:`~mempalace.backends.base.EmbedderIdentity`.
    """
    from .backends.base import EmbedderIdentity

    return EmbedderIdentity(
        model_name=current_model_name(model),
        dimension=probe_dimension(device=device, model=model),
    )
