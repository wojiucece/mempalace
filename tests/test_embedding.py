import pytest

import mempalace.embedding as embedding


@pytest.fixture(autouse=True)
def isolate_embedding_state(monkeypatch):
    monkeypatch.setattr(embedding, "_EF_CACHE", {})
    monkeypatch.setattr(embedding, "_WARNED", set())


def test_auto_picks_cuda(monkeypatch):
    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert embedding._resolve_providers("auto") == (
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda",
    )


def test_auto_falls_to_cpu(monkeypatch):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("auto") == (["CPUExecutionProvider"], "cpu")


def test_cuda_missing_warns_with_gpu_extra(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("cuda") == (["CPUExecutionProvider"], "cpu")
    assert "mempalace[gpu]" in caplog.text


def test_coreml_missing_warns_with_coreml_extra(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("coreml") == (["CPUExecutionProvider"], "cpu")
    assert "mempalace[coreml]" in caplog.text


def test_dml_missing_warns_with_dml_extra(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("dml") == (["CPUExecutionProvider"], "cpu")
    assert "mempalace[dml]" in caplog.text


def test_unknown_device_warns_once(monkeypatch, caplog):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])

    assert embedding._resolve_providers("bogus") == (["CPUExecutionProvider"], "cpu")
    assert embedding._resolve_providers("bogus") == (["CPUExecutionProvider"], "cpu")
    assert caplog.text.count("Unknown embedding_device") == 1


def test_onnxruntime_import_error_falls_back_to_cpu(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert embedding._resolve_providers("cuda") == (["CPUExecutionProvider"], "cpu")


def test_get_embedding_function_caches_by_resolved_provider_tuple(monkeypatch):
    class DummyEF:
        def __init__(self, preferred_providers, intra_op_num_threads=0):
            self.preferred_providers = preferred_providers

    monkeypatch.setattr(embedding, "_build_ef_class", lambda: DummyEF)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )

    first = embedding.get_embedding_function("cpu", "minilm")
    second = embedding.get_embedding_function("auto", "minilm")

    assert first is second
    assert first.preferred_providers == ["CPUExecutionProvider"]


def test_intra_op_session_options_caps_threads():
    so = embedding._intra_op_session_options(3)
    assert so is not None
    assert so.intra_op_num_threads == 3


def test_intra_op_session_options_uncapped_returns_none():
    assert embedding._intra_op_session_options(0) is None
    assert embedding._intra_op_session_options(-1) is None


def test_get_embedding_function_threads_cap_passed_to_minilm_ef(monkeypatch):
    captured = {}

    class DummyEF:
        def __init__(self, preferred_providers, intra_op_num_threads=0):
            captured["threads"] = intra_op_num_threads

    monkeypatch.setattr(embedding, "_build_ef_class", lambda: DummyEF)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    monkeypatch.setattr(embedding, "_resolve_intra_op_threads", lambda: 2)

    embedding.get_embedding_function("cpu", "minilm")

    assert captured["threads"] == 2


def test_get_embedding_function_threads_cap_passed_to_embeddinggemma(monkeypatch):
    captured = {}

    class DummyGemma:
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            captured["threads"] = intra_op_num_threads

    monkeypatch.setattr(embedding, "EmbeddinggemmaONNX", DummyGemma)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    monkeypatch.setattr(embedding, "_resolve_intra_op_threads", lambda: 4)

    embedding.get_embedding_function("cpu", "embeddinggemma")

    assert captured["threads"] == 4


def test_minilm_ef_model_override_applies_thread_cap(monkeypatch):
    """The ``_MempalaceONNX.model`` override must construct the ORT session
    with the configured ``intra_op_num_threads`` (#1068). We stub
    ``InferenceSession`` to capture the ``SessionOptions`` it receives, so the
    test never downloads or loads the real model."""
    import onnxruntime as ort

    captured = {}

    def fake_session(model_path, providers=None, sess_options=None):
        captured["sess_options"] = sess_options
        captured["providers"] = providers
        return object()

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    ef_cls = embedding._build_ef_class()
    ef = ef_cls(preferred_providers=["CPUExecutionProvider"], intra_op_num_threads=2)
    _ = ef.model  # triggers the cached_property build

    assert captured["sess_options"] is not None
    assert captured["sess_options"].intra_op_num_threads == 2
    assert "CoreMLExecutionProvider" not in captured["providers"]


def test_minilm_ef_model_override_falls_back_when_uncapped(monkeypatch):
    """With no cap (0), the override must defer to the parent build via
    ``super().model`` — not reach into ``cached_property`` internals (#1068
    review). Proves super() resolves the parent descriptor without error."""
    import onnxruntime as ort

    captured = {}

    def fake_session(model_path, providers=None, sess_options=None):
        captured["sess_options"] = sess_options
        return object()

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    ef_cls = embedding._build_ef_class()
    ef = ef_cls(preferred_providers=["CPUExecutionProvider"], intra_op_num_threads=0)
    session = ef.model  # cap <= 0 → super().model (upstream builder)

    assert session is not None
    # Upstream leaves intra_op at ORT's default (0 = unset), confirming we
    # deferred to it rather than applying our cap.
    assert captured["sess_options"].intra_op_num_threads == 0


def test_describe_device_uses_resolved_effective_device(monkeypatch):
    monkeypatch.setattr(
        embedding,
        "_resolve_providers",
        lambda device: (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
    )

    assert embedding.describe_device("auto") == "cuda"


def test_llamacpp_ef_name():
    """EF name 必须固定为 llamacpp（ChromaDB 用它当 collection 身份证）"""
    from mempalace.embedding import LlamacppEF

    ef = LlamacppEF(url="http://localhost:8080", timeout=60)
    assert ef.name() == "llamacpp"


def test_llamacpp_ef_call_batch(monkeypatch):
    """__call__([text1, text2]) → 两个 1024 维向量
    实测响应格式：根是 list，每项 {"index":N, "embedding":[[1024 floats]]}"""
    from unittest.mock import MagicMock, patch
    from mempalace.embedding import LlamacppEF

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = [
        {"index": 0, "embedding": [[0.1] * 1024]},
        {"index": 1, "embedding": [[0.2] * 1024]},
    ]
    with patch("requests.post", return_value=fake_resp) as mock_post:
        ef = LlamacppEF(url="http://localhost:8080", timeout=60)
        result = ef(["hello", "world"])

    assert len(result) == 2
    assert len(result[0]) == 1024
    assert result[0][0] == 0.1
    assert result[1][0] == 0.2
    # 验证 POST 到 /embeddings（带 s！），请求体里 content 是列表
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8080/embeddings"
    assert kwargs["json"] == {"content": ["hello", "world"]}


def test_llamacpp_ef_call_string_wraps_as_list(monkeypatch):
    """bare string → 自动包成单元素列表（避免按字符迭代）"""
    from unittest.mock import MagicMock, patch
    from mempalace.embedding import LlamacppEF

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = [{"index": 0, "embedding": [[0.1] * 1024]}]
    with patch("requests.post", return_value=fake_resp):
        ef = LlamacppEF(url="http://localhost:8080", timeout=60)
        result = ef("hello")
    assert len(result) == 1
    assert len(result[0]) == 1024


def test_llamacpp_ef_empty_input_skips_request(monkeypatch):
    """空输入 → 直接返回 []，不发请求"""
    from unittest.mock import patch
    from mempalace.embedding import LlamacppEF

    with patch("requests.post") as mock_post:
        ef = LlamacppEF(url="http://localhost:8080", timeout=60)
        assert ef([]) == []
        assert ef(None) == []
        mock_post.assert_not_called()


def test_llamacpp_ef_internal_batch_chunking(monkeypatch):
    """超过 _LLAMACPP_MAX_BATCH 的输入按 16 条/批切分，按顺序拼回。

    Regression: mempalace `repair --mode from-sqlite` 直接 upsert 整个 1449
    条 list，原实现单次 POST 导致 llama-server 端真在算但客户端 read timeout。
    本测试确保 EF 内部按 16 切片、多次 POST、结果按 input 顺序拼回。
    """
    from unittest.mock import patch, MagicMock
    from mempalace.embedding import LlamacppEF, _LLAMACPP_MAX_BATCH

    assert _LLAMACPP_MAX_BATCH == 16, "测试假设 batch=16；上限改了请同步更新"

    # 构造 20 条输入 → 应触发 2 次 POST：第一批 16 条 + 第二批 4 条
    inputs = [f"text-{i}" for i in range(20)]

    def fake_post(url, json, timeout):
        # 按 mempalace b9768 实测响应格式回 mock：响应根是 list，
        # 每项 {"index": i, "embedding": [[1024 floats]]}
        content = json["content"]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [
            # 用 input 字符串的索引值填进 vector[0]，便于验证顺序
            {"index": i, "embedding": [[float(int(text.split("-")[1]))] + [0.0] * 1023]}
            for i, text in enumerate(content)
        ]
        return resp

    with patch("requests.post", side_effect=fake_post) as mock_post:
        ef = LlamacppEF(url="http://localhost:8080", timeout=60)
        result = ef(inputs)

    # 必须发 2 次 HTTP
    assert mock_post.call_count == 2
    # 第一次 POST 16 条
    first_payload = mock_post.call_args_list[0].kwargs["json"]["content"]
    assert len(first_payload) == 16
    assert first_payload[0] == "text-0"
    assert first_payload[15] == "text-15"
    # 第二次 POST 4 条
    second_payload = mock_post.call_args_list[1].kwargs["json"]["content"]
    assert len(second_payload) == 4
    assert second_payload[0] == "text-16"
    assert second_payload[3] == "text-19"
    # 返回 20 条向量，且按 input 顺序拼回
    assert len(result) == 20
    assert result[0][0] == 0.0
    assert result[15][0] == 15.0
    assert result[16][0] == 16.0
    assert result[19][0] == 19.0
