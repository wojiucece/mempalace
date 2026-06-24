import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_resolve_paths_missing_modelscope_cache(monkeypatch):
    """MODELSCOPE_CACHE 没设 → fail-fast"""
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)
    from mempalace import _llamacpp_server

    with pytest.raises(RuntimeError, match="MODELSCOPE_CACHE"):
        _llamacpp_server._resolve_paths()


def test_resolve_paths_missing_gguf(tmp_path, monkeypatch):
    """GGUF 文件不存在 → fail-fast，错误信息提示 ModelScope 下载"""
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    from mempalace import _llamacpp_server

    with pytest.raises(RuntimeError, match=r"GGUF.*ModelScope|ModelScope.*GGUF"):
        _llamacpp_server._resolve_paths()


def test_resolve_paths_missing_llama_server(tmp_path, monkeypatch):
    """llama-server.exe 不存在 → fail-fast，错误信息含路径"""
    # 准备一个假的 GGUF 让 GGUF 校验通过
    gguf_dir = tmp_path / "models" / "Qwen" / "Qwen3-Embedding-0___6B-GGUF"
    gguf_dir.mkdir(parents=True)
    (gguf_dir / "Qwen3-Embedding-0.6B-Q8_0.gguf").write_bytes(b"fake")
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    # 将 _LLAMA_SERVER_BIN 指向不存在的路径以触发校验失败
    from mempalace import _llamacpp_server

    with patch.object(
        _llamacpp_server, "_LLAMA_SERVER_BIN", Path("Z:/nonexistent/llama-server.exe")
    ):
        with pytest.raises(RuntimeError, match="llama-server"):
            _llamacpp_server._resolve_paths()


def test_resolve_url_default(monkeypatch):
    monkeypatch.delenv("MEMPALACE_LLAMACPP_URL", raising=False)
    from mempalace import _llamacpp_server

    assert _llamacpp_server._resolve_url() == "http://localhost:8080"


def test_resolve_url_from_env(monkeypatch):
    monkeypatch.setenv("MEMPALACE_LLAMACPP_URL", "http://127.0.0.1:9999")
    from mempalace import _llamacpp_server

    assert _llamacpp_server._resolve_url() == "http://127.0.0.1:9999"


def test_resolve_url_rejects_0_0_0_0(monkeypatch):
    """0.0.0.0 不是合法 client target，必须拒绝"""
    monkeypatch.setenv("MEMPALACE_LLAMACPP_URL", "http://0.0.0.0:8080")
    from mempalace import _llamacpp_server

    with pytest.raises(RuntimeError, match="0.0.0.0|host"):
        _llamacpp_server._resolve_url()


def test_resolve_url_rejects_empty_hostname(monkeypatch):
    """URL 没 hostname（如 http://:8080） → fail-fast，对应 not parsed.hostname 分支"""
    monkeypatch.setenv("MEMPALACE_LLAMACPP_URL", "http://:8080")
    from mempalace import _llamacpp_server

    with pytest.raises(RuntimeError, match="host"):
        _llamacpp_server._resolve_url()


def test_resolve_url_rejects_no_port(monkeypatch):
    """URL 不带端口 → fail-fast（spawn 需要端口源）"""
    monkeypatch.setenv("MEMPALACE_LLAMACPP_URL", "http://localhost")
    from mempalace import _llamacpp_server

    with pytest.raises(RuntimeError, match="port"):
        _llamacpp_server._resolve_url()


def test_resolve_port():
    from mempalace import _llamacpp_server

    assert _llamacpp_server._resolve_port("http://localhost:8080") == 8080
    assert _llamacpp_server._resolve_port("http://127.0.0.1:9999") == 9999


def test_probe_existing_server_connection_refused():
    """端口空闲 → None"""
    from mempalace import _llamacpp_server

    # 用一个几乎肯定空闲的端口
    result = _llamacpp_server._probe_existing_server("http://127.0.0.1:1", timeout=0.5)
    assert result is None


def test_probe_existing_server_llama_server_response():
    """200 + {"status":"ok"} → 'ready'（实测 b9768 的 ready 响应）"""
    from mempalace import _llamacpp_server

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"status": "ok"}
    with patch("requests.get", return_value=fake_resp):
        result = _llamacpp_server._probe_existing_server("http://localhost:8080")
    assert result == "ready"


def test_probe_existing_server_loading():
    """503 + Loading model → 'loading'（同一个 server 还在 warmup，README 承诺的状态）"""
    from mempalace import _llamacpp_server

    fake_resp = MagicMock()
    fake_resp.status_code = 503
    fake_resp.json.return_value = {
        "error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}
    }
    with patch("requests.get", return_value=fake_resp):
        result = _llamacpp_server._probe_existing_server("http://localhost:8080")
    assert result == "loading"


def test_probe_existing_server_status_not_ok():
    """200 + JSON 但 status 不是 'ok' → 'foreign'"""
    from mempalace import _llamacpp_server

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"status": "weird"}
    with patch("requests.get", return_value=fake_resp):
        result = _llamacpp_server._probe_existing_server("http://localhost:8080")
    assert result == "foreign"


def test_probe_existing_server_foreign_response():
    """返回 200 但响应体不是 JSON（如被 Nginx 占了）→ 'foreign'"""
    from mempalace import _llamacpp_server

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.side_effect = ValueError("not JSON")
    fake_resp.text = "<html>Welcome to nginx</html>"
    with patch("requests.get", return_value=fake_resp):
        result = _llamacpp_server._probe_existing_server("http://localhost:8080")
    assert result == "foreign"
