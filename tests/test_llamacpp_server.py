import os
import pytest
from pathlib import Path
from unittest.mock import patch


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
    # patch _LLAMA_SERVER_BIN to point to a non-existent path
    from mempalace import _llamacpp_server
    with patch.object(_llamacpp_server, "_LLAMA_SERVER_BIN", Path("Z:/nonexistent/llama-server.exe")):
        with pytest.raises(RuntimeError, match="llama-server"):
            _llamacpp_server._resolve_paths()