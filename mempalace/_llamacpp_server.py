"""llama.cpp 子进程生命周期管理。

mempalace 切换到 llama.cpp embedding backend 后，需要按需启动一个本地
`llama-server.exe` 子进程并接管其生命周期。本模块封装这件事的所有
细节——路径解析、spawn 参数、Job Object 绑定、stdout/stderr 重定向、
health probe——让 embedding.py 只看到一个简单入口 ensure_running()。

设计原则：
- fail-fast：任一前置条件不满足立刻 RuntimeError，绝不静默回退
- 主 CLI 不污染：llama-server 的 C++ 日志统一沉到 ~/.mempalace/llamacpp_server.log
- 进程不留僵尸：Windows Job Object 绑定子进程，父死必带子
"""

from __future__ import annotations

import os
from pathlib import Path

# llama-server.exe 路径：fork 自留，硬编码（参见 ADR 已知技术债）
_LLAMA_SERVER_BIN = Path("D:/llama/llama-server.exe")

# 日志路径：所有 spawn 出来的 llama-server 实例 stdout/stderr 都重定向到这里
_LOG_PATH = Path.home() / ".mempalace" / "llamacpp_server.log"


def _resolve_paths() -> None:
    """校验环境变量和文件路径，缺一即 fail-fast。

    必须在 spawn 任何子进程前调用。三道关卡按上下文相关性排序：
    先校验环境变量（用户改 config 最容易出错的层），再校验 GGUF
    （文件大，最可能没下完），最后校验二进制（最稳定，几乎不变）。
    """
    cache = os.getenv("MODELSCOPE_CACHE")
    if not cache:
        raise RuntimeError(
            "环境变量 MODELSCOPE_CACHE 未设置。请设置为 ModelScope 缓存目录，例如 D:/modelscope"
        )

    gguf_path = (
        Path(cache)
        / "models"
        / "Qwen"
        / "Qwen3-Embedding-0___6B-GGUF"
        / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    )
    if not gguf_path.is_file():
        raise RuntimeError(
            f"GGUF 模型文件不存在：{gguf_path}。"
            "请在 ModelScope 下载 Qwen/Qwen3-Embedding-0___6B-GGUF 仓库的 Q8_0 量化版本。"
        )

    if not _LLAMA_SERVER_BIN.is_file():
        raise RuntimeError(
            f"llama-server.exe 不存在：{_LLAMA_SERVER_BIN}。"
            "请确认 llama.cpp 已安装到 D:/llama/，或修改 _llamacpp_server.py 里的 _LLAMA_SERVER_BIN。"
        )


# 模块顶层不再暴露具体 GGUF 路径常量——_resolve_paths() 内部重新构造，
# 是因为测试需要靠 monkeypatch 改 MODELSCOPE_CACHE 来覆盖三种失败分支。
