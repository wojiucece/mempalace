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

import ctypes
import os
import subprocess
import sys
from pathlib import Path

from urllib.parse import urlparse

import requests

# Win32 API 常量
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


# Win32 Job Object 结构体。提到模块顶层避免在 _bind_to_job_object 每次调用
# 都重建 class（Python 反模式 + 微小性能浪费）。三个 Structure 互相嵌套，
# 必须按依赖顺序定义。
class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


# 模块级 job handle：mempalace 进程整个生命周期共享一个 job，
# 所有 spawn 的 llama-server 绑到同一个 job 上。父死 → job 关闭 → 全部子进程一起死。
#
# 关于"绑直接子进程是否覆盖孙进程"：Windows Job Object 是继承式的。一旦
# llama-server 被加入 job，它后续 spawn 的任何子进程都自动在同一 job 里
# （除非显式 CREATE_BREAKAWAY_FROM_JOB）。llama.cpp 不用这个标志，所以
# 整个进程树都受 KILL_ON_JOB_CLOSE 保护，无须额外代码。
_job_handle: int | None = None

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

_LLAMACPP_DEFAULT_URL = "http://localhost:8080"
_LLAMACPP_DEFAULT_TIMEOUT = 60  # 秒，HTTP 请求超时


def _resolve_url() -> str:
    """读 MEMPALACE_LLAMACPP_URL，校验合法性。

    校验项：
    - host 不为空、不为 0.0.0.0（client 目标地址不能是 wildcard）
    - 必须带端口（spawn 时用作 --port 的来源）
    """
    url = os.getenv("MEMPALACE_LLAMACPP_URL", _LLAMACPP_DEFAULT_URL)
    parsed = urlparse(url)
    if not parsed.hostname or parsed.hostname == "0.0.0.0":
        raise RuntimeError(
            f"MEMPALACE_LLAMACPP_URL 的 host 必须是具体地址，不能为空或 0.0.0.0。当前值：{url}"
            f"。例如：MEMPALACE_LLAMACPP_URL=http://localhost:8080"
        )
    if parsed.port is None:
        raise RuntimeError(
            f"MEMPALACE_LLAMACPP_URL 必须显式带端口（spawn 时用作 --port 参数源）。当前值：{url}"
            f"。例如：MEMPALACE_LLAMACPP_URL=http://localhost:8080"
        )
    return url


def _resolve_port(url: str) -> int:
    """从 URL 抽出端口号。"""
    port = urlparse(url).port
    if port is None:
        raise RuntimeError(f"URL 必须带端口：{url}。例如：http://localhost:8080")
    return port


def _probe_existing_server(url: str, timeout: float = 2.0) -> str | None:
    """探测 URL 上是否已经有 llama-server 在跑。

    返回值（README 在 b9768 上承诺这两个状态码）：
      - "ready":   200 + {"status":"ok"}                  llama-server 就绪，可用
      - "loading": 503 + {"error":{"message":"Loading model",...}}
                                                          llama-server 还在加载（等就行）
      - "foreign": 其他响应（被无关进程占了）
      - None:      连接失败 / 超时（端口空闲，可以 spawn）
    """
    try:
        resp = requests.get(f"{url}/health", timeout=timeout)
    except requests.exceptions.RequestException:
        return None

    try:
        body = resp.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return "foreign"  # 不是 JSON → Nginx / Tomcat 默认页

    # llama-server 就绪
    if resp.status_code == 200 and isinstance(body, dict) and body.get("status") == "ok":
        return "ready"

    # llama-server 仍在加载模型（README 承诺的 503）
    if resp.status_code == 503 and isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict) and "Loading model" in str(err.get("message", "")):
            return "loading"

    return "foreign"


def _bind_to_job_object(proc: subprocess.Popen) -> None:
    """把子进程绑到 mempalace 的 Job Object 上。

    仅 Windows 实现。fork 实际只跑 Windows，Linux/Mac 直接 fail-fast
    退出，不画饼。第一次调用时创建 job 并设置 KILL_ON_JOB_CLOSE 标志，
    后续调用复用同一个 job。
    """
    if sys.platform != "win32":
        raise RuntimeError(
            "_bind_to_job_object 仅支持 Windows（Windows-only）。"
            "fork 当前不投资 Linux/Mac 上的 llama-server 进程清理。"
        )

    global _job_handle
    k32 = ctypes.windll.kernel32

    if _job_handle is None:
        # 创建 job
        _job_handle = k32.CreateJobObjectW(None, None)
        if not _job_handle:
            raise RuntimeError(f"CreateJobObjectW 失败，GetLastError={ctypes.get_last_error()}")

        # 设置 KILL_ON_JOB_CLOSE：mempalace 退出 → job 句柄关闭 → 子进程全杀
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
            _job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise RuntimeError(
                f"SetInformationJobObject 失败，GetLastError={ctypes.get_last_error()}"
            )

    # 把子进程加到 job
    proc_handle = k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, proc.pid)
    if not proc_handle:
        raise RuntimeError(
            f"OpenProcess 失败 (pid={proc.pid})，GetLastError={ctypes.get_last_error()}"
        )
    try:
        if not k32.AssignProcessToJobObject(_job_handle, proc_handle):
            raise RuntimeError(
                f"AssignProcessToJobObject 失败，GetLastError={ctypes.get_last_error()}"
            )
    finally:
        k32.CloseHandle(proc_handle)
