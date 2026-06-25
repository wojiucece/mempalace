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
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

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

# spawn 后等待 ready 的总预算
_READY_TIMEOUT_SECONDS = 30
_READY_POLL_INTERVAL = 0.1

# 模块级：已经 spawn 出来的 Popen 对象（避免重复 spawn）
_spawned_proc: subprocess.Popen | None = None


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
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.OpenProcess.restype = ctypes.c_void_p

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


def ensure_running() -> str:
    """保证有一个就绪的 llama-server，返回其 URL。

    幂等：调用多次只会真正 spawn 一次。流程：
      1. fail-fast 校验路径
      2. 探测目标 URL 上是否已有 llama-server
         - ready：复用
         - loading：等（已有 server 在 warmup，不要再 spawn）
         - foreign：报错（不抢占端口）
         - 空闲：spawn 自己的
      3. 轮询 /health 直到 ready（最长 30s）
    """
    _resolve_paths()
    url = _resolve_url()

    # 已经 spawn 过且还活着 → 复用
    global _spawned_proc
    if _spawned_proc is not None and _spawned_proc.poll() is None:
        return url

    probe_result = _probe_existing_server(url)
    if probe_result == "ready":
        logger.info("复用 %s 上已有的 llama-server", url)
        return url
    if probe_result == "loading":
        # 已经有一个 server 在加载（上次 mempalace 留的或用户手动启的）。
        # 不要再 spawn（端口会冲突），直接等它就绪。
        logger.info("%s 上已有 llama-server 在加载，等待就绪", url)
        _wait_until_ready(url)
        return url
    if probe_result == "foreign":
        raise RuntimeError(
            f"端口 {_resolve_port(url)} 被非 llama-server 进程占用。"
            f"请释放该端口，或修改 MEMPALACE_LLAMACPP_URL 指向其他端口。"
        )

    # 空闲，自己起一个
    _spawned_proc = _spawn(url)
    _bind_to_job_object(_spawned_proc)
    _wait_until_ready(url)
    return url


def _spawn(url: str) -> subprocess.Popen:
    """启动 llama-server 子进程，stdout/stderr 重定向到日志文件。

    注意：log_fh（open 返回的文件句柄）在传递给 Popen 后没有显式 close()。
    这是有意为之——子进程接管了该 fd，Python 端关闭句柄会导致子进程的
    stdout 也跟着损坏。Windows 上 Popen 内部会 DuplicateHandle，子进程
    持有引用时 Python 端的 close 是安全的，但显式 close 仍有风险。
    让 GC 在子进程退出后自然回收即可。
    """
    port = _resolve_port(url)
    cache = os.environ["MODELSCOPE_CACHE"]  # _resolve_paths 已校验存在
    gguf_path = (
        Path(cache)
        / "models"
        / "Qwen"
        / "Qwen3-Embedding-0___6B-GGUF"
        / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    )

    # 调优依据（i5-12500H + 16GB + CPU 推理）：
    #   --threads 8        避开 P 核（8 逻辑线程）+ E 核大小核调度陷阱
    #   --ctx-size 8192    embedding 上下文窗口；mempalace drawer 偶尔
    #                      1500+ token（CLAUDE.md 等长文档），8192 留余量
    #   --batch-size 4096 / --ubatch-size 4096
    #                      physical batch 必须 >= 单条 input token 数，
    #                      否则 llama-server 返 500 "input is too large to process"
    #   --mlock            锁热点页面在物理内存，避开 swap 抖动
    #   --parallel 2       i5-12500H 同时跑 2 个 batch 不堵塞
    #   不指定 --pooling   用模型默认 last-token pooling
    #                      （Qwen3-Embedding 官方 HuggingFace README 的训练时设置）
    args = [
        str(_LLAMA_SERVER_BIN),
        "-m",
        str(gguf_path),
        "--embedding",
        "--port",
        str(port),
        "--threads",
        "8",
        "--ctx-size",
        "8192",
        "--batch-size",
        "4096",
        "--ubatch-size",
        "4096",
        "--mlock",
        "--parallel",
        "2",
        "-ngl",
        "0",
    ]

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # append 模式：保留历史 spawn 的日志方便排查；不用 PIPE 避免缓冲区满阻塞
    log_fh = open(_LOG_PATH, "a", buffering=1, encoding="utf-8")
    log_fh.write(f"\n=== mempalace spawn at {time.time():.0f} ===\n")
    log_fh.flush()

    logger.info("spawn llama-server: %s", " ".join(args))
    proc = subprocess.Popen(
        args,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        # Windows: 不创建新控制台窗口（mempalace CLI 已经在终端里）
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return proc


def _wait_until_ready(url: str) -> None:
    """轮询 /health 直到返回 ready（200 + status:ok），最长 30s。

    loading 状态视为"继续等"。foreign 状态视为"被劫持"立刻 fail。
    """
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        probe = _probe_existing_server(url, timeout=1.0)
        if probe == "ready":
            logger.info("llama-server 在 %s 就绪", url)
            return
        if probe == "foreign":
            # 中途被无关进程顶替了端口（极端情况，但要 fail-fast）
            tail = _read_log_tail(20)
            raise RuntimeError(
                f"llama-server 还在加载时 {url} 被非 llama-server 进程顶替。\n"
                f"日志尾部 ({_LOG_PATH})：\n{tail}"
            )
        # loading / None：继续等
        time.sleep(_READY_POLL_INTERVAL)

    # 超时：把日志尾部贴进异常信息
    tail = _read_log_tail(20)
    raise RuntimeError(
        f"llama-server 在 {_READY_TIMEOUT_SECONDS}s 内未就绪。\n日志尾部 ({_LOG_PATH})：\n{tail}"
    )


def _read_log_tail(n_lines: int) -> str:
    """读 log 文件最后 n 行（拼到错误信息里）。

    策略：seek 到末尾前 4KB 读取，按 \\n 分割取最后 n 个非空行。
    不读整个文件——日志可能很大（spawn 历史累积），整读会慢。
    不要求 line buffering 完整——llama-server 可能正在 flush 中途，
    截断的第一行直接丢弃（一行 truncate 影响诊断价值 << 实现复杂度）。
    """
    try:
        size = _LOG_PATH.stat().st_size
        with open(_LOG_PATH, "rb") as f:
            # seek 到末尾前 4KB（或文件开头，取较大者）
            f.seek(max(0, size - 4096))
            chunk = f.read().decode("utf-8", errors="replace")
        # 拆行；如果 seek 落在某行中间，第一行残缺，扔掉
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        if size > 4096 and lines:
            lines = lines[1:]  # 第一行可能 truncate，扔
        if not lines:
            return "(日志为空，llama-server 可能还没产生输出)"
        return "\n".join(lines[-n_lines:])
    except FileNotFoundError:
        return "(日志文件不存在)"
