# -*- coding: utf-8 -*-
"""方案A 阶段1：自管理设备显存（直通分配，无池）

职责划分（见 `doc/CUDAC/显存管理.md` 第四节 D1/D2）：

    C 层（dev_memory.dll）   无状态薄封装：cudaMalloc/cudaFree/cudaMemcpy + 错误码 + 计数
    本模块                   句柄化 + 块表 + 双重释放检测 + 泄漏清单 + 上下文管理

**已实现**：
    阶段1  直通分配、同步传输、句柄与生命周期、错误上报、统计、无 CUDA 回退
    阶段2  size-class 内存池（命中即复用，稳态零 cudaMalloc）、trim、池驱逐、池统计

**不含**（后续阶段）：pinned/异步/双缓冲（阶段3）、
`__cuda_array_interface__` 互操作（阶段4）、多设备池隔离（阶段5）。

不依赖 cupy（NFR-6），只用到 `ctypes` + `numpy`。

典型用法：

    from eneuro.utils.dev_memory import DeviceAllocator
    from eneuro.utils.cuda_ops import CudaOps

    with DeviceAllocator() as mem:                       # 会话级
        da = mem.alloc_like(a_np, name="a")
        db = mem.alloc_like(b_np, name="b")
        dout = mem.alloc_like(a_np, name="out")
        da.upload(a_np); db.upload(b_np)                 # 一次进
        CudaOps().dev("add", da, db, dout)               # 多次算（复用现有算子）
        dout.download(out_np)                            # 一次出
    # 退出时自动归还；有残留会打印泄漏清单
"""
from __future__ import annotations

import atexit
import ctypes
import os
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
ENEURO_DIR = Path(__file__).resolve().parents[1]        # .../code/eneuro
DEFAULT_LIB_DIR = Path(os.environ.get("ENE_CUDA_LIB_DIR")
                       or (ENEURO_DIR / "cuda" / ("dll" if os.name == "nt" else "so")))
EXT = "dll" if os.name == "nt" else "so"
LIB_NAME = "dev_memory"

ENE_OK = 0
ENE_ERR_OOM = 1
ENE_ERR_INVALID = 2
ENE_ERR_DEVICE = 3
ENE_ERR_CUDA = 4
ENE_ERR_NODEVICE = 5


# ---------------------------------------------------------------------------
# 内存池的 size-class 划分（见 doc/CUDAC/显存管理.md 5.1）
#
#   s = 256 B 步进（≤ 4 KB）  —— 小块最密集，按 256 B 对齐把内部碎片压到最小
#   p = 2 的幂（4 KB ~ 1 MB） —— 桶数有限（9 个），内部碎片 ≤ 2x
#   e = 精确大小（> 1 MB）    —— 不再取整，避免大块白白浪费显存
#
# 「best-fit」在**桶级别**完成：选最小的、装得下的桶；
# 桶内所有块容量相同，取任意一个即可（O(1) pop），无需遍历比较。
#
# 注：设计文档原本写「> 1 MB 直通不进池」，这里改为「按精确大小分桶进池」。
# 原因是 NFR-3（稳态零 cudaMalloc）在典型张量尺寸（往往 > 1 MB）上必须也成立；
# 而「大块长期占显存」的问题由 pool_max_bytes 上限 + trim(older_than=) 解决，
# 比一刀切直通更精细。需要旧行为时传 pool_max_block=1<<20 即可。
_SMALL_STEP = 256
_SMALL_MAX = 4096
_LARGE_MIN = 1 << 20

BucketKey = tuple[str, int]                       # (kind, capacity)


def _size_class(nbytes: int) -> tuple[BucketKey, int]:
    """返回 (桶 key, 容量)。容量 ≥ nbytes，容量相同的块共用一个空闲链表。"""
    if nbytes <= _SMALL_STEP:
        return ("s", _SMALL_STEP), _SMALL_STEP
    if nbytes <= _SMALL_MAX:
        capacity = ((nbytes + _SMALL_STEP - 1) // _SMALL_STEP) * _SMALL_STEP
        return ("s", capacity), capacity
    if nbytes <= _LARGE_MIN:
        capacity = 1 << (nbytes - 1).bit_length()          # 向上取 2 的幂
        return ("p", capacity), capacity
    return ("e", nbytes), nbytes                           # 精确大小桶


# ---------------------------------------------------------------------------
# 异常体系（FR-1.2 / FR-4.4 / FR-8.2）
# ---------------------------------------------------------------------------
class DevMemError(RuntimeError):
    """显存管理相关错误基类"""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class OutOfMemoryError(DevMemError):
    """设备显存不足"""


class InvalidArgumentError(DevMemError):
    """参数非法（dtype / 连续性 / 长度不匹配等）"""


class DeviceError(DevMemError):
    """设备相关错误（设备号非法、跨设备访问）"""


class BufferReleasedError(DevMemError):
    """使用了已释放的句柄（含双重释放）"""


class AllocatorClosedError(DevMemError):
    """分配器已关闭"""


class LibraryNotFoundError(DevMemError):
    """找不到 dev_memory 动态库"""


_CODE_TO_EXC = {
    ENE_ERR_OOM: OutOfMemoryError,
    ENE_ERR_INVALID: InvalidArgumentError,
    ENE_ERR_DEVICE: DeviceError,
    ENE_ERR_CUDA: DevMemError,
    ENE_ERR_NODEVICE: DevMemError,
}


def _raise_for(code: int, message: str) -> None:
    exc = _CODE_TO_EXC.get(code, DevMemError)
    raise exc(f"{message} (code={code})", code=code)


def stream_ptr(stream: Optional[int]) -> int:
    """流句柄规范化：None / 0 -> 0（默认流）

    注意：这里的流是 CUDA 裸指针（int）。若要与 cupy 的流互操作，
    传 `cp.cuda.get_current_stream().ptr` 即可，两者是同一个东西。
    """
    return int(stream) if stream else 0


# ---------------------------------------------------------------------------
# ctypes 封装
# ---------------------------------------------------------------------------
class _Lib:
    """dev_memory.dll 的签名绑定"""

    def __init__(self, path: Path):
        self.path = path
        self.dll = ctypes.CDLL(str(path))
        self.dll.ene_mem_init.argtypes = [ctypes.c_int]
        self.dll.ene_mem_init.restype = ctypes.c_int
        self.dll.ene_mem_available.argtypes = []
        self.dll.ene_mem_available.restype = ctypes.c_int
        self.dll.ene_mem_device.argtypes = []
        self.dll.ene_mem_device.restype = ctypes.c_int
        self.dll.ene_mem_device_count.argtypes = []
        self.dll.ene_mem_device_count.restype = ctypes.c_int
        self.dll.ene_mem_last_code.argtypes = []
        self.dll.ene_mem_last_code.restype = ctypes.c_int
        self.dll.ene_mem_last_error.argtypes = []
        self.dll.ene_mem_last_error.restype = ctypes.c_char_p
        self.dll.ene_mem_alloc.argtypes = [ctypes.c_size_t]
        self.dll.ene_mem_alloc.restype = ctypes.c_void_p
        self.dll.ene_mem_free.argtypes = [ctypes.c_void_p]
        self.dll.ene_mem_free.restype = ctypes.c_int
        for name in ("ene_mem_upload", "ene_mem_download"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        self.dll.ene_mem_sync.argtypes = [ctypes.c_void_p]
        self.dll.ene_mem_sync.restype = ctypes.c_int
        for name in ("ene_mem_alloc_calls", "ene_mem_free_calls",
                     "ene_mem_upload_bytes", "ene_mem_download_bytes",
                     "ene_mem_host_alloc_calls"):
            fn = getattr(self.dll, name)
            fn.argtypes = []
            fn.restype = ctypes.c_ulonglong

        # ---- 阶段3：流 / 事件 / pinned 主机内存 / 异步传输 ----
        for name in ("ene_mem_stream_create", "ene_mem_event_create",
                     "ene_mem_event_create_timed", "ene_mem_host_alloc"):
            fn = getattr(self.dll, name)
            fn.argtypes = [] if name.startswith("ene_mem_stream") or "event_create" in name \
                else [ctypes.c_size_t]
            fn.restype = ctypes.c_void_p
        for name in ("ene_mem_stream_destroy", "ene_mem_event_destroy",
                     "ene_mem_event_sync", "ene_mem_event_done", "ene_mem_host_free"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
        for name in ("ene_mem_event_record", "ene_mem_event_wait"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        self.dll.ene_mem_event_elapsed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.dll.ene_mem_event_elapsed.restype = ctypes.c_float
        self.dll.ene_mem_is_pinned.argtypes = [ctypes.c_void_p]
        self.dll.ene_mem_is_pinned.restype = ctypes.c_int
        for name in ("ene_mem_upload_async", "ene_mem_download_async"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            fn.restype = ctypes.c_int

    # 统一的「取错误信息并抛异常」helper
    def _check(self, code: int, what: str) -> None:
        if code != ENE_OK:
            detail = self.dll.ene_mem_last_error()
            detail = detail.decode("utf-8", "replace") if detail else ""
            _raise_for(code, f"{what} 失败: {detail}")

    def last_error(self) -> tuple[int, str]:
        code = int(self.dll.ene_mem_last_code())
        msg = self.dll.ene_mem_last_error()
        return code, (msg.decode("utf-8", "replace") if msg else "")


def load_library(lib_dir: Optional[Path | str] = None) -> Optional[_Lib]:
    """加载 dev_memory 库；文件不存在时返回 None（由调用方决定是否回退）"""
    path = Path(lib_dir or DEFAULT_LIB_DIR) / f"{LIB_NAME}.{EXT}"
    if not path.exists():
        return None
    return _Lib(path)


# ---------------------------------------------------------------------------
# 块表
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class _BlockInfo:
    """一块显存的元数据（主机侧，FR-1.4）；设备端不加块头，保证指针对齐

    `eq=False` 是故意的：让 `list.remove` 走**身份比较**。
    不同块的 ptr 虽然不同，但其他字段可能一致，用值比较会删错。
    """

    ptr: int
    nbytes: int                                   # 调用方请求的字节数
    capacity: int                                 # 实际分配到的字节数（桶容量 ≥ nbytes）
    dtype: np.dtype
    device: int                                   # -1 表示主机回退模式
    name: str
    released: bool = False
    host: Optional[bytearray] = None              # 主机回退模式的后备内存
    tag: int = 0                                  # 便于在泄漏清单里区分
    cls: Optional[BucketKey] = None               # size-class 桶 key
    free_seq: int = 0                             # 入池顺序（FIFO 驱逐用）
    free_time: float = 0.0                        # 入池时刻（trim(older_than=) 用）

    @property
    def size(self) -> int:
        """按 dtype 计的元素个数（13 个算子都按 float32 解释）"""
        return self.nbytes // self.dtype.itemsize

    @property
    def wasted(self) -> int:
        """桶取整带来的内部碎片"""
        return self.capacity - self.nbytes

    def describe(self) -> str:
        where = "host" if self.device < 0 else f"cuda:{self.device}"
        bucket = "直通" if self.cls is None else f"桶{self.cls[0]}/{self.cls[1]}"
        return (f"#{self.tag:<4} {self.name:<12} ptr=0x{self.ptr:012x} "
                f"请求 {self.nbytes:>9} B / 实占 {self.capacity:>9} B "
                f"({self.size} × {self.dtype}) {where} {bucket}")


# ---------------------------------------------------------------------------
# 锁页（pinned）主机内存（阶段3）
# ---------------------------------------------------------------------------
class PinnedBuffer:
    """一块锁页（pinned）主机内存

    `.array` 是它上面的 numpy 视图（**零拷贝**）：可以先把数据算到这里，
    再交给 `upload_async` 做真正的异步 DMA；也可以作为 `download_async` 的目标。
    """

    __slots__ = ("_alloc", "_ptr", "nbytes", "dtype", "name", "array", "released",
                 "_backing")

    def __init__(self, alloc: "DeviceAllocator", ptr: int, nbytes: int,
                 dtype: Any, name: str, backing: Any = None):
        self._alloc = alloc
        self._ptr = ptr
        self.nbytes = nbytes
        self.dtype = np.dtype(dtype)
        self.name = name
        self.released = False
        self._backing = backing                      # 主机回退时保活 ctypes 数组
        # 用 ctypes 数组拿缓冲协议，再 zero-copy 地 view 成目标 dtype
        raw = (ctypes.c_char * nbytes).from_address(ptr)
        self.array = np.frombuffer(raw, dtype=np.uint8, count=nbytes).view(self.dtype)

    @property
    def ptr(self) -> int:
        if self.released:
            raise BufferReleasedError(f"pinned 块 {self.name!r} 已归还，不能再使用")
        return self._ptr

    def release(self) -> None:
        """归还到 pinned 池（重复调用会报错）"""
        self._alloc._pinned_release(self, quiet=False)

    def __enter__(self) -> "PinnedBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._alloc._pinned_release(self, quiet=True)
        return False

    def __del__(self) -> None:                           # pragma: no cover
        try:
            self._alloc._pinned_release(self, quiet=True)
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "released" if self.released else "live"
        return f"<PinnedBuffer {self.name!r} {self.nbytes}B {state}>"


# ---------------------------------------------------------------------------
# 句柄
# ---------------------------------------------------------------------------
class DeviceBuffer:
    """设备显存句柄（阶段1：直通分配，无池）

    裸指针只在 `ptr` 属性上暴露一次，且会拒绝已释放的块 —— 这是「句柄化」的落点（D1）。
    """

    __slots__ = ("_alloc", "_info")

    def __init__(self, alloc: "DeviceAllocator", info: _BlockInfo):
        self._alloc = alloc
        self._info = info

    # ---- 元数据 ----
    @property
    def nbytes(self) -> int:
        """请求的字节数（上传/下载按它校验）"""
        return self._info.nbytes

    @property
    def capacity(self) -> int:
        """实际占用显存的字节数（桶容量 ≥ nbytes，差额是内部碎片）"""
        return self._info.capacity

    @property
    def size(self) -> int:
        return self._info.size

    @property
    def dtype(self) -> np.dtype:
        return self._info.dtype

    @property
    def device(self) -> int:
        """-1 表示主机回退模式"""
        return self._info.device

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def released(self) -> bool:
        return self._info.released

    @property
    def ptr(self) -> int:
        """裸设备指针。仅供下发 kernel 使用（FR-5.2，O(1)）。"""
        if self._info.released:
            raise BufferReleasedError(f"块 {self._info.name!r} 已被释放，不能再取指针")
        return self._info.ptr

    def __int__(self) -> int:
        return self.ptr

    def __repr__(self) -> str:
        state = "released" if self._info.released else "live"
        cap = "" if self._info.capacity == self._info.nbytes else f" cap={self._info.capacity}B"
        return f"<DeviceBuffer {self._info.name!r} {self.nbytes}B{cap} {state}>"

    # ---- 传输 ----
    def upload(self, src: np.ndarray, *, stream: Optional[int] = None) -> "DeviceBuffer":
        """把主机数组拷进本块（H2D）。要求 dtype 一致、C 连续、字节数相等。"""
        self._alloc._upload(self._info, src, stream)
        return self

    def download(self, dst: np.ndarray, *, stream: Optional[int] = None) -> np.ndarray:
        """把本块拷回主机数组（D2H）。"""
        self._alloc._download(self._info, dst, stream)
        return dst

    def to_numpy(self, *, stream: Optional[int] = None) -> np.ndarray:
        """拷回一个新的 numpy 数组（便捷方法）"""
        out = np.empty(self._info.size, dtype=self._info.dtype)
        self.download(out, stream=stream)
        return out

    # ---- 异步传输（阶段3，只入队不等待）----
    def upload_async(self, src: np.ndarray, *, stream: Optional[int] = None) -> "DeviceBuffer":
        """异步 H2D：只入队，**不等完成**

        `src` 不是锁页内存时，会先拷进内部 pinned staging（主机内 memcpy）再异步发出；
        staging 登记为「在飞」，要到 `allocator.sync()` 或池回收时才复用。
        使用数据前必须 sync（或依赖同一流上的后续操作）。
        """
        self._alloc._upload_async(self._info, src, stream)
        return self

    def download_async(self, dst: np.ndarray, *, stream: Optional[int] = None) -> np.ndarray:
        """异步 D2H：只入队，不等完成

        目标**必须是锁页内存**（用 `allocator.pinned(...)` 的 `.array`）。
        往 pageable 内存做异步 D2H 实际上并不会异步，这里直接报错而不是骗你。
        """
        self._alloc._download_async(self._info, dst, stream)
        return dst

    # ---- 生命周期 ----
    def release(self) -> None:
        """显式归还。重复调用会报错（FR-4.4 双重释放检测）。"""
        self._alloc._release(self._info, quiet=False)

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # FR-4.3：异常路径也要回收
        self._alloc._release(self._info, quiet=True)
        return False

    def __del__(self) -> None:
        # 兜底自动回收；__del__ 不能抛异常
        try:
            self._alloc._release(self._info, quiet=True)
        except Exception:                            # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# 分配器
# ---------------------------------------------------------------------------
_LIVE_ALLOCATORS: "weakref.WeakSet[DeviceAllocator]" = weakref.WeakSet()
_ATEXIT_REGISTERED = False


def _atexit_check() -> None:                         # FR-4.5 泄漏自证
    for alloc in list(_LIVE_ALLOCATORS):
        if alloc.live_blocks:
            try:
                print(f"[dev_memory] 进程退出时仍有未释放显存:\n{alloc.leak_report()}")
            except Exception:                        # pragma: no cover
                pass


class DeviceAllocator:
    """设备显存分配器（阶段1：直通分配，无池）

    每个实例对应一张卡，并持有一张「块表」作为句柄的权威登记处。
    """

    def __init__(self, device: int = 0, *, lib_dir: Optional[Path | str] = None,
                 force_host: bool = False, name: str = "allocator",
                 pool: bool = True, pool_max_block: Optional[int] = None,
                 pool_max_bytes: Optional[int] = None,
                 chunk_bytes: int = 16 << 20):
        """
        Args:
            pool: 是否启用内存池；False 时退化为阶段1 的直通分配（每次 cudaMalloc）
            pool_max_block: 超过此容量的块不进池（None = 不限；传 1<<20 即设计文档原行为）
            pool_max_bytes: 池中空闲内存总量上限，超出按 FIFO 驱逐最旧的（None = 不限）
            chunk_bytes: 单次传输最大字节数，超过则自动分片（FR-3.3）
        """
        global _ATEXIT_REGISTERED

        self.name = name
        self._lock = threading.RLock()
        self._blocks: dict[int, _BlockInfo] = {}
        self._closed = False
        self._counter = 0
        self._peak_live_bytes = 0
        self._live_bytes = 0

        # ---- 内存池（阶段2）----
        self._pool = bool(pool)
        self._pool_max_block = None if pool_max_block is None else int(pool_max_block)
        self._pool_max_bytes = None if pool_max_bytes is None else int(pool_max_bytes)
        self._free: dict[BucketKey, list[_BlockInfo]] = {}
        self._free_bytes = 0
        self._free_seq = 0

        # ---- pinned staging 池（阶段3）----
        self._chunk_bytes = int(chunk_bytes)
        # nbytes -> [(ptr, nbytes, backing)]；存原始元组而不是 PinnedBuffer 对象，
        # 这样每次取出都是全新的句柄，旧句柄永远不会「复活」（与 DeviceBuffer 同理）
        self._pinned_free: dict[int, list[tuple[int, int, Any]]] = {}
        self._pinned_inflight: list[tuple[PinnedBuffer, int, int]] = []   # (buf, stream, event)
        self._pinned_bytes = 0

        # 本地统计（统一主机/设备两种模式；C 侧计数另有一份用于交叉验证）
        self._stat_allocs = 0
        self._stat_frees = 0
        self._stat_up_bytes = 0
        self._stat_down_bytes = 0
        self._stat_pool_hits = 0
        self._stat_pool_misses = 0
        self._stat_recycled = 0
        self._stat_evicted = 0
        self._stat_trimmed = 0
        self._stat_pinned_allocs = 0
        self._stat_pinned_hits = 0
        self._stat_async_up = 0
        self._stat_async_down = 0

        self._lib: Optional[_Lib] = None
        self._host_mode = True
        self._device = -1

        if not force_host:
            self._lib = load_library(lib_dir)
            if self._lib is not None:
                code = int(self._lib.dll.ene_mem_init(int(device)))
                if code == ENE_OK:
                    self._host_mode = False
                    self._device = int(self._lib.dll.ene_mem_device())
                elif code == ENE_ERR_NODEVICE:
                    # FR-10.1：无 CUDA 环境时回退到主机内存，API 不崩
                    self._host_mode = True
                    self._device = -1
                else:
                    err_code, msg = self._lib.last_error()
                    _raise_for(err_code, f"ene_mem_init(device={device}) 失败: {msg}")

        if not _ATEXIT_REGISTERED:
            atexit.register(_atexit_check)
            _ATEXIT_REGISTERED = True
        _LIVE_ALLOCATORS.add(self)

    # ---- 能力查询 ----
    @property
    def available(self) -> bool:
        """True 表示在真设备显存上分配；False 表示主机回退模式（FR-10.1）"""
        return not self._host_mode

    @property
    def device(self) -> int:
        return self._device

    @property
    def mode(self) -> str:
        return "host-fallback" if self._host_mode else f"cuda:{self._device}"

    # ---- 分配 ----
    def _bucket_for(self, nbytes: int) -> tuple[Optional[BucketKey], int]:
        """决定这次分配走哪个桶（None = 不进池）"""
        if not self._pool:
            return None, nbytes
        key, capacity = _size_class(nbytes)
        if self._pool_max_block is not None and capacity > self._pool_max_block:
            return None, nbytes
        return key, capacity

    def alloc_bytes(self, nbytes: int, *, dtype: Any = np.uint8,
                    name: str = "buffer") -> DeviceBuffer:
        self._ensure_open()
        if not isinstance(nbytes, (int, np.integer)) or nbytes <= 0:
            raise InvalidArgumentError(f"nbytes 必须是正整数，得到 {nbytes!r}")
        nbytes = int(nbytes)
        dt = np.dtype(dtype)
        key, capacity = self._bucket_for(nbytes)

        # ---- 快路径：池命中（FR-2.1 / NFR-3，完全不碰 CUDA）----
        if key is not None:
            with self._lock:
                bucket = self._free.get(key)
                if bucket:
                    old = bucket.pop()
                    if not bucket:
                        del self._free[key]
                    self._free_bytes -= old.capacity
                    self._stat_pool_hits += 1
                    return self._register(self._reuse_info(old, nbytes, dt, name, key))

        # ---- 慢路径：真正向驱动申请（cudaMalloc 约 0.25 ms/次）----
        self._stat_pool_misses += 1
        self._counter += 1
        info = _BlockInfo(ptr=0, nbytes=nbytes, capacity=capacity, dtype=dt,
                          device=-1, name=name, tag=self._counter, cls=key)

        if self._host_mode:
            buf = bytearray(capacity)
            info.host = buf
            info.ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            info.device = -1
        else:
            assert self._lib is not None
            ptr = self._lib.dll.ene_mem_alloc(ctypes.c_size_t(capacity))
            if not ptr:
                code, msg = self._lib.last_error()
                _raise_for(code, f"分配 {capacity} 字节失败: {msg}")
            if ptr % 256 != 0:                       # FR-1.3 自检（cudaMalloc 本身保证 ≥256B）
                self._lib.dll.ene_mem_free(ctypes.c_void_p(ptr))
                raise DevMemError(f"分配返回的指针未按 256B 对齐: 0x{ptr:x}")
            info.ptr = int(ptr)
            info.device = self._device

        return self._register(info)

    def _reuse_info(self, old: _BlockInfo, nbytes: int, dt: np.dtype,
                    name: str, key: BucketKey) -> _BlockInfo:
        """复用池中块：**新建** _BlockInfo，让旧句柄永远保持 released=True

        若就地把 old 的字段改回去，旧的 DeviceBuffer 会重新变成「可用」并指向新块 ——
        这是最典型的一类悬垂别名 bug。新建对象则旧句柄仍然会正确报错。
        """
        self._counter += 1
        return _BlockInfo(ptr=old.ptr, nbytes=nbytes, capacity=old.capacity, dtype=dt,
                          device=old.device, name=name, host=old.host,
                          tag=self._counter, cls=key)

    def _register(self, info: _BlockInfo) -> DeviceBuffer:
        with self._lock:
            self._blocks[info.ptr] = info
            self._stat_allocs += 1
            self._live_bytes += info.nbytes
            if self._live_bytes > self._peak_live_bytes:
                self._peak_live_bytes = self._live_bytes
        return DeviceBuffer(self, info)

    def alloc(self, size: int, *, dtype: Any = np.float32, name: str = "buffer") -> DeviceBuffer:
        """按元素个数分配。默认 float32 —— 与 13 个算子的内存解释一致。"""
        self._ensure_open()
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise InvalidArgumentError(f"size 必须是正整数，得到 {size!r}")
        dt = np.dtype(dtype)
        return self.alloc_bytes(int(size) * dt.itemsize, dtype=dt, name=name)

    def alloc_like(self, arr: np.ndarray, *, name: str = "buffer") -> DeviceBuffer:
        """按 numpy 数组的形状/类型分配（不做任何拷贝）"""
        if not isinstance(arr, np.ndarray):
            raise InvalidArgumentError(f"需要 numpy 数组，得到 {type(arr).__name__}")
        return self.alloc(arr.size, dtype=arr.dtype, name=name)

    # ---- 传输 ----
    def _upload(self, info: _BlockInfo, src: np.ndarray, stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "upload")
        arr = _check_host_array(src, info, "upload")
        if self._host_mode:
            ctypes.memmove(info.ptr, arr.ctypes.data, arr.nbytes)
        else:
            assert self._lib is not None
            code = int(self._lib.dll.ene_mem_upload(
                ctypes.c_void_p(info.ptr), ctypes.c_void_p(arr.ctypes.data),
                ctypes.c_size_t(arr.nbytes), ctypes.c_void_p(stream or 0)))
            self._lib._check(code, f"upload {arr.nbytes} 字节")
        self._stat_up_bytes += arr.nbytes

    def _download(self, info: _BlockInfo, dst: np.ndarray, stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "download")
        arr = _check_host_array(dst, info, "download")
        if self._host_mode:
            ctypes.memmove(arr.ctypes.data, info.ptr, arr.nbytes)
        else:
            assert self._lib is not None
            code = int(self._lib.dll.ene_mem_download(
                ctypes.c_void_p(arr.ctypes.data), ctypes.c_void_p(info.ptr),
                ctypes.c_size_t(arr.nbytes), ctypes.c_void_p(stream or 0)))
            self._lib._check(code, f"download {arr.nbytes} 字节")
        self._stat_down_bytes += arr.nbytes

    def sync(self, stream: Optional[int] = None) -> None:
        """等待该流上所有工作完成，并回收已完成的 pinned staging（阶段3）

        阶段1/2 的 `upload`/`download` 自带同步，不需要调用；阶段3 的
        `upload_async`/`download_async` 之后**必须** sync 或依赖同流依赖。
        """
        if self._host_mode:
            return
        assert self._lib is not None
        code = int(self._lib.dll.ene_mem_sync(ctypes.c_void_p(stream or 0)))
        self._lib._check(code, "sync")
        self._sweep_pinned(force_stream=stream_ptr(stream))

    # ---- 流与事件（阶段3）----
    def new_stream(self) -> int:
        """新建一条**非阻塞**流（返回裸指针）。用完请 destroy_stream。"""
        self._ensure_open()
        if self._host_mode:
            return 0
        assert self._lib is not None
        ptr = self._lib.dll.ene_mem_stream_create()
        if not ptr:
            code, msg = self._lib.last_error()
            _raise_for(code, f"创建流失败: {msg}")
        return int(ptr)

    def destroy_stream(self, stream: int) -> None:
        if self._host_mode or not stream:
            return
        assert self._lib is not None
        self._lib._check(int(self._lib.dll.ene_mem_stream_destroy(ctypes.c_void_p(stream))),
                         "destroy_stream")

    def create_event(self, *, timed: bool = False) -> int:
        """新建事件（默认关闭计时，更快）。用完请 destroy_event。"""
        if self._lib is None:
            raise DevMemError("主机回退模式下没有 CUDA 事件")
        fn = self._lib.dll.ene_mem_event_create_timed if timed else self._lib.dll.ene_mem_event_create
        ev = fn()
        if not ev:
            code, msg = self._lib.last_error()
            _raise_for(code, f"创建事件失败: {msg}")
        return int(ev)

    def record_event_on(self, event: int, stream: Optional[int] = None) -> None:
        assert self._lib is not None
        self._lib._check(int(self._lib.dll.ene_mem_event_record(
            ctypes.c_void_p(event), ctypes.c_void_p(stream or 0))), "record_event")

    def wait_event(self, event: int, stream: Optional[int] = None) -> None:
        """让 stream 上**后续**的工作都等该事件触发"""
        assert self._lib is not None
        self._lib._check(int(self._lib.dll.ene_mem_event_wait(
            ctypes.c_void_p(event), ctypes.c_void_p(stream or 0))), "wait_event")

    def event_sync(self, event: int) -> None:
        """主机阻塞等该事件"""
        assert self._lib is not None
        self._lib._check(int(self._lib.dll.ene_mem_event_sync(ctypes.c_void_p(event))),
                         "event_sync")

    def event_done(self, event: int) -> bool:
        """非阻塞查询事件是否已完成"""
        assert self._lib is not None
        return bool(self._lib.dll.ene_mem_event_done(ctypes.c_void_p(event)))

    def destroy_event(self, event: int) -> None:
        if self._lib is None or not event:
            return
        self._lib.dll.ene_mem_event_destroy(ctypes.c_void_p(event))

    def event_elapsed(self, start: int, end: int) -> float:
        """两个**计时事件**之间的 GPU 侧毫秒数（需用 create_event(timed=True) 创建）"""
        assert self._lib is not None
        return float(self._lib.dll.ene_mem_event_elapsed(
            ctypes.c_void_p(start), ctypes.c_void_p(end)))

    # ---- pinned 主机内存（阶段3）----
    def pinned(self, size: int, *, dtype: Any = np.float32,
               name: str = "pinned") -> PinnedBuffer:
        """取一块锁页主机内存（按精确字节数池化复用，FR-3.2）

        `buf.array` 是它的 numpy 视图，可直接填数/读数，零拷贝。
        """
        self._ensure_open()
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise InvalidArgumentError(f"size 必须是正整数，得到 {size!r}")
        return self._pinned_acquire(int(size) * np.dtype(dtype).itemsize,
                                    np.dtype(dtype), name)

    def _pinned_acquire(self, nbytes: int, dtype: Any = np.uint8,
                        name: str = "pinned") -> PinnedBuffer:
        def take() -> Optional[PinnedBuffer]:
            bucket = self._pinned_free.get(nbytes)
            if not bucket:
                return None
            self._stat_pinned_hits += 1
            ptr, nb, backing = bucket.pop()
            if not bucket:
                del self._pinned_free[nbytes]
            return PinnedBuffer(self, ptr, nb, dtype, name, backing=backing)

        hit = take()
        if hit is not None:
            return hit

        # 池里没有 → 先扫一遍「在飞」的，可能已有完成的了（这样 pinned 用量自然有界）
        self._sweep_pinned()
        hit = take()
        if hit is not None:
            return hit

        if self._host_mode:
            raw = (ctypes.c_char * nbytes)()             # 主机模式：普通内存即可
            return PinnedBuffer(self, ctypes.addressof(raw), nbytes, dtype, name, backing=raw)

        assert self._lib is not None
        ptr = self._lib.dll.ene_mem_host_alloc(ctypes.c_size_t(nbytes))
        if not ptr:
            code, msg = self._lib.last_error()
            _raise_for(code, f"分配 {nbytes} 字节锁页内存失败: {msg}")
        self._pinned_bytes += nbytes
        self._stat_pinned_allocs += 1
        return PinnedBuffer(self, int(ptr), nbytes, dtype, name)

    def _pinned_release(self, buf: PinnedBuffer, *, quiet: bool) -> None:
        if buf.released:
            if quiet:
                return
            raise BufferReleasedError(f"pinned 块 {buf.name!r} 已归还 —— 重复归还")
        buf.released = True
        entry = (buf._ptr, buf.nbytes, buf._backing)
        buf._backing = None
        with self._lock:
            if self._closed:
                drop = True
            else:
                self._pinned_free.setdefault(buf.nbytes, []).append(entry)
                drop = False
        if drop:
            self._free_pinned_native(*entry)

    def _free_pinned_native(self, ptr: int, nbytes: int, backing: Any) -> None:
        self._pinned_bytes -= nbytes
        if backing is not None:
            return                                       # 主机回退：交回 GC
        if self._lib is not None:
            self._lib.dll.ene_mem_host_free(ctypes.c_void_p(ptr))

    def _sweep_pinned(self, force_stream: Optional[int] = None) -> int:
        """回收已完成的在飞 pinned 缓冲；force_stream 指定时无条件回收该流的"""
        if not self._pinned_inflight or self._lib is None:
            return 0
        remaining = []
        freed = 0
        for buf, sid, ev in self._pinned_inflight:
            if force_stream is not None:
                done = (sid == force_stream)
            else:
                done = self.event_done(ev)
            if done:
                self.destroy_event(ev)
                self._pinned_release(buf, quiet=True)
                freed += 1
            else:
                remaining.append((buf, sid, ev))
        self._pinned_inflight = remaining
        return freed

    # ---- 异步传输（阶段3）----
    def _issue_copy(self, dst_ptr: int, src_ptr: int, nbytes: int,
                    stream: Optional[int], *, up: bool) -> None:
        """分片下发拷贝（FR-3.3）：单次不超过 chunk_bytes"""
        assert self._lib is not None
        s = ctypes.c_void_p(stream or 0)
        fn = self._lib.dll.ene_mem_upload_async if up else self._lib.dll.ene_mem_download_async
        what = "upload_async" if up else "download_async"
        off = 0
        while off < nbytes:
            n = min(self._chunk_bytes, nbytes - off)
            code = int(fn(ctypes.c_void_p(dst_ptr + off), ctypes.c_void_p(src_ptr + off),
                          ctypes.c_size_t(n), s))
            self._lib._check(code, f"{what} {n} 字节")
            off += n

    def _upload_async(self, info: _BlockInfo, src: np.ndarray,
                      stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "upload_async")
        arr = _check_host_array(src, info, "upload_async")
        nbytes = arr.nbytes

        if self._host_mode:
            ctypes.memmove(info.ptr, arr.ctypes.data, nbytes)
            self._stat_up_bytes += nbytes
            return

        assert self._lib is not None
        host_ptr = int(arr.ctypes.data)
        sid = stream_ptr(stream)
        if self._lib.dll.ene_mem_is_pinned(ctypes.c_void_p(host_ptr)):
            self._issue_copy(info.ptr, host_ptr, nbytes, stream, up=True)
        else:
            # pageable 源 -> 先落到 pinned staging（主机内 memcpy，约 10 GB/s），再异步 H2D
            pin = self._pinned_acquire(nbytes, np.uint8, "staging")
            ctypes.memmove(pin._ptr, host_ptr, nbytes)
            self._issue_copy(info.ptr, pin._ptr, nbytes, stream, up=True)
            ev = self.create_event()
            self.record_event_on(ev, stream)             # 记在 copy 流上
            self._pinned_inflight.append((pin, sid, ev))
        self._stat_up_bytes += nbytes
        self._stat_async_up += 1

    def _download_async(self, info: _BlockInfo, dst: np.ndarray,
                        stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "download_async")
        arr = _check_host_array(dst, info, "download_async")

        if self._host_mode:
            ctypes.memmove(arr.ctypes.data, info.ptr, arr.nbytes)
            self._stat_down_bytes += arr.nbytes
            return

        assert self._lib is not None
        dst_ptr = int(arr.ctypes.data)
        if not self._lib.dll.ene_mem_is_pinned(ctypes.c_void_p(dst_ptr)):
            raise InvalidArgumentError(
                "download_async 需要锁页（pinned）目标：请用 allocator.pinned(n) 拿 `.array`，"
                "或改用同步的 download()")
        self._issue_copy(dst_ptr, info.ptr, arr.nbytes, stream, up=False)
        self._stat_down_bytes += arr.nbytes
        self._stat_async_down += 1

    # ---- 双缓冲流水线（阶段3）----
    def double_buffer(self, size: int, *, dtype: Any = np.float32,
                      slots: int = 2, name: str = "pipe") -> "DoubleBuffer":
        """创建双缓冲流水线（FR-3.4）"""
        self._ensure_open()
        if self._host_mode:
            raise DevMemError("主机回退模式不支持双缓冲流水线（需要 CUDA 流与事件）")
        return DoubleBuffer(self, size, dtype=dtype, slots=slots, name=name)

    # ---- 生命周期 ----
    def _release(self, info: _BlockInfo, *, quiet: bool) -> None:
        recycle = False
        victims: list[_BlockInfo] = []
        with self._lock:
            if info.released:
                if quiet:
                    return
                raise BufferReleasedError(
                    f"块 {info.name!r}(ptr=0x{info.ptr:x}) 已被释放 —— 双重释放")
            info.released = True
            self._blocks.pop(info.ptr, None)
            self._stat_frees += 1
            self._live_bytes -= info.nbytes

            # FR-2.1：可池化的块留在池里等复用，不归还驱动
            if info.cls is not None and not self._closed:
                self._free_seq += 1
                info.free_seq = self._free_seq
                info.free_time = time.monotonic()
                self._free.setdefault(info.cls, []).append(info)
                self._free_bytes += info.capacity
                self._stat_recycled += 1
                recycle = True
                victims = self._evict_locked()        # FR-2.6：超上限则驱逐最旧的
        if not recycle:
            self._free_native(info, quiet=quiet)
        for victim in victims:
            self._free_native(victim, quiet=True)

    def _free_native(self, info: _BlockInfo, *, quiet: bool) -> None:
        """真正归还给驱动/GC。**不得在持锁时调用**（FR-11.2：cudaFree 会阻塞）"""
        if info.host is not None:
            info.host = None                          # 交回 GC
            return
        if self._lib is None:
            return
        code = int(self._lib.dll.ene_mem_free(ctypes.c_void_p(info.ptr)))
        if code != ENE_OK and not quiet:
            err_code, msg = self._lib.last_error()
            _raise_for(err_code, f"释放 0x{info.ptr:x} 失败: {msg}")

    def _evict_locked(self) -> list[_BlockInfo]:
        """池中空闲字节超上限时按 FIFO 挑出要归还的块（锁内只挑不 free）"""
        if self._pool_max_bytes is None or self._free_bytes <= self._pool_max_bytes:
            return []
        idle = sorted((b for bucket in self._free.values() for b in bucket),
                      key=lambda b: b.free_seq)
        victims: list[_BlockInfo] = []
        for block in idle:
            if self._free_bytes <= self._pool_max_bytes:
                break
            self._free[block.cls].remove(block)       # eq=False → 按身份比较
            self._free_bytes -= block.capacity
            self._stat_evicted += 1
            victims.append(block)
        for key in [k for k, v in self._free.items() if not v]:
            del self._free[key]
        return victims

    def trim(self, *, older_than: Optional[float] = None) -> dict[str, int]:
        """把池中的空闲块真正归还驱动（FR-2.5）

        `older_than=None` 全部归还；`older_than=30` 只归还已空闲 30 秒以上的块。
        """
        now = time.monotonic()
        victims: list[_BlockInfo] = []
        with self._lock:
            for key in list(self._free):
                keep = []
                for block in self._free[key]:
                    if older_than is None or (now - block.free_time) >= older_than:
                        self._free_bytes -= block.capacity
                        victims.append(block)
                    else:
                        keep.append(block)
                if keep:
                    self._free[key] = keep
                else:
                    del self._free[key]
            self._stat_trimmed += len(victims)
        for block in victims:
            self._free_native(block, quiet=True)
        return {"blocks": len(victims),
                "bytes": sum(b.capacity for b in victims)}

    @property
    def live_blocks(self) -> list[_BlockInfo]:
        with self._lock:
            return list(self._blocks.values())

    @property
    def free_blocks(self) -> list[_BlockInfo]:
        """池中空闲块（仅供检查/统计）"""
        with self._lock:
            return [b for bucket in self._free.values() for b in bucket]

    def close(self) -> None:
        """释放本分配器名下所有设备显存与锁页内存，幂等

        会先把已知的流同步完，避免释放正在被 kernel 使用的显存。
        """
        if not self._host_mode and self._lib is not None:
            for sid in {sid for _, sid, _ in self._pinned_inflight}:
                self._lib.dll.ene_mem_sync(ctypes.c_void_p(sid))
            self._lib.dll.ene_mem_sync(ctypes.c_void_p(0))

        with self._lock:
            if self._closed:
                return
            self._closed = True                       # 先置位：_release 便不再回池
            pending = list(self._blocks.values())
            pooled = [b for bucket in self._free.values() for b in bucket]
            pinned = [t for bucket in self._pinned_free.values() for t in bucket]
            inflight = [(b._ptr, b.nbytes, b._backing) for b, _, _ in self._pinned_inflight]
            events = [ev for _, _, ev in self._pinned_inflight]
            self._free.clear()
            self._free_bytes = 0
            self._pinned_free.clear()
            self._pinned_inflight = []
        for info in pending:
            self._release(info, quiet=True)           # 走正常回收路径（已关闭 → 直接 free）
        for info in pooled:
            self._free_native(info, quiet=True)
        for ev in events:
            self.destroy_event(ev)
        for entry in pinned + inflight:
            self._free_pinned_native(*entry)
        _LIVE_ALLOCATORS.discard(self)

    def leak_report(self) -> str:
        live = self.live_blocks
        with self._lock:
            free_blocks = sum(len(b) for b in self._free.values())
            free_bytes = self._free_bytes
        if live:
            total = sum(b.nbytes for b in live)
            lines = [f"[{self.name} / {self.mode}] 未释放块 {len(live)} 个，共 {total} 字节:"]
            lines += [f"  - {b.describe()}" for b in live]
            text = "\n".join(lines)
        else:
            text = f"[{self.name}] 无未释放块"
        if free_blocks:
            text += (f"\n  提示: 池中另有 {free_blocks} 个空闲块（共 {free_bytes} 字节），"
                     f"可 trim() 归还驱动")
        return text

    # ---- 统计 ----
    def stats(self) -> dict[str, Any]:
        with self._lock:
            free_blocks = sum(len(b) for b in self._free.values())
            free_bytes = self._free_bytes
            idle = [b for bucket in self._free.values() for b in bucket]
            largest_free = max((b.capacity for b in idle), default=0)
            classes = {f"{k[0]}/{k[1]}": {"blocks": len(v), "bytes": sum(b.capacity for b in v)}
                       for k, v in sorted(self._free.items()) if v}
            hits, misses = self._stat_pool_hits, self._stat_pool_misses
            evicted, trimmed = self._stat_evicted, self._stat_trimmed
            pinned_free = sum(len(v) for v in self._pinned_free.values())
            pinned_inflight = len(self._pinned_inflight)

        requests = hits + misses
        out: dict[str, Any] = {
            "mode": self.mode,
            "device": self._device,
            "pool_enabled": self._pool,
            "alloc_calls": self._stat_allocs,
            "free_calls": self._stat_frees,
            "live_blocks": len(self._blocks),
            "live_bytes": self._live_bytes,
            "peak_live_bytes": self._peak_live_bytes,
            "upload_bytes": self._stat_up_bytes,
            "download_bytes": self._stat_down_bytes,
            # ---- 池（FR-2.4）----
            "pool_hits": hits,
            "pool_misses": misses,
            "hit_rate": (hits / requests) if requests else 0.0,
            "recycled_blocks": self._stat_recycled,
            "free_blocks": free_blocks,
            "free_bytes": free_bytes,
            "largest_free_block": largest_free,
            # 碎片率 = 1 - 最大可服务块 / 空闲总字节（见设计文档 5.1）
            "fragmentation": (1.0 - largest_free / free_bytes) if free_bytes else 0.0,
            "evicted_blocks": evicted,
            "trimmed_blocks": trimmed,
            "free_classes": classes,
            # ---- pinned staging / 异步（阶段3）----
            "pinned_allocs": self._stat_pinned_allocs,
            "pinned_hits": self._stat_pinned_hits,
            "pinned_free_blocks": pinned_free,
            "pinned_inflight": pinned_inflight,
            "pinned_bytes": self._pinned_bytes,
            "async_upload_calls": self._stat_async_up,
            "async_download_calls": self._stat_async_down,
        }
        if self._lib is not None:
            # C 侧计数：用于交叉验证「热路径真的没再 cudaMalloc」（NFR-3）
            out["native_alloc_calls"] = int(self._lib.dll.ene_mem_alloc_calls())
            out["native_free_calls"] = int(self._lib.dll.ene_mem_free_calls())
            out["native_upload_bytes"] = int(self._lib.dll.ene_mem_upload_bytes())
            out["native_download_bytes"] = int(self._lib.dll.ene_mem_download_bytes())
            out["native_host_alloc_calls"] = int(self._lib.dll.ene_mem_host_alloc_calls())
        return out

    # ---- 内部校验 ----
    def _ensure_open(self) -> None:
        if self._closed:
            raise AllocatorClosedError("分配器已关闭，不能再使用")

    def _ensure_live(self, info: _BlockInfo, what: str) -> None:
        if info.released:
            raise BufferReleasedError(
                f"{what}: 块 {info.name!r} 已被释放，不能再使用")

    # ---- 上下文管理 ----
    def __enter__(self) -> "DeviceAllocator":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:                       # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 双缓冲流水线（阶段3）
# ---------------------------------------------------------------------------
class _PipelineSlot:
    """双缓冲流水线的一个槽"""

    __slots__ = ("_pipe", "index", "buffer", "copy_stream", "compute_stream",
                 "_ev_copy", "_ev_compute", "_used")

    def __init__(self, pipe: "DoubleBuffer", index: int, buffer: DeviceBuffer,
                 ev_copy: int, ev_compute: int):
        self._pipe = pipe
        self.index = index
        self.buffer = buffer
        self.copy_stream = pipe.copy_stream
        self.compute_stream = pipe.compute_stream
        self._ev_copy = ev_copy
        self._ev_compute = ev_compute
        self._used = False

    def upload_async(self, src: np.ndarray) -> "_PipelineSlot":
        """异步把 src 传进本槽，并保证本槽 compute_stream 上后续工作等它完成"""
        mem = self._pipe._alloc
        if self._used:
            # 复用本槽前，先等上一轮在本槽上的计算跑完，否则会覆写正在算的数据
            mem.wait_event(self._ev_compute, self.copy_stream)
        self.buffer.upload_async(src, stream=self.copy_stream)
        mem.record_event_on(self._ev_copy, self.copy_stream)
        mem.wait_event(self._ev_copy, self.compute_stream)
        self._used = True
        return self

    def done(self) -> None:
        """声明本槽本轮的计算已全部提交（记录事件，供下一轮复用前等待）"""
        self._pipe._alloc.record_event_on(self._ev_compute, self.compute_stream)

    def __repr__(self) -> str:
        return f"<Slot #{self.index} {self.buffer.name!r}>"


class DoubleBuffer:
    """双缓冲流水线：让「传第 k+1 块」与「算第 k 块」重叠（FR-3.4）

    两条**非阻塞**流 + 每槽一对事件：

        copy_stream    —— H2D 按序排队
        compute_stream —— kernel 按序排队

    同步规则（缺一不可）：

    * 在某槽上计算**之前**：让 `compute_stream` 等该槽本轮的 copy 事件；
    * 复用某槽**之前**：让 `copy_stream` 等该槽上一轮的 compute 事件。

    用法：

        with mem.double_buffer(count, dtype=np.float32, slots=2, name="pipe") as pipe:
            for chunk in chunks:
                slot = pipe.next()
                slot.upload_async(chunk)                    # 异步 H2D 到 slot.buffer
                ops.dev("add", slot.buffer, db, dout, stream=slot.compute_stream)
                slot.done()                                 # 提交完成事件
            pipe.sync()                                     # 等两条流跑完
    """

    def __init__(self, alloc: DeviceAllocator, size: int, *,
                 dtype: Any = np.float32, slots: int = 2, name: str = "pipe"):
        if slots < 2:
            raise InvalidArgumentError(f"双缓冲至少需要 2 个槽，得到 {slots}")
        self._alloc = alloc
        self.copy_stream = alloc.new_stream()
        self.compute_stream = alloc.new_stream()
        self._slots = [
            _PipelineSlot(self, i,
                          alloc.alloc(size, dtype=dtype, name=f"{name}[{i}]"),
                          alloc.create_event(), alloc.create_event())
            for i in range(slots)
        ]
        self._cursor = 0
        self._closed = False

    def next(self) -> _PipelineSlot:
        """轮转取下ー个槽（调用方需保证不会与仍在使用的槽冲突，槽数 ≥ 2 时自然成立）"""
        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._slots)
        return slot

    def sync(self) -> None:
        """等两条流都完成，并回收 staging"""
        for sid in (self.copy_stream, self.compute_stream):
            self._alloc.sync(sid)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sync()
        mem = self._alloc
        for slot in self._slots:
            mem.destroy_event(slot._ev_copy)
            mem.destroy_event(slot._ev_compute)
            slot.buffer.release()
        mem.destroy_stream(self.copy_stream)
        mem.destroy_stream(self.compute_stream)

    def __enter__(self) -> "DoubleBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _check_host_array(arr: Any, info: _BlockInfo, what: str) -> np.ndarray:
    """上传/下载前的三项校验 —— 和 dev 算子一样，宁可直接报错也不做隐式拷贝/错算"""
    if not isinstance(arr, np.ndarray):
        raise InvalidArgumentError(f"{what}: 需要 numpy 数组，得到 {type(arr).__name__}")
    if arr.dtype != info.dtype:
        raise InvalidArgumentError(
            f"{what}: dtype 不匹配（块为 {info.dtype}，数组为 {arr.dtype}）")
    if not arr.flags.c_contiguous:
        raise InvalidArgumentError(
            f"{what}: 需要 C 连续数组，请先 np.ascontiguousarray（那是显式拷贝）")
    if arr.nbytes != info.nbytes:
        raise InvalidArgumentError(
            f"{what}: 字节数不匹配（块为 {info.nbytes}，数组为 {arr.nbytes}）")
    return arr
