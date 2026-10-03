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
                     "ene_mem_upload_bytes", "ene_mem_download_bytes"):
            fn = getattr(self.dll, name)
            fn.argtypes = []
            fn.restype = ctypes.c_ulonglong

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
                 pool_max_bytes: Optional[int] = None):
        """
        Args:
            pool: 是否启用内存池；False 时退化为阶段1 的直通分配（每次 cudaMalloc）
            pool_max_block: 超过此容量的块不进池（None = 不限；传 1<<20 即设计文档原行为）
            pool_max_bytes: 池中空闲内存总量上限，超出按 FIFO 驱逐最旧的（None = 不限）
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
        """等待流上所有工作完成（阶段1 的传输本身已同步，这里供 kernel 之后使用）"""
        if self._host_mode:
            return
        assert self._lib is not None
        code = int(self._lib.dll.ene_mem_sync(ctypes.c_void_p(stream or 0)))
        self._lib._check(code, "sync")

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
        """释放本分配器名下所有显存（活跃块 + 池中空闲块），幂等"""
        with self._lock:
            if self._closed:
                return
            self._closed = True                       # 先置位：_release 便不再回池
            pending = list(self._blocks.values())
            pooled = [b for bucket in self._free.values() for b in bucket]
            self._free.clear()
            self._free_bytes = 0
        for info in pending:
            self._release(info, quiet=True)           # 走正常回收路径（已关闭 → 直接 free）
        for info in pooled:
            self._free_native(info, quiet=True)
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
        }
        if self._lib is not None:
            # C 侧计数：用于交叉验证「热路径真的没再 cudaMalloc」（NFR-3）
            out["native_alloc_calls"] = int(self._lib.dll.ene_mem_alloc_calls())
            out["native_free_calls"] = int(self._lib.dll.ene_mem_free_calls())
            out["native_upload_bytes"] = int(self._lib.dll.ene_mem_upload_bytes())
            out["native_download_bytes"] = int(self._lib.dll.ene_mem_download_bytes())
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
