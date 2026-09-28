# -*- coding: utf-8 -*-
"""方案A 阶段1：自管理设备显存（直通分配，无池）

职责划分（见 `doc/CUDAC/显存管理.md` 第四节 D1/D2）：

    C 层（dev_memory.dll）   无状态薄封装：cudaMalloc/cudaFree/cudaMemcpy + 错误码 + 计数
    本模块                   句柄化 + 块表 + 双重释放检测 + 泄漏清单 + 上下文管理

**阶段1 范围**：直通分配（无池）、同步传输、句柄与生命周期、错误上报、统计、无 CUDA 回退。
**不含**（后续阶段）：内存池（阶段2）、pinned/异步/双缓冲（阶段3）、
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
@dataclass
class _BlockInfo:
    """一块显存的元数据（主机侧，FR-1.4）；设备端不加块头，保证指针对齐"""

    ptr: int
    nbytes: int
    dtype: np.dtype
    device: int                                   # -1 表示主机回退模式
    name: str
    released: bool = False
    host: Optional[bytearray] = None              # 主机回退模式的后备内存
    tag: int = 0                                  # 便于在泄漏清单里区分

    @property
    def size(self) -> int:
        """按 dtype 计的元素个数（13 个算子都按 float32 解释）"""
        return self.nbytes // self.dtype.itemsize

    def describe(self) -> str:
        where = "host" if self.device < 0 else f"cuda:{self.device}"
        return (f"#{self.tag:<4} {self.name:<12} ptr=0x{self.ptr:012x} {self.nbytes:>10} B "
                f"({self.size} × {self.dtype}) {where}")


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
        return self._info.nbytes

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
        return f"<DeviceBuffer {self._info.name!r} {self.nbytes}B {state}>"

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
                 force_host: bool = False, name: str = "allocator"):
        global _ATEXIT_REGISTERED

        self.name = name
        self._lock = threading.RLock()
        self._blocks: dict[int, _BlockInfo] = {}
        self._closed = False
        self._counter = 0
        self._peak_live_bytes = 0

        # 本地统计（统一主机/设备两种模式；C 侧计数另有一份用于交叉验证）
        self._stat_allocs = 0
        self._stat_frees = 0
        self._stat_up_bytes = 0
        self._stat_down_bytes = 0

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
    def alloc_bytes(self, nbytes: int, *, dtype: Any = np.uint8,
                    name: str = "buffer") -> DeviceBuffer:
        self._ensure_open()
        if not isinstance(nbytes, (int, np.integer)) or nbytes <= 0:
            raise InvalidArgumentError(f"nbytes 必须是正整数，得到 {nbytes!r}")
        nbytes = int(nbytes)

        self._counter += 1
        info = _BlockInfo(ptr=0, nbytes=nbytes, dtype=np.dtype(dtype),
                          device=-1, name=f"{name}", tag=self._counter)

        if self._host_mode:
            buf = bytearray(nbytes)
            info.host = buf
            info.ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            info.device = -1
        else:
            assert self._lib is not None
            ptr = self._lib.dll.ene_mem_alloc(ctypes.c_size_t(nbytes))
            if not ptr:
                code, msg = self._lib.last_error()
                _raise_for(code, f"分配 {nbytes} 字节失败: {msg}")
            if ptr % 256 != 0:                       # FR-1.3 自检（cudaMalloc 本身保证 ≥256B）
                self._lib.dll.ene_mem_free(ctypes.c_void_p(ptr))
                raise DevMemError(f"分配返回的指针未按 256B 对齐: 0x{ptr:x}")
            info.ptr = int(ptr)
            info.device = self._device

        with self._lock:
            self._blocks[info.ptr] = info
            self._stat_allocs += 1
            live = sum(b.nbytes for b in self._blocks.values())
            self._peak_live_bytes = max(self._peak_live_bytes, live)
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
        with self._lock:
            if info.released:
                if quiet:
                    return
                raise BufferReleasedError(
                    f"块 {info.name!r}(ptr=0x{info.ptr:x}) 已被释放 —— 双重释放")
            info.released = True
            self._blocks.pop(info.ptr, None)
            self._stat_frees += 1

        if info.host is not None:
            info.host = None                          # 交回 GC
            return
        if self._lib is None:
            return
        code = int(self._lib.dll.ene_mem_free(ctypes.c_void_p(info.ptr)))
        if code != ENE_OK and not quiet:
            err_code, msg = self._lib.last_error()
            _raise_for(err_code, f"释放 0x{info.ptr:x} 失败: {msg}")

    @property
    def live_blocks(self) -> list[_BlockInfo]:
        with self._lock:
            return list(self._blocks.values())

    def close(self) -> None:
        """释放本分配器名下所有未归还的块（幂等）"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._blocks.values())
        for info in pending:
            self._release(info, quiet=True)
        _LIVE_ALLOCATORS.discard(self)

    def leak_report(self) -> str:
        live = self.live_blocks
        if not live:
            return f"[{self.name}] 无未释放块"
        total = sum(b.nbytes for b in live)
        lines = [f"[{self.name} / {self.mode}] 未释放块 {len(live)} 个，共 {total} 字节:"]
        lines += [f"  - {b.describe()}" for b in live]
        return "\n".join(lines)

    # ---- 统计 ----
    def stats(self) -> dict[str, Any]:
        live = self.live_blocks
        out: dict[str, Any] = {
            "mode": self.mode,
            "device": self._device,
            "alloc_calls": self._stat_allocs,
            "free_calls": self._stat_frees,
            "live_blocks": len(live),
            "live_bytes": sum(b.nbytes for b in live),
            "peak_live_bytes": self._peak_live_bytes,
            "upload_bytes": self._stat_up_bytes,
            "download_bytes": self._stat_down_bytes,
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
