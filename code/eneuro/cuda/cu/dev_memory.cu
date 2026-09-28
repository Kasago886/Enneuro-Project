// dev_memory.cu —— 方案A 阶段1：自管理设备显存（直通分配，无池）
//
// 设计取舍（详见 doc/CUDAC/显存管理.md 第四节 D1/D2）：
//   · C 层保持「无状态薄封装」：只做 cudaMalloc/cudaMemcpy + 错误码 + 计数，
//     **不维护块表**。句柄化、块表、双重释放检测、泄漏清单全部由 Python 层的
//     DeviceAllocator 承担（块表是 O(1) 的 dict，放 Python 侧更简单也更安全）。
//   · 元数据放主机侧，设备端不加块头 —— 保证 cudaMalloc 返回的 256B 对齐不被破坏。
//   · 阶段1 的传输是**同步语义**（内部 memcpyAsync + StreamSynchronize）；
//     阶段3 只需去掉那次 synchronize 就能变成真正的异步流水线。
//   · cudaMalloc 本身保证 ≥256B 对齐（Python 侧会自检兜底）。
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#define ENE_EXPORT extern "C" __declspec(dllexport)
#else
#define ENE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ---- 错误码 ----------------------------------------------------------------
enum {
    ENE_OK           = 0,
    ENE_ERR_OOM      = 1,   // 显存不足
    ENE_ERR_INVALID  = 2,   // 参数非法（空指针 / 长度为 0 / 设备号越界）
    ENE_ERR_DEVICE   = 3,   // 设备号非法
    ENE_ERR_CUDA     = 4,   // 其他 CUDA 错误
    ENE_ERR_NODEVICE = 5    // 无可用 CUDA 设备（上层据此回退主机内存）
};

// ---- 状态 ------------------------------------------------------------------
#if defined(_WIN32)
#define ENE_TLS __declspec(thread)
#else
#define ENE_TLS __thread
#endif

// 错误信息按线程保存，避免多线程互相覆盖（FR-11.3）
static ENE_TLS char g_last_msg[256];
static ENE_TLS int  g_last_code = ENE_OK;

static int                g_available = 0;
static int                g_device    = -1;
static unsigned long long g_alloc_calls = 0;
static unsigned long long g_free_calls  = 0;
static unsigned long long g_up_bytes    = 0;
static unsigned long long g_down_bytes  = 0;

static void ene_set_error(int code, const char *msg) {
    g_last_code = code;
#if defined(_MSC_VER)
    strncpy_s(g_last_msg, sizeof(g_last_msg), msg ? msg : "", _TRUNCATE);
#else
    strncpy(g_last_msg, msg ? msg : "", sizeof(g_last_msg) - 1);
    g_last_msg[sizeof(g_last_msg) - 1] = '\0';
#endif
}

static void ene_set_cuda_error(cudaError_t err) {
    if (err == cudaErrorMemoryAllocation) {
        ene_set_error(ENE_ERR_OOM, "out of device memory");
    } else {
        ene_set_error(ENE_ERR_CUDA, cudaGetErrorString(err));
    }
}

// ---- 初始化 / 能力查询 ------------------------------------------------------
ENE_EXPORT
int ene_mem_init(int device) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        g_available = 0;
        ene_set_error(ENE_ERR_NODEVICE,
                      err == cudaSuccess ? "no CUDA device found" : cudaGetErrorString(err));
        return ENE_ERR_NODEVICE;
    }
    if (device < 0) {
        device = 0;                                  // FR-8.2：默认用 0 号卡
    }
    if (device >= count) {
        char buf[128];
        snprintf(buf, sizeof(buf), "device index %d out of range (count=%d)", device, count);
        ene_set_error(ENE_ERR_DEVICE, buf);
        return ENE_ERR_DEVICE;
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        ene_set_cuda_error(err);
        return g_last_code;
    }
    g_device    = device;
    g_available = 1;
    ene_set_error(ENE_OK, "");
    return ENE_OK;
}

ENE_EXPORT
int ene_mem_available(void) { return g_available; }

ENE_EXPORT
int ene_mem_device(void) { return g_device; }

ENE_EXPORT
int ene_mem_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

// ---- 错误信息 --------------------------------------------------------------
ENE_EXPORT
int ene_mem_last_code(void) { return g_last_code; }

ENE_EXPORT
const char *ene_mem_last_error(void) { return g_last_msg; }

ENE_EXPORT
const char *ene_mem_error_string(int code) {
    switch (code) {
        case ENE_OK:           return "ok";
        case ENE_ERR_OOM:      return "out of device memory";
        case ENE_ERR_INVALID:  return "invalid argument";
        case ENE_ERR_DEVICE:   return "invalid device";
        case ENE_ERR_CUDA:     return "cuda error";
        case ENE_ERR_NODEVICE: return "no cuda device";
        default:               return "unknown";
    }
}

// ---- 分配 / 释放（直通，无池）----------------------------------------------
ENE_EXPORT
void *ene_mem_alloc(size_t bytes) {
    if (!g_available) {
        ene_set_error(ENE_ERR_NODEVICE, "device memory not initialized");
        return NULL;
    }
    if (bytes == 0) {
        ene_set_error(ENE_ERR_INVALID, "bytes must be > 0");
        return NULL;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        ene_set_cuda_error(err);                     // FR-1.2：区分 OOM 与其他错误
        return NULL;
    }
    g_alloc_calls++;
    ene_set_error(ENE_OK, "");
    return ptr;
}

ENE_EXPORT
int ene_mem_free(void *ptr) {
    if (ptr == NULL) {
        ene_set_error(ENE_ERR_INVALID, "free(NULL)");
        return ENE_ERR_INVALID;
    }
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        ene_set_cuda_error(err);
        return g_last_code;
    }
    g_free_calls++;
    ene_set_error(ENE_OK, "");
    return ENE_OK;
}

// ---- 传输（阶段1：同步语义）------------------------------------------------
// 内部用 Async + StreamSynchronize：stream 参数此时就生效，
// 阶段3 去掉那次 synchronize 即为异步版本，签名不用变（FR-3.1）。
ENE_EXPORT
int ene_mem_upload(void *d_ptr, const void *h_ptr, size_t bytes, void *stream) {
    if (d_ptr == NULL || h_ptr == NULL) {
        ene_set_error(ENE_ERR_INVALID, "upload: null pointer");
        return ENE_ERR_INVALID;
    }
    if (bytes == 0) {
        ene_set_error(ENE_ERR_INVALID, "upload: bytes must be > 0");
        return ENE_ERR_INVALID;
    }
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err = cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, s);
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(s);
    }
    if (err != cudaSuccess) {
        // FR-3.5：失败必须上报；调用方收到错误后应把该块标记为「内容不可信」
        ene_set_cuda_error(err);
        return g_last_code;
    }
    g_up_bytes += (unsigned long long)bytes;
    ene_set_error(ENE_OK, "");
    return ENE_OK;
}

ENE_EXPORT
int ene_mem_download(void *h_ptr, const void *d_ptr, size_t bytes, void *stream) {
    if (h_ptr == NULL || d_ptr == NULL) {
        ene_set_error(ENE_ERR_INVALID, "download: null pointer");
        return ENE_ERR_INVALID;
    }
    if (bytes == 0) {
        ene_set_error(ENE_ERR_INVALID, "download: bytes must be > 0");
        return ENE_ERR_INVALID;
    }
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err = cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, s);
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(s);
    }
    if (err != cudaSuccess) {
        ene_set_cuda_error(err);
        return g_last_code;
    }
    g_down_bytes += (unsigned long long)bytes;
    ene_set_error(ENE_OK, "");
    return ENE_OK;
}

ENE_EXPORT
int ene_mem_sync(void *stream) {
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    if (err != cudaSuccess) {
        ene_set_cuda_error(err);
        return g_last_code;
    }
    ene_set_error(ENE_OK, "");
    return ENE_OK;
}

// ---- 统计（供基准脚本验证「稳态零分配」NFR-3）------------------------------
ENE_EXPORT
unsigned long long ene_mem_alloc_calls(void) { return g_alloc_calls; }

ENE_EXPORT
unsigned long long ene_mem_free_calls(void) { return g_free_calls; }

ENE_EXPORT
unsigned long long ene_mem_upload_bytes(void) { return g_up_bytes; }

ENE_EXPORT
unsigned long long ene_mem_download_bytes(void) { return g_down_bytes; }

ENE_EXPORT
void ene_mem_reset_counters(void) {
    g_alloc_calls = 0;
    g_free_calls  = 0;
    g_up_bytes    = 0;
    g_down_bytes  = 0;
}
