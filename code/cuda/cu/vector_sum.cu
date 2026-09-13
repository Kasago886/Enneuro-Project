// vector_sum.cu  (归约求和: 返回 sum(a))
// 两阶段归约: 块内共享内存折半归约 -> 每个块只做一次 atomicAdd
#include <cuda_runtime.h>
#include <stdio.h>

#if defined(_WIN32)
#define ENE_EXPORT extern "C" __declspec(dllexport)
#else
#define ENE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ void sum_kernel(const float *a, float *out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// 返回值即为求和结果
ENE_EXPORT
float launch_sum(const float *a, int n) {
    float *d_a = NULL, *d_out = NULL;
    size_t bytes = (size_t)n * sizeof(float);
    float result = 0.0f;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_a, d_out, n);

    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_out);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return result;
}
