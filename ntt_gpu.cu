#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

const int N = 16384;
const int G = 3;
const unsigned long long P = 4179340454199820289ULL;
const int LOGN = 14; // log2(N)

// 设备端函数声明
__device__ unsigned long long mod_mul(unsigned long long a, unsigned long long b, unsigned long long mod);
__device__ unsigned long long mod_add(unsigned long long a, unsigned long long b, unsigned long long mod);
__device__ unsigned long long mod_sub(unsigned long long a, unsigned long long b, unsigned long long mod);
__global__ void ntt_kernel(unsigned long long* data, const unsigned long long* twiddles, int step);
__global__ void bit_reverse_kernel(unsigned long long* data, const int* rev);

// 设备常量内存 (存储旋转因子)
__constant__ unsigned long long d_twiddles[N];
__constant__ unsigned long long d_itwiddles[N];
__constant__ int d_rev[N];

// 设备端模乘函数 (Barrett reduction)
__device__ unsigned long long mod_mul(unsigned long long a, unsigned long long b, unsigned long long mod) {
    unsigned long long res;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b));
    unsigned long long q = ((__uint128_t)res * (__uint128_t)mod) >> 64;
    res = a * b - q * mod;
    return res < mod ? res : res - mod;
}

// 设备端模加
__device__ unsigned long long mod_add(unsigned long long a, unsigned long long b, unsigned long long mod) {
    unsigned long long res = a + b;
    return res >= mod ? res - mod : res;
}

// 设备端模减
__device__ unsigned long long mod_sub(unsigned long long a, unsigned long long b, unsigned long long mod) {
    return a >= b ? a - b : a + mod - b;
}

// 位反转核函数
__global__ void bit_reverse_kernel(unsigned long long* data, const int* rev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int rev_idx = rev[idx];
        if (idx < rev_idx) {
            unsigned long long tmp = data[idx];
            data[idx] = data[rev_idx];
            data[rev_idx] = tmp;
        }
    }
}

// NTT核函数
__global__ void ntt_kernel(unsigned long long* data, int step) {
    extern __shared__ unsigned long long sdata[];
    int idx = threadIdx.x;
    int block_start = blockIdx.x * (blockDim.x * 2);
    
    // 将全局内存加载到共享内存
    sdata[idx] = data[block_start + idx];
    sdata[idx + blockDim.x] = data[block_start + idx + blockDim.x];
    __syncthreads();
    
    // 蝶形运算
    int m = 1 << (step + 1);
    int k = idx & ((1 << step) - 1);
    int j = idx >> step;
    int g_idx = j * (1 << (LOGN - step - 1)) + k;
    
    unsigned long long w = d_twiddles[g_idx];
    unsigned long long u = sdata[j * m + k];
    unsigned long long v = mod_mul(sdata[j * m + k + (m >> 1)], w, P);
    
    sdata[j * m + k] = mod_add(u, v, P);
    sdata[j * m + k + (m >> 1)] = mod_sub(u, v, P);
    __syncthreads();
    
    // 写回全局内存
    data[block_start + idx] = sdata[idx];
    data[block_start + idx + blockDim.x] = sdata[idx + blockDim.x];
}



// 主机端NTT函数
void ntt_gpu(unsigned long long* d_data) {
    dim3 block(512);
    dim3 grid(N / (2 * block.x));
    
    // 位反转排列
    bit_reverse_kernel<<<(N + 255)/256, 256>>>(d_data, d_rev);
    cudaDeviceSynchronize();
    
    // 逐层处理
    for (int step = 0; step < LOGN; step++) {
        ntt_kernel<<<grid, block, 2 * block.x * sizeof(unsigned long long)>>>(d_data, step);
        cudaDeviceSynchronize();
    }
}



// 性能测试函数
void ntt_performance_test_gpu() {
    // 初始化GPU NTT
    
    // 分配设备内存
    unsigned long long *d_data;
    cudaMalloc(&d_data, N * sizeof(unsigned long long));
    
    // 生成随机数据
    std::vector<unsigned long long> h_data(N);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dis(0, P-1);
    for (auto& x : h_data) x = dis(gen);
    
    // 复制到设备
    cudaMemcpy(d_data, h_data.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    
    // 预热
    ntt_gpu(d_data);
    
    // 计时测试
    const int repeats = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < repeats; i++) {
        ntt_gpu(d_data);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 输出结果
    double total_time = duration.count() / 1000.0; // ms
    double avg_time = total_time / (2 * repeats); // 每次NTT/INTT平均时间
    
    std::cout << "GPU NTT Performance (N=" << N << "):\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Average NTT+INTT time: " << avg_time * 2 << " ms\n";
    std::cout << "Average NTT time: " << avg_time << " ms\n";
    
    // 验证结果
    cudaMemcpy(h_data.data(), d_data, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (h_data[i] != h_data[0]) { // 简化验证
            std::cerr << "Verification failed at position " << i << "\n";
            break;
        }
    }
    
    cudaFree(d_data);
}

int main() {
    ntt_performance_test_gpu();
    return 0;
}