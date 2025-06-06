#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

typedef unsigned long long ULL;
typedef __uint128_t ULL128;

const int N = 16384;
const ULL P = 4179340454199820289ULL;
const ULL G = 3;

// Modular multiplication
inline __device__ ULL qmul(ULL a, ULL b, ULL mod) {
    return (ULL)((ULL128)a * b % mod);
}

// Modular exponentiation (used on host)
inline __device__ ULL qpow(ULL x, ULL y, ULL mod) {
    ULL res = 1;
    #pragma unroll
    while (y) {
        if (y & 1) res = qmul(res, x, mod);
        x = qmul(x, x, mod);
        y >>= 1;
    }
    return res;
}




inline ULL qmul1(ULL a, ULL b, ULL mod) {
    return (ULL)((ULL128)a * b % mod);
}

inline ULL qpow1(ULL x, ULL y, ULL mod) {
    ULL res = 1;
    
    while (y) {
        if (y & 1) res = qmul1(res, x, mod);
        x = qmul1(x, x, mod);
        y >>= 1;
    }
    return res;
}

// Bit-reversal function
__device__ int bit_reverse(int x, int bits) {
    int rev = 0;
    for (int i = 0; i < bits; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    return rev;
}

// Kernel for in-place bit-reversal permutation
__global__ void bit_reverse_permute_inplace(ULL* d_x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int rev_i = bit_reverse(i, 14); // log2(16384) = 14
        if (i < rev_i) {
            ULL temp = d_x[i];
            d_x[i] = d_x[rev_i];
            d_x[rev_i] = temp;
        }
    }
}

// Kernel for NTT stage
__global__ void ntt_stage_kernel(ULL* d_x, ULL* d_twiddle, int start_s, int s, int N, ULL P) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N / 2) {
        int m = 1 << s;
        int k = m / 2;
        int group_id = t / k;
        int local_j = t % k;
        int i = group_id * m;
        int idx1 = i + local_j;
        int idx2 = i + local_j + k;
        ULL g = d_twiddle[start_s + local_j];
        ULL x1 = d_x[idx1];
        ULL x2 = d_x[idx2];
        ULL tmp = qmul(x2, g, P);
        ULL diff = x1 >= tmp ? x1 - tmp : x1 + P - tmp;
        ULL sum = (x1 + tmp) % P;
        d_x[idx1] = sum;
        d_x[idx2] = diff;
    }
}

// Kernel for INTT stage
__global__ void intt_stage_kernel(ULL* d_x, ULL* d_twiddle_inv, int start_s, int s, int N, ULL P) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N / 2) {
        int m = 1 << s;
        int k = m / 2;
        int group_id = t / k;
        int local_j = t % k;
        int i = group_id * m;
        int idx1 = i + local_j;
        int idx2 = i + local_j + k;
        ULL g = d_twiddle_inv[start_s + local_j];
        ULL x1 = d_x[idx1];
        ULL x2 = d_x[idx2];
        ULL tmp = qmul(x2, g, P);
        ULL diff = x1 >= tmp ? x1 - tmp : x1 + P - tmp;
        ULL sum = (x1 + tmp) % P;
        d_x[idx1] = sum;
        d_x[idx2] = diff;
    }
}

// Kernel for scaling by N_inv
__global__ void scale_by_n_inv(ULL* d_x, ULL n_inv, int N, ULL P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_x[i] = qmul(d_x[i], n_inv, P);
    }
}

int main() {
    // Generate random input data
    std::vector<ULL> h_a_original(N);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<ULL> dis(0, P - 1);
    for (int i = 0; i < N; i++) {
        h_a_original[i] = dis(gen);
    }

    // Precompute twiddle factors for NTT
    std::vector<ULL> h_twiddle_total(16383);
    int idx = 0;
    for (int s = 1; s <= 14; s++) {
        int m = 1 << s;
        ULL exponent = (P - 1) / m;
        int k = m / 2;
        for (int j = 0; j < k; j++) {
            h_twiddle_total[idx + j] = qpow1(G, j * exponent, P);
        }
        idx += k;
    }

    // Precompute inverse twiddle factors for INTT
    ULL G_inv = qpow1(G, P - 2, P); // G^{-1} mod P
    std::vector<ULL> h_twiddle_inv_total(16383);
    idx = 0;
    for (int s = 1; s <= 14; s++) {
        int m = 1 << s;
        ULL exponent = (P - 1) / m;
        int k = m / 2;
        for (int j = 0; j < k; j++) {
            h_twiddle_inv_total[idx + j] = qpow1(G_inv, j * exponent, P);
        }
        idx += k;
    }

    // Compute N_inv for scaling
    ULL N_inv = qpow1(N, P - 2, P); // N^{-1} mod P

    // Allocate device memory
    ULL *d_x, *d_twiddle, *d_twiddle_inv;
    cudaMalloc(&d_x, N * sizeof(ULL));
    cudaMalloc(&d_twiddle, h_twiddle_total.size() * sizeof(ULL));
    cudaMalloc(&d_twiddle_inv, h_twiddle_inv_total.size() * sizeof(ULL));
    cudaMemcpy(d_twiddle, h_twiddle_total.data(), h_twiddle_total.size() * sizeof(ULL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddle_inv, h_twiddle_inv_total.data(), h_twiddle_inv_total.size() * sizeof(ULL), cudaMemcpyHostToDevice);

    // Timing and verification loop
    const int repeats = 10;
    long long total_duration = 0;
    int threads_per_block = 1024;
    int blocks_bitrev = (N + threads_per_block - 1) / threads_per_block;
    int blocks_ntt = ((N / 2) + threads_per_block - 1) / threads_per_block;
    int blocks_scale = (N + threads_per_block - 1) / threads_per_block;

    // Store NTT output for verification
    std::vector<ULL> h_a_ntt(N);
    bool is_correct = true;

    for (int rep = 0; rep < repeats; rep++) {
        // Copy input to device
        cudaMemcpy(d_x, h_a_original.data(), N * sizeof(ULL), cudaMemcpyHostToDevice);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // NTT
        bit_reverse_permute_inplace<<<blocks_bitrev, threads_per_block>>>(d_x, N);
        cudaDeviceSynchronize();
        for (int s = 1; s <= 14; s++) {
            int start_s = (1 << (s - 1)) - 1;
            ntt_stage_kernel<<<blocks_ntt, threads_per_block>>>(d_x, d_twiddle, start_s, s, N, P);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_duration += milliseconds * 1000; // Convert to microseconds

        cudaEventDestroy(start);
        cudaEventDestroy(stop);





        // Check correctness of NTT
        // Copy NTT result back for INTT
        cudaMemcpy(h_a_ntt.data(), d_x, N * sizeof(ULL), cudaMemcpyDeviceToHost);

        // INTT
        bit_reverse_permute_inplace<<<blocks_bitrev, threads_per_block>>>(d_x, N);
        cudaDeviceSynchronize();
        for (int s = 1; s <= 14; s++) {
            int start_s = (1 << (s - 1)) - 1;
            intt_stage_kernel<<<blocks_ntt, threads_per_block>>>(d_x, d_twiddle_inv, start_s, s, N, P);
            cudaDeviceSynchronize();
        }
        scale_by_n_inv<<<blocks_scale, threads_per_block>>>(d_x, N_inv, N, P);
        cudaDeviceSynchronize();

        // Copy INTT result back
        std::vector<ULL> h_a_intt(N);
        cudaMemcpy(h_a_intt.data(), d_x, N * sizeof(ULL), cudaMemcpyDeviceToHost);

        // Verify correctness
        for (int i = 0; i < N; i++) {
            if (h_a_intt[i] != h_a_original[i]) {
                is_correct = false;
                std::cout << "Verification failed at index " << i << ": expected " << h_a_original[i] << ", got " << h_a_intt[i] << "\n";
                break;
            }
        }



    }

    // Output results
    double average_time = static_cast<double>(total_duration) / repeats;
    std::cout << "N: " << N << ", Repeats: " << repeats << "\n";
    std::cout << "Total time: " << total_duration << " us\n";
    std::cout << "Average time per NTT+INTT: " << average_time << " us\n";
    std::cout << "Verification: " << (is_correct ? "Passed" : "Failed") << "\n";

    // Clean up
    cudaFree(d_x);
    cudaFree(d_twiddle);
    cudaFree(d_twiddle_inv);
    return 0;
}