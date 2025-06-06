#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <cstdint>

#include <chrono>
#include <random>
#include <algorithm>

typedef unsigned __int128 uint128_t; // 高版本gcc支持128位无符号数，也可以考虑使用gmp库
const int N = 16384, G = 3;          // 模数为P，NTT长度为N，素数的原根为G
const unsigned long long P = 4179340454199820289;

std::vector<int> r(N);

// 初始化位逆序表
void init_r()
{
    for (int i = 0; i < N; i++)
    {
        r[i] = (r[i >> 1] >> 1) | ((i & 1) << 13); // N=2^14, 13=14-1
    }
}

unsigned long long qmul(unsigned long long a, unsigned long long b, unsigned long long mod)
{
    __uint128_t res = (__uint128_t)a * b;
    return (unsigned long long)(res % mod);
}

unsigned long long qpow(unsigned long long x, unsigned long long y) // 快速模幂算法
{
    unsigned long long res(1);
    while (y)
    {
        if (y & 1)
            res = qmul(res, y, P);
        x = qmul(x, x, P);
        y >>= 1;
    }
    return res;
}

void ntt(std::vector<unsigned long long> &x, int lim)
{ // ntt的简单实现
    int i, j, k, m;
    unsigned long long gn, g, tmp;
    for (i = 0; i < lim; ++i)
    {
        if (r[i] < i)
        {
            std::swap(x[i], x[r[i]]);
        }
    }
    for (m = 2; m <= lim; m <<= 1)
    {
        k = m >> 1;
        gn = qpow(G, (P - 1) / m);
        for (i = 0; i < lim; i += m)
        {
            g = 1;
            for (j = 0; j < k; j++, g = qmul(g, gn, P))
            {
                tmp = qmul(x[i + j + k], g, P);
                x[i + j + k] = (x[i + j] >= tmp ? (x[i + j] - tmp) : (x[i + j] + P - tmp));
                x[i + j] = (x[i + j] + tmp) % P;
            }
        }
    }
}

int main()
{
    // 初始化位逆序表
    init_r();

    // 生成随机数引擎
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dis(0, P - 1);

    // 创建原始数据
    std::vector<unsigned long long> a_original(N);
    for (int i = 0; i < N; i++)
    {
        a_original[i] = dis(gen);
    }

    // 工作向量
    std::vector<unsigned long long> a_work(N);

    const int repeats = 10;
    long long total_duration = 0;

    for (int i = 0; i < repeats; i++)
    {
        // 复制数据到工作向量
        std::copy(a_original.begin(), a_original.end(), a_work.begin());

        auto start = std::chrono::high_resolution_clock::now();

        // 执行NTT变换
        ntt(a_work, N);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_duration += duration.count();
    }

    double average_time = static_cast<double>(total_duration) / repeats;
    std::cout << "N: " << N << ", Repeats: " << repeats << "\n";
    std::cout << "Total time: " << total_duration << " us\n";
    std::cout << "Average time per NTT: " << average_time << " us\n";
    std::cout << "Average time per polynomial mult: " << average_time << " us\n";

    return 0;
}