import math

def NTT(P, q, root):
    n = len(P)
    if n == 1:
        return P
    omega = pow(root, (q-1) // n, q)
    Pe = P[::2]
    Po = P[1::2]
    ye,yo = NTT(Pe, q, root),NTT(Po, q, root)
    y = [0] * n
    half = n // 2
    w = 1  # 初始旋转因子
    for j in range(half):
        t = (w * yo[j]) % q
        y[j] = (ye[j] + t) % q
        y[j + half] = (ye[j] - t) % q
        w = (w * omega) % q
    
    return y


def INTT(y, q, root):
    """
    逆数论变换（INTT）
    
    参数:
        y (list[int]): NTT结果
        q (int): 模数（与NTT相同）
        root (int): 原根（与NTT相同）
    
    返回:
        list[int]: 逆变换后的系数（模q）
    """
    n = len(y)
    # 计算逆变换参数：inv_root = root^{-1} mod q
    inv_root = pow(root, q-2, q)  # 费马小定理求逆元
    # 调用NTT（参数为inv_root），最后乘以n^{-1} mod q
    result = NTT(y, q, inv_root)
    inv_n = pow(n, q-2, q)  # n的逆元
    return [(x * inv_n) % q for x in result]

# 示例测试
if __name__ == "__main__":
    # 常用参数：q = 998244353（费马素数），原根 root = 3
    q = 998244353
    root = 3
    
    # 输入多项式系数（长度必须是2的幂）
    P = [1, 2, 3, 4]
    
    # 计算NTT
    ntt_result = NTT(P, q, root)
    print("NTT结果（模{}）:".format(q))
    print(ntt_result)
    print(INTT(ntt_result, q, root))



          
