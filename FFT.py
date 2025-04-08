# A(x) = 2 + 3x + x^2
# B(x) = 1 + 2x^2
import math
import cmath
#coeff
def generate_roots_of_unity(n):

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n 必须是 2 的正整数次幂（如 2, 4, 8...）")
        
    roots = []
    for k in range(n):
        # 计算角度 θ = 2πk/n
        theta = 2 * math.pi * k / n
        # 生成复数根 e^(iθ)
        root = cmath.exp(1j * theta)
        # 四舍五入消除浮点误差
        root = complex(round(root.real, 10), round(root.imag, 10))
        roots.append(root)
    return roots




C = [5,3,1,8,2,6,8,5]
D = [3,3,5,7,2,3,9,4]



def FFT(P):
    n = len(P)
    if n==1:
        return P
    omega = cmath.exp(1j * 2 * math.pi / n)
    Pe = P[::2]
    Po = P[1::2]
    ye,yo = FFT(Pe),FFT(Po)
    y = [0] *n
    for j in range(int(n/2)):
        y[j] = ye[j] + (omega**j) * yo[j]
        y[j + int(n/2)] = ye[j] - (omega**j) * yo[j]
    return y





def iFFT(P):
    n = len(P)
    if n==1:
        return P
    omega = cmath.exp(1j * (-2) * math.pi / n)
    Pe = P[::2]
    Po = P[1::2]
    ye,yo = iFFT(Pe),iFFT(Po)
    y = [0] *n
    for j in range(int(n/2)):
        y[j] = ye[j] + (omega**j) * yo[j]
        y[j + int(n/2)] = ye[j] - (omega**j) * yo[j]
    return [i/2 for i in y]




C = [5,3,1,8,2,6,8,5]
D = [3,3,5,7,2,3,9,4]
def polyMult(poly1, poly2):
    length = len(poly1)
    pointC = FFT(C+[0]*length)
    pointD = FFT(D+[0]*length)
    pointMult = [i*j for (i,j) in zip(pointC,pointD)]
    print(iFFT(pointMult))
    return iFFT(pointMult)


def polyMult2(poly1, poly2):
    length = len(poly1)
    res = [0] * (length * 2)
    for i in range(length):
        for j in range(length):
            res[i+j] += poly1[i] * poly2[j]
    return res


def com(A, B):
    length = len(A)
    C = [abs(i[0]-i[1]) for i in zip(A,B)]
    print(C)
    if sum(C)>1e-10*length:
        print("error")
    else:
        print("pass")

polyMult(C,D)
    
polyMult2(C,D)

com(polyMult(C,D),polyMult2(C,D))



    
