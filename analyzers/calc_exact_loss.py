import numpy as np
from scipy import integrate

def exact_ising_free_energy_per_spin(T, J=1.0):
    """
    计算二维正方晶格 Ising 模型的精确自由能 (Onsager Solution)
    """
    beta = 1.0 / T
    K = beta * J
    
    # 定义 Onsager 积分的被积函数
    def integrand(x, y):
        term1 = np.cosh(2*K)**2
        term2 = np.sinh(2*K) * (np.cos(x) + np.cos(y))
        return np.log(term1 - term2)
        
    # 双重数值积分
    integral, error = integrate.dblquad(integrand, 0, np.pi, lambda x: 0, lambda x: np.pi)
    
    # 计算每个自旋的自由能 f
    f = -T * np.log(2) - (T / (2 * np.pi**2)) * integral
    return f

# 设定系统尺寸
L = 32
N = L**2

# 测试你关心的几个关键温度
temperatures = [3.0, 2.65, 2.269185, 2.0]

print("-" * 70)
# 修改表头，增加总 Loss 列
print(f"{'温度 T':<10} | {'f (F/N)':<15} | {'单点 Loss (f/T)':<18} | {'总 Loss (L=32)':<15}")
print("-" * 70)

for T in temperatures:
    f_exact = exact_ising_free_energy_per_spin(T)
    
    # 每个自旋的理论 Loss 极小值
    loss_per_spin = f_exact / T
    
    # 给定尺寸 L 的总 Loss 理论值
    # 总 Loss = N * (f / T)
    total_loss_exact = N * loss_per_spin
    
    print(f"{T:<10.6f} | {f_exact:<15.6f} | {loss_per_spin:<18.6f} | {total_loss_exact:<15.6f}")

print("-" * 70)