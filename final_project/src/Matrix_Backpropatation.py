import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ==============================================================================
# 1. 基础数学工具与目标函数
# ==============================================================================

def sym(A):
    """对称化算子: (A + A^T) / 2"""
    return 0.5 * (A + A.transpose(-2, -1))

def func_log(x): 
    """目标函数 f(x) = log(x)"""
    return torch.log(x)

def func_log_prime(x): 
    """目标函数导数 f'(x) = 1/x"""
    return 1.0 / x

# ==============================================================================
# 2. 核心算法实现 (DK/B vs Ionescu)
# ==============================================================================

def get_loewner_matrix(lambda_vals, fn, fn_prime, epsilon=1e-12):
    """
    [Unified Framework 核心] Loewner 矩阵计算
    关键改进：当 λi ≈ λj 时，使用导数 f'(λ) 替换差商，避免数值爆炸。
    """
    n = lambda_vals.shape[0]
    li = lambda_vals.unsqueeze(1)
    lj = lambda_vals.unsqueeze(0)
    denom = li - lj
    num = fn(li) - fn(lj)
    
    L = torch.zeros((n, n), dtype=lambda_vals.dtype, device=lambda_vals.device)
    
    # 1. 正常情况：使用差商 (Difference Quotient)
    mask_diff = torch.abs(denom) > epsilon
    L[mask_diff] = num[mask_diff] / denom[mask_diff]
    
    # 2. 退化情况 (Degenerate)：使用导数 (Derivative)
    mask_close = ~mask_diff
    if mask_close.any():
        mid_lambda = (li + lj) / 2
        L[mask_close] = fn_prime(mid_lambda[mask_close])
    return L

def gradient_dkb(grad_Y, U, S, fn, fn_prime):
    """
    【推荐】Daleckii-Krein/Bhatia 公式 (Unified Framework)
    Paper 2 Eq (17): dX = U * (L ⊙ (U^T * dY * U)) * U^T
    特点：稳定、快速。
    """
    # 1. 计算 Loewner 矩阵 (O(N^2))
    L = get_loewner_matrix(S, fn, fn_prime)
    
    # 2. 旋转梯度 (O(N^3))
    C = U.t() @ grad_Y @ U
    
    # 3. Hadamard 乘积与回旋 (O(N^3))
    grad_X = U @ (L * C) @ U.t()
    return grad_X

def gradient_ionescu(grad_Y, U, S, fn, fn_prime, epsilon=0): 
    """
    【不推荐】Ionescu 公式 (Paper 1 / Paper 2 Eq 19)
    Paper 2 Eq (19): dX = U * [2 sym(K^T ⊙ (C f(Λ))) + I ⊙ (f'(Λ) C)] * U^T
    特点：显式依赖 1/(λi - λj)，在特征值接近时数值不稳定，且计算量更大。
    """
    n = S.shape[0]
    li = S.unsqueeze(1)
    lj = S.unsqueeze(0)
    denom = li - lj
    
    # 构造 K 矩阵：这是数值不稳定的根源
    K = torch.zeros((n, n), dtype=S.dtype, device=S.device)
    # 为了展示其缺陷，我们不做平滑处理，只避免绝对的除零
    mask_nonzero = torch.abs(denom) > epsilon 
    K[mask_nonzero] = 1.0 / denom[mask_nonzero] 
    
    C = U.t() @ grad_Y @ U
    f_Lambda = torch.diag(fn(S))
    f_prime_Lambda_vec = fn_prime(S)
    
    # 复杂的中间项计算
    # Term 1: 2 * sym(K^T ⊙ (C @ f(Λ)))
    term1 = 2 * sym(K.t() * (C @ f_Lambda))
    
    # Term 2: diag(f'(Λ) * diag(C))
    term2 = torch.diag(f_prime_Lambda_vec * torch.diagonal(C))
    
    grad_X = U @ (term1 + term2) @ U.t()
    return grad_X

# ==============================================================================
# 3. 辅助实验模块 (Ground Truth & Benchmark)
# ==============================================================================

def compute_numerical_gradient(X_input, grad_Y_fixed, epsilon=1e-7):
    """通过中心有限差分法 (Finite Difference) 计算 Ground Truth"""
    N = X_input.shape[0]
    grad_num = torch.zeros_like(X_input)
    
    for i in range(N):
        for j in range(N):
            delta = torch.zeros_like(X_input)
            # 保持对称性扰动
            if i == j:
                delta[i, j] = epsilon
            else:
                delta[i, j] = epsilon
                delta[j, i] = epsilon
            
            # f(X + h)
            S_p, U_p = torch.linalg.eigh(X_input + delta)
            val_p = (U_p @ torch.diag(func_log(S_p)) @ U_p.t() * grad_Y_fixed).sum()
            
            # f(X - h)
            S_m, U_m = torch.linalg.eigh(X_input - delta)
            val_m = (U_m @ torch.diag(func_log(S_m)) @ U_m.t() * grad_Y_fixed).sum()
            
            grad_num[i, j] = (val_p - val_m) / (2 * epsilon)
            
    return grad_num

def benchmark_speed():
    """速度基准测试"""
    print("\n[Phase 3] Running Speed Benchmark...")
    sizes = [64, 128, 256, 512, 1024]
    loops = {64: 200, 128: 100, 256: 50, 512: 20, 1024: 5} # 动态调整循环次数
    
    time_dkb = []
    time_ionescu = []
    
    # 自动选择设备 (CPU/CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Device: {device}")

    for n in sizes:
        num_loops = loops[n]
        
        # 准备数据 (预热)
        temp = torch.randn(n, n, dtype=torch.float64, device=device)
        U_fixed, _ = torch.linalg.qr(temp)
        S_fixed = torch.linspace(1.0, 10.0, n, dtype=torch.float64, device=device)
        grad_Y_fixed = sym(torch.randn(n, n, dtype=torch.float64, device=device))
        
        # 预热 JIT
        _ = gradient_dkb(grad_Y_fixed, U_fixed, S_fixed, func_log, func_log_prime)
        _ = gradient_ionescu(grad_Y_fixed, U_fixed, S_fixed, func_log, func_log_prime)
        
        # 计时 DK/B
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_loops):
            _ = gradient_dkb(grad_Y_fixed, U_fixed, S_fixed, func_log, func_log_prime)
        if device.type == 'cuda': torch.cuda.synchronize()
        avg_dkb = (time.time() - start) / num_loops
        time_dkb.append(avg_dkb)
        
        # 计时 Ionescu
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_loops):
            _ = gradient_ionescu(grad_Y_fixed, U_fixed, S_fixed, func_log, func_log_prime)
        if device.type == 'cuda': torch.cuda.synchronize()
        avg_ion = (time.time() - start) / num_loops
        time_ionescu.append(avg_ion)
        
        print(f"Size {n}: DK/B = {avg_dkb*1000:.2f}ms | Ionescu = {avg_ion*1000:.2f}ms")

    return sizes, time_dkb, time_ionescu

# ==============================================================================
# 4. 主程序
# ==============================================================================

def main():
    # 设置
    torch.manual_seed(42)
    np.random.seed(42)
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # 定义特征值间隙序列 (从 0.1 到 1e-16)
    gaps = np.logspace(-1, -16, 100)
    
    norm_dkb = []
    norm_ionescu = []
    error_dkb = []
    error_ionescu = []
    valid_gaps = []

    # 构造基矩阵 (Double Precision 保证验证精度)
    N = 4
    temp = torch.randn(N, N, dtype=torch.float64)
    U_fixed, _ = torch.linalg.qr(temp)
    grad_Y_fixed = sym(torch.randn(N, N, dtype=torch.float64))
    S_base = torch.linspace(2.0, 5.0, N, dtype=torch.float64)

    print("[Phase 1 & 2] Running Stability & Validation Experiments...")

    for gap in gaps:
        # 构造特征值接近的矩阵
        S = S_base.clone()
        S[1] = S[0] + gap
        X = U_fixed @ torch.diag(S) @ U_fixed.t()
        
        # 模拟前向 EVD
        S_calc, U_calc = torch.linalg.eigh(X)
        
        # 计算两种梯度
        g_dkb = gradient_dkb(grad_Y_fixed, U_calc, S_calc, func_log, func_log_prime)
        g_ion = gradient_ionescu(grad_Y_fixed, U_calc, S_calc, func_log, func_log_prime)
        
        norm_dkb.append(torch.norm(g_dkb).item())
        norm_ionescu.append(torch.norm(g_ion).item())
        
        # 计算 Ground Truth (仅在数值微分有效的区间)
        if gap > 1e-8:
            g_gt = compute_numerical_gradient(X, grad_Y_fixed)
            norm_gt = torch.norm(g_gt).item()
            error_dkb.append(torch.norm(g_dkb - g_gt).item() / norm_gt)
            error_ionescu.append(torch.norm(g_ion - g_gt).item() / norm_gt)
            valid_gaps.append(gap)

    # 运行速度测试
    sizes, t_dkb, t_ion = benchmark_speed()

    # ================= 绘图 =================
    print("Plotting results...")
    
    # 图 1: 相对误差验证
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(valid_gaps, error_dkb, 'r-o', markersize=4, label='DK/B (Proposed)')
    ax1.plot(valid_gaps, error_ionescu, 'b--x', markersize=4, label='Ionescu (Baseline)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.invert_xaxis()
    ax1.set_xlabel(r'Eigenvalue Gap $|\lambda_i - \lambda_j|$')
    ax1.set_ylabel('Relative Error vs Numerical GT')
    ax1.set_title('Accuracy Verification\n(Both correct when Gap is large)')
    ax1.legend()

    # 图 2: 梯度爆炸检测
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(gaps, norm_dkb, 'r-', lw=2.5, label='DK/B Gradient Norm')
    ax2.plot(gaps, norm_ionescu, 'b--', lw=1.5, label='Ionescu Gradient Norm')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.invert_xaxis()
    ax2.set_xlabel(r'Eigenvalue Gap $|\lambda_i - \lambda_j|$')
    ax2.set_ylabel(r'Gradient Norm $||\nabla X||_F$')
    ax2.set_title('Stability Check (Small Gaps)\nIonescu Explodes!')
    ax2.annotate('Explosion', xy=(1e-15, norm_ionescu[-1]), xytext=(1e-11, norm_ionescu[-1]/10),
                 arrowprops=dict(facecolor='black', shrink=0.05), color='red')
    ax2.legend()

    # 图 3: 计算速度
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(sizes, t_dkb, 'r-o', lw=2, label='DK/B Time')
    ax3.plot(sizes, t_ion, 'b--s', lw=2, label='Ionescu Time')
    ax3.fill_between(sizes, t_dkb, t_ion, color='green', alpha=0.1, label='Efficiency Gain')
    ax3.set_xlabel('Matrix Dimension (N)')
    ax3.set_ylabel('Execution Time (s)')
    ax3.set_title('Computational Efficiency')
    ax3.legend()
    speedup = (t_ion[-1] - t_dkb[-1]) / t_ion[-1] * 100
    ax3.text(sizes[2], (t_ion[-1]+t_dkb[-1])/2, f"~{speedup:.1f}% Faster", 
             color='green', fontweight='bold', ha='center')

    # 保存结果
    root_dir = Path(__file__).resolve().parent.parent / 'plots' / 'plots_matrix_backprop'
    root_dir.mkdir(exist_ok=True)
    
    fig1.savefig(root_dir / '1_Validation.png', dpi=300)
    fig2.savefig(root_dir / '2_Stability.png', dpi=300)
    fig3.savefig(root_dir / '3_Speed.png', dpi=300)
    
    print(f"Done! Plots saved to: {root_dir}")
    plt.show()

if __name__ == "__main__":
    main()