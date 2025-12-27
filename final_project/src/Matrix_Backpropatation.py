import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 1. 核心数学模块 (DK/B vs Ionescu)
# ==============================================================================

def sym(A):
    """对称化算子: (A + A^T) / 2"""
    return 0.5 * (A + A.transpose(-2, -1))

def get_loewner_matrix(lambda_vals, fn, fn_prime, epsilon=1e-6):
    """
    DK/B 方法的核心：Loewner 矩阵计算
    关键点：当 λi ≈ λj 时，自动切换为导数 f'(λ)，避免除零错误。
    """
    n = lambda_vals.shape[0]
    li = lambda_vals.unsqueeze(1)
    lj = lambda_vals.unsqueeze(0)
    denom = li - lj
    num = fn(li) - fn(lj)
    
    L = torch.zeros((n, n), dtype=lambda_vals.dtype)
    
    # 1. 正常情况：使用差商
    mask_diff = torch.abs(denom) > epsilon
    L[mask_diff] = num[mask_diff] / denom[mask_diff]
    
    # 2. 简并情况 (Degenerate)：使用导数进行数值修补
    mask_close = ~mask_diff
    if mask_close.any():
        mid_lambda = (li + lj) / 2
        L[mask_close] = fn_prime(mid_lambda[mask_close])
    return L

def gradient_dkb(grad_Y, U, S, fn, fn_prime):
    """【推荐】Daleckii-Krein/Bhatia 公式 (稳定)"""
    L = get_loewner_matrix(S, fn, fn_prime)
    C = U.t() @ grad_Y @ U
    grad_X = U @ (L * C) @ U.t()
    return grad_X

def gradient_ionescu(grad_Y, U, S, fn, fn_prime, epsilon=0): 
    """【不推荐】Ionescu 公式 (不稳定，仅用于反面教材)"""
    n = S.shape[0]
    li = S.unsqueeze(1)
    lj = S.unsqueeze(0)
    denom = li - lj
    
    K = torch.zeros((n, n), dtype=S.dtype)
    # Ionescu 方法显式依赖 1/(λi - λj)。
    # 为了展示它会爆炸，这里我们尽可能保留除法，只过滤绝对的 0
    mask_nonzero = torch.abs(denom) > epsilon 
    K[mask_nonzero] = 1.0 / denom[mask_nonzero] 
    
    C = U.t() @ grad_Y @ U
    f_Lambda = torch.diag(fn(S))
    f_prime_Lambda_vec = fn_prime(S)
    
    # 对应论文 Eq. 19
    term1 = 2 * sym(K.t() * (C @ f_Lambda))
    term2 = torch.diag(f_prime_Lambda_vec * torch.diagonal(C))
    grad_X = U @ (term1 + term2) @ U.t()
    return grad_X

def func_log(x): return torch.log(x)
def func_log_prime(x): return 1.0 / x

# ==============================================================================
# 2. Ground Truth 模块 (有限差分法)
# ==============================================================================

def compute_numerical_gradient(X_input, grad_Y_fixed, epsilon=1e-7):
    """
    通过数值微分求真值。
    注意：当特征值间距小于 1e-8 时，此方法自身也会失效，因此只在前半段使用。
    """
    N = X_input.shape[0]
    grad_num = torch.zeros_like(X_input)
    
    for i in range(N):
        for j in range(N):
            delta = torch.zeros_like(X_input)
            # 对称扰动以保持矩阵结构
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

# ==============================================================================
# 3. 统一实验与绘图模块
# ==============================================================================

def main():
    # 设置
    torch.manual_seed(42)
    np.random.seed(42)

    plt.style.use('default') # 先重置为白色
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['axes.grid'] = True # 开启网格

    # 参数：特征值间距从 0.1 逐渐缩小到 1e-16
    gaps = np.logspace(-1, -16, 100)
    
    # 结果容器
    norm_dkb = []
    norm_ionescu = []
    error_dkb_vs_gt = []
    error_ionescu_vs_gt = []
    gt_valid_gaps = []

    # 固定基矩阵（Double Precision）
    N = 4
    temp = torch.randn(N, N, dtype=torch.float64)
    U_fixed, _ = torch.linalg.qr(temp)
    grad_Y_fixed = sym(torch.randn(N, N, dtype=torch.float64))
    S_base = torch.linspace(2.0, 5.0, N, dtype=torch.float64)

    print("正在运行模拟实验 (这可能需要几秒钟)...")

    for gap in gaps:
        # 1. 构造病态矩阵 X
        S = S_base.clone()
        S[1] = S[0] + gap # 强行让两个特征值非常接近
        X = U_fixed @ torch.diag(S) @ U_fixed.t()
        
        # 为了更真实的模拟，重新做一次 EVD（模拟前向传播）
        S_calc, U_calc = torch.linalg.eigh(X)
        
        # 2. 计算两种梯度
        g_dkb = gradient_dkb(grad_Y_fixed, U_calc, S_calc, func_log, func_log_prime)
        g_ion = gradient_ionescu(grad_Y_fixed, U_calc, S_calc, func_log, func_log_prime)
        
        norm_dkb.append(torch.norm(g_dkb).item())
        norm_ionescu.append(torch.norm(g_ion).item())
        
        # 3. 计算 Ground Truth (仅在数值稳定的区间计算，避免 GT 自身失效)
        if gap > 1e-8:
            g_gt = compute_numerical_gradient(X, grad_Y_fixed)
            norm_gt = torch.norm(g_gt).item()
            
            # 记录相对误差
            error_dkb_vs_gt.append(torch.norm(g_dkb - g_gt).item() / norm_gt)
            error_ionescu_vs_gt.append(torch.norm(g_ion - g_gt).item() / norm_gt)
            gt_valid_gaps.append(gap)

    # ================= 绘图 =================
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')

    # --- 图 1: 与 Ground Truth 的误差对比 (Valid Region) ---
    ax1.plot(gt_valid_gaps, error_dkb_vs_gt, 'r-o', lw=2, label='DK/B Error (Proposed)')
    ax1.plot(gt_valid_gaps, error_ionescu_vs_gt, 'b--x', lw=2, label='Ionescu Error')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.invert_xaxis() # 左边更难
    ax1.set_xlabel(r'Eigenvalue Separation $|\lambda_i - \lambda_j|$')
    ax1.set_ylabel('Relative Error vs Ground Truth')
    ax1.set_title('Phase 1: Validation (Gap > 1e-8)\nBoth are correct mathematically')
    ax1.legend()
    
    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='white')

    # --- 图 2: 极端情况下的梯度范数 (Stability Check) ---
    ax2.plot(gaps, norm_dkb, 'r-', lw=3, label='DK/B Gradient Norm')
    ax2.plot(gaps, norm_ionescu, 'b--', lw=2, label='Ionescu Gradient Norm')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.invert_xaxis() # 左边更难
    ax2.set_xlabel(r'Eigenvalue Separation $|\lambda_i - \lambda_j|$')
    ax2.set_ylabel(r'Gradient Norm $||\nabla X||_F$')
    ax2.set_title('Phase 2: The Crash (Gap < 1e-8)\nIonescu Explodes to Infinity')
    
    # 添加标注箭头
    ax2.annotate('Numerical Explosion!\n(Training Crash)', xy=(1e-15, 1e14), xytext=(1e-10, 1e10),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='darkred')
    
    ax2.legend()

    fig1.tight_layout()
    fig2.tight_layout()
    root_dir = Path(__file__).resolve().parent.parent
    root_dir = Path.joinpath(root_dir, 'plots/Matrix_Backpropagation')
    fig1.savefig(Path.joinpath(root_dir, 'Validation.png'), dpi=300, facecolor='white')
    fig2.savefig(Path.joinpath(root_dir, 'Crash.png'), dpi=300, facecolor='white')
    plt.show()

if __name__ == "__main__":
    main()