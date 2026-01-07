import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==============================================================================
# 1. Ground Truth (Double Precision)
# ==============================================================================
def get_ground_truth_grad(A, G):
    """
    使用双精度计算精确梯度。
    """
    A_64 = A.double().detach().requires_grad_(False)
    G_64 = G.double().detach()
    
    # Forward
    L, U = torch.linalg.eigh(A_64)
    L = L.clamp(min=1e-12)
    # 【修复】使用 diag_embed 支持 Batch
    Y_64 = U @ torch.diag_embed(L.sqrt()) @ U.mT
    
    # Backward: Solve Lyapunov Equation
    def mat_op(X):
        return X @ Y_64 + Y_64 @ X
    
    # CG Solver for GT
    X = torch.zeros_like(G_64)
    R = G_64 - mat_op(X)
    P = R.clone()
    rsold = torch.sum(R * R, dim=(-2, -1), keepdim=True)
    
    for _ in range(1000): 
        AP = mat_op(P)
        alpha = rsold / (torch.sum(P * AP, dim=(-2, -1), keepdim=True) + 1e-20)
        X = X + alpha * P
        R = R - alpha * AP
        rsnew = torch.sum(R * R, dim=(-2, -1), keepdim=True)
        if rsnew.max().sqrt() < 1e-12: break
        P = R + (rsnew / rsold) * P
        rsold = rsnew
        
    return X.float()

# ==============================================================================
# 2. Baseline 1: Ionescu (Unstable)
# ==============================================================================
class IonescuSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        L, U = torch.linalg.eigh(A)
        # 【修复】使用 diag_embed 替代 diag，支持 batch
        Y = U @ torch.diag_embed(L.clamp(min=1e-12).sqrt()) @ U.mT
        ctx.save_for_backward(L, U)
        return Y

    @staticmethod
    def backward(ctx, G):
        L, U = ctx.saved_tensors
        # L shape: [B, N] or [N]
        # U shape: [B, N, N] or [N, N]
        
        # 【修复】安全的广播写法，支持 Batch
        L_col = L.unsqueeze(-1) # [B, N, 1]
        L_row = L.unsqueeze(-2) # [B, 1, N]
        
        # 构造 K 矩阵 (The instability source)
        denom = L_col - L_row
        # 强制不稳定性：不使用 mask，允许数值爆炸
        K = 1.0 / (denom + 1e-30)
        K.diagonal(dim1=-2, dim2=-1).fill_(0)
        
        sqrt_L = L_col.sqrt() # [B, N, 1]
        dL_dSigma = 0.5 / sqrt_L 
        
        U_G_U = U.mT @ G @ U
        P = K * U_G_U
        
        # Ionescu Unstable Formulation
        # F_ij = (sqrt(li) - sqrt(lj)) * K
        sqrt_L_row = L_row.sqrt()
        F_ij = (sqrt_L - sqrt_L_row) * K 
        grad_rotated = F_ij * U_G_U 
        
        # Diagonal fix
        # 对角线部分单独处理: f'(lambda) * diag(U^T G U)
        grad_diag = (0.5 / L.sqrt()).unsqueeze(-1) * U_G_U.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        # 将对角线填回矩阵
        grad_rotated = grad_rotated.diagonal_scatter(grad_diag.squeeze(-1), dim1=-2, dim2=-1)
        
        grad_input = U @ grad_rotated @ U.mT
        return grad_input

# ==============================================================================
# 3. Baseline 2: Unified (Stable DK/B) - 【已修复】
# ==============================================================================
class DarleySqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        L, U = torch.linalg.eigh(A)
        # 【修复】使用 diag_embed 支持 Batch
        Y = U @ torch.diag_embed(L.clamp(min=1e-12).sqrt()) @ U.mT
        ctx.save_for_backward(L, U)
        return Y

    @staticmethod
    def backward(ctx, G):
        L, U = ctx.saved_tensors
        
        # 【修复】安全的 Batch 广播
        L_col = L.unsqueeze(-1) # [B, N, 1]
        L_row = L.unsqueeze(-2) # [B, 1, N]
        
        # Robust Divided Differences
        num = L_col.sqrt() - L_row.sqrt()
        denom = L_col - L_row
        
        # Mask for small gaps
        mask = torch.abs(denom) < 1e-6
        
        # 1. 计算差分项 (避免除以0)
        term_diff = num / (denom + 1e-20) 
        
        # 2. 计算导数项 f'(L)
        term_deriv = 0.5 / L_col.sqrt()
        
        # 3. 组合: mask 为 True 选导数，False 选差分
        F = torch.where(mask, term_deriv, term_diff)
        
        grad_rotated = F * (U.mT @ G @ U)
        return U @ grad_rotated @ U.mT

# ==============================================================================
# 4. Ours: Newton-Schulz + CG (SVD-Free)
# ==============================================================================
class NewtonSchulzSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, num_iters=15):
        # Batch-wise Norm
        normA = torch.norm(A, p='fro', dim=(-2, -1), keepdim=True)
        Y = A / normA
        Z = torch.eye(A.shape[-1], device=A.device).unsqueeze(0).expand_as(A)
        
        for _ in range(num_iters):
            T = 0.5 * (3.0 * torch.eye(A.shape[-1], device=A.device) - Z @ Y)
            Y = Y @ T
            Z = T @ Z
            
        Y = Y * normA.sqrt()
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, G):
        Y, = ctx.saved_tensors
        
        # Implicit Differentiation: Solve XY + YX = G
        def mat_op(X): return X @ Y + Y @ X
        
        X = torch.zeros_like(G)
        R = G - mat_op(X)
        P = R.clone()
        rsold = torch.sum(R * R, dim=(-2, -1), keepdim=True)
        
        for _ in range(50): 
            AP = mat_op(P)
            alpha = rsold / (torch.sum(P * AP, dim=(-2, -1), keepdim=True) + 1e-20)
            X = X + alpha * P
            R = R - alpha * AP
            rsnew = torch.sum(R * R, dim=(-2, -1), keepdim=True)
            if rsnew.max() < 1e-10: break
            P = R + (rsnew / rsold) * P
            rsold = rsnew
            
        return X, None

# ==============================================================================
# Main Experiment
# ==============================================================================
def run_experiments():
    dim = 64 
    gaps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] 
    
    err_ionescu = []
    err_darley = []
    err_ours = []
    
    print("Starting Stability Experiment...")
    for gap in tqdm(gaps):
        # 1. Construct Matrix with Specific Gap (Unbatched for stability)
        eigs = torch.linspace(2, 10, dim, device=device).double()
        eigs[0] = 1.0
        eigs[1] = 1.0 + gap
        
        Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device, dtype=torch.double))
        A_double = Q @ torch.diag(eigs) @ Q.T
        A = A_double.float().unsqueeze(0).requires_grad_(True) # [1, 64, 64]
        
        G = torch.randn_like(A)
        G = (G + G.mT) / 2
        
        # 2. Get Ground Truth
        grad_gt = get_ground_truth_grad(A, G)
        gt_norm = torch.norm(grad_gt)
        
        # 3. Test Baseline 1 (Ionescu)
        A.grad = None
        try:
            Y = IonescuSqrt.apply(A)
            Y.backward(G)
            err = torch.norm(A.grad - grad_gt) / gt_norm
            err_ionescu.append(min(err.item(), 100.0)) 
        except:
            err_ionescu.append(100.0)
            
        # 4. Test Baseline 2 (Unified)
        A.grad = None
        Y = DarleySqrt.apply(A)
        Y.backward(G)
        err = torch.norm(A.grad - grad_gt) / gt_norm
        err_darley.append(err.item())
        
        # 5. Test Ours
        A.grad = None
        Y = NewtonSchulzSqrt.apply(A)
        Y.backward(G)
        err = torch.norm(A.grad - grad_gt) / gt_norm
        err_ours.append(err.item())

    # Plotting Stability
    plt.figure(figsize=(8, 6))
    plt.loglog(gaps, err_ionescu, 'r-o', label='Baseline 1 (Ionescu)', linewidth=2)
    plt.loglog(gaps, err_darley, 'g--s', label='Baseline 2 (Unified)', linewidth=2)
    plt.loglog(gaps, err_ours, 'b-^', label='Ours (SVD-Free NS)', linewidth=2)
    
    plt.gca().invert_xaxis() 
    plt.xlabel(r'Eigenvalue Gap $\delta$')
    plt.ylabel('Relative Gradient Error')
    plt.title('Gradient Stability under Degeneracy')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Stability.png')
    print("Stability plot saved.")
    
    # ---------------------------------------------------------
    # Speed Experiment
    # ---------------------------------------------------------
    print("Starting Speed Experiment...")
    batch_sizes = [32, 64, 128, 256, 512]
    t_unified = []
    t_ours = []
    
    for bs in tqdm(batch_sizes):
        A = torch.randn(bs, 64, 64, device=device)
        A = A @ A.mT + 1e-2 * torch.eye(64, device=device).unsqueeze(0)
        A.requires_grad_(True)
        G = torch.randn_like(A)
        
        # Warmup
        DarleySqrt.apply(A)
        
        # Unified
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            Y = DarleySqrt.apply(A)
            Y.backward(G)
        torch.cuda.synchronize()
        t_unified.append((time.time() - start)*100) # ms
        
        # Ours
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            Y = NewtonSchulzSqrt.apply(A)
            Y.backward(G)
        torch.cuda.synchronize()
        t_ours.append((time.time() - start)*100) # ms

    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes, t_unified, 'r-o', label='Baseline (SVD-based)')
    plt.plot(batch_sizes, t_ours, 'b-^', label='Ours (SVD-Free NS)')
    plt.xlabel('Batch Size')
    plt.ylabel('Total Time (ms) for 10 iters')
    plt.title('Runtime Scalability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Speed.png')
    print("Speed plot saved.")

if __name__ == "__main__":
    run_experiments()