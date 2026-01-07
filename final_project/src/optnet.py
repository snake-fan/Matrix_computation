import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import cvxopt
from cvxopt import solvers
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ==========================================
# 1. 核心算子 (保持数学逻辑不变)
# ==========================================
solvers.options['show_progress'] = False

def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.double)

def to_tensor(numpy_array, dtype=torch.float64):
    return torch.from_numpy(numpy_array).type(dtype)

class OptNetFunction(Function):
    @staticmethod
    def forward(ctx, Q, p, G, h):
        n_vars = Q.shape[0]
        Q_np, p_np = to_numpy(Q), to_numpy(p)
        G_np, h_np = to_numpy(G), to_numpy(h)
        args = [cvxopt.matrix(Q_np), cvxopt.matrix(p_np), 
                cvxopt.matrix(G_np), cvxopt.matrix(h_np)]
        try:
            sol = solvers.qp(*args)
            status = sol['status']
        except ValueError:
            status = 'failed'

        if status != 'optimal':
            z_star = torch.zeros(n_vars, dtype=torch.float64)
            lambda_star = torch.zeros(G.shape[0], dtype=torch.float64)
        else:
            z_star = to_tensor(np.array(sol['x'])).view(-1)
            lambda_star = to_tensor(np.array(sol['z'])).view(-1)

        ctx.save_for_backward(z_star, lambda_star, Q, p, G, h)
        return z_star

    @staticmethod
    def backward(ctx, grad_z):
        z_star, lam, Q, p, G, h = ctx.saved_tensors
        n = z_star.size(0)
        n_ineq = G.size(0)
        slacks = G @ z_star - h
        
        # KKT 矩阵构建
        block_1_2 = G.t() @ torch.diag(lam)
        row1 = torch.cat([Q, block_1_2], dim=1)
        row2 = torch.cat([G, torch.diag(slacks)], dim=1)
        KKT_matrix = torch.cat([row1, row2], dim=0)
        KKT_matrix += 1e-6 * torch.eye(KKT_matrix.shape[0], dtype=torch.float64)

        # 求解方程组
        rhs = torch.cat([-grad_z, torch.zeros(n_ineq, dtype=torch.float64)])
        try:
            adjoint_vec = torch.linalg.solve(KKT_matrix.t(), rhs)
        except RuntimeError:
            adjoint_vec = torch.zeros_like(rhs)
        
        d_z_tilde = adjoint_vec[:n]      
        d_lam_tilde = adjoint_vec[n:]    

        # 梯度计算
        grad_Q = 0.5 * (torch.outer(d_z_tilde, z_star) + torch.outer(z_star, d_z_tilde))
        grad_p = d_z_tilde
        grad_G = torch.outer(lam * d_lam_tilde, z_star) + torch.outer(lam, d_z_tilde)
        grad_h = - lam * d_lam_tilde

        return grad_Q, grad_p, grad_G, grad_h

# ==========================================
# 2. 模型定义 (公平初始化)
# ==========================================

class OptNetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 【公平初始化】：使用标准正态分布随机初始化
        # 没有任何人为设定的 -3.0 或 trick
        # 为了防止一开始就无解(Infeasible)，我们只保证 h 初始为正数 (原点可行)
        # 这就像初始化 Bias 为 0 或 0.1 一样，是标准操作
        self.G = nn.Parameter(torch.randn(1, 2).double())
        self.h = nn.Parameter(torch.rand(1).double() + 0.1) 
        self.Q = torch.eye(2, dtype=torch.float64) 

    def forward(self, x):
        outputs = []
        p_batch = -x 
        for i in range(x.size(0)):
            z = OptNetFunction.apply(self.Q, p_batch[i], self.G, self.h)
            outputs.append(z)
        return torch.stack(outputs)

class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 标准 MLP：3层，ReLU激活
        # 参数量比 OptNet 多得多，给它足够的容量去拟合
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).double()
        
        # 【公平初始化】：PyTorch 默认的 Kaiming/Xavier 初始化

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 训练与可视化
# ==========================================
def get_ground_truth(x):
    # 真实规则: z1 + z2 <= 1
    outputs = []
    w_true = torch.tensor([1.0, 1.0], dtype=torch.float64)
    for i in range(x.size(0)):
        xi = x[i]
        if xi.sum() <= 1.0:
            outputs.append(xi) # 墙内保持不动
        else:
            diff = (xi.sum() - 1.0) / 2.0
            outputs.append(xi - diff) # 墙外投影
    return torch.stack(outputs)

def train_fair():
    # 1. 数据生成：均匀分布 (Uniform)
    # 覆盖 [-3, 3] 区域，公平地包含墙内和墙外数据
    torch.manual_seed(999)
    n_samples = 150
    X_train = (torch.rand(n_samples, 2).double() * 6) - 3.0
    Y_train = get_ground_truth(X_train)

    opt_model = OptNetLayer()
    mlp_model = BaselineMLP()

    # 优化器
    opt_optim = torch.optim.SGD(opt_model.parameters(), lr=0.05, momentum=0.9)
    mlp_optim = torch.optim.Adam(mlp_model.parameters(), lr=0.01) # Adam 对 MLP 收敛更快
    
    criterion = nn.MSELoss()

    print("=== 开始公平对比训练 ===")
    
    # 训练循环
    epochs = 100
    for epoch in range(epochs):
        # Train OptNet
        opt_optim.zero_grad()
        opt_pred = opt_model(X_train)
        loss_opt = criterion(opt_pred, Y_train)
        loss_opt.backward()
        opt_optim.step()
        
        # Train MLP
        mlp_optim.zero_grad()
        mlp_pred = mlp_model(X_train)
        loss_mlp = criterion(mlp_pred, Y_train)
        loss_mlp.backward()
        mlp_optim.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: OptNet Loss={loss_opt.item():.5f}, MLP Loss={loss_mlp.item():.5f}")

    # ==========================
    # 可视化 (绘制位移线)
    # ==========================
    # 生成测试网格
    grid_x = np.linspace(-3, 3, 15)
    grid_y = np.linspace(-3, 3, 15)
    xx, yy = np.meshgrid(grid_x, grid_y)
    X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).double()

    with torch.no_grad():
        opt_out = opt_model(X_test)
        mlp_out = mlp_model(X_test)

    # 绘图函数
    def plot_displacement(ax, model_name, X, Y_pred, G=None, h=None):
        ax.set_title(model_name)
        # 1. 画真实墙
        x_range = np.linspace(-3, 3, 100)
        ax.plot(x_range, 1.0 - x_range, 'b-', linewidth=4, alpha=0.3, label='True Wall')
        
        # 2. 画学到的墙 (仅 OptNet)
        if G is not None:
            G_np = G.detach().numpy().ravel()
            h_np = h.detach().numpy()
            if abs(G_np[1]) > 1e-4:
                y_pred_line = (h_np[0] - G_np[0] * x_range) / G_np[1]
                ax.plot(x_range, y_pred_line, 'r--', linewidth=2, label='Learned Wall')

        # 3. 绘制位移线段 (Input -> Output)
        # 这是一个非常关键的可视化，可以看到点是怎么移动的
        lines = []
        colors = []
        X_np = X.numpy()
        Y_np = Y_pred.numpy()
        
        for i in range(len(X_np)):
            lines.append([X_np[i], Y_np[i]])
            # 如果位移很小，用灰色；如果位移大，用红色
            dist = np.linalg.norm(X_np[i] - Y_np[i])
            if dist < 0.05:
                colors.append('lightgray') # 不动点 (Safe)
            else:
                colors.append('red')       # 投影点 (Violating)
        
        lc = LineCollection(lines, colors=colors, linewidths=1, alpha=0.7)
        ax.add_collection(lc)
        
        # 画端点
        ax.scatter(X_np[:, 0], X_np[:, 1], s=10, c='gray', alpha=0.5, marker='.') # 起点
        # ax.scatter(Y_np[:, 0], Y_np[:, 1], s=20, c='black', marker='.') # 终点

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True)
        ax.legend(loc='lower left')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    plot_displacement(ax1, f"OptNet (Loss: {loss_opt.item():.5f})", 
                      X_test, opt_out, opt_model.G, opt_model.h)
    
    plot_displacement(ax2, f"MLP (Loss: {loss_mlp.item():.5f})", 
                      X_test, mlp_out)

    plt.tight_layout()
    plt.savefig('optnet_fair_comparison.png')
    print("公平对比图已保存为 optnet_fair_comparison.png")
    plt.show()

if __name__ == "__main__":
    train_fair()