import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ==========================================
# 1. 准备数据 (Ground Truth)
# ==========================================
def get_true_projection(x):
    # 真实的墙：x + y <= 1 (也就是 w=[1,1], b=1)
    w_true = torch.tensor([1.0, 1.0])
    b_true = torch.tensor([1.0])
    w_norm_sq = (w_true ** 2).sum()
    
    # 投影逻辑：如果点在墙外 (x·w > b)，就投影回墙上
    val = x @ w_true - b_true
    violation = F.relu(val) # 只取正数部分（违反量）
    
    # 投影公式: x_new = x - (violation * w) / ||w||^2
    return x - (violation.view(-1, 1) * w_true) / w_norm_sq

torch.manual_seed(42)
# 生成训练数据：我们让数据分布得广一点，甚至有一部分在"真墙"外面
# 真墙在 x+y=1，我们生成 [-3, 3] 范围内的点
X_train = torch.randn(300, 2) * 2 
Y_train = get_true_projection(X_train)

# ==========================================
# 2. 定义模型 (OptNet vs MLP)
# ==========================================

# --- 模型 A: OptNet (学习约束逻辑) ---
class RobustOptLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 【关键调整 1】：初始化
        # 我们把墙的截距 b 设为 -3.0。
        # 真实的墙是 b=1.0 (在原点右上方)。
        # 现在的墙 b=-3.0 (在原点左下方)。
        # 这意味着绝大多数数据点一开始都在"墙外" (violating constraints)。
        # 这样能保证一开始就有巨大的梯度，强迫参数迅速更新。
        self.w = nn.Parameter(torch.tensor([0.5, 0.5])) 
        self.b = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, x):
        w_norm_sq = (self.w ** 2).sum()
        
        # 计算违反程度
        val = x @ self.w - self.b
        violation = F.relu(val)
        
        # 投影
        return x - (violation.view(-1, 1) * self.w) / (w_norm_sq + 1e-6)

# --- 模型 B: MLP (传统神经网络) ---
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 增加一点容量，看看能不能拟合得更好
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 输出也是坐标
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 训练过程
# ==========================================
opt_model = RobustOptLayer()
mlp_model = BaselineMLP()

# 【关键调整 2】：优化器参数
# OptNet 的参数物理意义明确，可以用较大的学习率 (0.05)
# MLP 是非凸的黑盒，学习率小一点更稳定 (0.01)
opt_optimizer = optim.Adam(opt_model.parameters(), lr=0.05)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

criterion = nn.MSELoss()

print("开始训练对比实验...")
epochs = 500 # 【关键调整 3】：增加训练轮数
loss_history_opt = []
loss_history_mlp = []

for epoch in range(epochs):
    # 1. 训练 OptNet
    opt_optimizer.zero_grad()
    opt_pred = opt_model(X_train)
    loss_opt = criterion(opt_pred, Y_train)
    loss_opt.backward()
    opt_optimizer.step()
    loss_history_opt.append(loss_opt.item())
    
    # 2. 训练 MLP
    mlp_optimizer.zero_grad()
    mlp_pred = mlp_model(X_train)
    loss_mlp = criterion(mlp_pred, Y_train)
    loss_mlp.backward()
    mlp_optimizer.step()
    loss_history_mlp.append(loss_mlp.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: OptNet Loss={loss_opt.item():.5f}, MLP Loss={loss_mlp.item():.5f}")

print(f"训练结束.\nOptNet 最终参数: w={opt_model.w.data.numpy()}, b={opt_model.b.data.item():.2f}")
print(f"真实参数: w=[1.0, 1.0], b=1.0")

# ==========================================
# 4. 可视化对比 (生成高清大图)
# ==========================================
def plot_line(w, b, color, label, style='-', alpha=1.0, linewidth=2):
    w = w.detach().numpy()
    b = b.detach().numpy()
    # 扩大画图范围以免线断掉
    x_vals = np.linspace(-6, 6, 200)
    
    if abs(w[1]) > 1e-5:
        y_vals = (b - w[0] * x_vals) / w[1]
        plt.plot(x_vals, y_vals, color=color, label=label, linestyle=style, alpha=alpha, linewidth=linewidth)
    else:
        plt.axvline(x=b/w[0], color=color, label=label, linestyle=style, alpha=alpha, linewidth=linewidth)

plt.figure(figsize=(16, 7))

# --- 测试数据 (生成一圈新数据来测试泛化) ---
# 这是一个"圆环"形状的数据，很多点都在墙外，更能看出投影效果
theta = torch.linspace(0, 2 * np.pi, 100)
r = 3.5
X_test = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
# 加上一些随机噪点
X_test = torch.cat([X_test, torch.randn(50, 2)*3], dim=0)

# ---------------------------
# 子图 1: OptNet
# ---------------------------
plt.subplot(1, 2, 1)
plt.title("OptNet: Learned Constraint (Logic)", fontsize=14)

# 1. 画真实的墙 (背景)
plot_line(torch.tensor([1.0, 1.0]), torch.tensor([1.0]), 'blue', 'True Wall', linewidth=10, alpha=0.2)

# 2. 画 OptNet 学到的墙
plot_line(opt_model.w, opt_model.b, 'red', 'Learned Wall', '-', linewidth=3)

# 3. 画输入点 (灰色)
plt.scatter(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.3, label='Input Points')

# 4. 画 OptNet 的输出 (红色叉)
opt_out = opt_model(X_test).detach()
plt.scatter(opt_out[:, 0], opt_out[:, 1], c='red', marker='x', s=40, label='OptNet Output')

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)

# ---------------------------
# 子图 2: MLP
# ---------------------------
plt.subplot(1, 2, 2)
plt.title("MLP: Function Approximation (Memory)", fontsize=14)

# 1. 画真实的墙 (作为参考)
plot_line(torch.tensor([1.0, 1.0]), torch.tensor([1.0]), 'blue', 'True Wall', linewidth=10, alpha=0.2)

# 2. 画输入点 (灰色)
plt.scatter(X_test[:, 0], X_test[:, 1], c='gray', alpha=0.3, label='Input Points')

# 3. 画 MLP 的输出 (绿色三角)
mlp_out = mlp_model(X_test).detach()
plt.scatter(mlp_out[:, 0], mlp_out[:, 1], c='green', marker='^', s=40, label='MLP Output')

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('optnet_vs_mlp_improved.png')