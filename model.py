import torch  # 导入 PyTorch 主库，提供张量(Tensor)和自动求导等
import torch.nn as nn  # 导入神经网络模块，通常用 nn 别名，包含层、损失函数等
import torch.nn.functional as F  # 导入函数式接口，如 F.silu、F.relu 等，无需实例化层
from utils import DropPath  # 从本项目的 utils.py 导入 DropPath 类（随机深度正则化）

# 尝试导入编译好的 CUDA 加速内核（来自 clifford_thrust 包）
# 这个包不是标准库，需要单独编译安装，官方针对特定 GPU 优化
try:
    from clifford_thrust import LayerNorm2d, CliffordInteraction
    print("✅ Successfully loaded accelerated Clifford kernels (CUDA).")
    has_acceleration = True  # 标记：成功加载加速内核
except ImportError:
    # 如果导入失败（没安装或编译失败），就走纯 PyTorch 实现
    print("⚠️ 'clifford_thrust' not found. Falling back to pure PyTorch implementation (Slower).")
    has_acceleration = False  # 标记：没有加速内核


class LayerNorm2d_PyTorch(nn.Module):
    # 自定义的二维层归一化（Layer Normalization），纯 PyTorch 实现
    # 对每个样本、每个空间位置，在通道维上做归一化
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()  # 调用父类 nn.Module 的初始化，必须写
        # 可学习参数：缩放 gamma 和偏移 beta，形状为 (num_channels,)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps  # 防止除零的小常数

    def forward(self, x):
        # x 形状假设为 (B, C, H, W)
        # 在第 1 维（通道维）上求均值，keepdim=True 保持维度，方便广播
        u = x.mean(1, keepdim=True)  # 形状 (B, 1, H, W)
        # 计算方差：先减去均值，平方，再在通道维求均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 形状 (B, 1, H, W)
        # 归一化：(x - 均值) / sqrt(方差 + eps)
        x = (x - u) / torch.sqrt(s + self.eps)
        # 应用可学习的缩放和偏移
        # self.weight[:, None, None] 把 (C,) 变成 (C, 1, 1)，以便与 (B, C, H, W) 广播
        """
        为什么要加这一步？因为强行把所有层都变成均值 0 方差 1 可能损失表达能力。
        加两个可学习参数，网络可以学到“我其实想要均值 0.5、方差 2”这样的分布。
        """
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CliffordInteraction_PyTorch(nn.Module):
    """
    纯 PyTorch 实现的 Clifford 交互模块。
    参数:
        cli_mode: 'full', 'wedge', 'inner'  # 选择使用哪种几何积成分
        ctx_mode:
            - 'diff': C = C_local - H (离散拉普拉斯，即 z2 - z1)
            - 'abs' : C = C_local
            - 'others': 待添加
    """
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2]):
        super().__init__()
        self.dim = dim  # 通道数，向量数
        """
        (0.2, -1.3, 0.8, 2.1)
        0.2·e₁ + (-1.3)·e₂ + 0.8·e₃ + 2.1·e
        """
        self.cli_mode = cli_mode  # 交互模式
        self.ctx_mode = ctx_mode  # 上下文模式
        self.act = nn.SiLU()  # SiLU 激活函数，也叫 Swish

        self.shifts = shifts  # 要使用的通道位移量，例如 [1,2]
        # 过滤掉大于等于通道数的位移，防止 roll 后重叠无意义
        self.shifts = [s for s in self.shifts if s < dim]

        # 每个位移会产生一个分支，每个分支通道数为 dim
        self.branch_dim = dim * len(self.shifts)

        self.proj_ = None  # 先占位

        if self.cli_mode == 'full':
            # 如果 full，同时使用 wedge 和 inner，所以拼接后的通道数是 branch_dim * 2
            cat_dim = self.branch_dim * 2
        elif self.cli_mode in ['wedge', 'inner']:
            # 只用一种，通道数就是 branch_dim
            cat_dim = self.branch_dim
        else:
            raise ValueError(f"Invalid cli_mode: {cli_mode}")
        # 用 1x1 卷积把拼接后的特征投影回 dim 通道
        self.proj_ = nn.Conv2d(cat_dim, dim, kernel_size=1)

    def forward(self, z1, z2):
        # z1 来自 get_state，z2 来自 get_context_local
        if self.ctx_mode == 'diff':
            C = z2 - z1  # 差分：局部上下文减去状态，类似离散拉普拉斯
        elif self.ctx_mode == 'abs':
            C = z2  # 直接用上下文
        # 注意：如果 ctx_mode 是其他值，C 未定义，会报错

        feats = []  # 存放各分支特征
        for s in self.shifts:
            # torch.roll 在通道维（dim=1）上循环移位 s 个位置
            #“通道 i”和“通道 i+1”发生交互
            C_shifted = torch.roll(C, shifts=s, dims=1)
            if self.cli_mode in ['wedge', 'full']:
                # 楔积（外积）的离散近似：z1 * C_shifted - C * z1_shifted
                z1_shifted = torch.roll(z1, shifts=s, dims=1)
                wedge = z1 * C_shifted - C * z1_shifted
                feats.append(wedge)
            if self.cli_mode in ['inner', 'full']:
                # 内积（点积）的离散近似：先逐元素相乘，再过激活函数
                inner = self.act(z1 * C_shifted)
                feats.append(inner)
        # 把所有分支在通道维拼接
        x_ = torch.cat(feats, dim=1)
        # 1x1 卷积融合并投影回 dim
        out = self.proj_(x_)
        return out
"""
把每个空间位置想象成一个小城市，有 128 个指标（人口、GDP、温度、湿度……）。

“通道”就是这 128 个指标。

这个城市的状态就是一个 128 维向量。

z1 是城市当前状态，z2 是周边城市的平均状态。

C = z2 - z1 是“本地和周边的差异”。

通道移位 = 把指标列表错开一位，让“人口”和“GDP”配对，而不是“人口”和“人口”配对。

内积 = 看两个城市在哪些指标上相似。

外积 = 看两个城市在哪些指标上形成“反差结构”。

这就是为什么作者说“把通道当作向量的分量”——每个通道是一个抽象维度，整个特征向量描述一个位置的状态。
"""


class CliffordAlgebraBlock(nn.Module):
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2], enable_gFFNG=False,
                 num_heads=1, mlp_ratio=0., drop=0., drop_path=0.1, init_values=1e-5, enable_cuda=False):
        super().__init__()

        """
        z_state：这个位置“自己是什么”,z1

        z_context_local：这个位置“周围是什么”,z2
        """
        # 从输入特征生成“状态” z_state，用 1x1 卷积
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        # 生成“局部上下文” z_context_local：两个 3x3 深度可分离卷积 + BN + SiLU
        self.get_context_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),  # 深度卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),  # 再一个深度卷积
            nn.BatchNorm2d(dim),  # 批归一化
            nn.SiLU()             # 激活
        )
        self.enable_gFFNG = enable_gFFNG  # 是否启用全局几何前馈网络（gFFNG）

        if enable_cuda:
            # 如果启用 CUDA 加速，使用编译好的 LayerNorm2d 和 CliffordInteraction
            self.norm = LayerNorm2d(dim)
            self.clifford_interaction_local = CliffordInteraction(dim, cli_mode, ctx_mode, shifts)
            if self.enable_gFFNG:
                # 全局交互使用 full 模式，固定的 shifts=[1,2]
                self.clifford_interaction_global = CliffordInteraction(dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2])
        else:
            # 否则使用纯 PyTorch 实现
            self.norm = LayerNorm2d_PyTorch(dim)
            self.clifford_interaction_local = CliffordInteraction_PyTorch(dim, cli_mode, ctx_mode, shifts)
            if self.enable_gFFNG:
                self.clifford_interaction_global = CliffordInteraction_PyTorch(dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2])

        self.act = nn.SiLU()  # 激活函数
        # 门控：把 x_ln 和 g_feat_total 拼接后，用 1x1 卷积生成门控信号
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        # 可学习的缩放因子 gamma，形状 (1, dim, 1, 1)，初始值很小 (1e-5)，让残差分支初始接近 0
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        # DropPath 正则化，如果 drop_path > 0 则使用，否则用 Identity 不操作
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  # 残差连接：保存输入
        x_ln = self.norm(x)  # 层归一化

        # 生成状态和局部上下文
        z_state = self.get_state(x_ln)
        z_context_local = self.get_context_local(x_ln)
        # 局部 Clifford 交互
        g_feat_total = self.clifford_interaction_local(z_state, z_context_local)

        if self.enable_gFFNG:
            # 全局上下文：对空间维求均值，然后扩展到原尺寸
            z_context_global = x_ln.mean(dim=[-2, -1], keepdim=True).expand_as(x_ln)
            # 全局 Clifford 交互
            g_feat_global = self.clifford_interaction_global(z_state, z_context_global)
            # 合并局部和全局特征
            g_feat_total = g_feat_total + g_feat_global

        # 门控机制：拼接原始归一化特征和几何交互特征
        combined = torch.cat([x_ln, g_feat_total], dim=1)
        gate = torch.sigmoid(self.gate_fc(combined))  # sigmoid 得到 0~1 的门控值
        # 混合：对 x_ln 过 SiLU，再加上门控后的几何特征
        x_mixed = F.silu(x_ln) + gate * g_feat_total
        # 残差连接：shortcut + DropPath(gamma * x_mixed)
        x = shortcut + self.drop_path(self.gamma * x_mixed)

        return x


class GeometricStem(nn.Module):
    # 负责把原始输入变成网络主体能处理的形式
    #原始图像是 (B, 3, 32, 32),但后面的 CliffordAlgebraBlock 期待输入是 (B, 128, 16, 16)
    def __init__(self, in_chans=3, embed_dim=128, patch_size=2):
        super().__init__()
        # patch_size指“降采样倍数”
        if patch_size == 1:
            # 不降采样，用两层 3x3 卷积
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=1, padding=1, bias=False),
            )
        elif patch_size == 2:
            # 用 stride=2 的 3x3 卷积降采样 2 倍
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        elif patch_size == 4:
            # 用两层 stride=2 卷积，总共降采样 4 倍
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            )
        else:
            # 其他 patch_size，直接用对应大小的卷积核和步长
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.BatchNorm2d(embed_dim)  # 最后接 BN

    def forward(self, x):
        x = self.proj(x)  # 投影/降采样
        x = self.norm(x)  # 归一化
        return x


class CliffordNet(nn.Module):
    def __init__(self, num_classes=10, patch_size=4, embed_dim=128, cli_mode='full', ctx_mode='diff',
                 shifts=[1, 2], depth=6, num_heads=1, mlp_ratio=0., drop_rate=0.,
                 drop_path_rate=0.1, enable_cuda=False):
        super().__init__()

        # 图像首先经过 GeometricStem 进行分块嵌入/降采样
        self.patch_embed = GeometricStem(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)
        # dpr: drop path rate 的列表，从 0 到 drop_path_rate 线性增加
        # torch.linspace(0, drop_path_rate, depth) 生成 depth 个等间距值
        # .item() 把每个张量转成 Python 浮点数
        #torch.linspace(0, 0.1, 6) 生成 [0, 0.02, 0.04, 0.06, 0.08, 0.1]。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 构建 depth 个 CliffordAlgebraBlock
        self.blocks = nn.ModuleList([
            CliffordAlgebraBlock(
                dim=embed_dim,
                cli_mode=cli_mode,
                ctx_mode=ctx_mode,
                shifts=shifts,
                num_heads=num_heads,
                drop_path=dpr[i],  # 每个块使用不同的 drop_path 概率
                enable_cuda=enable_cuda,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)  # 最后的层归一化，用于分类头
        self.head = nn.Linear(embed_dim, num_classes)  # 线性分类头
        self.apply(self._init_weights)  # 递归地对所有子模块应用权重初始化

    def _init_weights(self, m):
        # 自定义初始化函数
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # 对卷积和线性层，用截断正态分布初始化权重，标准差 0.02
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            # 对 LayerNorm，权重初始化为 1，偏置为 0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)  # 分块嵌入
        for block in self.blocks:
            x = block(x)  # 依次通过每个 CliffordAlgebraBlock
        return x

    def forward(self, x):
        x = self.forward_features(x)  # 提取特征
        x = x.mean(dim=[-2, -1])  # 全局平均池化：对空间维 (H, W) 求均值，形状从 (B,C,H,W) 变成 (B,C)
        x = self.norm(x)  # 层归一化
        x = self.head(x)  # 分类头，输出 (B, num_classes)
        return x