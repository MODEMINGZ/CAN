import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DropPath
# 注意：这里直接导入 clifford_thrust，没有 try/except 回退
# 说明这个模块假定你已经安装了 CUDA 加速内核
from clifford_thrust import LayerNorm2d, CliffordInteraction


class gFFN(nn.Module):
    """
    Geometric Feed-Forward Network (gFFN)

    用 Clifford 几何交互替代标准 MLP 的独立模块。
    可以作为即插即用组件，放进各种骨干网络里。

    参数:
        dim (int): 特征维度（通道数）。
        cli_mode (str): 'full', 'wedge', 'inner'，控制用内积/外积。
        ctx_mode (str): 'diff'（拉普拉斯差分）或 'abs'。
        gffn_mode (str):
            - 'l': 仅局部（卷积上下文）
            - 'g': 仅全局（全局平均上下文）
            - 'h': 混合（局部 + beta * 全局）
        shifts (list): 通道移位的量。
    """
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', gffn_mode='h',
                 shifts=[1, 2, 4], drop_path=0., init_values=1e-5, enable_cuda=False):
        super().__init__()

        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.gffn_mode = gffn_mode.lower()  # 统一转小写，防止传入 'L'/'H' 这种

        # 归一化层，直接来自编译好的 CUDA 内核
        self.norm = LayerNorm2d(dim)
        # 把 CliffordInteraction 赋给一个局部变量，方便下面复用
        InteractionLayer = CliffordInteraction

        # 从归一化特征生成“状态” z_state，1x1 卷积
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)

        # 如果模式是局部或混合，创建局部上下文分支
        if self.gffn_mode in ['l', 'h']:
            # 两层深度可分离 3x3 卷积 + BN + SiLU，感受野 5x5
            self.get_context_local = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU()
            )
            # 局部几何交互模块
            self.inter_local = InteractionLayer(dim, cli_mode=cli_mode, ctx_mode=ctx_mode, shifts=shifts)
        else:
            # 只用全局模式时，不创建局部分支
            self.get_context_local = None
            self.inter_local = None

        # 如果模式是全局或混合，创建全局交互模块
        if self.gffn_mode in ['g', 'h']:
            # 全局交互固定用 full 模式 + shifts=[1,2]
            self.inter_global = InteractionLayer(dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2])
        else:
            self.inter_global = None

        # 混合模式下，用一个可学习的 beta 平衡局部和全局
        if self.gffn_mode == 'h':
            # 初始值 0.5，表示一开始局部和全局各占一半
            self.beta = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        # 门控：拼接 x_ln 和 g_feat_total 后投影回 dim
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        # gamma：残差缩放，初始值很小（1e-5），让网络初始接近恒等映射
        self.gamma = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        # DropPath 正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x                     # 残差连接保存输入
        x_ln = self.norm(x)              # 归一化
        z_state = self.get_state(x_ln)   # 生成状态
        g_feat_total = None              # 先占位，后面根据模式赋值

        # --- 局部分支 ---
        if self.get_context_local is not None:
            z_ctx_local = self.get_context_local(x_ln)        # 局部上下文
            g_feat_local = self.inter_local(z_state, z_ctx_local)  # 局部几何交互

        # --- 全局分支 ---
        if self.inter_global is not None:
            # 对空间维求均值，得到全局上下文，再扩展到原尺寸
            z_ctx_global = x_ln.mean(dim=[-2, -1], keepdim=True).expand_as(x_ln)
            g_feat_global = self.inter_global(z_state, z_ctx_global)  # 全局几何交互

        # --- 根据模式合并 ---
        if self.gffn_mode == 'l':
            g_feat_total = g_feat_local                        # 只用局部
        elif self.gffn_mode == 'g':
            g_feat_total = g_feat_global                       # 只用全局
        elif self.gffn_mode == 'h':
            # 混合：局部 + beta * 全局。beta 可学习，网络自己决定权重
            g_feat_total = g_feat_local + self.beta * g_feat_global

        # --- 门控融合 ---
        combined = torch.cat([x_ln, g_feat_total], dim=1)   # 拼接，通道数 2*dim
        gate = torch.sigmoid(self.gate_fc(combined))        # 门控信号 0~1

        # --- 混合与残差 ---
        x_mixed = F.silu(x_ln) + gate * g_feat_total        # 主干 + 门控几何信息
        x = shortcut + self.drop_path(self.gamma * x_mixed)  # 残差连接

        return x