import torch
import torch.nn as nn

from utils import DropPath
# 注意：这里也是直接导入，没有 try/except 回退
# 说明 model_hier.py 假定你已经装了 clifford_thrust
from clifford_thrust import LayerNorm2d, CliffordInteraction


class LayerNorm2d_PyTorch(nn.Module):
    # 和 model.py 里完全一样，纯 PyTorch 的 2D LayerNorm
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CliffordInteraction_PyTorch(nn.Module):
    # 和 model.py 里完全一样，纯 PyTorch 的几何交互
    def __init__(self, dim, cli_mode="full", ctx_mode="diff", shifts=(1, 2)):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.act = nn.SiLU()
        self.shifts = [s for s in shifts if s < dim]
        self.branch_dim = dim * len(self.shifts)

        if self.cli_mode == "full":
            cat_dim = self.branch_dim * 2
        elif self.cli_mode in ("wedge", "inner"):
            cat_dim = self.branch_dim
        else:
            raise ValueError(f"Invalid cli_mode: {cli_mode}")
        self.proj_ = nn.Conv2d(cat_dim, dim, kernel_size=1)

    def forward(self, z1, z2):
        if self.ctx_mode == "diff":
            c = z2 - z1
        elif self.ctx_mode == "abs":
            c = z2
        else:
            raise ValueError(f"Invalid ctx_mode: {self.ctx_mode}")

        feats = []
        for s in self.shifts:
            c_shifted = torch.roll(c, shifts=s, dims=1)
            if self.cli_mode in ("wedge", "full"):
                z1_shifted = torch.roll(z1, shifts=s, dims=1)
                wedge = z1 * c_shifted - c * z1_shifted
                feats.append(wedge)
            if self.cli_mode in ("inner", "full"):
                inner = self.act(z1 * c_shifted)
                feats.append(inner)
        x_ = torch.cat(feats, dim=1)
        out = self.proj_(x_)
        return out


class MultiScaleContext(nn.Module):
    # 多尺度上下文模块：用不同膨胀率的深度卷积捕捉不同范围的上下文
    # 注意：这个类在文件里被注释掉了（# self.get_context = MultiScaleContext(dim)）
    # 说明作者试过但最终没用，是备选方案
    def __init__(self, dim):
        super().__init__()
        # 膨胀率 1：感受野 3x3
        self.dw3_d1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        # 膨胀率 2：感受野 5x5
        self.dw3_d2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False)
        # 膨胀率 3：感受野 7x7
        self.dw3_d3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False)
        # 把三个尺度的特征拼接后融合
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.dw3_d1(x)
        x2 = self.dw3_d2(x)
        x3 = self.dw3_d3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CliffordAlgebraBlock(nn.Module):
    # 和 model.py 里的 Block 几乎一样，但更简洁
    # 区别：
    #   1. 没有 enable_gFFNG（全局交互分支）
    #   2. 只有局部上下文，且固定为两层深度卷积
    #   3. 参数命名略有不同（get_context 而不是 get_context_local）
    def __init__(
        self,
        dim,
        cli_mode="full",
        ctx_mode="diff",
        shifts=(1, 2),
        drop_path=0.1,
        init_values=1e-5,
        enable_cuda=False,
    ):
        super().__init__()
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        # self.get_context = MultiScaleContext(dim)  # 备选方案，被注释掉了

        if enable_cuda:
            self.norm = LayerNorm2d(dim)
            self.clifford_interaction = CliffordInteraction(dim, cli_mode, ctx_mode, shifts)
        else:
            self.norm = LayerNorm2d_PyTorch(dim)
            self.clifford_interaction = CliffordInteraction_PyTorch(dim, cli_mode, ctx_mode, shifts)

        self.act = nn.SiLU()
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x)
        z_state = self.get_state(x_ln)       # 状态
        z_context = self.get_context(x_ln)   # 局部上下文
        g_feat = self.clifford_interaction(z_state, z_context)  # 几何交互
        gate = torch.sigmoid(self.gate_fc(torch.cat([x_ln, g_feat], dim=1)))  # 门控
        x_mixed = self.act(x_ln) + gate * g_feat
        x_mixed = self.gamma * x_mixed
        x = shortcut + self.drop_path(x_mixed)
        return x


class GeometricStem(nn.Module):
    # 和 model.py 里完全一样，入口模块
    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        if patch_size == 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=1, padding=1, bias=False),
            )
        elif patch_size == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class StageDownsample(nn.Module):
    # 阶段之间的降采样模块：把特征图空间缩小 2 倍，通道数变成 out_dim
    # 支持三种模式：avgpool、conv、patch
    def __init__(self, in_dim, out_dim, mode="avgpool"):
        super().__init__()
        if mode == "avgpool":
            # 平均池化降采样，再用 1x1 卷积改通道数
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        elif mode == "conv":
            # 深度卷积 stride=2 降采样，再 1x1 改通道
            self.down = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.SiLU(),
            )
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        elif mode == "patch":
            # 用 2x2 stride=2 卷积，一次性完成降采样 + 改通道
            self.down = nn.Identity()
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Invalid downsample mode: {mode}")

    def forward(self, x):
        x = self.down(x)   # 空间降采样
        x = self.proj(x)   # 通道变换
        return x


class HierarchicalCliffordNet(nn.Module):
    """
    stage_depths: 每个阶段的 Block 数量，例如 (2, 2, 4)
    stage_dims:
        - None + dim_policy='double': [embed_dim, 2*embed_dim, 4*embed_dim, ...]
        - None + dim_policy='constant': [embed_dim] * num_stages
        - 显式列表/元组：每个阶段的通道数
    downsample_mode: 'avgpool', 'conv', 或 'patch'
    """

    def __init__(
        self,
        num_classes=100,
        in_chans=3,
        patch_size=1,
        embed_dim=32,
        cli_mode="full",
        ctx_mode="diff",
        shifts=(1, 2),
        stage_depths=(3, 4, 5),      # 三个阶段，分别堆 3、4、5 个 Block
        stage_dims=None,             # 每个阶段的通道数，None 则按 dim_policy 自动生成
        dim_policy="double",         # 通道翻倍策略
        downsample_mode="conv",      # 阶段间降采样方式
        drop_path_rate=0.1,
        enable_cuda=False,
    ):
        super().__init__()
        if len(stage_depths) == 0:
            raise ValueError("stage_depths must not be empty")

        # --- 决定每个阶段的通道数 ---
        if stage_dims is None:
            if dim_policy == "double":
                # 每个阶段通道翻倍：[32, 64, 128]
                stage_dims = [embed_dim * (2 ** i) for i in range(len(stage_depths))]
            elif dim_policy == "constant":
                # 所有阶段通道相同：[32, 32, 32]
                stage_dims = [embed_dim for _ in stage_depths]
            else:
                raise ValueError(f"Invalid dim_policy: {dim_policy}")
        else:
            if len(stage_dims) != len(stage_depths):
                raise ValueError("stage_dims must have the same length as stage_depths")
            stage_dims = list(stage_dims)

        self.stage_depths = list(stage_depths)
        self.stage_dims = stage_dims

        # --- 入口 Stem ---
        self.patch_embed = GeometricStem(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)

        # 如果 Stem 输出通道和第一阶段通道不一致，用 1x1 卷积对齐
        self.stem_proj = (
            nn.Conv2d(embed_dim, stage_dims[0], kernel_size=1) if embed_dim != stage_dims[0] else nn.Identity()
        )

        # --- DropPath 速率列表，所有阶段的所有 Block 共享一条线性递增曲线 ---
        total_blocks = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()       # 存放每个阶段的 Block 序列
        self.downsamples = nn.ModuleList()  # 存放阶段间的降采样模块

        dp_idx = 0  # 用于在 dpr 列表里按顺序取 drop_path 概率
        for stage_idx, (depth, dim) in enumerate(zip(stage_depths, stage_dims)):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    CliffordAlgebraBlock(
                        dim=dim,
                        cli_mode=cli_mode,
                        ctx_mode=ctx_mode,
                        shifts=shifts,
                        drop_path=dpr[dp_idx],
                        enable_cuda=enable_cuda,
                    )
                )
                dp_idx += 1
            # 把这一阶段的所有 Block 打包成一个 Sequential
            self.stages.append(nn.Sequential(*blocks))

            # 如果不是最后一个阶段，加一个降采样模块
            if stage_idx < len(stage_depths) - 1:
                self.downsamples.append(
                    StageDownsample(
                        in_dim=stage_dims[stage_idx],
                        out_dim=stage_dims[stage_idx + 1],
                        mode=downsample_mode,
                    )
                )

        # --- 分类头 ---
        self.norm = nn.LayerNorm(stage_dims[-1])
        self.head = nn.Linear(stage_dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 和 model.py 一样的初始化
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)       # Stem
        x = self.stem_proj(x)         # 通道对齐

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)              # 通过这一阶段的所有 Block
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)  # 降采样到下一阶段

        return x

    def forward(self, x):
        x = self.forward_features(x)  # 提取特征
        x = x.mean(dim=[-2, -1])      # 全局平均池化
        x = self.norm(x)              # LayerNorm
        x = self.head(x)              # 分类
        return x