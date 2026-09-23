import torch
import torch.nn as nn
import numpy as np
import random
import os


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        #manual 手动的
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #让 cuDNN（NVIDIA 的加速库）使用确定性算法。同一输入每次输出完全一样。
        torch.backends.cudnn.deterministic = True
        #关闭自动寻找最快卷积算法的功能。因为 benchmark=True 会引入随机性，不利于复现。
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():#如果没 CUDA，检查 MPS（Apple Silicon 的 GPU 加速后端）。
        torch.mps.manual_seed(seed)
    print(f"Global seed set to {seed}")


#随机丢弃整个样本的某条路径（比如残差分支），常用于 Transformer、ConvNeXt 等结构。
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob#丢弃概率和保留概率
    """
    x.shape 是张量的形状，比如 (batch_size, channels, height, width)。

    x.shape[0] 是 batch 大小。

    (1,) * (x.ndim - 1)：如果 x.ndim=4，则 (1,1,1)。x.ndim 是维度数

    结果：一个形状 (batch_size, 1, 1, 1) 的 0/1 掩码。
    """
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    #.bernoulli_(keep_prob)：原地操作（后缀 _ 表示 in-place），用伯努利分布填充
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    """
    如果保留概率大于 0 且需要缩放，把掩码除以 keep_prob。

    为什么？假设 keep_prob=0.9，掩码为 1 的概率 0.9，为 0 的概率 0.1。期望是 0.9 * 1 + 0.1 * 0 = 0.9。除以 0.9 后，非零值变成 1/0.9 ≈ 1.111，期望变成 0.9 * 1.111 = 1。这叫保持期望不变，类似 Dropout 的 inverted dropout。

    .div_() 也是原地除法。
    """
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)    