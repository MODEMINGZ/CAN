from model import CliffordNet  # 从 model.py 导入主模型类


def gen_shifts(n):
    # 生成 n 个位移量：1, 2, 4, 8, 16, ...
    # 1 << i 是位运算左移，等价于 2 的 i 次方
    # i=0 时 1<<0 = 1
    # i=1 时 1<<1 = 2
    # i=2 时 1<<2 = 4
    # 返回一个列表，比如 n=3 时返回 [1, 2, 4]
    return [1 << i for i in range(n)]


def gen_shifts_fibonacci(n):
    # 生成斐波那契数列的位移量：1, 2, 3, 5, 8, ...
    # 注意：这个函数在本文件里没有被用到，是作者的备选方案
    a, b = 1, 2  # 初始两个数
    for _ in range(n):
        yield a       # 每次产出当前的 a
        a, b = b, a + b  # 更新：a 变成 b，b 变成 a+b
    # yield 让这个函数变成生成器，用的时候需要 list(...) 或 for 循环


def cliffordnet_12_2(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # 命名规则：cliffordnet_{深度}_{shifts个数}
    # 这个变体：12 层深，2 个 shifts
    # 注释里写 "Nano"，是作者的命名，表示这是一个小模型
    shifts = gen_shifts(2)  # [1, 2]
    return CliffordNet(
        enable_cuda=enable_cuda,      # 是否用 CUDA 加速内核
        num_classes=num_classes,      # 分类数，默认 CIFAR-100
        patch_size=patch_size,        # 降采样倍率，2 表示 32→16
        embed_dim=embed_dim,          # 通道数 128
        cli_mode='full',              # 同时用内积和外积
        ctx_mode='diff',              # 上下文用差分
        shifts=shifts,                # 通道移位量 [1, 2]
        depth=12,                     # 堆叠 12 个 CliffordAlgebraBlock
        drop_path_rate=0.3            # DropPath 最大概率 0.3
    )


def cliffordnet_12_3(num_classes=100, patch_size=1, embed_dim=160, enable_cuda=False):
    # 12 层深，3 个 shifts
    # 注意这个变体和其他的区别：patch_size=1（不降采样），embed_dim=160（更宽）
    shifts = gen_shifts(3)  # [1, 2, 4]
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,        # 1 表示不降采样，保持 32x32
        embed_dim=embed_dim,          # 160 比 128 更宽，参数量更大
        cli_mode='full',
        ctx_mode='diff',
        shifts=shifts,                # [1, 2, 4]
        depth=12,
        drop_path_rate=0.3
    )


def cliffordnet_12_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # 12 层深，5 个 shifts
    # 注释里写 "Lite"
    shifts = gen_shifts(5)  # [1, 2, 4, 8, 16]
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,        # 2，32→16
        embed_dim=embed_dim,          # 128
        cli_mode='full',              # 内积+外积都用
        ctx_mode='diff',
        shifts=shifts,                # 5 个移位，几何交互分支更多
        depth=12,
        drop_path_rate=0.3
    )


def cliffordnet_18_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # 18 层深，5 个 shifts
    # 比其他 12 层的更深
    shifts = gen_shifts(5)  # [1, 2, 4, 8, 16]
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        cli_mode='full',
        ctx_mode='diff',
        shifts=shifts,
        depth=18,                     # 18 层
        drop_path_rate=0.3
    )


def cliffordnet_32_3(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # 32 层深，3 个 shifts
    # 注释里写 "Small"
    shifts = gen_shifts(3)  # [1, 2, 4]
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        cli_mode='full',
        ctx_mode='diff',
        shifts=shifts,
        depth=32,                     # 32 层
        drop_path_rate=0.3
    )


def cliffordnet_32_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # 32 层深，5 个 shifts
    # 注释里写 "Small"
    # 注意：cli_mode='inner'，只用内积，不用外积
    shifts = gen_shifts(5)  # [1, 2, 4, 8, 16]
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        cli_mode='inner',             # 只用内积（相似性），不用外积
        ctx_mode='diff',
        shifts=shifts,
        depth=32,
        drop_path_rate=0.3
    )


def cliffordnet_64_5(num_classes=100, patch_size=2, embed_dim=128, enable_cuda=False):
    # 64 层深，5 个 shifts
    # 注释里写 "Deep"，是最大的变体
    # cli_mode='inner'，drop_path_rate=0.4（更深所以正则化更强）
    shifts = gen_shifts(5)  # [1, 2, 4, 8, 16]
    return CliffordNet(
        enable_cuda=enable_cuda,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        cli_mode='inner',             # 只用内积
        ctx_mode='diff',
        shifts=shifts,
        depth=64,                     # 64 层，非常深
        drop_path_rate=0.4            # 更深的网络需要更强的正则化
    )