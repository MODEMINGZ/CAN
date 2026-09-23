import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
import argparse
from model import CliffordNet
from networks import cliffordnet_12_2, cliffordnet_12_5, cliffordnet_32_3 
from utils import seed_everything
# from hybrid_model import clifford_hybrid_nano

# --- Configuration ---
"""
这个脚本做的是标准的 CIFAR-100 图像分类训练：

解析命令行参数（是否启用 CUDA 加速）

设置随机种子、选择设备

准备 CIFAR-100 数据（下载、增强、DataLoader）

初始化模型（cliffordnet_12_2，失败则回退到 CliffordNet）

定义损失函数、优化器、学习率调度器

循环训练 200 个 epoch，每个 epoch 训练一遍、评估一遍

打印总耗时



"""
@dataclass
class TrainingConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 200
    weight_decay: float = 0.1
    num_workers: int = 4 if torch.cuda.is_available() else 0
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Dataset params
    data_root: str = './data'
    random_erasing_prob: float = 0.25
    
    num_classes: int = 100
    patch_size: int = 2
    embed_dim: int = 128
    
    # Checkpoint
    save_path: str = 'cliffordnet_cifar100.pth'

# --- Utils ---
def get_device(device_str: str) -> torch.device:
    print(f"Using Device: {device_str.upper()}")
    return torch.device(device_str)


"""
统计可训练参数量。

p.numel()：张量元素个数。

p.requires_grad：是否可训练。通常所有参数都是 True。

sum(... for ...)：生成器表达式求和
"""
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Data Pipeline ---
"""
训练集增强：

RandomCrop(32, padding=4)：先四周填 4 像素，再随机裁回 32x32。模拟平移。

RandomHorizontalFlip()：随机水平翻转，默认概率 0.5。

AutoAugment(CIFAR10)：自动增强策略，来自 Google 的 AutoAugment，这里用 CIFAR10 的策略（虽然数据集是 CIFAR100，但图像尺寸一样，可以通用）。

ToTensor()：把 PIL 图像变成 torch.Tensor，形状从 (H,W,C) 变成 (C,H,W)，数值从 0-255 缩放到 0-1。

Normalize(mean, std)：按通道减均值除标准差。这三个均值/标准差是 CIFAR-100 的统计值。

RandomErasing(p=0.25)：随机选一块区域涂成随机值或 0，模拟遮挡。

测试集只做 ToTensor 和 Normalize，不做随机增强。

"""
def get_transforms(cfg: TrainingConfig):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=cfg.random_erasing_prob)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    return transform_train, transform_test

def get_dataloaders(cfg: TrainingConfig):
    print("Preparing Data...")
    train_transform, test_transform = get_transforms(cfg)
    
    trainset = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True, transform=test_transform
    )
    #pin_memory=True：把数据放在锁页内存，加速 CPU 到 GPU 的拷贝（有 CUDA 时有用）。
    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    
    return trainloader, testloader

# --- Training Engine ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()#把模型设为训练模式。
    #累计损失、正确数、总样本数。
    running_loss = 0.0
    correct = 0
    total = 0
    #包装 DataLoader，显示进度条。
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)

    #每次取一批。inputs 形状 (B,3,32,32)，labels 形状 (B,)。
    for inputs, labels in pbar:
        #把数据移到 GPU/CPU。
        inputs, labels = inputs.to(device), labels.to(device)

        #清空上一轮梯度。PyTorch 默认会累加梯度，所以每轮必须清零。
        optimizer.zero_grad()
        """
        outputs = model(inputs)：前向传播。调用模型的 forward。

        loss = criterion(outputs, labels)：计算交叉熵损失。

        loss.backward()：反向传播，计算所有参数的梯度。

                optimizer.step()：根据梯度更新参数。
        """
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #.item() 把单元素张量转成 Python 浮点数。
        """
        loss 是一个张量（tensor），里面只有一个数字，比如 tensor(3.45)。

        .item() 把这个单元素张量转成 Python 的普通浮点数 3.45。

        running_loss 是累计损失，+= 就是不断累加。
        张量会保留计算图，导致显存爆炸。.item() 切断计算图，只取值。
        """
        running_loss += loss.item()
        #outputs 形状 (B,100)，max(1) 在第 1 维（类别维）找最大值，返回 (最大值, 索引)。索引就是预测类别。_ 表示忽略最大值。
        _, predicted = outputs.max(1)
        #累加样本数。
        total += labels.size(0)
        #predicted.eq(labels) 逐元素比较，返回布尔张量；.sum() 统计 True 个数；.item() 转 Python 数。
        #.eq() 是 element-wise equality（逐元素相等比较）。返回一个布尔张量：
        """
        [1==1, 0==0, 2==5, 1==1]
        = [True, True, False, True]
        """
        correct += predicted.eq(labels).sum().item()#这批预测对了几个

        #在进度条后面显示当前平均损失和准确率。
        #累计平均损失、累计准确率
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")

@torch.no_grad()#装饰器，关闭自动求导。评估时不需要梯度，省显存、加速。
def evaluate(model, loader, device, epoch, best_acc, save_path='best_model.pth'):

#设为评估模式。Dropout、BatchNorm、DropPath 行为改变。比如 DropPath 里 self.training=False，直接返回输入。
    model.eval()
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f"Epoch {epoch} Test Acc: {acc:.2f}%")
    
    if acc > best_acc:
        print(f"🔥 New record! Accuracy improved from {best_acc:.2f}% to {acc:.2f}%")
        best_acc = acc

    # torch.save(model.state_dict(), save_path)
    return best_acc


# --- Main Execution ---
def main(enable_cuda=False):

    # 1. Setup
    cfg = TrainingConfig()
    seed_everything()
    device = get_device(cfg.device)
    
    # 2. Data
    trainloader, testloader = get_dataloaders(cfg)
    
    # 3. Model Initialization
    print("Initializing Model...")

    try:
        from networks import cliffordnet_12_2
        model = cliffordnet_12_2(
            num_classes=cfg.num_classes, 
            patch_size=cfg.patch_size, 
            embed_dim=cfg.embed_dim,
            enable_cuda=enable_cuda
        )
    except ImportError:
        print("Warning: model not found, using generic CliffordNet.")
        model = CliffordNet(
            num_classes=cfg.num_classes,
            img_size=32, 
            patch_size=cfg.patch_size, 
            embed_dim=cfg.embed_dim,
            depth=12, 
            enable_cuda=enable_cuda
        )
      
    model = model.to(device)
    
    print(f"Model built. Learnable Parameters: {count_parameters(model):,}")

    # 4. Optimization Components
    criterion = nn.CrossEntropyLoss()#分类任务标准损失，内部包含 LogSoftmax + NLLLoss。
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 5. Training Loop
    print(f"Start training for {cfg.epochs} epochs...")
    start_time = time.time()
    best_acc = 0.0 
    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, cfg.epochs)
        best_acc = evaluate(model, testloader, device, epoch, best_acc, save_path=cfg.save_path)
        scheduler.step()

    total_time = time.time() - start_time
    print(f"Training Finished. Total time: {total_time/60:.2f} mins")


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="supports CUDA acceleration")
    parser.add_argument('--enable_cuda', action='store_true', help='Whether to enable CUDA acceleration (default: False)')
    args = parser.parse_args()    
    print(f"Enable CUDA acceleration: {args.enable_cuda}")
    main(enable_cuda=args.enable_cuda)
