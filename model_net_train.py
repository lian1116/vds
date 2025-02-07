import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR
from model_net import *
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    """
    自定义数据集类，用于加载 jiguang 文件夹下的图片数据
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: 根目录（jiguang）
        :param transform: 数据预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # 获取所有文件夹名称（即类别）
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}  # 类别名称到索引的映射
        self.images = self._load_images()  # 加载所有图片路径和标签

    def _load_images(self):
        """
        加载所有图片路径和标签
        """
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    images.append((img_path, self.class_to_idx[cls_name]))  # (图片路径, 标签)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # 打开图片并转换为 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image, label

def train_val_data_process():
    """
    加载 jiguang 文件夹下的数据，并划分为训练集和验证集
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(size=224),  # 调整图片大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(  # 标准化
            mean=[0.485, 0.456, 0.406],  # 均值
            std=[0.229, 0.224, 0.225]  # 标准差
        )
    ])

    # 加载自定义数据集
    dataset = CustomImageDataset(root_dir='jiguang', transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))  # 80% 训练集
    val_size = len(dataset) - train_size  # 20% 验证集
    print(val_size)
    generator1 = torch.Generator().manual_seed(666)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size],generator=generator1)

    # 训练集加载器
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=128,  # 批次大小
        shuffle=True,  # 打乱数据
        num_workers=2  # 多进程加载
    )

    # 验证集加载器
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, val_dataloader
\


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 定义训练使用的设备，有GPU则用，没有则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器进行模型参数更新，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    scheduler = MultiStepLR(optimizer, milestones=[50, 200], gamma=0.5)
    # 损失函数为交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备内
    model = model.to(device)
    # 复制当前模型参数(w,b等)，以便将最好的模型参数权重保存下来
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失值列表
    train_loss_all = []
    # 验证集损失值列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数
        # 训练集损失值
        train_loss = 0.0
        # 训练集精确度
        train_corrects = 0
        # 验证集损失值
        val_loss = 0.0
        # 验证集精确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)  # batch_size*28*28*1的tensor数据
            # 将标签放入到训练设备中
            b_y = b_y.to(device)  # batch_size大小的向量tensor数据
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)  # 输出为：batch_size大小的行和10列组成的矩阵
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)  # batch_size大小的向量表示属于物品的标签
            # 计算每一个batch的损失函数，向量形式的交叉熵损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0，防止梯度累积
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加，该批次的loss值乘于该批次数量得到批次总体loss值，在将其累加得到轮次总体loss值
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        scheduler.step()
        print(epoch, scheduler.get_last_lr())

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度val_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一轮次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 打印每一轮次的loss值和准确度
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    os.makedirs("./model_save", exist_ok=True)  # 创建模型保存目录
    os.makedirs("./data_save", exist_ok=True)  # 创建数据保存目录

    # 保存最优模型
    torch.save(best_model_wts, "./model_save/ResNet18_best_model.pth")

    # 将训练过程数据保存为表格
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })
    train_process.to_csv("./data_save/train_process.csv", index=False)
    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, "./model_save/ResNet18_best_model.pth")

    return train_process


def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  # 表示一行两列的第一张图
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)  # 表示一行两列的第二张图
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载需要的模型

    path = "jiguang"
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 获取验证集的所有标签
    val_labels = [label for _, label in val_data]
    # model = Net()
    model = resnet2_simple(num_classes=8)

    train_process = train_model_process(model, train_data, val_data, num_epochs=400)  # 注：由于硬件限制，只设置训练10轮(可修改)

    matplot_acc_loss(train_process)