import torch
import torch.utils.data as data
from torchvision.datasets import FashionMNIST

from model_net import *
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import pandas as pd
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

def test_data_process():
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

    return  val_dataloader


def test_model_process(model, test_dataloader, output_csv="test_results.csv"):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 将模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 用于存储预测标签和实际标签
    predictions = []
    actual_labels = []

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

            # 将预测标签和实际标签存入列表
            predictions.extend(pre_lab.cpu().numpy())  # 将预测标签从 GPU 移到 CPU 并转换为 numpy 数组
            actual_labels.extend(test_data_y.cpu().numpy())  # 将实际标签从 GPU 移到 CPU 并转换为 numpy 数组

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

    # 将预测标签和实际标签保存到 CSV 文件
    results_df = pd.DataFrame({
        "Predicted Label": predictions,
        "Actual Label": actual_labels
    })
    results_df.to_csv(output_csv, index=False)
    print(f"测试结果已保存到 {output_csv}")
if __name__ == "__main__":
    # 加载模型
    model = resnet2_simple(num_classes=8)
    model.load_state_dict(torch.load('./model_save/ResNet18_best_model.pth'))  # 调用训练好的参数权重
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)