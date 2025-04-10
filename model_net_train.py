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
    Custom dataset class to load image data from the 'jiguang' folder
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Root directory ('jiguang')
        :param transform: Data preprocessing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # Get all folder names (i.e., classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}  # Mapping class names to indices
        self.images = self._load_images()  # Load all image paths and labels

    def _load_images(self):
        """
        Load all image paths and labels
        """
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    images.append((img_path, self.class_to_idx[cls_name]))  # (image path, label)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB format
        if self.transform:
            image = self.transform(image)
        return image, label

def train_val_data_process():
    """
    Load data from the 'jiguang' folder and split it into training and validation sets
    """
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(size=224),  # Resize the image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize
            mean=[0.485, 0.456, 0.406],  # Mean
            std=[0.229, 0.224, 0.225]  # Standard deviation
        )
    ])

    # Load custom dataset
    dataset = CustomImageDataset(root_dir='jiguang', transform=transform)

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% training set
    val_size = len(dataset) - train_size  # 20% validation set
    print(val_size)
    generator1 = torch.Generator().manual_seed(666)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator1)

    # Training data loader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=128,  # Batch size
        shuffle=True,  # Shuffle data
        num_workers=2  # Multi-process loading
    )

    # Validation data loader
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # Define the device for training, use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use Adam optimizer for model parameter updates with a learning rate of 0.00002
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    scheduler = MultiStepLR(optimizer, milestones=[50, 200], gamma=0.5)
    # Loss function: Cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    # Move model to the training device
    model = model.to(device)
    # Copy the current model parameters (weights, biases, etc.) to save the best model parameters
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize variables
    best_acc = 0.0  # Best accuracy
    train_loss_all = []  # List of training loss values
    val_loss_all = []  # List of validation loss values
    train_acc_all = []  # List of training accuracy values
    val_acc_all = []  # List of validation accuracy values
    since = time.time()  # Start time

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Initialize parameters for this epoch
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # Train and compute for each mini-batch
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # Move features and labels to the training device
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()  # Set model to training mode

            # Forward pass: input is a batch, output is a prediction for the batch
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)  # Find the class with the highest score
            loss = criterion(output, b_y)  # Compute cross-entropy loss

            optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            train_loss += loss.item() * b_x.size(0)  # Accumulate loss
            train_corrects += torch.sum(pre_lab == b_y.data)  # Count correct predictions
            train_num += b_x.size(0)  # Increment number of samples processed

        scheduler.step()  # Update learning rate schedule
        print(epoch, scheduler.get_last_lr())

        # Validation phase
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()  # Set model to evaluation mode
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        # Save loss and accuracy values for each epoch
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        # Print loss and accuracy for this epoch
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())  # Save best model weights

        # Print time taken for training and validation
        time_use = time.time() - since
        print("Training and validation time: {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    os.makedirs("./model_save", exist_ok=True)  # Create directory for saving the model
    os.makedirs("./data_save", exist_ok=True)  # Create directory for saving the data

    torch.save(best_model_wts, "./model_save/ResNet18_best_model.pth")  # Save the best model

    # Save training process data as a table
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })
    train_process.to_csv("./data_save/train_process.csv", index=False)

    return train_process


def matplot_acc_loss(train_process):
    # Plot the loss and accuracy for training and validation after each iteration
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  # First graph in a 1x2 grid
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)  # Second graph in a 1x2 grid
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load the necessary model

    path = "jiguang"
    # Load the dataset
    train_data, val_data = train_val_data_process()
    # Get all labels from the validation set
    val_labels = [label for _, label in val_data]
    # model = Net()
    model = resnet2_simple(num_classes=8)

    train_process = train_model_process(model, train_data, val_data, num_epochs=400)  # Note: 10 epochs due to hardware constraints (can be adjusted)

    matplot_acc_loss(train_process)
