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
    Custom dataset class for loading image data from the 'jiguang' folder
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Root directory ('jiguang')
        :param transform: Data preprocessing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # Get all folder names (i.e., categories)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}  # Mapping category names to indices
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

def test_data_process():
    """
    Load data from the 'jiguang' folder and split it into training and validation sets
    """
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(size=224),  # Resize images
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
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size],generator=generator1)

    # Training data loader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=128,  # Batch size
        shuffle=True,  # Shuffle data
        num_workers=2  # Use multiple processes for loading data
    )

    # Validation data loader
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    return  val_dataloader


def test_model_process(model, test_dataloader, output_csv="test_results.csv"):
    # Set the device to be used for testing, GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Move model to the selected device
    model = model.to(device)

    # Initialize parameters
    test_corrects = 0.0
    test_num = 0

    # Store predicted and actual labels
    predictions = []
    actual_labels = []

    # Only perform forward propagation without calculating gradients to save memory and speed up execution
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # Move features to the testing device
            test_data_x = test_data_x.to(device)
            # Move labels to the testing device
            test_data_y = test_data_y.to(device)
            # Set model to evaluation mode
            model.eval()
            # Perform forward propagation, input is the test dataset, output is the prediction for each sample
            output = model(test_data_x)
            # Find the index of the maximum value in each row
            pre_lab = torch.argmax(output, dim=1)
            # If prediction is correct, increase the correct count
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # Add all test samples to the total count
            test_num += test_data_x.size(0)

            # Store predicted and actual labels in lists
            predictions.extend(pre_lab.cpu().numpy())  # Move predicted labels from GPU to CPU and convert to numpy array
            actual_labels.extend(test_data_y.cpu().numpy())  # Move actual labels from GPU to CPU and convert to numpy array

    # Calculate the test accuracy
    test_acc = test_corrects.double().item() / test_num
    print("Test accuracy: ", test_acc)

    # Save the predicted and actual labels to a CSV file
    results_df = pd.DataFrame({
        "Predicted Label": predictions,
        "Actual Label": actual_labels
    })
    results_df.to_csv(output_csv, index=False)
    print(f"Test results saved to {output_csv}")

if __name__ == "__main__":
    # Load the model
    model = resnet2_simple(num_classes=8)
    model.load_state_dict(torch.load('./model_save/ResNet18_best_model.pth'))  # Load the trained model weights
    # Load test data
    test_dataloader = test_data_process()
    # Call the function to test the model
    test_model_process(model, test_dataloader)
