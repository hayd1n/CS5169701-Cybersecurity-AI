import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# Set hyperparameters
batch_size: int = 512
learning_rate: float = 50e-4
num_epochs: int = 20
image_size: Tuple[int, int] = (128, 128)  # Adjust according to your image size

# Dataset path
data_dir: str = (
    "C:\\Users\\hayden\\.conda\\envs\\virus_pic\\gray_virus\\class_5_output_image"
)
model_path: str = (
    "C:\\Users\\hayden\\.conda\\envs\\virus_pic\\gray_virus\\cnn_model.pth"
)

# Data preprocessing
transform: transforms.Compose = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.Resize(image_size),  # Resize the image to a uniform size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize
    ]
)

# Load the entire dataset
full_dataset: datasets.ImageFolder = datasets.ImageFolder(
    root=data_dir, transform=transform
)

# Get all labels
labels: List[int] = [label for _, label in full_dataset]

train_indices: List[int]
val_indices: List[int]
train_indices, val_indices = train_test_split(
    range(len(labels)),
    test_size=0.3,  # 30% of the data for the validation set
    stratify=labels,  # Stratified sampling based on labels
    random_state=42,  # Fixed random seed
)

# Create the split datasets
train_dataset: Subset = Subset(full_dataset, train_indices)
val_dataset: Subset = Subset(full_dataset, val_indices)

# Create DataLoader
train_loader: DataLoader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
val_loader: DataLoader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False
)

print(f"Total images: {len(full_dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(CNN, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1: nn.Linear = nn.Linear(
            64 * (image_size[0] // 4) * (image_size[1] // 4), 128
        )
        self.fc2: nn.Linear = nn.Linear(128, num_classes)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Get the number of classes
num_classes: int = len(full_dataset.classes)

# Initialize model, loss function, and optimizer
device: str = "cuda" if torch.cuda.is_available() else "cpu"
model: CNN = CNN(num_classes=num_classes).to(device)
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if a saved model exists
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    print("No pre-trained model found. Starting training from scratch.")


# Validate the model
def validate(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy: float = 100 * correct / total
    return accuracy


# Train the model
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> Tuple[List[float], List[float]]:
    model.train()
    accuracy_list: List[float] = []
    loss_list: List[float] = []

    for epoch in range(num_epochs):
        running_loss: float = 0.0
        # Wrap the DataLoader with tqdm to show a progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the tqdm description
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

        # Validate accuracy
        accuracy: float = validate(model, val_loader)
        accuracy_list.append(accuracy)
        loss_list.append(running_loss / len(train_loader))

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%"
        )

        # Save the model after each epoch
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("Training complete.")
    return accuracy_list, loss_list


# Start training and validation
accuracy_list, loss_list = train(
    model, train_loader, val_loader, criterion, optimizer, num_epochs
)


# Plot and save accuracy and loss images
def plot_metrics(
    accuracy_list: List[float], loss_list: List[float], acc_path: str, loss_path: str
) -> None:
    # Plot accuracy
    plt.figure()
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, marker="o")
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.savefig(acc_path)
    print(f"Accuracy plot saved to {acc_path}")

    # Plot loss
    plt.figure()
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker="o", color="red")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(loss_path)
    print(f"Loss plot saved to {loss_path}")


# Plot and save accuracy and loss images
acc_plot_path: str = (
    "C:\\Users\\hayden\\.conda\\envs\\virus_pic\\gray_virus\\accuracy_plot.png"
)
loss_plot_path: str = (
    "C:\\Users\\hayden\\.conda\\envs\\virus_pic\\gray_virus\\loss_plot.png"
)
plot_metrics(accuracy_list, loss_list, acc_plot_path, loss_plot_path)
