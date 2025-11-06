"""
Week 2 - Days 1-3: Understanding CNNs
Building a CNN from scratch in PyTorch to classify images from MNIST or CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification.
    Demonstrates convolution, pooling, and fully connected layers.
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # After 3 pooling layers: 28->14->7->3
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification (32x32 RGB images).
    """
    
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # After 4 pooling: 32->16->8->4->2
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = self.pool(F.relu(self.conv4(x)))  # 4x4 -> 2x2
        
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_mnist_data(batch_size: int = 64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_cifar10_data(batch_size: int = 64):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_cnn(model: nn.Module, train_loader: DataLoader, 
              criterion: nn.Module, optimizer: optim.Optimizer,
              device: torch.device, epochs: int = 10) -> list:
    """Train a CNN model."""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_loss:.4f}\n')
    
    return losses


def evaluate_cnn(model: nn.Module, test_loader: DataLoader, 
                device: torch.device) -> float:
    """Evaluate a CNN model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def visualize_predictions(model: nn.Module, test_loader: DataLoader,
                         device: torch.device, num_samples: int = 10):
    """Visualize model predictions."""
    model.eval()
    
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images[:num_samples])
        _, predicted = torch.max(outputs, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(num_samples):
        ax = axes[i // 5, i % 5]
        img = images[i].cpu().squeeze()
        if img.dim() == 3:  # CIFAR-10 (RGB)
            img = img.permute(1, 2, 0)
            img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
            img = torch.clamp(img, 0, 1)
            ax.imshow(img)
        else:  # MNIST (grayscale)
            ax.imshow(img, cmap='gray')
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}', 
                    color=color)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def train_mnist_cnn():
    """Train CNN on MNIST dataset."""
    print("=" * 60)
    print("Training CNN on MNIST Dataset")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining model...")
    losses = train_cnn(model, train_loader, criterion, optimizer, device, epochs=5)
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy = evaluate_cnn(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Visualize
    output_dir = Path('week2/day1-3/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = visualize_predictions(model, test_loader, device)
    plt.savefig(output_dir / 'mnist_predictions.png')
    print(f"\nPredictions saved to {output_dir / 'mnist_predictions.png'}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss (MNIST CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_dir / 'mnist_training_loss.png')
    print(f"Training loss saved to {output_dir / 'mnist_training_loss.png'}")


def train_cifar10_cnn():
    """Train CNN on CIFAR-10 dataset."""
    print("\n" + "=" * 60)
    print("Training CNN on CIFAR-10 Dataset")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_data(batch_size=64)
    
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create model
    model = CIFAR10CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining model...")
    losses = train_cnn(model, train_loader, criterion, optimizer, device, epochs=10)
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy = evaluate_cnn(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Visualize
    output_dir = Path('week2/day1-3/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = visualize_predictions(model, test_loader, device)
    plt.savefig(output_dir / 'cifar10_predictions.png')
    print(f"\nPredictions saved to {output_dir / 'cifar10_predictions.png'}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss (CIFAR-10 CNN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_dir / 'cifar10_training_loss.png')
    print(f"Training loss saved to {output_dir / 'cifar10_training_loss.png'}")


if __name__ == "__main__":
    # Train on MNIST (faster, good for learning)
    train_mnist_cnn()
    
    # Uncomment to train on CIFAR-10 (slower, more complex)
    # train_cifar10_cnn()

