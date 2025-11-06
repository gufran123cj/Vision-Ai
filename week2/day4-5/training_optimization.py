"""
Week 2 - Days 4-5: Training, Evaluation & Optimization
Loss functions, optimizers, learning rate scheduling, overfitting/underfitting,
data augmentation, and regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn.functional as F


# SimpleCNN class definition (from week2/day1-3/cnn_from_scratch.py)
class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_mnist_data(batch_size: int = 64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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


def train_with_different_optimizers():
    """Compare different optimizers."""
    print("=" * 60)
    print("Comparing Different Optimizers")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    optimizers_config = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop
    }
    
    results = {}
    
    for opt_name, opt_class in optimizers_config.items():
        print(f"\nTraining with {opt_name} optimizer...")
        
        # Create model
        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        if opt_name == 'SGD':
            optimizer = opt_class(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = opt_class(model.parameters(), lr=0.001)
        
        # Train
        losses = []
        for epoch in range(3):  # Short training for comparison
            model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(train_loader))
        
        # Evaluate
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
        results[opt_name] = {'losses': losses, 'accuracy': accuracy}
        print(f"{opt_name} - Final Accuracy: {accuracy:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opt_name, result in results.items():
        plt.plot(result['losses'], label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    opt_names = list(results.keys())
    accuracies = [results[opt]['accuracy'] for opt in opt_names]
    plt.bar(opt_names, accuracies)
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.grid(True, axis='y')
    
    output_dir = Path('week2/day4-5/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer_comparison.png')
    print(f"\nComparison saved to {output_dir / 'optimizer_comparison.png'}")


def train_with_learning_rate_scheduler():
    """Demonstrate learning rate scheduling."""
    print("\n" + "=" * 60)
    print("Learning Rate Scheduling")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Learning rate schedulers
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    }
    
    for scheduler_name, scheduler in schedulers.items():
        print(f"\nTraining with {scheduler_name}...")
        
        # Reset model
        model = SimpleCNN(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        elif scheduler_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        lr_history = []
        losses = []
        
        for epoch in range(10):
            model.train()
            epoch_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(train_loader))
            lr_history.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{scheduler_name} - Training Loss')
        ax1.grid(True)
        
        ax2.plot(lr_history)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title(f'{scheduler_name} - LR Schedule')
        ax2.grid(True)
        
        output_dir = Path('week2/day4-5/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / f'lr_schedule_{scheduler_name}.png')
        print(f"LR schedule plot saved to {output_dir / f'lr_schedule_{scheduler_name}.png'}")


def demonstrate_data_augmentation():
    """Demonstrate data augmentation techniques."""
    print("\n" + "=" * 60)
    print("Data Augmentation")
    print("=" * 60)
    
    # Original transform
    transform_original = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Augmented transform
    transform_augmented = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset_original = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_original
    )
    train_dataset_augmented = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_augmented
    )
    
    # Visualize augmentations
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    for i in range(5):
        # Original
        img_orig, _ = train_dataset_original[i]
        axes[0, i].imshow(img_orig.squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Augmented samples
        for j in range(2):
            img_aug, _ = train_dataset_augmented[i]
            axes[j + 1, i].imshow(img_aug.squeeze(), cmap='gray')
            axes[j + 1, i].set_title(f'Augmented {j + 1}')
            axes[j + 1, i].axis('off')
    
    output_dir = Path('week2/day4-5/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_augmentation_examples.png')
    print(f"Augmentation examples saved to {output_dir / 'data_augmentation_examples.png'}")


def train_with_regularization():
    """Compare models with and without regularization."""
    print("\n" + "=" * 60)
    print("Regularization Techniques")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Model without regularization
    model_no_reg = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.001)
    
    # Model with regularization (Dropout + Weight Decay)
    model_with_reg = SimpleCNN(num_classes=10).to(device)
    optimizer_with_reg = optim.Adam(model_with_reg.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train both models
    def train_model(model, optimizer, epochs=10):
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            
            # Evaluation
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
            
            test_accuracies.append(100 * correct / total)
        
        return train_losses, test_accuracies
    
    print("Training model without regularization...")
    losses_no_reg, acc_no_reg = train_model(model_no_reg, optimizer_no_reg)
    
    print("Training model with regularization...")
    losses_with_reg, acc_with_reg = train_model(model_with_reg, optimizer_with_reg)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses_no_reg, label='No Regularization')
    ax1.plot(losses_with_reg, label='With Regularization')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(acc_no_reg, label='No Regularization')
    ax2.plot(acc_with_reg, label='With Regularization')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    output_dir = Path('week2/day4-5/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'regularization_comparison.png')
    print(f"\nRegularization comparison saved to {output_dir / 'regularization_comparison.png'}")


if __name__ == "__main__":
    train_with_different_optimizers()
    train_with_learning_rate_scheduler()
    demonstrate_data_augmentation()
    train_with_regularization()
    
    print("\n" + "=" * 60)
    print("All optimization demonstrations completed!")
    print("=" * 60)

