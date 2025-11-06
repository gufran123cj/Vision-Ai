"""
Week 1 - Days 3-4: Introduction to PyTorch
Setting up PyTorch environment, mastering Tensors, autograd, and torch.nn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class SimpleDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class SimpleNeuralNetworkPyTorch(nn.Module):
    """
    Simple neural network using PyTorch's nn.Module.
    Demonstrates the basic building blocks of PyTorch.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNeuralNetworkPyTorch, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def demonstrate_tensors():
    """Demonstrate PyTorch tensors and operations."""
    print("=" * 60)
    print("PyTorch Tensors Demonstration")
    print("=" * 60)
    
    # Create tensors
    print("\n1. Creating Tensors:")
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    print(f"x = {x}")
    print(f"y = {y}")
    
    # Tensor operations
    print("\n2. Tensor Operations:")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")
    print(f"x.dot(y) = {x.dot(y)}")
    
    # Matrix operations
    print("\n3. Matrix Operations:")
    A = torch.randn(3, 4)
    B = torch.randn(4, 2)
    C = torch.matmul(A, B)
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"C = A @ B shape: {C.shape}")
    
    # GPU availability
    print(f"\n4. Device Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")


def demonstrate_autograd():
    """Demonstrate automatic differentiation (autograd)."""
    print("\n" + "=" * 60)
    print("Automatic Differentiation (Autograd)")
    print("=" * 60)
    
    # Create a tensor with requires_grad=True
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # Define a function
    z = x**2 + y**2 + x * y
    
    print(f"\nFunction: z = x² + y² + x*y")
    print(f"x = {x.item()}, y = {y.item()}")
    print(f"z = {z.item()}")
    
    # Compute gradients
    z.backward()
    
    print(f"\nGradients:")
    print(f"∂z/∂x = {x.grad.item()}")  # Should be 2x + y = 2*2 + 3 = 7
    print(f"∂z/∂y = {y.grad.item()}")  # Should be 2y + x = 2*3 + 2 = 8


def train_model(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                epochs: int = 100) -> list:
    """Train a PyTorch model."""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % (epochs // 10) == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return losses


def demonstrate_pytorch_model():
    """Demonstrate building and training a model with PyTorch."""
    print("\n" + "=" * 60)
    print("Building and Training a PyTorch Model")
    print("=" * 60)
    
    # Create synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(float)  # Binary classification
    
    # Create dataset and dataloader
    dataset = SimpleDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = SimpleNeuralNetworkPyTorch(input_size=2, hidden_size=8, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\nTraining model...")
    losses = train_model(model, train_loader, criterion, optimizer, epochs=200)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_X = torch.FloatTensor(X)
        predictions = model(test_X)
        accuracy = ((predictions > 0.5).float().squeeze() == 
                   torch.FloatTensor(y)).float().mean()
        print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss (PyTorch Model)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('week1/day3-4/pytorch_training_loss.png')
    print("\nTraining loss plot saved to 'week1/day3-4/pytorch_training_loss.png'")


if __name__ == "__main__":
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Run demonstrations
    demonstrate_tensors()
    demonstrate_autograd()
    demonstrate_pytorch_model()

