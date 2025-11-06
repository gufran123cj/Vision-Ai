"""
Week 1 - Days 1-2: From ML to Deep Learning
Understanding the core mechanics of how neural networks learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple


class SimpleNeuralNetwork:
    """
    A simple neural network implementation from scratch to understand
    the fundamentals: forward pass, backward pass, and gradient descent.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.01):
        """
        Initialize a simple 2-layer neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # Store activations for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Output predictions (batch_size, output_size)
        """
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """
        Backward pass (backpropagation) to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            output: Network predictions
        """
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights using gradient descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
        loss = -(1 / m) * np.sum(y_true * np.log(y_pred) + 
                                 (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              verbose: bool = True) -> List[float]:
        """
        Train the neural network.
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            List of loss values during training
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses


def demonstrate_neural_network():
    """Demonstrate the neural network with a simple classification problem."""
    print("=" * 60)
    print("Neural Network Demonstration: XOR Problem")
    print("=" * 60)
    
    # Create XOR dataset (non-linearly separable)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, 
                             learning_rate=0.5)
    
    print("\nTraining neural network...")
    losses = nn.train(X, y, epochs=2000, verbose=True)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions:")
    print("=" * 60)
    predictions = nn.forward(X)
    
    for i in range(len(X)):
        print(f"Input: {X[i]}, True: {y[i][0]}, Predicted: {predictions[i][0]:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('week1/day1-2/training_loss.png')
    print("\nTraining loss plot saved to 'week1/day1-2/training_loss.png'")


if __name__ == "__main__":
    demonstrate_neural_network()

