"""
Week 3 - Days 4-7: Image Segmentation
Semantic and instance segmentation using U-Net and other architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (Expanding Path)
        self.dec4 = self._conv_block(1024 + 512, 512)
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Convolutional block: Conv -> ReLU -> Conv -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net."""
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upsample(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final layer
        x = self.final(x)
        
        return x


class SegmentationDataset(Dataset):
    """Simple dataset for segmentation demonstration."""
    
    def __init__(self, images: list, masks: list, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform:
            # Apply same transform to both image and mask
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask


def create_synthetic_data(num_samples: int = 100, img_size: int = 256):
    """Create synthetic images and masks for demonstration."""
    images = []
    masks = []
    
    for _ in range(num_samples):
        # Create random image
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Create mask (circle in center)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        center = (img_size // 2, img_size // 2)
        radius = np.random.randint(30, 80)
        cv2.circle(mask, center, radius, 1, -1)
        
        images.append(img)
        masks.append(mask)
    
    return images, masks


def train_unet(model: nn.Module, train_loader: DataLoader, 
               device: torch.device, epochs: int = 10):
    """Train U-Net model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return losses


def visualize_segmentation(model: nn.Module, image: np.ndarray, 
                          device: torch.device, save_path: str = None):
    """Visualize segmentation results."""
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    return prediction


def demonstrate_semantic_segmentation():
    """Demonstrate semantic segmentation with U-Net."""
    print("=" * 60)
    print("Semantic Segmentation with U-Net")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\nCreating synthetic dataset...")
    images, masks = create_synthetic_data(num_samples=50, img_size=256)
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = SegmentationDataset(images, masks, transform=transform)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    model = UNet(in_channels=3, num_classes=2).to(device)
    print(f"\nModel created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining U-Net...")
    losses = train_unet(model, train_loader, device, epochs=5)
    
    # Visualize results
    output_dir = Path('week3/day4-7/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on a sample image
    test_image = images[0]
    prediction = visualize_segmentation(
        model, test_image, device, 
        str(output_dir / 'segmentation_result.png')
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('U-Net Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_dir / 'unet_training_loss.png')
    print(f"\nTraining loss saved to {output_dir / 'unet_training_loss.png'}")


def demonstrate_instance_segmentation():
    """Demonstrate instance segmentation (using YOLO segmentation)."""
    print("\n" + "=" * 60)
    print("Instance Segmentation")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Load YOLO segmentation model
        model = YOLO('yolov8n-seg.pt')  # Segmentation model
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
        cv2.circle(test_image, (450, 200), 50, (255, 0, 0), -1)
        
        output_dir = Path('week3/day4-7/output')
        test_image_path = str(output_dir / 'test_seg_image.jpg')
        cv2.imwrite(test_image_path, test_image)
        
        # Run segmentation
        results = model(test_image_path)
        result_image = results[0].plot()
        
        # Save result
        result_path = str(output_dir / 'instance_segmentation_result.jpg')
        cv2.imwrite(result_path, result_image)
        print(f"Instance segmentation result saved to {result_path}")
        
    except ImportError:
        print("⚠ ultralytics not installed. Install with: pip install ultralytics")


def explain_segmentation_differences():
    """Explain the difference between semantic and instance segmentation."""
    print("\n" + "=" * 60)
    print("Semantic vs Instance Segmentation")
    print("=" * 60)
    
    print("""
    SEMANTIC SEGMENTATION:
    - Assigns a class label to each pixel
    - Does not distinguish between different instances of the same class
    - Example: All pixels of "person" are labeled the same
    
    INSTANCE SEGMENTATION:
    - Identifies each individual object instance
    - Distinguishes between different instances of the same class
    - Example: Each "person" gets a unique label
    
    USE CASES:
    - Semantic: Scene understanding, autonomous driving road detection
    - Instance: Object counting, tracking, medical image analysis
    """)


if __name__ == "__main__":
    demonstrate_semantic_segmentation()
    demonstrate_instance_segmentation()
    explain_segmentation_differences()
    
    print("\n" + "=" * 60)
    print("Image segmentation demonstrations completed!")
    print("=" * 60)

