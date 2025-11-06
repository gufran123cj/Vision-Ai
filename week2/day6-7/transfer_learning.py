"""
Week 2 - Days 6-7: Transfer Learning with Pre-trained Models
Using ResNet, VGG, MobileNet as feature extractors or for fine-tuning
(torchvision >= 0.13 uyumlu)
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)
import matplotlib.pyplot as plt


# ----------------------------
# Veri yükleme (CIFAR-10)
# ----------------------------
def load_cifar10_data(batch_size: int = 64, img_size: int = 224):
    """
    CIFAR-10'u ImageNet-ön-eğitimli modellerle uyumlu boyuta getirir.
    Not: Normalize değerlerini manuel vermek yerine weights.transforms() kullanacağız.
    Burada sadece resize + augmentation yapıyoruz; normalize'ı DataLoader içinde ekleyeceğiz.
    """
    transform_train = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=4),
        T.ToTensor(),
        # Normalize'ı modelin kendi transforms'ı ekleyecek.
    ])

    transform_test = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


# ----------------------------
# Yardımcı: transform zincirini weights'e göre tamamla
# ----------------------------
def attach_normalize_transform(dataloader, weights):
    """
    DataLoader içindeki dataset.transform sonuna, ilgili ağırlıkların
    ImageNet mean/std normalize adımını ekler (torchvision >= 0.13 uyumlu).
    """
    base_tf = dataloader.dataset.transform
    mean = weights.meta.get('mean', (0.485, 0.456, 0.406))
    std  = weights.meta.get('std',  (0.229, 0.224, 0.225))

    # base_tf zaten Resize/Flip/Crop/ToTensor içeriyor olmalı.
    # Sadece Normalize ekliyoruz.
    dataloader.dataset.transform = torchvision.transforms.Compose([
        base_tf,
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    return dataloader



# ----------------------------
# Modeller
# ----------------------------
class ResNetFeatureExtractor(nn.Module):
    """ResNet18: tüm gövde dondurulur, sadece son katman (fc) eğitim alır."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.weights = ResNet18_Weights.DEFAULT
        m = resnet18(weights=self.weights)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)


class ResNetFineTuned(nn.Module):
    """ResNet18: erken katmanlar dondurulur, son blok (layer4) + fc serbest bırakılır."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.weights = ResNet18_Weights.DEFAULT
        m = resnet18(weights=self.weights)
        # Tümünü dondur
        for p in m.parameters():
            p.requires_grad = False
        # Sadece layer4 + fc'yi aç
        for p in m.layer4.parameters():
            p.requires_grad = True
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        for p in m.fc.parameters():
            p.requires_grad = True
        self.model = m

    def forward(self, x):
        return self.model(x)


# ----------------------------
# Eğitim & Değerlendirme
# ----------------------------
def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                device: torch.device, epochs: int = 5, model_name: str = "Model"):
    criterion = nn.CrossEntropyLoss()
    # Sadece eğitilebilir parametreleri optimize et
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        avg_loss = running / max(1, len(train_loader))
        train_losses.append(avg_loss)

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits = model(images)
                preds = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / max(1, total)
        test_accuracies.append(acc)

        print(f'{model_name} - Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%')

    return train_losses, test_accuracies


# ----------------------------
# Karşılaştırmalar
# ----------------------------
def compare_transfer_learning_approaches():
    print("=" * 60)
    print("Transfer Learning: Feature Extraction vs Fine-tuning")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Veri
    train_loader, test_loader = load_cifar10_data(batch_size=32, img_size=224)

    # Model 1: Feature Extractor
    model_fe = ResNetFeatureExtractor(num_classes=10).to(device)
    # Doğru normalize
    train_loader_fe = attach_normalize_transform(train_loader, model_fe.weights)
    test_loader_fe  = attach_normalize_transform(test_loader,  model_fe.weights)
    losses_fe, acc_fe = train_model(
        model_fe, train_loader_fe, test_loader_fe, device, epochs=5, model_name="Feature Extractor"
    )

    # Model 2: Fine-tuned
    model_ft = ResNetFineTuned(num_classes=10).to(device)
    train_loader_ft = attach_normalize_transform(train_loader, model_ft.weights)
    test_loader_ft  = attach_normalize_transform(test_loader,  model_ft.weights)
    losses_ft, acc_ft = train_model(
        model_ft, train_loader_ft, test_loader_ft, device, epochs=5, model_name="Fine-tuned"
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses_fe, label='Feature Extractor', marker='o')
    ax1.plot(losses_ft, label='Fine-tuned', marker='s')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison'); ax1.legend(); ax1.grid(True)

    ax2.plot(acc_fe, label='Feature Extractor', marker='o')
    ax2.plot(acc_ft, label='Fine-tuned', marker='s')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison'); ax2.legend(); ax2.grid(True)

    out = Path('week2/day6-7/output'); out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out / 'transfer_learning_comparison.png')
    print(f"\nComparison saved to {out / 'transfer_learning_comparison.png'}")


def compare_pretrained_architectures():
    print("\n" + "=" * 60)
    print("Comparing Different Pretrained Architectures")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = load_cifar10_data(batch_size=32, img_size=224)

    configs = {
        'ResNet18':  (resnet18,        ResNet18_Weights.DEFAULT,  'resnet'),
        'ResNet50':  (resnet50,        ResNet50_Weights.DEFAULT,  'resnet'),
        'VGG16':     (vgg16,           VGG16_Weights.DEFAULT,     'vgg'),
        'MobileNetV2': (mobilenet_v2,  MobileNet_V2_Weights.DEFAULT, 'mobilenet'),
    }

    results = {}

    for name, (ctor, weights, family) in configs.items():
        print(f"\nTesting {name}...")
        try:
            model = ctor(weights=weights)

            # Gövdeyi dondur
            for p in model.parameters():
                p.requires_grad = False

            # Son sınıflandırıcıyı 10 sınıfa uyarlama
            if family == 'resnet':
                in_f = model.fc.in_features
                model.fc = nn.Linear(in_f, 10)
            elif family == 'vgg':
                in_f = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(in_f, 10)
            elif family == 'mobilenet':
                in_f = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_f, 10)

            # Yalnızca sınıflandırıcı parametrelerini optimize et
            params = [p for p in model.parameters() if p.requires_grad]
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(params, lr=1e-3)

            # Doğru normalize: her modelin kendi weights'i!
            tr_loader = attach_normalize_transform(train_loader, weights)
            te_loader = attach_normalize_transform(test_loader,  weights)

            # Kısa eğitim (2 epoch)
            model.train()
            for _ in range(2):
                for images, labels in tr_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    loss = criterion(model(images), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Değerlendirme
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in te_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    preds = model(images).argmax(1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            acc = 100.0 * correct / max(1, total)
            results[name] = acc
            print(f"{name} - Accuracy: {acc:.2f}%")

        except Exception as e:
            print(f"Error with {name}: {e}")

    if results:
        plt.figure(figsize=(10, 6))
        names = list(results.keys()); scores = list(results.values())
        plt.bar(names, scores)
        plt.ylabel('Test Accuracy (%)'); plt.title('Pretrained Architecture Comparison')
        plt.xticks(rotation=45); plt.grid(True, axis='y')

        out = Path('week2/day6-7/output'); out.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(out / 'architecture_comparison.png')
        print(f"\nComparison saved to {out / 'architecture_comparison.png'}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Starting transfer learning demonstrations")
    print("=" * 60)

    compare_transfer_learning_approaches()
    compare_pretrained_architectures()

    print("\n" + "=" * 60)
    print("Transfer learning demonstrations completed!")
    print("=" * 60)
