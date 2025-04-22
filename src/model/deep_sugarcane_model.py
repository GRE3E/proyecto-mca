import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
data_yaml = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dataset.yaml')

class SugarCaneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # Leer clases desde dataset.yaml
        import yaml
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'dataset.yaml')
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data['names']
        # Solo considerar carpetas que estén en class_names
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                        self.samples.append((os.path.join(class_path, fname), label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DeepSugarCaneNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*20*20, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_dataloaders(train_dir, val_dir, batch_size=16, img_size=320):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = SugarCaneDataset(train_dir, transform)
    val_ds = SugarCaneDataset(val_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train_model(train_dir, val_dir, model_path=None, epochs=30, batch_size=16, img_size=320, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size, img_size)
    if model_path:
        model = DeepSugarCaneNet(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = DeepSugarCaneNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(epochs):
        try:
            model.train()
            running_loss = 0
            total_batches = len(train_loader)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()*images.size(0)
                print(f"Época {epoch+1}/{epochs} - Lote {batch_idx+1}/{total_batches} - Pérdida: {loss.item():.4f}")
            train_loss = running_loss/len(train_loader.dataset)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'best_sugarcane_model.pth'))
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        except Exception as e:
            print(f"Error durante el entrenamiento en la época {epoch+1}: {e}")
            break
    plot_metrics(history)
    return model, history

def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()*images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = loss/total
    acc = correct/total
    return avg_loss, acc

def plot_metrics(history, save_dir=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
    else:
        plt.show()
    plt.close()

def predict_image(image_path, weights_path=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSugarCaneNet(num_classes=2)
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), 'best_sugarcane_model.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return pred.item()