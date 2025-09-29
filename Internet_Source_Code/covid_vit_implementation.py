#!/usr/bin/env python3
"""
COVID-19 Vision Transformer Implementation
Based on Vision Transformer code found from lucidrains/vit-pytorch GitHub repository
https://github.com/lucidrains/vit-pytorch

This implementation follows the assignment requirement to:
"find appropriate code on github, on Kaggle, or via search" and execute it

Code adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
Dataset: COVID-19 chest X-ray from ieee8023/covid-chestxray-dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from vit_pytorch import ViT
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# =============================================================================
# COVID-19 Dataset Class (adapted for ViT)
# =============================================================================
class COVID19Dataset(Dataset):
    """COVID-19 chest X-ray dataset for Vision Transformer"""
    
    def __init__(self, csv_file, img_dir, transform=None, is_validation=False):
        """
        Args:
            csv_file (str): Path to CSV file with image filenames and labels
            img_dir (str): Directory with chest X-ray images  
            transform (callable): Optional transform to be applied on images
            is_validation (bool): Whether this is validation dataset
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_validation = is_validation
        
        print(f"Loaded dataset: {len(self.data)} samples")
        if 'Binary_Label' in self.data.columns:
            print(f"Label distribution:")
            print(self.data['Binary_Label'].value_counts())
        elif 'label' in self.data.columns:
            print(f"Label distribution:")  
            print(self.data['label'].value_counts())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = str(self.data.iloc[idx]['filename'])
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), 0)
        
        # Get label - check for Binary_Label or label columns
        if 'Binary_Label' in self.data.columns:
            label_str = str(self.data.iloc[idx]['Binary_Label'])
            # Convert COVID-19/Pneumonia to 1/0
            if 'COVID' in label_str.upper():
                label = 1
            else:
                label = 0
        elif 'label' in self.data.columns:
            label = int(str(self.data.iloc[idx]['label']))
        else:
            label = 0  # Default fallback
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# =============================================================================
# Vision Transformer Model (using found implementation)
# =============================================================================
class COVID19ViT(nn.Module):
    """
    Vision Transformer for COVID-19 classification
    Using implementation from lucidrains/vit-pytorch repository
    """
    
    def __init__(self, num_classes=2, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(COVID19ViT, self).__init__()
        
        # Vision Transformer from lucidrains/vit-pytorch
        self.vit = ViT(
            image_size=image_size,      # 224x224 input images
            patch_size=patch_size,      # 16x16 patches
            num_classes=num_classes,    # Binary classification: COVID vs Pneumonia
            dim=dim,                    # Token dimension
            depth=depth,                # Number of transformer layers
            heads=heads,                # Number of attention heads
            mlp_dim=mlp_dim,           # MLP dimension
            dropout=0.1,
            emb_dropout=0.1
        )
        
    def forward(self, x):
        return self.vit(x)

# =============================================================================
# Data transforms for Vision Transformer
# =============================================================================
def get_transforms():
    """Get data transformations for Vision Transformer"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# =============================================================================
# Training functions
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Update progress bar
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.4f}'
        })
    
    if scheduler:
        scheduler.step()
    
    epoch_loss = running_loss / total_samples
    epoch_acc = float(running_corrects) / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Store predictions for detailed metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = float(running_corrects) / total_samples
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# =============================================================================
# Main training function
# =============================================================================
def main():
    """Main function to train Vision Transformer on COVID-19 dataset"""
    
    print("=" * 60)
    print("COVID-19 Vision Transformer Implementation")
    print("Based on code from lucidrains/vit-pytorch GitHub repository")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    data_dir = 'data/processed'
    img_dir = 'covid-chestxray-dataset/images'
    
    # Check if data exists
    train_csv = os.path.join(data_dir, 'train_split.csv')
    test_csv = os.path.join(data_dir, 'test_split.csv')
    val_csv = os.path.join(data_dir, 'validation_split.csv')
    
    if not all(os.path.exists(f) for f in [train_csv, test_csv, val_csv]):
        print("Dataset splits not found. Creating splits...")
        from create_dataset_splits import main as create_splits
        create_splits()
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = COVID19Dataset(train_csv, img_dir, transform=train_transform)
    val_dataset = COVID19Dataset(val_csv, img_dir, transform=val_transform, is_validation=True)
    test_dataset = COVID19Dataset(test_csv, img_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model (using found ViT implementation)
    print("\n🏗️  Creating Vision Transformer model...")
    model = COVID19ViT(num_classes=2, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Training loop
    print("\n🚀 Starting training...")
    num_epochs = 5  # Reduced for demonstration
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, 'best_covid_vit.pth')
            print(f"New best validation accuracy: {best_val_acc:.4f}")
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\n📈 Classification Report:")
    class_names = ['Pneumonia', 'COVID-19']
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('COVID-19 vs Pneumonia Classification\nVision Transformer Results')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('covid_vit_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'covid_vit_confusion_matrix.png'")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('covid_vit_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as 'covid_vit_training_curves.png'")
    
    # Save results
    results = {
        'model_name': 'Vision Transformer (ViT)',
        'dataset': 'COVID-19 Chest X-ray',
        'implementation_source': 'lucidrains/vit-pytorch GitHub repository',
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'num_epochs': num_epochs,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('covid_vit_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Best validation accuracy: {best_val_acc:.4f}")
    print(f"📊 Test accuracy: {test_acc:.4f}")
    print("📁 Results saved to 'covid_vit_results.json'")
    print("\n🎯 Implementation source: lucidrains/vit-pytorch GitHub repository")
    print("📊 Dataset: COVID-19 chest X-ray from ieee8023/covid-chestxray-dataset")
    
    return model, results

if __name__ == "__main__":
    model, results = main()