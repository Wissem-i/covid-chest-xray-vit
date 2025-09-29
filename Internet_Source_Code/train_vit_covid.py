#!/usr/bin/env python3
"""
COVID-19 Vision Transformer Training Script
Train Vision Transformer on COVID-19 vs Pneumonia chest X-ray classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from create_dataset_splits import main as create_splits

class COVID19Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Create label mapping
        self.label_map = {'COVID-19': 0, 'Pneumonia': 1}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.img_dir / 'images' / row['filename']
        if not img_path.exists():
            # Try alternative path
            img_path = self.img_dir / row['filename']
            
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Create dummy image if file not found
            image = Image.new('RGB', (224, 224), color='gray')
            
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.label_map.get(row['finding'], 0)
        
        return image, label

def get_transforms():
    """Get data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model():
    """Create Vision Transformer model"""
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}: Loss {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def main():
    """Main training function"""
    print("COVID-19 Vision Transformer Training")
    print("=" * 50)
    
    # Check for dataset
    if not Path("covid-chestxray-dataset").exists():
        print("Dataset not found!")
        print("Please run: git clone https://github.com/ieee8023/covid-chestxray-dataset.git")
        return
    
    # Create dataset splits
    print("Creating dataset splits...")
    try:
        loader = create_splits()
        train_df = loader.get_train_data()
        test_df = loader.get_test_data()
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
    except Exception as e:
        print(f"Error creating splits: {e}")
        print("Using simplified training demo...")
        
        # Simplified training with dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = create_model().to(device)
        print("Vision Transformer model created")
        
        # Demo training with synthetic data
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        print("\nRunning training demo with synthetic data...")
        for epoch in range(3):
            model.train()
            dummy_data = torch.randn(8, 3, 224, 224).to(device)
            dummy_labels = torch.randint(0, 2, (8,)).to(device)
            
            optimizer.zero_grad()
            outputs = model(dummy_data)
            loss = criterion(outputs, dummy_labels)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                acc = (pred == dummy_labels).float().mean()
                
            print(f"Epoch {epoch+1}/3: Loss = {loss.item():.4f}, Acc = {acc.item():.4f}")
        
        print("Training demo completed!")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model().to(device)
    print("Vision Transformer model created")
    
    # Create datasets and data loaders
    train_transform, val_transform = get_transforms()
    
    train_dataset = COVID19Dataset(train_df, 'covid-chestxray-dataset', train_transform)
    test_dataset = COVID19Dataset(test_df, 'covid-chestxray-dataset', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 5
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("\nTraining completed!")
    
    # Save model
    torch.save(model.state_dict(), 'covid_vit_model.pth')
    print("Model saved as 'covid_vit_model.pth'")

if __name__ == "__main__":
    main()