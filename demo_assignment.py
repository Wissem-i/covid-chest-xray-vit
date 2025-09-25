#!/usr/bin/env python3
"""
Demo script for PSU Week 1 Assignment - COVID-19 ViT Classification
Shows that the code actually works and can be executed
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def check_requirements():
    """Check if all required packages are available"""
    print("🔍 Checking Requirements...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'), 
        ('timm', 'PyTorch Image Models'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow')
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All requirements satisfied!")
        return True

def demonstrate_vision_transformer():
    """Demonstrate Vision Transformer model creation"""
    print("\n🤖 Testing Vision Transformer Model...")
    
    try:
        import timm
        
        # Create a ViT model (this proves the architecture works)
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        print(f"  ✅ ViT Model Created: {model.__class__.__name__}")
        print(f"  ✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✅ Forward Pass: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ViT Test Failed: {e}")
        return False

def demonstrate_data_processing():
    """Demonstrate data processing capabilities"""
    print("\n📊 Testing Data Processing...")
    
    try:
        # Test our dataset splitting module
        from create_dataset_splits import create_patient_level_splits, verify_splits
        
        # Create dummy medical data
        dummy_data = pd.DataFrame({
            'Patient ID': [f'P{i//3:03d}' for i in range(30)],  # 3 images per patient
            'Image Index': [f'img_{i:03d}.png' for i in range(30)],
            'Finding Labels': ['COVID-19'] * 15 + ['Pneumonia'] * 15,
            'Binary_Label': ['COVID-19'] * 15 + ['Pneumonia'] * 15
        })
        
        print(f"  ✅ Created dummy dataset: {len(dummy_data)} samples")
        
        # Test splitting
        train_df, test_df, val_df = create_patient_level_splits(dummy_data)
        print(f"  ✅ Train/Test/Val split: {len(train_df)}/{len(test_df)}/{len(val_df)}")
        
        # Verify splits
        verify_splits(train_df, test_df, val_df)
        print("  ✅ Patient-level splitting verified!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data Processing Test Failed: {e}")
        return False

def check_dataset_availability():
    """Check if COVID-19 dataset is available"""
    print("\n💾 Checking Dataset Availability...")
    
    dataset_path = Path('covid-chestxray-dataset')
    if dataset_path.exists():
        metadata_path = dataset_path / 'metadata.csv'
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            print(f"  ✅ Dataset found: {len(df)} images")
            print(f"  ✅ Metadata: {metadata_path}")
            return True
        else:
            print("  ⚠️  Dataset folder exists but no metadata.csv")
    else:
        print("  ⚠️  Dataset not found")
        print("  💡 To download: git clone https://github.com/ieee8023/covid-chestxray-dataset.git")
    
    return False

def main():
    """Main demonstration function"""
    print("🚀 PSU Week 1 Assignment - COVID-19 ViT Classification Demo")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Requirements
    if check_requirements():
        success_count += 1
    
    # Test 2: Vision Transformer
    if demonstrate_vision_transformer():
        success_count += 1
        
    # Test 3: Data Processing
    if demonstrate_data_processing():
        success_count += 1
    
    # Test 4: Dataset
    if check_dataset_availability():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📈 Demo Results: {success_count}/{total_tests} tests passed")
    
    if success_count >= 3:
        print("✅ Code is ready for submission!")
        print("\n💡 Next steps:")
        print("  1. Download dataset: git clone https://github.com/ieee8023/covid-chestxray-dataset.git")
        print("  2. Create splits: python create_dataset_splits.py") 
        print("  3. Train model: python vit_covid19_classifier.py")
    else:
        print("❌ Some issues need to be resolved")
        print("💡 Install missing requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()