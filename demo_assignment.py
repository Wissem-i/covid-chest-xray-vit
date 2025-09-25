#!/usr/bin/env python3
"""
COVID-19 Chest X-ray Classification - Assignment Demo
Student implementation for PSU Week 1 assignment
"""

import sys
import importlib
import pandas as pd
import numpy as np
from pathlib import Path

def check_packages():
    """Check if required packages are installed"""
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'timm': 'PyTorch Image Models',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow'
    }
    
    missing = []
    print("Checking required packages:")
    for package, name in packages.items():
        try:
            importlib.import_module(package)
            print(f"  Found: {name}")
        except ImportError:
            print(f"  Missing: {name}")
            missing.append(package)
    
    if missing:
        print(f"\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    print("All packages installed successfully!")
    return True

def test_vision_transformer():
    """Test Vision Transformer model creation"""
    try:
        import timm
        import torch
        
        print("\nTesting Vision Transformer model:")
        # Create a simple ViT model for testing
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"  Model created successfully")
        print(f"  Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"  Error creating model: {e}")
        return False

def test_data_processing():
    """Test data processing and splitting functionality"""
    print("\nTesting data processing:")
    
    # Create dummy dataset for testing
    np.random.seed(42)
    n_samples = 30
    
    data = []
    patients = [f"patient_{i//3:03d}" for i in range(n_samples)]
    labels = np.random.choice(['COVID-19', 'Pneumonia'], n_samples)
    
    for i in range(n_samples):
        data.append({
            'patient_id': patients[i],
            'image_path': f'image_{i:03d}.jpg',
            'label': labels[i]
        })
    
    df = pd.DataFrame(data)
    print(f"  Created sample dataset: {len(df)} images")
    
    # Test patient-level splitting
    unique_patients = df['patient_id'].unique()
    print(f"  Number of patients: {len(unique_patients)}")
    
    # Split patients (not images) to prevent data leakage
    np.random.shuffle(unique_patients)
    n_train = int(0.6 * len(unique_patients))
    n_test = int(0.2 * len(unique_patients))
    
    train_patients = unique_patients[:n_train]
    test_patients = unique_patients[n_train:n_train+n_test]
    val_patients = unique_patients[n_train+n_test:]
    
    train_df = df[df['patient_id'].isin(train_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    
    print(f"  Split completed: {len(train_df)}/{len(test_df)}/{len(val_df)} samples")
    
    # Verify no patient overlap
    train_set = set(train_df['patient_id'])
    test_set = set(test_df['patient_id'])
    val_set = set(val_df['patient_id'])
    
    overlaps = [
        len(train_set.intersection(test_set)),
        len(train_set.intersection(val_set)), 
        len(test_set.intersection(val_set))
    ]
    
    if sum(overlaps) == 0:
        print("  Patient-level splitting verified - no data leakage!")
        return True
    else:
        print("  Warning: Patient overlap detected!")
        return False

def check_dataset():
    """Check if COVID dataset is available"""
    dataset_path = Path("covid-chestxray-dataset")
    if dataset_path.exists():
        print(f"\nDataset found at: {dataset_path}")
        return True
    else:
        print(f"\nDataset not found.")
        print("To download: git clone https://github.com/ieee8023/covid-chestxray-dataset.git")
        return False

def main():
    """Run the complete demo"""
    print("COVID-19 Vision Transformer Assignment - Demo")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Package installation
    if check_packages():
        tests_passed += 1
    
    # Test 2: Vision Transformer
    if test_vision_transformer():
        tests_passed += 1
    
    # Test 3: Data processing
    if test_data_processing():
        tests_passed += 1
    
    # Test 4: Dataset availability
    if check_dataset():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("All systems working correctly!")
    else:
        print("Some components need attention.")

if __name__ == "__main__":
    main()