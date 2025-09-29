#!/usr/bin/env python3
"""
Dataset verification script for COVID-19 chest X-ray dataset.
This script verifies the dataset is properly downloaded and structured.
"""

import os
import csv
from pathlib import Path

def verify_dataset():
    """Verify the COVID-19 dataset is properly downloaded and structured."""
    print("COVID-19 Dataset Verification")
    print("=" * 40)
    
    # Check if dataset directory exists
    dataset_dir = Path("covid-chestxray-dataset")
    if not dataset_dir.exists():
        print("❌ Dataset directory not found: covid-chestxray-dataset/")
        return False
    
    print(f"✅ Dataset directory found: {dataset_dir}")
    
    # Check images directory
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        print("❌ Images directory not found: covid-chestxray-dataset/images/")
        return False
    
    # Count images
    image_files = list(images_dir.glob("*"))
    image_count = len([f for f in image_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    print(f"✅ Images directory found with {image_count} image files")
    
    # Check metadata file
    metadata_file = dataset_dir / "metadata.csv"
    if not metadata_file.exists():
        print("❌ Metadata file not found: covid-chestxray-dataset/metadata.csv")
        return False
    
    # Check metadata file size and content
    metadata_size = metadata_file.stat().st_size
    print(f"✅ Metadata file found: {metadata_size} bytes")
    
    # Quick metadata validation
    try:
        with open(metadata_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            row_count = sum(1 for row in reader)
        print(f"✅ Metadata CSV has {len(headers)} columns and {row_count} rows")
        print(f"   Sample headers: {headers[:5]}...")
    except Exception as e:
        print(f"⚠️  Warning: Could not fully validate metadata: {e}")
    
    # Check other important files
    important_files = ["README.md", "SCHEMA.md"]
    for filename in important_files:
        file_path = dataset_dir / filename
        if file_path.exists():
            print(f"✅ {filename} found ({file_path.stat().st_size} bytes)")
        else:
            print(f"⚠️  {filename} not found (optional)")
    
    # Summary
    print("\nDataset Verification Summary:")
    print(f"📁 Dataset location: {dataset_dir.absolute()}")
    print(f"📊 Total images: {image_count}")
    print(f"📄 Metadata size: {metadata_size:,} bytes")
    
    if image_count >= 900 and metadata_size > 500000:  # Expected ~930 images, ~599KB metadata
        print("✅ Dataset appears complete and ready for use!")
        return True
    else:
        print("⚠️  Dataset may be incomplete")
        return False

def check_parent_paper():
    """Check for parent paper PDF."""
    print("\nParent Paper Check:")
    print("=" * 20)
    
    pdf_file = Path("COVID19_ViT_Parent_Paper.pdf")
    if pdf_file.exists():
        size = pdf_file.stat().st_size
        print(f"✅ Parent paper found: {size:,} bytes")
        return True
    else:
        print("📄 Parent paper not found (manual download required)")
        print("   Download from: https://arxiv.org/pdf/2003.11597.pdf")
        print("   Save as: COVID19_ViT_Parent_Paper.pdf")
        return False

def main():
    """Main verification function."""
    print("COVID-19 ViT Project - Setup Verification\n")
    
    dataset_ok = verify_dataset()
    paper_ok = check_parent_paper()
    
    print("\n" + "=" * 50)
    if dataset_ok:
        print("🎉 SUCCESS: Dataset is properly downloaded and ready!")
    else:
        print("❌ Dataset verification failed")
        
    if not paper_ok:
        print("📋 TODO: Download parent paper from ArXiv")
    
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run demo: python demo_covid_vit.py")
    print("3. Run training: python covid_vit_implementation.py")

if __name__ == "__main__":
    main()