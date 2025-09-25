"""
Chest X-ray Dataset Splitting Script
Creates train/test/validation splits for the chest X-ray classification task
Supports both NIH dataset (45GB) and smaller COVID-19 dataset (500MB)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm
import json

def load_covid_dataset(data_dir):
    """Load COVID-19 Chest X-ray dataset (smaller alternative)"""
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        print("Please ensure the COVID-19 chest X-ray dataset is downloaded")
        print("Download from: https://github.com/ieee8023/covid-chestxray-dataset")
        return None
    
    print(f"Found COVID-19 dataset at: {data_dir}")
    df = pd.read_csv(metadata_path)
    
    # Clean the data for binary classification
    print(f"Original dataset size: {len(df)} images")
    
    # Filter for X-ray images only (exclude CT scans)
    df_xray = df[df['modality'] == 'X-ray'].copy()
    print(f"X-ray images only: {len(df_xray)} images")
    
    # Filter for PA and AP views only and remove missing data
    df_filtered = df_xray[df_xray['view'].isin(['PA', 'AP'])].copy()
    df_filtered = df_filtered.dropna(subset=['patientid', 'filename']).copy()
    print(f"After filtering views and cleaning: {len(df_filtered)} images")
    
    # Create binary labels: COVID/Disease vs Normal
    def create_binary_label(finding):
        if pd.isna(finding) or finding == '':
            return 'Normal'
        finding_str = str(finding).lower()
        if 'covid-19' in finding_str or 'covid' in finding_str:
            return 'COVID-19'
        elif any(disease in finding_str for disease in ['pneumonia', 'viral', 'bacterial']):
            return 'Pneumonia'
        else:
            return 'Normal'
    
    df_filtered['Binary_Label'] = df_filtered['finding'].apply(create_binary_label)
    
    # Focus on COVID vs Pneumonia (remove Normal for balanced classification)
    df_disease = df_filtered[df_filtered['Binary_Label'].isin(['COVID-19', 'Pneumonia'])].copy()
    print(f"COVID-19 and Pneumonia cases: {len(df_disease)} images")
    
    # Balance the dataset
    covid_samples = df_disease[df_disease['Binary_Label'] == 'COVID-19']
    pneumonia_samples = df_disease[df_disease['Binary_Label'] == 'Pneumonia']
    
    print(f"COVID-19 cases: {len(covid_samples)}")
    print(f"Pneumonia cases: {len(pneumonia_samples)}")
    
    min_samples = min(len(covid_samples), len(pneumonia_samples))
    
    if min_samples > 0:
        balanced_covid = covid_samples.sample(n=min_samples, random_state=42)
        balanced_pneumonia = pneumonia_samples.sample(n=min_samples, random_state=42)
        df_balanced = pd.concat([balanced_covid, balanced_pneumonia], ignore_index=True)
        print(f"Balanced dataset: {len(df_balanced)} images ({min_samples} per class)")
    else:
        df_balanced = df_disease
        print(f"Using all available data: {len(df_balanced)} images")
    
    # Rename columns to match expected format
    df_balanced['Patient ID'] = df_balanced['patientid']
    df_balanced['Image Index'] = df_balanced['filename']
    df_balanced['Finding Labels'] = df_balanced['finding']
    
    return df_balanced
    
    print(f"Final dataset size: {len(df_filtered)} images")
    print("Class distribution:")
    print(df_filtered['Binary_Label'].value_counts())
    
    return df_filtered

def load_nih_metadata(data_dir):
    """Load NIH Chest X-ray dataset metadata"""
    metadata_path = os.path.join(data_dir, 'Data_Entry_2017_v2020.csv')
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        print("Please ensure the NIH dataset is downloaded and extracted")
        return None
    
    df = pd.read_csv(metadata_path)
    
    # Clean the data
    print(f"Original dataset size: {len(df)} images")
    
    # Remove images with 'No Finding' label for our classification task
    # We want to focus on Normal vs Pneumonia/COVID detection
    df_filtered = df[~df['Finding Labels'].str.contains('No Finding', na=False)]
    print(f"After removing 'No Finding': {len(df_filtered)} images")
    
    # Create binary labels: Normal vs Disease (Pneumonia, COVID, etc.)
    def create_binary_label(finding_labels):
        if pd.isna(finding_labels) or finding_labels == '':
            return 'Normal'
        elif any(disease in finding_labels.lower() for disease in ['pneumonia', 'covid', 'infiltration']):
            return 'Disease'
        else:
            return 'Normal'
    
    df_filtered['Binary_Label'] = df_filtered['Finding Labels'].apply(create_binary_label)
    
    # Balance the dataset (take equal samples from each class)
    normal_samples = df_filtered[df_filtered['Binary_Label'] == 'Normal']
    disease_samples = df_filtered[df_filtered['Binary_Label'] == 'Disease']
    
    # Take minimum of both classes to balance
    min_samples = min(len(normal_samples), len(disease_samples), 10000)  # Cap at 10k for manageable size
    
    balanced_normal = normal_samples.sample(n=min_samples, random_state=42)
    balanced_disease = disease_samples.sample(n=min_samples, random_state=42)
    
    df_balanced = pd.concat([balanced_normal, balanced_disease], ignore_index=True)
    
    print(f"Balanced dataset size: {len(df_balanced)} images")
    print(f"Normal cases: {len(balanced_normal)}")
    print(f"Disease cases: {len(balanced_disease)}")
    
    return df_balanced

def create_patient_level_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create train/test/validation splits at patient level to prevent data leakage
    Standard 70/15/15 split for medical imaging
    """
    # Get unique patients
    unique_patients = df['Patient ID'].unique()
    print(f"Number of unique patients: {len(unique_patients)}")
    
    # First split: separate validation patients (15%)
    train_test_patients, val_patients = train_test_split(
        unique_patients,
        test_size=val_size,
        random_state=random_state,
        stratify=None  # Can't stratify on patient level easily
    )
    
    # Second split: separate train (70%) and test (15%) patients
    # From remaining 85%, we want 70% train and 15% test
    # So test should be 15%/(100%-15%) = 15%/85% ≈ 0.176
    adjusted_test_size = test_size / (1 - val_size)
    
    train_patients, test_patients = train_test_split(
        train_test_patients,
        test_size=adjusted_test_size,
        random_state=random_state
    )
    
    # Create the splits based on patient IDs
    train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
    test_df = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)
    val_df = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    
    return train_df, test_df, val_df

def verify_splits(train_df, test_df, val_df):
    """Verify that the splits are properly separated and balanced"""
    
    total_samples = len(train_df) + len(test_df) + len(val_df)
    
    print("\n=== DATASET SPLIT VERIFICATION ===")
    print(f"Total samples: {total_samples:,}")
    print(f"Training: {len(train_df):,} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"Test: {len(test_df):,} samples ({len(test_df)/total_samples*100:.1f}%)")
    print(f"Validation: {len(val_df):,} samples ({len(val_df)/total_samples*100:.1f}%)")
    
    # Check for patient overlap
    train_patients = set(train_df['Patient ID'].unique())
    test_patients = set(test_df['Patient ID'].unique())
    val_patients = set(val_df['Patient ID'].unique())
    
    overlap_train_test = train_patients.intersection(test_patients)
    overlap_train_val = train_patients.intersection(val_patients)
    overlap_test_val = test_patients.intersection(val_patients)
    
    print(f"\n=== PATIENT OVERLAP CHECK ===")
    print(f"Train-Test overlap: {len(overlap_train_test)} patients")
    print(f"Train-Validation overlap: {len(overlap_train_val)} patients") 
    print(f"Test-Validation overlap: {len(overlap_test_val)} patients")
    
    if len(overlap_train_test) == 0 and len(overlap_train_val) == 0 and len(overlap_test_val) == 0:
        print("✅ No patient overlap detected - splits are clean!")
    else:
        print("❌ Patient overlap detected - data leakage possible!")
    
    # Check class distribution
    print(f"\n=== CLASS DISTRIBUTION ===")
    for split_name, split_df in [("Training", train_df), ("Test", test_df), ("Validation", val_df)]:
        class_counts = split_df['Binary_Label'].value_counts()
        print(f"{split_name}:")
        for label, count in class_counts.items():
            percentage = count / len(split_df) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")

def save_splits(train_df, test_df, val_df, output_dir):
    """Save the split datasets to CSV files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files
    train_path = os.path.join(output_dir, 'train_split.csv')
    test_path = os.path.join(output_dir, 'test_split.csv')
    val_path = os.path.join(output_dir, 'validation_split.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n=== SPLITS SAVED ===")
    print(f"Training split: {train_path}")
    print(f"Test split: {test_path}")
    print(f"Validation split: {val_path}")
    
    # Save split statistics
    stats = {
        'total_samples': len(train_df) + len(test_df) + len(val_df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'validation_samples': len(val_df),
        'train_percentage': len(train_df) / (len(train_df) + len(test_df) + len(val_df)) * 100,
        'test_percentage': len(test_df) / (len(train_df) + len(test_df) + len(val_df)) * 100,
        'validation_percentage': len(val_df) / (len(train_df) + len(test_df) + len(val_df)) * 100,
        'random_seed': 42
    }
    
    stats_path = os.path.join(output_dir, 'split_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Split statistics: {stats_path}")

class DataProtection:
    """Class to protect validation set from accidental access"""
    
    def __init__(self, output_dir):
        self.train_path = os.path.join(output_dir, 'train_split.csv')
        self.test_path = os.path.join(output_dir, 'test_split.csv') 
        self.val_path = os.path.join(output_dir, 'validation_split.csv')
        self._validation_accessed = False
        
    def get_train_data(self):
        """Get training data"""
        return pd.read_csv(self.train_path)
    
    def get_test_data(self):
        """Get test data"""
        return pd.read_csv(self.test_path)
    
    def get_validation_data(self, confirm_final_evaluation=False):
        """
        Get validation data - only for final evaluation!
        
        Args:
            confirm_final_evaluation: Must be True to access validation data
        """
        if not confirm_final_evaluation:
            raise ValueError(
                "🚫 Validation data access denied!\n"
                "Validation set should ONLY be used for final evaluation.\n"
                "Set confirm_final_evaluation=True if you're ready for final testing."
            )
        
        if self._validation_accessed:
            print("⚠️  WARNING: Validation set has been accessed before!")
        
        self._validation_accessed = True
        print("🔓 Accessing validation set for final evaluation...")
        return pd.read_csv(self.val_path)

def main():
    """Main function to create dataset splits - works with real datasets only"""
    
    # Dataset locations to check
    possible_dirs = [
        'covid-chestxray-dataset',           # Current directory
        'data/covid-chestxray-dataset',      # Data subfolder
        'data/raw/covid-chestxray-dataset',  # Full path
        'data/raw/NIH_ChestXray',            # NIH dataset
    ]
    
    output_dir = 'data/processed'
    
    print("🏥 Chest X-ray Dataset Splitting (70/15/15)")
    print("=" * 50)
    
    # Try to find a real dataset
    df = None
    dataset_type = None
    found_dir = None
    
    for data_dir in possible_dirs:
        if os.path.exists(os.path.join(data_dir, 'metadata.csv')):
            print(f"📁 Found COVID-19 dataset at: {data_dir}")
            df = load_covid_dataset(data_dir)
            dataset_type = "COVID-19"
            found_dir = data_dir
            break
        elif os.path.exists(os.path.join(data_dir, 'Data_Entry_2017_v2020.csv')):
            print(f"📁 Found NIH dataset at: {data_dir}")
            df = load_nih_metadata(data_dir)
            dataset_type = "NIH"
            found_dir = data_dir
            break
    
    if df is None:
        print("❌ No dataset found!")
        print("\n📥 Please download one of these datasets:")
        print("\n🦠 Option 1: COVID-19 Chest X-ray Dataset (Recommended - 500MB)")
        print("   git clone https://github.com/ieee8023/covid-chestxray-dataset.git")
        print("   Or download ZIP from: https://github.com/ieee8023/covid-chestxray-dataset")
        print("\n🏥 Option 2: NIH Chest X-ray Dataset (Large - 45GB)")
        print("   Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("\n📁 Place the dataset in one of these locations:")
        for dir_path in possible_dirs:
            print(f"   - {dir_path}")
        return None
    
    print(f"✅ Using {dataset_type} dataset from: {found_dir}")
    print(f"📊 Total samples: {len(df)}")
    
    # Create patient-level splits (70/15/15)
    train_df, test_df, val_df = create_patient_level_splits(df)
    
    # Verify splits
    verify_splits(train_df, test_df, val_df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation_split.csv'), index=False)
    
    # Save metadata
    split_info = {
        'dataset_type': dataset_type,
        'dataset_path': found_dir,
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'validation_samples': len(val_df),
        'split_ratios': {
            'train': f"{len(train_df)/len(df)*100:.1f}%",
            'test': f"{len(test_df)/len(df)*100:.1f}%", 
            'validation': f"{len(val_df)/len(df)*100:.1f}%"
        },
        'class_distribution': {
            'train': train_df['Binary_Label'].value_counts().to_dict(),
            'test': test_df['Binary_Label'].value_counts().to_dict(),
            'validation': val_df['Binary_Label'].value_counts().to_dict()
        }
    }
    
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Splits saved to: {output_dir}")
    print(f"📊 Split info saved to: {os.path.join(output_dir, 'split_info.json')}")
    
    # Initialize data loader for protected access
    loader = DataProtection(output_dir)
    print(f"\n📚 DataLoader initialized. Use loader.get_train_data() and loader.get_test_data()")
    print(f"⚠️  Validation data requires confirm_final_evaluation=True")
    
    return loader
    print("📊 Loading NIH dataset metadata...")
    df = load_nih_metadata(data_dir)
    
    if df is None:
        print("❌ Failed to load dataset. Please check data directory.")
        return
    
    # Create patient-level splits
    print("\n🔀 Creating patient-level splits...")
    train_df, test_df, val_df = create_patient_level_splits(
        df, 
        test_size=0.2, 
        val_size=0.1, 
        random_state=42
    )
    
    # Verify splits
    verify_splits(train_df, test_df, val_df)
    
    # Save splits
    print("\n💾 Saving split files...")
    save_splits(train_df, test_df, val_df, output_dir)
    
    # Create data protection example
    print("\n🔒 Setting up data protection...")
    data_manager = DataProtection(output_dir)
    
    # Example usage
    print("\n📋 Example usage:")
    print("✅ Accessing training data:")
    train_sample = data_manager.get_train_data()
    print(f"   Training set shape: {train_sample.shape}")
    
    print("✅ Accessing test data:")
    test_sample = data_manager.get_test_data()
    print(f"   Test set shape: {test_sample.shape}")
    
    print("🚫 Attempting to access validation data incorrectly:")
    try:
        val_sample = data_manager.get_validation_data()
    except ValueError as e:
        print(f"   {e}")
    
    print("\n✅ Dataset splitting completed successfully!")
    print(f"📁 Output files saved to: {output_dir}")
    
if __name__ == "__main__":
    main()
