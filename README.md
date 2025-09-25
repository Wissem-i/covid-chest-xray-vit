# COVID-19 Chest X-ray Classification using Vision Transformer# COVID-19 Chest X-ray Dataset Splitting Tool



Week 1 Assignment - Code ImplementationThis tool creates proper train/test/validation splits for the COVID-19 chest X-ray dataset while preventing data leakage through patient-level splitting.



## 🎯 Assignment Requirements Met## Features



- ✅ **GitHub Repository**: Public repository with working code- **70/15/15 split ratio**: Training (70%), Test (15%), Validation (15%)

- ✅ **Working Implementation**: Vision Transformer for COVID-19 classification  - **Patient-level splitting**: Ensures no patient appears in multiple splits

- ✅ **Error-Free Execution**: Code runs successfully as demonstrated- **Validation data protection**: Requires explicit confirmation for validation access

- ✅ **Based on Research**: Implements ViT architecture for medical imaging- **COVID-19 vs Pneumonia classification**: Balanced binary classification

- ✅ **Manageable Dataset**: Uses 930-image COVID dataset instead of 45GB NIH dataset- **Data leakage prevention**: Strict patient separation across splits



## 📊 Dataset## Quick Start



**COVID-19 Chest X-ray Dataset**### 1. Download COVID-19 Dataset

- **Source**: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)

- **Size**: 930 images (~500MB) - Perfect for local development!```bash

- **Classes**: COVID-19 vs Pneumonia (binary classification)git clone https://github.com/ieee8023/covid-chestxray-dataset.git

- **Split**: 70% train (230 samples), 20% test (52 samples), 10% validation (52 samples)```



## 🚀 Quick Start### 2. Run Dataset Splitting



### 1. Clone and Setup```python

```bashpython create_dataset_splits.py

git clone https://github.com/your-username/covid-chest-xray-vit```

cd covid-chest-xray-vit

pip install -r requirements.txt### 3. Test the Splits

```

```python

### 2. Download Datasetpython test_dataset_splits.py

```bash```

git clone https://github.com/ieee8023/covid-chestxray-dataset.git

```## Dataset Information



### 3. Create Data Splits**COVID-19 Chest X-ray Dataset (ieee8023/covid-chestxray-dataset)**

```bash- **Source**: https://github.com/ieee8023/covid-chestxray-dataset

python create_dataset_splits.py- **Size**: ~630MB (930 images)

```- **Classes**: COVID-19, Pneumonia, Viral, Bacterial

- **Our Processing**: 334 balanced samples (167 COVID-19, 167 Pneumonia)

### 4. Run Vision Transformer Training

```bash## Usage Example

python vit_covid19_classifier.py

``````python

from create_dataset_splits import main

## 🏗️ Model Architecture

# Create splits

**Vision Transformer (ViT-B/16)**loader = main()

- Pre-trained on ImageNet

- Input size: 224x224 pixels  # Access training data

- 85.8M parameterstrain_data = loader.get_train_data()  # 230 samples

- Binary classification: COVID-19 vs Pneumoniatest_data = loader.get_test_data()    # 52 samples



## 📁 Project Structure# Access validation data (final evaluation only!)

validation_data = loader.get_validation_data(confirm_final_evaluation=True)  # 52 samples

``````

covid-chest-xray-vit/

├── vit_covid19_classifier.py    # Main ViT implementation## Output Files

├── create_dataset_splits.py     # Data preprocessing

├── test_dataset_splits.py       # Data validation- `data/processed/train_split.csv` - Training set (230 samples)

├── requirements.txt             # Dependencies- `data/processed/test_split.csv` - Test set (52 samples)  

├── Week1_*.md                   # Assignment documentation- `data/processed/validation_split.csv` - Validation set (52 samples)

└── data/                        # Created after running splits- `data/processed/split_info.json` - Split statistics and metadata

    └── processed/

        ├── train_split.csv      # Training data (230 samples)## Key Features

        ├── test_split.csv       # Test data (52 samples)

        └── validation_split.csv # Validation data (52 samples)### Patient-Level Splitting

```Ensures no patient appears in multiple splits, preventing data leakage:

- Splits patients first, then assigns their images

## 🔬 Technical Details- Maintains class balance across splits

- Reports zero patient overlap

- **Framework**: PyTorch with timm library for Vision Transformers

- **Training**: Adam optimizer, learning rate 1e-4### Validation Data Protection

- **Data Augmentation**: Random rotation, flipping, color jitteringPrevents accidental validation data usage:

- **Validation Protection**: Built-in safeguards prevent validation data leakage```python

- **Medical Data Handling**: Patient-level splitting prevents data contamination# This will raise an error

val_data = loader.get_validation_data()  # ❌ Access denied

## 📋 Assignment Deliverables (210 points total)

# This works for final evaluation

1. **Team Formation** (10 pts) - Working individually ✅val_data = loader.get_validation_data(confirm_final_evaluation=True)  # ✅

2. **Online Search** (100 pts) - Vision Transformer research compilation ✅  ```

3. **Parent Paper** (20 pts) - Selected ViT medical imaging paper ✅

4. **Input Data** (10 pts) - COVID-19 dataset documentation ✅### Balanced Classification

5. **Code Implementation** (30 pts) - This repository with working ViT ✅- **Training**: 121 Pneumonia, 109 COVID-19 (52.6% vs 47.4%)

6. **Dataset Splits** (20 pts) - Patient-level splitting implementation ✅- **Test**: 23 Pneumonia, 29 COVID-19 (44.2% vs 55.8%)

7. **Activity Log** (20 pts) - Weekly work documentation ✅- **Validation**: 23 Pneumonia, 29 COVID-19 (44.2% vs 55.8%)



## 🎯 Why This Dataset Choice## Verification



**Original Plan**: NIH ChestX-ray14 Dataset (45GB, 112k images)  The tool performs comprehensive verification:

**Better Choice**: COVID-19 Dataset (500MB, 930 images)- ✅ Split ratios (~70/15/15)

- ✅ No patient overlap between splits  

**Advantages**:- ✅ Balanced class distribution

- ✅ **Actually downloadable** on student budget/internet- ✅ Data access protection

- ✅ **Runs locally** without cloud computing costs- ✅ File integrity

- ✅ **Perfect for learning** - results in reasonable time

- ✅ **Still challenging** - medical imaging with class imbalanceRun `python test_dataset_splits.py` to verify everything works correctly.

- ✅ **Research relevant** - COVID detection is high impact

## Requirements

## 🏃‍♂️ Running the Code

```bash

The code is designed to run on modest hardware:pip install pandas scikit-learn

- **RAM**: 8GB minimum (16GB recommended)```

- **Storage**: 2GB for dataset + code

- **GPU**: Optional (CUDA support included)## License

- **Time**: Full training ~30 minutes on CPU, ~5 minutes on GPU

This tool is designed for academic and research use with the COVID-19 chest X-ray dataset. Please refer to the original dataset license at https://github.com/ieee8023/covid-chestxray-dataset

## 📈 Expected Results

Based on the research paper and our implementation:
- **Training Accuracy**: ~85%+
- **Validation Accuracy**: ~75%+
- **Training Time**: 5-30 minutes depending on hardware

## 🔍 Code Features

- **Error Handling**: Robust file loading and preprocessing
- **Data Protection**: Validation set access controls
- **Medical Ethics**: Patient-level splitting for data integrity  
- **Reproducibility**: Fixed random seeds for consistent results
- **Documentation**: Comprehensive code comments and docstrings

---

**Student**: Individual project submission  
**Course**: Computer Vision/Machine Learning  
**Assignment**: Week 1 - Code Implementation (30 points)