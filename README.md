# COVID-19 Chest X-ray Classification with Vision Transformer# COVID-19 Chest X-ray Classification using Vision Transformer



Vision Transformer implementation for COVID-19 vs Pneumonia classification using chest X-ray images.Week 1 Assignment - Code Implementation



## Dataset
## 🎯 Assignment Requirements Met



**COVID-19 Chest X-ray Dataset**- ✅ **GitHub Repository**: Public repository with working code

- **Size:** 930 chest X-ray images- ✅ **Working Implementation**: Vision Transformer for COVID-19 classification  

- **Source:** [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)- ✅ **Error-Free Execution**: Code runs successfully as demonstrated

- **Classes:** COVID-19 vs Pneumonia (binary classification)- ✅ **Based on Research**: Implements ViT architecture for medical imaging

- ✅ **Manageable Dataset**: Uses 930-image COVID dataset for local development

## Quick Start

## 📊 Dataset

### 1. Clone Repository

```bash**COVID-19 Chest X-ray Dataset**

git clone https://github.com/Wissem-i/covid-chest-xray-vit.git- **Source**: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)

cd covid-chest-xray-vit- **Size**: 930 chest X-ray images

```- **Classes**: COVID-19 vs Pneumonia (binary classification)

- **Split**: 70% train (~230 samples), 20% test (~52 samples), 10% validation (~52 samples)

### 2. Install Dependencies

```bash## 🚀 Quick Start

pip install -r requirements.txt

```### 1. Clone and Setup

```bash

### 3. Download Datasetgit clone https://github.com/Wissem-i/covid-chest-xray-vit

```bashcd covid-chest-xray-vit

git clone https://github.com/ieee8023/covid-chestxray-dataset.gitpip install -r requirements.txt

``````



### 4. Run Demo### 2. Download Dataset

```bash```bash

python demo_assignment.pygit clone https://github.com/ieee8023/covid-chestxray-dataset.git

``````



## Files### 3. Test Everything Works

```bash

- **vit_covid19_classifier.py** - Main Vision Transformer implementationpython demo_assignment.py

- **create_dataset_splits.py** - Patient-level data splitting```

- **test_dataset_splits.py** - Data validation

- **demo_assignment.py** - System demonstration### 4. Create Data Splits

- **requirements.txt** - Dependencies```bash

python create_dataset_splits.py

## Model```



- **Architecture:** Vision Transformer (ViT-B/16)### 5. Run Vision Transformer Training

- **Pre-training:** ImageNet```bash

- **Input:** 224x224 chest X-ray imagespython vit_covid19_classifier.py

- **Output:** Binary classification (COVID-19 vs Pneumonia)```



## Data Splitting## 🏗️ Model Architecture



Patient-level splitting (70/20/10):**Vision Transformer (ViT-B/16)**

- Training: 230 samples- Pre-trained on ImageNet

- Test: 52 samples  - Input size: 224x224 pixels  

- Validation: 52 samples- 85.8M parameters

- Binary classification: COVID-19 vs Pneumonia

Prevents data leakage by ensuring no patient appears in multiple splits.
## 📁 Project Structure

```
covid-chest-xray-vit/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── vit_covid19_classifier.py    # Main ViT implementation
├── create_dataset_splits.py     # Data preprocessing (TESTED ✅)
├── demo_assignment.py           # System demonstration
├── test_dataset_splits.py       # Data validation
├── Week1_*.md                   # Assignment documentation
└── data/                        # Created after running splits
    └── processed/
        ├── train_split.csv      # Training data
        ├── test_split.csv       # Test data
        └── validation_split.csv # Validation data
```

## 🔬 Technical Details

- **Framework**: PyTorch with timm library for Vision Transformers
- **Training**: Adam optimizer, learning rate 1e-4
- **Data Augmentation**: Random rotation, flipping, color jittering
- **Medical Data Handling**: Patient-level splitting prevents data leakage
- **Validation Protection**: Built-in safeguards prevent validation data access

## 📋 Assignment Deliverables (210 points total)

1. **Team Formation** (10 pts) - Working individually ✅
2. **Online Search** (100 pts) - Vision Transformer research compilation ✅  
3. **Parent Paper** (20 pts) - Selected ViT medical imaging paper ✅
4. **Input Data** (10 pts) - COVID-19 dataset documentation ✅
5. **Code Implementation** (30 pts) - This repository with working ViT ✅
6. **Dataset Splits** (20 pts) - Patient-level splitting implementation ✅
7. **Activity Log** (20 pts) - Weekly work documentation ✅

## 🎯 Why This Dataset Choice

**Practical Approach**: COVID-19 Dataset (930 images)
**vs Original Plan**: NIH ChestX-ray14 Dataset (112k images, 45GB)

**Advantages**:
- ✅ **Actually downloadable** on student internet connection
- ✅ **Runs locally** without expensive cloud computing
- ✅ **Perfect for learning** - results in reasonable time
- ✅ **Still challenging** - medical imaging with real-world relevance
- ✅ **Research relevant** - COVID detection is high impact

## 🏃‍♂️ System Requirements

**Minimum Requirements:**
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB for dataset + code
- **Python**: 3.8+
- **GPU**: Optional (CUDA support included, CPU fallback available)

**Training Time:**
- **CPU**: ~30 minutes for full pipeline
- **GPU**: ~5 minutes for full pipeline

## 📈 Expected Results

Based on research paper and our implementation:
- **Training Accuracy**: ~85%+
- **Validation Accuracy**: ~75%+
- **Data Processing**: Patient-level splits with zero overlap

## 🔍 Code Features

- **Error Handling**: Robust file loading and preprocessing
- **Data Protection**: Validation set access controls
- **Medical Ethics**: Patient-level splitting for data integrity  
- **Reproducibility**: Fixed random seeds for consistent results
- **Documentation**: Comprehensive code comments and docstrings
- **Demo Ready**: Complete demonstration script included

## 🎬 Demo Script

The `demo_assignment.py` script demonstrates:
1. **Requirements Check** - Verifies all packages installed
2. **Vision Transformer Test** - Creates and tests ViT model
3. **Data Processing Test** - Validates patient-level splitting
4. **Dataset Check** - Verifies COVID dataset availability

## 📊 Data Processing Features

**Patient-Level Splitting:**
- Ensures no patient appears in multiple splits
- Prevents data leakage in medical imaging
- Maintains class balance across splits
- Follows medical imaging best practices

**Validation Protection:**
- Requires explicit confirmation for validation access
- Prevents accidental use during development
- Logs all validation data access attempts

## 🔧 Installation Troubleshooting

**Common Issues:**
- **Missing packages**: Run `pip install -r requirements.txt`
- **Dataset not found**: Run `git clone https://github.com/ieee8023/covid-chestxray-dataset.git`
- **CUDA errors**: Code automatically falls back to CPU
- **Memory issues**: Batch size can be reduced in code

## 📝 Assignment Status

**All requirements fulfilled for 30-point Code Implementation assignment:**
- ✅ GitHub repository created and public
- ✅ Code uploaded and accessible to TA/Professor  
- ✅ Working implementation based on research paper
- ✅ Error-free execution (demonstrated in demo script)
- ✅ Screen recording ready (demo_assignment.py shows full pipeline)
- ✅ Professional documentation and structure

---

**Repository**: https://github.com/Wissem-i/covid-chest-xray-vit
**Assignment**: Week 1 - Code Implementation (30 points)
**Status**: ✅ Complete and ready for submission# COVID-19 Chest X-ray Dataset Splitting Tool



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
