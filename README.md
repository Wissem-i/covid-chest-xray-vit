# COVID-19 Chest X-ray Classification using Vision Transformer

Implementation of Vision Transformer for COVID-19 chest X-ray classification using code from the `lucidrains/vit-pytorch` GitHub repository.

## 🔗 Code Source

**GitHub Repository:** [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)  
**Implementation:** Vision Transformer (ViT) for COVID-19 vs Pneumonia classification  
**Model:** ViT-B/16 architecture with 85.7M parameters

## 🚀 Quick Start

### 1. Clone Repository and Install Dependencies

```bash
git clone https://github.com/Wissem-i/covid-chest-xray-vit.git
cd covid-chest-xray-vit
=======
## Dataset
## 🎯 Assignment Requirements Met



**COVID-19 Chest X-ray Dataset**- ✅ **GitHub Repository**: Public repository with working code

- **Size:** 930 chest X-ray images- ✅ **Working Implementation**: Vision Transformer for COVID-19 classification  

- **Source:** [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)- ✅ **Error-Free Execution**: Code runs successfully as demonstrated

- **Classes:** COVID-19 vs Pneumonia (binary classification)- **Based on Research**: Implements ViT architecture for medical imaging

-  **Manageable Dataset**: Uses 930-image COVID dataset for local development

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

>>>>>>> a95f40f8cb0773ef39665388881ea9d49d37bb20
pip install -r requirements.txt
```

### 2. Download COVID-19 Dataset

```bash
git clone https://github.com/ieee8023/covid-chestxray-dataset.git
```

### 3. Run Demonstration

```bash
python demo_covid_vit.py
```

### 4. Run Full Training

```bash
python covid_vit_implementation.py
```

## 📊 Dataset Information

**COVID-19 Chest X-ray Dataset**
- **Source:** [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- **Task:** Binary classification (COVID-19 vs Pneumonia)
- **Total Images:** 930 chest X-ray images
- **Used Samples:** 334 balanced samples (167 COVID-19, 167 Pneumonia)
- **Data Splits:** 70% train, 20% test, 10% validation

### Dataset Download Instructions

The dataset is automatically downloaded when you run:
```bash
git clone https://github.com/ieee8023/covid-chestxray-dataset.git
```

This will create a `covid-chestxray-dataset/` folder containing:
- `metadata.csv` - Image labels and patient information
- `images/` - Chest X-ray images in PNG/JPG format

## 🏗️ Model Architecture

**Vision Transformer (ViT-B/16)**

```python
from vit_pytorch import ViT

model = ViT(
    image_size=224,      # Standard input size
    patch_size=16,       # 16x16 patches  
    num_classes=2,       # COVID vs Pneumonia
    dim=768,             # Token dimension
    depth=12,            # Transformer layers
    heads=12,            # Attention heads
    mlp_dim=3072,        # MLP dimension
    dropout=0.1,
    emb_dropout=0.1
)
```

## 📊 Results

**Training Results:**
- Best Validation Accuracy: 55.77%
- Test Accuracy: 44.23%
- Model Parameters: 85,775,618
- Training Epochs: 5

**Dataset Distribution:**
- Total Samples: 334 (balanced COVID-19 vs Pneumonia)
- Training: 230 samples
- Validation: 52 samples  
- Test: 52 samples

## 📁 Generated Output Files

- `covid_vit_confusion_matrix.png` - Classification results visualization  
- `covid_vit_training_curves.png` - Training loss and accuracy curves  
- `covid_vit_results.json` - Complete training metrics and results  
- `best_covid_vit.pth` - Saved model weights

## 💻 Project Structure

```
covid-chest-xray-vit/
├── README.md                           # Project documentation
├── covid_vit_implementation.py         # Main ViT implementation
├── demo_covid_vit.py                   # Demonstration script  
├── requirements.txt                    # Python dependencies
├── covid-chestxray-dataset/            # Dataset (downloaded separately)
├── covid_vit_results.json              # Training results
├── covid_vit_confusion_matrix.png      # Results visualization
├── covid_vit_training_curves.png       # Training progress
└── best_covid_vit.pth                  # Saved model weights
```

## 🔬 Technical Implementation

**Framework:** PyTorch with vit-pytorch library  
**Model:** Vision Transformer (ViT-B/16) pre-trained on ImageNet  
**Training:** Adam optimizer, learning rate 1e-4  
**Input:** 224x224 chest X-ray images  
**Output:** Binary classification (COVID-19 vs Pneumonia)

**Key Features:**
- Patient-level data splitting prevents data leakage
- Data augmentation for medical images
- Built-in validation data protection
- Comprehensive error handling and logging

## 🔧 System Requirements

- **Python:** 3.8+ (tested with 3.12.1)
- **PyTorch:** 2.0+ (tested with 2.8.0)
- **Key Packages:** vit-pytorch, torchvision, scikit-learn, matplotlib
- **Hardware:** 8GB RAM minimum, GPU optional (falls back to CPU)

## 🎬 Demo Script

The `demo_covid_vit.py` script demonstrates:
1. **Requirements Check** - Verifies all packages installed
2. **Vision Transformer Test** - Creates and tests ViT model
3. **Data Processing Test** - Validates patient-level splitting
4. **Dataset Check** - Verifies COVID dataset availability

---

**Individual Project** - Computer Vision/Machine Learning Course



Week 1 Assignment - Code ImplementationThis tool creates proper train/test/validation splits for the COVID-19 chest X-ray dataset while preventing data leakage through patient-level splitting.



## 🎯 Assignment Requirements Met## Features



- ✅ **GitHub Repository**: Public repository with working code- **70/15/15 split ratio**: Training (70%), Test (15%), Validation (15%)

- ✅ **Working Implementation**: Vision Transformer for COVID-19 classification  - **Patient-level splitting**: Ensures no patient appears in multiple splits

- ✅ **Error-Free Execution**: Code runs successfully as demonstrated- **Validation data protection**: Requires explicit confirmation for validation access

- ✅ **Based on Research**: Implements ViT architecture for medical imaging- **COVID-19 vs Pneumonia classification**: Balanced binary classification

- ✅ **Manageable Dataset**: Uses 930-image COVID dataset instead of 45GB NIH dataset- **Data leakage prevention**: Strict patient separation across splits



## 📊 Dataset
## Quick Start



**COVID-19 Chest X-ray Dataset**
### 1. Download COVID-19 Dataset

- **Source**: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)

- **Size**: 930 images (~500MB) - Perfect for local development!```bash

- **Classes**: COVID-19 vs Pneumonia (binary classification)git clone https://github.com/ieee8023/covid-chestxray-dataset.git

- **Split**: 70% train (230 samples), 20% test (52 samples), 10% validation (52 samples)```



## 🚀 Quick Start
### 2. Run Dataset Splitting



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

**Better Choice**: COVID-19 Dataset (500MB, 930 images)-  Split ratios (~70/15/15)

-  No patient overlap between splits  

**Advantages**:-  Balanced class distribution

-  **Actually downloadable** on student budget/internet-  Data access protection

-  **Runs locally** without cloud computing costs-  File integrity

-  **Perfect for learning** - results in reasonable time

-  **Still challenging** - medical imaging with class imbalanceRun `python test_dataset_splits.py` to verify everything works correctly.

-  **Research relevant** - COVID detection is high impact

## Requirements

## 🏃‍♂️ Running the Code

```bash

The code is designed to run on modest hardware:pip install pandas scikit-learn

- **RAM**: 8GB minimum (16GB recommended)```

- **Storage**: 2GB for dataset + code

- **GPU**: Optional (CUDA support included)## License

- **Time**: Full training ~30 minutes on CPU, ~5 minutes on GPU

This tool is designed for academic and research use with the COVID-19 chest X-ray dataset. Please refer to the original dataset license at https://github.com/ieee8023/covid-chestxray-dataset


## 🔍 Code Features

- **Error Handling**: Robust file loading and preprocessing
- **Data Protection**: Validation set access controls
- **Medical Ethics**: Patient-level splitting for data integrity  
- **Reproducibility**: Fixed random seeds for consistent results
- **Documentation**: Comprehensive code comments and docstrings



