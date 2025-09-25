# Week 1: Input Data Documentation

## Dataset I'm Using

**Dataset Name:** COVID-19 Chest X-ray Dataset

**Where to get it:** https://github.com/ieee8023/covid-chestxray-dataset

**What it contains:** 
- About 930 chest X-ray images
- From multiple medical institutions globally
- Both PA and AP view chest X-rays
- All images are in PNG/JPG format
- Comes with a CSV metadata file that has labels and patient info

## About the Dataset

The images are chest X-rays from patients with COVID-19, pneumonia, and other respiratory conditions. Each image has metadata including the diagnosis, patient age, sex, and other clinical information.

**What makes this good for my project:**
- Small size - I can actually download and work with this locally
- Has COVID-19 labels which is perfect for my research paper
- Other researchers have used it so I know it works
- Free to download from GitHub
- Good variety of cases for training

**Potential issues:**
- Smaller dataset means less training data
- Images from different hospitals might have different quality/equipment
- Limited to certain disease types

## How I'll Use It

I'm planning to focus on binary classification: COVID-19 vs Pneumonia. This gives me a manageable problem to work on with the Vision Transformer architecture.

## My Plan

I'm planning to use a subset of this data since 112k images is probably too much for me to handle with my computer. I'll probably take around 10,000 normal images and 10,000 with diseases to make it balanced.

### Data Access and Availability

**Access Status:** Freely available - no registration required

**Download Information:**
- **Direct Download Link:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **File Format:** .tar.gz archives (12 parts) + metadata CSV files
- **Documentation:** https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

### Dataset Split Information

**Original Paper Split:**
- Training: 70% - 78,468 images  
- Validation: 15% - 16,818 images
- Testing: 15% - 16,834 images

**Our Planned Split:**
- Training: 70% - for model training (78,484 images)
- Test: 20% - for performance evaluation (22,424 images)  
- Validation (Unseen): 10% - final validation (11,212 images)

### Data Preprocessing Requirements

**Required Preprocessing:**
1. **Image resizing:** Resize from 1024x1024 to 224x224 (ViT standard input)
2. **Normalization:** Convert pixel values to [0,1] range, apply ImageNet statistics
3. **Data augmentation:** Random horizontal flip, rotation (±15°), brightness/contrast adjustment
4. **Label processing:** Convert multi-label to binary classification (Normal vs. Pneumonia/COVID)
5. **Train/test splitting:** Ensure patient-level splitting (no patient appears in multiple splits)

**Tools/Libraries Needed:**
- **pandas** for metadata manipulation and CSV processing
- **PIL/torchvision** for image loading and preprocessing
- **PyTorch** for data loading and tensor operations
- **scikit-learn** for stratified splitting and preprocessing utilities
- **numpy** for numerical operations

### Additional Datasets (if any)

**Secondary Dataset 1:**
- **Name:** COVID-19 Chest X-ray Dataset
- **Purpose:** Supplement COVID-19 positive cases (NIH dataset has limited COVID cases)
- **Source:** https://github.com/ieee8023/covid-chestxray-dataset
- **Access:** Freely available GitHub repository

**Secondary Dataset 2:**
- **Name:** RSNA Pneumonia Detection Challenge Dataset
- **Purpose:** Additional pneumonia cases for robust training
- **Source:** https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
- **Access:** Kaggle competition data (free download)

### Data Ethics and Usage

**License:** NIH dataset is public domain (CC0 1.0 Universal)

**Usage Restrictions:** 
- No commercial use restrictions
- Must acknowledge NIH Clinical Center as source
- Cannot re-identify patients (data is already anonymized)

**Citation Required:** 
Xiaosong Wang et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks for Weakly-Supervised Classification and Localization of Common Thorax Diseases." IEEE CVPR 2017

**Privacy Considerations:** 
- All patient identifiers have been removed
- Ages >89 are recorded as 90 to maintain anonymity
- No direct patient consent issues (retrospective analysis of clinical data)

### Implementation Plan

**Data Acquisition Timeline:**
- Week 1: Download NIH dataset (45GB - may take 6-12 hours)
- Week 1-2: Verify data integrity and explore dataset structure  
- Week 2: Implement preprocessing pipeline
- Week 2: Create train/test/validation splits with patient-level separation

**Verification Steps:**
1. Dataset accessibility confirmed (NIH website active)
2. Dataset downloaded successfully (45GB download in progress)
3. Data format matches paper description (PNG images + CSV metadata)
4. Sample size matches reported numbers (112,120 images)
5. Can reproduce basic statistics from paper (disease distributions)
6. Preprocessing pipeline works correctly
4. **Contact authors:** Reach out to paper authors for dataset access guidance

---

## File Attachments and Links

### Provided Materials:
1. **Dataset Link:** https://nihcc.app.box.com/v/ChestXray-NIHCC
2. **Data Documentation:** https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
3. **Kaggle Mirror:** https://www.kaggle.com/nih-chest-xrays/data
4. **Paper with Dataset Details:** https://arxiv.org/abs/1705.02315

### Status Checklist:
- [x] Primary dataset identified and accessible
- [x] Alternative datasets identified (Kaggle, CheXpert)
- [x] Download/access method confirmed (direct download, no registration)
- [x] Data preprocessing requirements understood (resize, normalize, augment)
- [x] Tools and libraries identified (PyTorch, torchvision, pandas)
- [x] Data splits planned (70/20/10 with patient-level separation)

---

**Completion Date:** September 21, 2025
**Data Access Confirmed:** Yes - NIH dataset is freely available
**Ready for Implementation:** Yes - download in progress, preprocessing pipeline ready

**Notes:** 
- Large dataset (45GB) requires significant storage space and download time
- Patient-level splitting is crucial to prevent data leakage
- Consider using subset for initial development due to dataset size
- Multiple high-quality alternative datasets available if needed
