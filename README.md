# GSLM-DSM: A Deep Learning Framework for Sequence Analysis

## Overview
GSLM-DSM (Genomic Sequence Language Model - Deep Sequence Model) is a deep learning framework designed for genomic sequence analysis, leveraging convolutional neural networks (TextCNN as the default backbone) to process dual-modal sequence features (GPN-MSA and SpliceBERT). This repository contains the core configuration, environment setup, and data organization guidelines for replicating and extending the model.
## Environment Setup
### Prerequisites
- Anaconda/Miniconda (Python 3.9+)
- CUDA 12.1 (compatible with PyTorch 2.5.1)
- GPU with CUDA support (recommended: NVIDIA RTX 3090/4090 or Tesla V100/A100)

### Step 1: Create Conda Environment
Use the provided GSLM-DSM.yml file to build the environment with all dependencies:
```javascript
# Create environment from YAML file
conda env create -f GSLM-DSM.yml

# Activate the environment
conda activate GSLM-DSM
```
### Step 2: Verify Environment
Confirm all packages are installed correctly:
```javascript
# Check PyTorch/CUDA compatibility
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check key dependencies
python -c "import numpy, pandas, scikit-learn; print('All core packages installed')"
```

## Data Preparation
### Data Structure
All input data should be placed in the input/ directory of the project root. The framework requires dual-modal sequence features (GPN-MSA and SpliceBERT) in .npy format and label files in .csv format.
### Sequence Feature Files (.npy)
Two types of sequence features are required, with fixed dimensions and lengths:

|Feature Type|Length|Dimension|File Format|
|------|------|------|------|
|GPN-MSA|128|768|npy|
|SpliceBERT|503|512|npy|

### Label Files (.csv)
Labels are stored in CSV files and must be aligned with the order of sequence features (1:1 mapping between feature rows and label rows).

|Data Split|Label Path|
|------|------|
|Training Set|input/Balance_train_CB.csv|
|Test Set (Full)|input/Balance_test_CB.csv|
|Test Set (Non-Canonical)|input/test_noncanonical.csv|
|Test Set (Canonical)|input/test_canonical.csv|

### Model-Specific Parameters
|Parameter|Purpose|
|------|------|
|embedding_size_DLM1|Embedding dimension for GPN-MSA (fixed to 768, match feature dimension)|
|DLM_seq_len1|Sequence length for GPN-MSA (fixed to 128, match feature length)|
|embedding_size_DLM2|Embedding dimension for SpliceBERT (fixed to 512)|
|DLM_seq_len2|Sequence length for SpliceBERT (fixed to 503)|

## Notes
1. All file paths in sta_config.py are relative to the project root (replace absolute paths with input/ for portability).
2. Ensure CUDA 12.1 is installed and compatible with the PyTorch version in GSLM-DSM.yml.
3. For cross-validation (--CV True), the output will be saved to result/sta_test.csv; for hold-out testing, results are saved to the same path with test set metrics.
4. Feature files must have consistent sample ordering with label files to avoid misalignment.

## Citation
If you use GSLM-DSM in your research, please cite the relevant work (xxxx).

## Contact
For technical issues, please contact the repository maintainer (q24301229@stu.ahu.edu.cn) with details of your environment and error logs.






