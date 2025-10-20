# AMP-DFM: Antimicrobial Peptide Discrete Flow Matching

This repository contains the code, data and implementation details for an MSc thesis at Imperial College London. The model is a discrete flow matching model with reweighted jump rates using trained classifiers on antimicrobial peptide property data. The code and implementation details are inspired by the paper by Tong et al.

The data processing and protein structure prediction were processed in separate repositories. This codebase is a refactored version of the original so some details may differ from those shared in the report.

## Overview

AMP-DFM is a generative model for designing antimicrobial peptides with desired properties. The framework combines:

- **Discrete Flow Matching**: A continuous-time generative model that learns to transform noise into valid peptide sequences through a learned flow
- **Multi-Objective Guidance**: XGBoost classifiers trained on antimicrobial activity, haemolysis, and cytotoxicity data guide generation towards peptides with optimal properties
- **Species-Specific Models**: Separate classifiers for generic antimicrobial activity and species-specific activity against E. coli, P. aeruginosa, and S. aureus

The model architecture uses a U-Net-style CNN with dilated convolutions and supports both unconditional and conditional generation modes.

## Repository Structure

```
amp_dfm/
├── src/ampdfm/                    # Core source code
│   ├── classifiers/               # XGBoost classifier implementations
│   ├── dfm/                       # Discrete flow matching components
│   ├── evaluation/                # Model evaluation utilities
│   └── utils/                     # Tokenisation, embeddings, and utilities
├── scripts/                       # Executable scripts for training and sampling
│   ├── data_preprocessing/        # Data clustering, embedding, and dataset prep
│   ├── classifiers/               # Classifier training scripts
│   ├── dfm/                       # DFM training and sampling scripts
│   └── mog/                       # Multi-objective guided sampling scripts
├── configs/                       # YAML configuration files
│   ├── classifiers/               # Classifier training configs
│   ├── flow_matching/             # DFM training configs
│   └── mog/                       # Multi-objective guidance configs
├── data/                          # Datasets and embeddings
│   ├── clustered/                 # Train/val/test splits after clustering
│   ├── dfm/                       # Tokenised datasets for DFM training
│   ├── embeddings/                # ESM2 embeddings and clustering outputs
│   └── filtered/                  # Filtered and processed raw data
├── checkpoints/                   # Trained model weights
│   ├── dfm/                       # DFM model checkpoints
│   └── classifiers/               # XGBoost classifier models
├── outputs/                       # Generated outputs and results
│   ├── classifiers/               # Classifier evaluation metrics
│   ├── peptides/                  # Generated peptide sequences
│   └── model_comparison/          # Comparative analysis results
└── documentation/                 # Environment specifications
```

## Installation

### Prerequisites

- Linux environment (tested on Red Hat Enterprise Linux 8.5)
- CUDA 12.4+ for GPU support
- Anaconda or Miniconda

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd amp_dfm
```

2. Create the conda environment from the provided specification:
```bash
conda env create -f documentation/amp-dfm.yml
```

Alternatively, create a new environment and install key dependencies:
```bash
conda create -n amp-dfm python=3.9
conda activate amp-dfm

# Install from conda
conda install -c pytorch -c bioconda -c nvidia -c conda-forge \
    pytorch=2.4.0 pytorch-cuda=12.4 \
    xgboost=2.1.4 scikit-learn=1.6.1 \
    biopython=1.85 mmseqs2=17.b804f \
    matplotlib seaborn pandas numpy scipy

# Install from pip
pip install fair-esm==2.0.0 transformers datasets torchdiffeq
```

3. Activate the environment:
```bash
source ~/anaconda3/bin/activate amp-dfm
```

### Key Dependencies

- **PyTorch 2.4.0** with CUDA 12.4 support
- **ESM2** (fair-esm 2.0.0) for protein embeddings
- **XGBoost 2.1.4** for property classifiers
- **MMseqs2** for sequence clustering
- **BioPython 1.85** for sequence handling
- **torchdiffeq** for ODE solving in flow matching

## Usage

### 1. Data Preprocessing

Generate ESM2 embeddings for sequences:
```bash
cd /rds/general/user/kja24/home/amp_dfm
source ~/anaconda3/bin/activate amp-dfm
qsub scripts/data_preprocessing/generate_embeddings.sh
```

Cluster sequences to prevent data leakage:
```bash
qsub scripts/data_preprocessing/mmseqs_cluster.sh
```

Assign train/val/test splits based on clusters:
```bash
qsub scripts/data_preprocessing/assign_cluster_split.sh
```

Prepare tokenised datasets for DFM training:
```bash
qsub scripts/data_preprocessing/prepare_ampdfm_uncond_dataset.sh
qsub scripts/data_preprocessing/prepare_ampdfm_cond_dataset.sh
```

### 2. Training Classifiers

Train XGBoost classifiers for peptide property prediction:

```bash
# Generic antimicrobial activity
qsub scripts/classifiers/antimicrobial_activity_generic_xgboost.sh

# Species-specific antimicrobial activity
qsub scripts/classifiers/antimicrobial_activity_ecoli_xgboost.sh
qsub scripts/classifiers/antimicrobial_activity_paeruginosa_xgboost.sh
qsub scripts/classifiers/antimicrobial_activity_saureus_xgboost.sh

# Safety classifiers
qsub scripts/classifiers/haemolysis_xgboost.sh
qsub scripts/classifiers/cytotoxicity_xgboost.sh
```

### 3. Training Flow Matching Models

Train the unconditional DFM:
```bash
qsub scripts/dfm/ampdfm_unconditional.sh
```

Fine-tune a conditional DFM:
```bash
qsub scripts/dfm/ampdfm_conditional_finetune.sh
```

### 4. Sampling Peptides

#### Unguided Sampling
```bash
qsub scripts/dfm/ampdfm_uncond_sample.sh
```

#### Multi-Objective Guided Sampling

Generate peptides with classifier guidance for specific targets:

```bash
# Generic antimicrobial peptides
qsub scripts/mog/ampdfm_mog_generic.sh

# E. coli-specific peptides
qsub scripts/mog/ampdfm_mog_ecoli.sh

# P. aeruginosa-specific peptides
qsub scripts/mog/ampdfm_mog_paeruginosa.sh

# S. aureus-specific peptides
qsub scripts/mog/ampdfm_mog_saureus.sh
```

Edit the corresponding YAML config files in `configs/mog/` to adjust:
- Number of samples (`n_samples`)
- Sequence length range (`len_min`, `len_max`)
- Guidance strength (`importance` weights)
- Homopolymer penalty (`homopolymer_gamma`)

### 5. Model Comparison

Compare unconditional vs conditional models:
```bash
qsub scripts/dfm/ampdfm_uncond_vs_cond.sh
```

## Configuration

All experiments are controlled via YAML configuration files in the `configs/` directory. Key parameters include:

### Classifier Training (`configs/classifiers/`)
- `task`: Property to predict (antimicrobial_activity, haemolysis, cytotoxicity)
- `variant`: Species variant for antimicrobial activity (generic, ecoli, paeruginosa, saureus)
- `n_estimators`: Number of boosting rounds
- `max_depth`: Maximum tree depth
- `learning_rate`: Boosting learning rate

### Flow Matching Training (`configs/flow_matching/`)
- `ckpt`: Path to checkpoint for fine-tuning
- `vocab_size`: Amino acid vocabulary size (24)
- `embed_dim`: Token embedding dimension (1024)
- `hidden_dim`: Hidden layer dimension (512)
- `learning_rate`: Optimiser learning rate
- `epochs`: Number of training epochs

### Multi-Objective Guidance (`configs/mog/`)
- `ckpt`: Path to trained DFM checkpoint
- `amp_variant`: Species variant (generic, ecoli, paeruginosa, saureus)
- `n_samples`: Number of peptides to generate
- `n_batches`: Split generation into batches
- `len_min`, `len_max`: Sequence length range
- `importance`: Weight for each objective [antimicrobial, non-haemolysis, non-cytotoxicity]
- `homopolymer_gamma`: Penalty strength for homopolymer runs

## Outputs

### Generated Peptides
- FASTA files: `outputs/peptides/<variant>/<run_name>.fa`
- Score CSV files: `outputs/peptides/<variant>/<run_name>_scores.csv`

### Classifier Outputs
- Model weights: `checkpoints/classifiers/<task>/<variant>/model.json`
- Metadata: `checkpoints/classifiers/<task>/<variant>/metadata.pkl`
- Evaluation metrics: `outputs/classifiers/<task>/<variant>/test_results.csv`
- ROC curves: `outputs/classifiers/<task>/<variant>/roc_curve_data.csv`

### DFM Checkpoints
- Unconditional: `checkpoints/dfm/ampdfm_unconditional_epoch200.ckpt`
- Conditional: `checkpoints/dfm/ampdfm_conditional_finetuned.ckpt`

## Model Architecture

### Discrete Flow Matching Model
- **Backbone**: U-Net-style CNN with 6 convolutional blocks
- **Time Embedding**: Gaussian Fourier projection
- **Token Embedding**: Learned amino acid embeddings (dimension 1024)
- **Dilated Convolutions**: Kernel size 9 with dilation rates [1, 1, 4, 16, 64, 1]
- **Output**: Logits over amino acid vocabulary at each position

### XGBoost Classifiers
- **Input**: ESM2 embeddings (1280-dimensional)
- **Architecture**: Gradient boosted decision trees
- **Output**: Binary classification (active/inactive, toxic/non-toxic)
- **Optimisation**: 5-fold cross-validation with early stopping

## HPC Usage

This codebase is designed for use on an HPC cluster with PBS job scheduling. Key points:

- All compute-intensive jobs use PBS scripts (`.sh` files in `scripts/`)
- Jobs require GPU resources (1 GPU per job)
- Activate the conda environment in each script before running Python code
- Monitor job outputs in `.pbs.OU` files (ignored by git)

Example PBS header:
```bash
#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -N job_name
#PBS -j oe
#PBS -o /path/to/output/directory/
```

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{amp-dfm-2025,
  author = {[Author Name]},
  title = {Discrete Flow Matching for Antimicrobial Peptide Design},
  school = {Imperial College London},
  year = {2025}
}
```

And the original discrete flow matching paper:

```bibtex
@article{tong2024discrete,
  title={Discrete Flow Matching},
  author={Tong, Alexander and Campbell, Avery and Roose, Jan and Matejovicova, Nikolay and De Bortoli, Valentin and Doucet, Arnaud and Bengio, Yoshua and Huguet, Guillaume},
  journal={arXiv preprint arXiv:2407.15595},
  year={2024}
}
```

## License

This project is for academic use as part of an MSc thesis at Imperial College London.

## Contact

For questions or issues, please open an issue on the repository or contact the author.
