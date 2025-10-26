# AMP-DFM: Discrete Flow Matching for Multi-Property Antimicrobial Peptides

This project involved developing AMP-DFM, a generative antimicrobial peptide model designed to address limitations in current AMP generation approaches - notably, the increased toxicity risk from optimising solely for antimicrobial potency during generation.

![AMP-DFM Overview](documentation/dfm.drawio.png)

A generative discrete flow matching model was used to create realistic and diverse peptides. The sampling process was guided by trained classifiers for haemolysis, cytotoxicity and antimicrobial activity. Peptides were steered towards Pareto-optimal trade-offs across these properties with the goal of producing candidate sequences more likely to succeed in clinical settings.

Other parts of the analysis such as peptide structure prediction, comparison with other models (generative + classifiers) and data collation are omitted from this repository. This repository contains only the main analysis and results.

## Setup

Create and activate the conda environment from the provided environment yaml file:

```bash
git clone https://github.com/kjayres/AMP-DFM
cd amp_dfm
conda env create -f documentation/amp-dfm.yaml
conda activate amp-dfm
```

## Analytical Pipeline

The model was developed through the following steps:

1. **Data Preprocessing**: Sequences were clustered using MMseqs2 at 80% identity to prevent data leakage. ESM-2 (650M) embeddings were generated for classifier training. Tokenised datasets were prepared for the generative models.

```bash
# Cluster sequences and assign train/val/test splits
python scripts/data_preprocessing/mmseqs_cluster.py
python scripts/data_preprocessing/assign_cluster_split.py

# Generate ESM-2 embeddings for classifier training
python scripts/data_preprocessing/generate_embeddings.py

# Prepare tokenized datasets for generative models
python scripts/data_preprocessing/prepare_ampdfm_uncond_dataset.py
python scripts/data_preprocessing/prepare_ampdfm_cond_dataset.py
```

2. **Classifier Training**: XGBoost classifiers were trained on ESM-2 embeddings to predict antimicrobial activity (generic and organism-specific), haemolysis, and cytotoxicity.

```bash
# Main classifiers
python scripts/classifiers/train_classifiers.py \
    --config configs/classifiers/antimicrobial_activity_generic_xgboost.yaml
python scripts/classifiers/train_classifiers.py \
    --config configs/classifiers/haemolysis_xgboost.yaml
python scripts/classifiers/train_classifiers.py \
    --config configs/classifiers/cytotoxicity_xgboost.yaml

# Organism-specific antimicrobial activity classifiers
python scripts/classifiers/train_classifiers.py \
    --config configs/classifiers/antimicrobial_activity_ecoli_xgboost.yaml
python scripts/classifiers/train_classifiers.py \
    --config configs/classifiers/antimicrobial_activity_paeruginosa_xgboost.yaml
python scripts/classifiers/train_classifiers.py \
    --config configs/classifiers/antimicrobial_activity_saureus_xgboost.yaml
```

3. **Model Training**: A time-conditioned CNN was trained to estimate transition probabilities along a mixture path that evolves sequences from a uniform distribution towards data through single-position edits. Training minimised the generalised KL divergence between the teacher posterior and the model posterior which allows for novel peptide generation.

```bash
# Unconditional training
python scripts/dfm/ampdfm_unconditional.py \
    --config configs/flow_matching/ampdfm_unconditional.yaml

# Conditional fine-tuning (optional)
python scripts/dfm/ampdfm_conditional_finetune.py \
    --config configs/flow_matching/ampdfm_conditional_finetune.yaml

# Unguided sampling
python scripts/dfm/ampdfm_uncond_sample.py \
    --config configs/flow_matching/ampdfm_uncond_sample.yaml
```

4. **Multi-Objective Guidance**: During sampling, classifiers scored single-position edit candidates across the three objectives. Proposals were reweighted using importance weights, penalised for homopolymer formation, and sampled via Euler jumps weighted by the guided transition rates.

```bash
# Generic antimicrobial activity
python scripts/mog/ampdfm_mog.py --config configs/mog/ampdfm_mog_generic.yaml

# Organism-specific variants
python scripts/mog/ampdfm_mog.py --config configs/mog/ampdfm_mog_ecoli.yaml
python scripts/mog/ampdfm_mog.py --config configs/mog/ampdfm_mog_paeruginosa.yaml
python scripts/mog/ampdfm_mog.py --config configs/mog/ampdfm_mog_saureus.yaml
```

### Generation Parameters

The generation process can be customised through config file parameters and command-line options:

**Config file parameters:**
- `amp_variant`: Target organism (generic, ecoli, paeruginosa, saureus)
- `importance`: Weighting for each objective [antimicrobial, haemolysis, cytotoxicity].
- `homopolymer_gamma`: Penalty strength for homopolymer sequences to avoid repetitive patterns
- `n_samples`: Total number of peptides to generate
- `n_batches`: Number of batches to split generation into
- `len_min` and `len_max`: Peptide length range
- `seq_length` (optional): Fixes sequence length (overrides `len_min`/`len_max` when set)

**Command-line options:**
- `--T`: Number of sampling steps
- `--beta`: Guidance reweighting scale
- `--lambda_`: Trade‑off for directional score vs average rank
- `--Phi_init`, `--Phi_min`, `--Phi_max`: Hypercone angle (radians)
- `--tau`, `--alpha_r`, `--eta`: Adaptation controls for the hypercone angle (EMA target and update rate)
- `--num_div`: Simplex discretisation for importance weight vectors

Example Usage:
```bash
python scripts/mog/ampdfm_mog.py \
  --config configs/mog/ampdfm_mog_generic.yaml \
  --T 150 \
  --beta 2.0 \
  --lambda_ 1.0 \
  --Phi_init 0.785 \
  --Phi_min 0.262 \
  --Phi_max 1.309 \
  --tau 0.3 \
  --alpha_r 0.5 \
  --eta 1.0 \
  --num_div 64
```

## Outputs

### Checkpoints

- DFM model checkpoints are saved under `checkpoints/dfm/`:
  - Unconditional model: `checkpoints/dfm/ampdfm_unconditional_epoch200.ckpt`
  - Conditional fine-tuned model: `checkpoints/dfm/ampdfm_conditional_finetuned.ckpt`

- Classifier checkpoints are saved under `checkpoints/classifiers/`:
  - Antimicrobial activity (organism-specific): `checkpoints/classifiers/antimicrobial_activity/<variant>/model.json` with `metadata.pkl`
  - Haemolysis and cytotoxicity: `checkpoints/classifiers/<task>/model.json` with `metadata.pkl`

### Peptides

Generated peptides are saved as fasta and CSV files which contain scores/probabilities provided by the classifiers:
- Guided (MOG):
  - **FASTA**: `outputs/peptides/<variant>/<run_name>.fa`
  - **CSV**: `outputs/peptides/<variant>/<run_name>_scores.csv`
- Unguided:
  - **FASTA**: `outputs/peptides/unguided/unconditional_samples.fa`
  - **CSV**: `outputs/peptides/unguided/unconditional_samples_scores.csv`

## Citation

The code for this repo and the generative model is largely based on the work of Chen et al. and Lipman et al. The design of the antimicrobial activity classifiers is based on the work of Soares et al. and Szymczak et al. The design of the haemolysis classifier is based on the work of Capecchi et al.

If this code is of any use, you may be interested in the relevant papers:

```bibtex
@misc{chen2025multiobjectiveguideddiscreteflowmatching,
      title={Multi-Objective-Guided Discrete Flow Matching for Controllable Biological Sequence Design}, 
      author={Tong Chen and Yinuo Zhang and Sophia Tang and Pranam Chatterjee},
      year={2025},
      eprint={2505.07086},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.07086}
}

@misc{lipman2024flowmatchingguidecode,
      title={Flow Matching Guide and Code}, 
      author={Yaron Lipman and Marton Havasi and Peter Holderrieth and Neta Shaul and Matt Le and Brian Karrer and Ricky T. Q. Chen and David Lopez-Paz and Heli Ben-Hamu and Itai Gat},
      year={2024},
      eprint={2412.06264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06264}
}

@misc{soares2025targetedampgenerationcontrolled,
      title={Targeted AMP generation through controlled diffusion with efficient embeddings}, 
      author={Diogo Soares and Leon Hetzel and Paulina Szymczak and Fabian Theis and Stephan Günnemann and Ewa Szczurek},
      year={2025},
      eprint={2504.17247},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.17247}
}

@article{szymczak2023discovering,
      title={Discovering highly potent antimicrobial peptides with deep generative model HydrAMP},
      author={Szymczak, Paulina and Możejko, Marcin and Grzegorzek, Tomasz and Bauer, Radosław and Neubauer, Damian and Michalski, Mateusz and Sroka, Jacek and Setny, Piotr and Kamysz, Wojciech and Szczurek, Ewa},
      journal={Nature Communications},
      volume={14},
      number={1},
      pages={1453},
      year={2023},
      publisher={Nature Publishing Group},
      doi={10.1038/s41467-023-36994-z},
      url={https://doi.org/10.1038/s41467-023-36994-z}
}

@article{capecchi2021machine,
      title={Machine learning designs non-hemolytic antimicrobial peptides},
      author={Capecchi, Alice and Cai, Xingguang and Personne, Hippolyte and Köhler, Thilo and van Delden, Christian and Reymond, Jean-Louis},
      journal={Chemical Science},
      year={2021},
      volume={12},
      number={26},
      pages={9221--9232},
      publisher={The Royal Society of Chemistry},
      doi={10.1039/D1SC01713F},
      url={http://dx.doi.org/10.1039/D1SC01713F}
}
```
