# Hierarchical HMM Deconvolution of MHC Class II Binding Motifs

Unsupervised deconvolution of mixed MHC class II peptide repertoires into distinct,
allele-specific binding motifs using an ensemble of anchor-aware Hidden Markov Models.

This repository contains the full pipeline accompanying the thesis *"Peptide Motif Deconvolution and Immunogenicity 
Prediction with Hidden Markov Model."*
It covers everything from data generation and preprocessing to hierarchical model
training, motif recovery, and benchmarking.

---

## Table of contents

- [Background](#background)
- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Expected data layout](#expected-data-layout)
- [Pipeline](#pipeline)
  - [`000_make_iedb_dataset.py` — IEDB preprocessing](#000_make_iedb_datasetpy--iedb-preprocessing)
  - [`001_simulated_data..py` — synthetic data generation](#001_simulated_datapy--synthetic-data-generation)
  - [`002_make_per_allele_per_length_split.py` — per-length / k-fold split](#002_make_per_allele_per_length_splitpy--per-length--k-fold-split)
  - [`003_deconvolution_main.py` — hierarchical deconvolution](#003_deconvolution_mainpy--hierarchical-deconvolution)
- [The `deconvolution_pipeline` package](#the-deconvolution_pipeline-package)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [References & acknowledgements](#references--acknowledgements)

---

## Background

Major Histocompatibility Complex (MHC) class II molecules present peptides to CD4⁺ T
cells. Unlike class I, the class II binding groove is open at both ends, so ligands vary
in length (typically 12-25 residues) and only a central **9-mer core** determines binding,
anchored at positions P1, P4, P6, and P9.

In real immunopeptidomics experiments, eluted-ligand datasets are *mixtures*: thousands of
peptides bound by several co-expressed HLA alleles, with no per-peptide allele label.
**Motif deconvolution** is the task of splitting such a mixture back into subsets, each
corresponding to a single allele's binding motif. Classical profile-based methods tend to
average distinct sub-motifs together and lose minor but biologically meaningful signals.

This project implements a **hierarchical HMM** approach to that problem. Hidden Markov
models naturally handle variable-length sequences and yield interpretable, position-specific
emission distributions — well suited to modelling conserved anchor positions.

## Method overview

The deconvolution engine combines five ideas:

1. **Anchor-aware HMM topology.** A left-to-right chain of nine core states (P1–P9) is
   flanked by two cyclic states that absorb variable-length N- and C-terminal flanks via
   self-transitions, removing the need for explicit gap penalties. Anchor states use
   restricted emission distributions concentrated on a few dominant amino acids; cycle
   states use broad, near-uniform distributions.
2. **Ensembles, not single models.** For each node, an ensemble of independently
   initialized HMMs is trained (Viterbi by default). Models are ranked by log-probability,
   and scoring uses the median length-normalized log-probability across the top model
   fraction.
3. **Background (trash) filtering.** A parallel ensemble is trained on random sequences
   drawn from the SwissProt amino-acid background. Peptides scored higher by the background
   ensemble than by the signal ensemble are discarded as uninformative.
4. **Viterbi-embedding + clustering split.** Each peptide is encoded from its Viterbi
   alignment to the top models as a one-hot positional feature vector, projected with PCA
  and partitioned with KMeans into branches.
5. **Iterative enrichment.** Branch assignments are refined over several iterations:
   peptides are reassigned to their best-scoring branch and ensembles are retrained, until
   branches are either kept (Jensen–Shannon distance above the merge threshold and
   information content above the retention threshold) or collapsed.

Recovered branches are compared to experimentally derived reference PSSMs
(NetMHCIIpan / MHCMotifViewer `.mat` format) using per-position Jensen–Shannon similarity
and Pearson correlation.

## Repository structure

```
Washu_project/
├── 000_make_iedb_dataset.py                 # Step 0: build per-allele datasets from raw IEDB export
├── 001_simulated_data..py                   # Step 1: generate synthetic peptide mixtures with known motifs
├── 002_make_per_allele_per_length_split.py  # Step 2: per-length 5-fold train/test split + sequence logos
├── 003_deconvolution_main.py                # Step 3: run the full hierarchical deconvolution
├── .gitignore
└── deconvolution_pipeline/                  # core library
    ├── training_parameters.py               # all configuration (paths, architecture, training, thresholds)
    ├── data_reading_methods.py              # data loading, allele balancing/collapsing, weights
    ├── peptides_utils.py                     # helpers: allele class detection, logos, dict utils
    ├── hmm_logic_cluster_split_trash.py     # core engine: build/train HMMs, ensembles, trash, hierarchy
    ├── hmm_diversity_split.py               # Viterbi positional embedding + PCA + KMeans branch split
    ├── hmm_metrics.py                        # reference PSSMs, JS/PCC/IC/KLD metrics, MetricsTracker
    ├── hmm_cluster_diagnostics.py           # diagnostics of cluster-split quality
    └── hmm_visualization_methods.py         # sequence logos, HMM graph rendering, PCA/UMAP plots
```

## Installation

> **Important — custom pomegranate fork required.**
> The code does **not** run on the standard PyPI `pomegranate`. It relies on a customized
> version that adds `DiscreteDistributionAnchor` and `DiscreteDistributionCycle`, based on
> the codebase at <https://github.com/artyomovlab/MHC_predictor>. Install that fork (and its
> required pomegranate build) before installing the remaining dependencies below.

```bash
# 1. Clone
git clone https://github.com/Annaknowshowtocode/Washu_project.git
cd Washu_project

# 2. (Recommended) create an environment
conda create -n hmm-decon python=3.9
conda activate hmm-decon

# 3. Install the custom pomegranate fork first (see the MHC_predictor repo for build steps)

# 4. Install the remaining dependencies
pip install pandas numpy scipy scikit-learn seaborn matplotlib \
            logomaker mhcgnomes mhcnames dask umap-learn pyvis \
            networkx colorcet param tqdm
```

Python 3.9 is recommended for compatibility with the pomegranate fork.

## Expected data layout

The pipeline package (`training_parameters.py`) resolves paths from environment variables,
falling back to folders next to the repository root:

| Variable            | Default                  | Holds                                            |
| ------------------- | ------------------------ | ------------------------------------------------ |
| `HMM_DATA_DIR`      | `<repo>/data`            | input data                                       |
| `HMM_MAT_DIR`       | `<repo>/data/reference_matrices` | reference PSSM `.mat` files (for benchmarking) |
| `HMM_PEPTIDES_DIR`  | `<repo>/data/peptides`   | peptide files                                    |
| `HMM_RESULTS_DIR`   | `<repo>/results`         | all outputs                                      |

A typical working tree for the experiment `simple_model_enrichment` looks like:

```
data/
├── mhc_ligand_full.csv                       # raw IEDB export (input to step 0)
├── reference_matrices/*.mat                  # gold-standard PSSMs (optional, for benchmarking)
└── simple_model_enrichment/
    ├── data#<allele>.csv                      # produced by step 0
    ├── simulated_data/<scenario>/*.csv        # produced by step 1
    └── per_length_per_kfold_split/            # produced by step 2 (consumed by step 3)
        └── <allele>#binders#{train,test}.csv
results/
└── simple_model_enrichment/                   # produced by step 3
```

## Pipeline

Run the scripts in order. Steps 0 and 1 produce input data; step 2 prepares it for training;
step 3 performs the deconvolution. (Steps 0–1 are independent — use step 0 for real IEDB
data and/or step 1 for controlled synthetic validation.)

### `000_make_iedb_dataset.py` — IEDB preprocessing

Builds clean, per-allele binder/non-binder tables from a raw IEDB export.

- **Input:** `data/mhc_ligand_full.csv` — the full *MHC Ligand* export from the
  [Immune Epitope Database](https://www.iedb.org/).
- **What it does:** keeps linear peptides composed only of the 20 standard amino acids;
  removes selected assay types (following Sarkizova et al.); merges quantitative (nM) and
  qualitative measurements; labels a peptide as a **binder** when IC₅₀ ≤ 500 nM (or a
  positive qualitative call); deduplicates; subsets to a configurable list of target HLA
  alleles; normalizes allele names with `mhcgnomes`.
- **Output:** one CSV per allele, `data/<experiment>/data#<compact_allele>.csv`
  (`;`-separated), plus a dataset-size bar plot.

### `001_simulated_data..py` — synthetic data generation

Generates peptide mixtures with a **fully controlled** motif structure, used to validate the
method against a known ground truth.

- **Input:** none — everything is parameterized in the script.
- **What it does:** for each of several built-in scenarios (`two_distinct_motifs`,
  `three_different_motifs`, `five_different_motifs`, `random_cores`, …) it builds peptides
  from a 9-residue core with motif-specific anchor positions plus random flanks. Core start
  positions are drawn from a center-weighted distribution; peptide lengths from
  per-motif length distributions; and a tunable fraction of cores (default **25% noise**) is
  replaced by random sequence. All motif-ratio combinations in 0.05 steps that sum to 1 are
  generated (12,000 sequences each).
- **Output:** `data/<experiment>/simulated_data/<scenario>/sim_data_motifs_*.csv` with
  columns `peptide, core, core_start, length, allele (Dummy_allele_i), noize, binder`,
  plus sequence-logo PNGs under `logos/` for verification.
- **Configure:** `CLASS_TO_GENERATE`, `scenario`, `DATA_PATH_BASE`, noise levels.

### `002_make_per_allele_per_length_split.py` — per-length / k-fold split

Prepares preprocessed per-allele data for training.

- **Input:** the `data#<allele>.csv` files from step 0 (or step 1 output).
- **What it does:** keeps binders only, groups peptides by length (8–30), and creates a
  **5-fold** train/test split per length (`sklearn.KFold`). Also renders example sequence
  logos for selected alleles/lengths.
- **Output:** `data/<experiment>/per_length_per_kfold_split/<allele>#binders#train.csv`
  and `…#test.csv` (columns `peptide, length, split`), plus logos under `logos/`.

### `003_deconvolution_main.py` — hierarchical deconvolution

The main driver that runs the full method.

- **Input:** the `per_length_per_kfold_split/` tables from step 2; optionally reference
  `.mat` PSSMs in `reference_matrices/` for benchmarking.
- **What it does:**
  1. loads train/test data and **collapses all alleles into a single "dummy" allele** (the
     mixture the model must deconvolve without labels);
  2. exports a plain peptide list (`peptides_for_benchmarking.txt`) for the GibbsCluster/MoDec
     comparison;
  3. prepares and trains an **ensemble of root HMMs**, reorders them by score, and saves
     root visualizations;
  4. runs **hierarchical training** (`hierarchically_train_splited_models`, default
     `total_layers_to_do=3`) with background filtering, clustering-based
     splitting, and enrichment.
- **Output:** under `results/<experiment>/<SUBFOLDER>/` — trained models, sequence logos
  per branch, peptide-by-model score-matrix heatmaps, PCA/UMAP projections, alignment
  examples (SVG), and metrics tables.
- **Configure:** `SUBFOLDER`, `experiment_name`, `num_runs`, and `total_layers_to_do` /
  `n_branches` in the call to `hierarchically_train_splited_models`.

## The `deconvolution_pipeline` package

| Module | Role |
| ------ | ---- |
| **`training_parameters.py`** | All configuration via [`param`](https://param.holoviz.org/) classes: data scenarios (`SimulatedDataParams`, `PreprocessedIEDBDataParams`, `MixMHCpredDataParams`, …), `ModelTrainingParams`, the ready-to-use `SimpleModelClassIIParams`, and `ExperimentParams` that ties name, data, and result paths together. |
| **`data_reading_methods.py`** | Reads per-allele/k-fold/length data, balances peptides while preserving length distributions, collapses alleles into one dummy mixture, builds train/test sets, and computes length-based sample weights. |
| **`peptides_utils.py`** | Small helpers: MHC class detection (`defineClass`), sequence-logo creation, dict (de)nesting, background frequency tables. |
| **`hmm_logic_cluster_split_trash.py`** | The heart of the project. Builds HMMs from parameters (`build_model_based_on_params`), trains single and ensemble models, performs background/trash training and filtering, and orchestrates hierarchical splitting + enrichment (`make_single_split_and_enrich`, `hierarchically_train_splited_models`). |
| **`hmm_diversity_split.py`** | Encodes peptides from their Viterbi anchor alignment into positional one-hot features, then PCA + KMeans to split a node's peptides into branches. |
| **`hmm_metrics.py`** | Loads reference PSSMs, computes per-position Jensen–Shannon similarity, Pearson correlation, information content, KL divergence from background, confusion-style metrics, and provides `MetricsTracker` to log every training step. |
| **`hmm_cluster_diagnostics.py`** | Visual diagnostics of how cleanly a node's peptides separate into branches (per-cluster allele composition, logos). |
| **`hmm_visualization_methods.py`** | Sequence logos, interpretable HMM graph rendering (NetworkX / PyVis), PCA & UMAP projections, and split-diagnostic figures. |

## Configuration

Key knobs live in `ModelTrainingParams` / `SimpleModelClassIIParams`
(`deconvolution_pipeline/training_parameters.py`):

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `algorithm` | `viterbi` | training algorithm (`viterbi` or `baum_welch`) |
| `num_runs` | `20`  | models per ensemble |
| `groups` × `states_per_group` | `7 × 1` | core-state count |
| `anchor_top_aas` | `5` | dominant amino acids per anchor state |
| `emission_pseudocount` | `0.1` | emission smoothing |
| `enrichment_steps` | `3` | branch-reassignment iterations |
| `trash_model_information_threshold` | `0.35` bits | min IC to keep a branch |
| `same_model_distance_threshold` | `0.05` | min JS distance to keep branches separate |
| `edge_inertia` / `distribution_inertia` | `0.4` / `0.4` | training inertia |

## Outputs

A completed run produces, per branch and per training step:

- **Sequence logos** of recovered motifs (mean emission of the top model fraction).
- **Peptide-by-model score-matrix heatmaps** showing block structure of the mixture.
- **PCA and UMAP projections** of the score space, coloured by true allele and by cluster.
- **Reference-comparison metrics** (JS similarity, Pearson correlation, IC, KLD) and
  branch→allele assignment tables.

## References & acknowledgements

-The HMM component of this work is based on and extends the interpretable
HMM framework developed in:
Kleverov D.A., Shalyto A.A., Artyomov M. A method for constructing interpretable
hidden Markov models for the task of finding peptide binding sites in protein
sequences. // Scientific and Technical Journal of Information Technologies, Mechanics
and Optics. — 2023. — Vol. 23, No. 5. — P. 989–1000.

Polezhaeva V.A., Kleverov D.A., Shalyto A.A., Artyomov M. Classification of peptide
sequences using hidden Markov models that account for negative examples. //
Scientific and Technical Journal of Information Technologies, Mechanics and Optics. —
2025. — Vol. 25, No. 5. — P. 888–901.
DOI: https://doi.org/10.17586/2226-1494-2025-25-5-888-901

- Built on top of the HMM codebase at
  [`artyomovlab/MHC_predictor`](https://github.com/artyomovlab/MHC_predictor) and a
  customized [pomegranate](https://github.com/jmschrek/pomegranate) build.
- Reference matrices follow the NetMHCIIpan / MHC Motif Atlas format
  (Tadros et al., *Nucleic Acids Res.* 2023).
- GibbsCluster: Andreatta, Alvarez & Nielsen, *Nucleic Acids Res.* 2017.
- Data: the [Immune Epitope Database (IEDB)](https://www.iedb.org/).


