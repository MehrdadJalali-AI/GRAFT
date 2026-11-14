# GRAFT

GRAFT is an experimental framework for graph oversampling in semi-supervised node classification. It studies a practical question that ordinary accuracy metrics can miss: when synthetic minority nodes are connected to a graph, how often do those new edges change predictions for existing, confident nodes?

The repository implements the GRAFT attachment rule and its baselines, runs repeatable experiments, and includes the processed datasets and result files used by the project. Manuscripts, PDFs, submission archives, and manuscript-related documents are intentionally excluded.

## What is evaluated

For each train/validation/test split, the experiment first creates an **edge-free reference**: synthetic minority-node features are added to the training set but remain isolated from the graph. It then compares edge-attaching approaches against that matched reference.

The implemented methods are:

- **ERM**, class reweighting, and post-hoc logit adjustment;
- **SMOTE** and **conditional WGAN-GP** feature generation;
- **no-edge** oversampling, used as the matched reference;
- **similarity attachment** and **ego attachment** baselines; and
- **GRAFT**, which balances feature similarity against a certificate-aware exposure cost when selecting attachment targets.

Experiments report Macro-F1, balanced accuracy, minority recall, and collateral prediction changes among a protected cohort of confident test nodes. The code also evaluates propagation and parameter-drift bounds, together with frozen-model and retrained-model prediction certificates.

## Repository layout

| Path | Description |
| --- | --- |
| `code/graft/core.py` | SGC propagation, convex logistic-regression training, bounds, certificates, SMOTE, and attachment rules. |
| `code/graft/data.py` | Raw-dataset loading, feature construction, caching, and long-tail split creation. |
| `code/graft/gan.py` | Conditional WGAN-GP feature generator. |
| `code/run_experiments.py` | Main experiment runner. |
| `code/make_figures_tables.py` | Builds figures, summary JSON, and tabular assets from result CSVs. |
| `code/make_fig4.py` | Builds the certificate-coverage figure and its supporting values. |
| `data/` | Processed graph caches (`.npz`) containing adjacency, features, labels, and class names. |
| `results/` | Current experimental CSVs and the corresponding configuration files. |
| `results_v1/` | Earlier archived experiment outputs. |

## Requirements

Use Python 3.10 or newer. The experiments require NumPy, SciPy, pandas, scikit-learn, Matplotlib, PyTorch, and PyTorch Geometric.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas scikit-learn matplotlib torch torch-geometric
```

PyTorch and PyTorch Geometric wheels can be platform-specific. If the last command fails, install the compatible PyTorch and PyG wheels for your operating system and Python version, then rerun the command for the remaining packages.

## Quick start

Clone the repository, create the environment above, then inspect the included result files:

```bash
git clone https://github.com/MehrdadJalali-AI/GRAFT.git
cd GRAFT
ls results
```

To generate figure assets from the included results without rerunning training:

```bash
mkdir -p artifacts/fig4 artifacts/report-assets
python code/make_fig4.py results artifacts/fig4
python code/make_figures_tables.py results artifacts/report-assets
```

`make_fig4.py` writes PNG and PDF versions of the certificate figure. `make_figures_tables.py` writes PDF figures, LaTeX table fragments, and `tables/summary.json` under the supplied output directory.

## Running experiments

The runner uses this interface:

```text
python code/run_experiments.py RAW_DIR OUT_DIR [DATASET ...]
```

Supported dataset identifiers are `facebook`, `github`, `lastfm`, and `twitch_en`. Omitting dataset identifiers runs all four.

For a small smoke run on LastFM with one split:

```bash
git clone --depth 1 https://github.com/benedekrozemberczki/musae raw/musae
git clone --depth 1 https://github.com/benedekrozemberczki/FEATHER raw/FEATHER

mkdir -p artifacts
GRAFT_SEEDS=0 python code/run_experiments.py raw artifacts/results lastfm
```

For the standard five-seed setup, leave `GRAFT_SEEDS` unset (the default is `0,1,2,3,4`) or set it explicitly:

```bash
GRAFT_SEEDS=0,1,2,3,4 python code/run_experiments.py raw artifacts/results \
  lastfm twitch_en facebook github
```

The runner caches processed datasets in a sibling `data/` directory next to the requested output directory. For example, `artifacts/results` produces or reuses `artifacts/data/<dataset>.npz`. The repository already includes processed caches for Facebook, LastFM, and Twitch-EN; use a separate `artifacts/` directory to avoid overwriting the checked-in results.

## Input data and preprocessing

Raw inputs are read from the MUSAE and FEATHER repositories using the paths specified in `code/graft/data.py`. The preprocessing pipeline:

1. loads node labels, edges, and sparse feature lists;
2. symmetrizes the graph and removes self-loops;
3. converts sparse features into 128-dimensional truncated-SVD embeddings;
4. L2-normalizes feature rows; and
5. saves a compressed cache with the sparse adjacency representation, features, labels, and class names.

Each processed `.npz` file contains `data`, `indices`, `indptr`, and `shape` for the CSR adjacency matrix, plus `X`, `y`, and `classes`.

## Experimental protocol

The default configuration uses two propagation steps, five attachment edges per synthetic node, a 50-node similarity candidate pool, 20 labelled examples in each majority class, and a 10:1 majority/minority label ratio. The regularization parameter is selected on the validation split.

GRAFT considers `gamma` values from `0.25` to `16`. Among models within one Macro-F1 point of the best validation score, it selects the model with the lowest label-free disagreement on the validation cohort. Exact settings used for each checked-in dataset are stored in `results/config_<dataset>.json`.

## Result files

For each dataset, the main runner writes:

| File | Contents |
| --- | --- |
| `runs_<dataset>.csv` | Per-seed outcomes for selected methods, predictive metrics, collateral-change metrics, and certificate checks. |
| `sweep_<dataset>.csv` | Per-seed sweep results for attachment degree and GRAFT's `gamma` parameter. |
| `gcn_<dataset>.csv` | GCN transfer-study outcomes for the SMOTE and GRAFT-SMOTE variants. |
| `config_<dataset>.json` | The experiment constants and selected regularization setting. |

The `results/` directory holds the current run. `results_v1/` preserves a prior set of CSV outputs for comparison.

## Reproducibility notes

- Set `GRAFT_SEEDS` to a comma-separated list to control the repeated splits, for example `GRAFT_SEEDS=0,1`.
- The code sets PyTorch's default dtype to `float64` for the bound calculations.
- Running the GCN transfer study requires `torch_geometric`; the remaining SGC/logistic-regression pipeline uses the packages listed above.
- Results are written to the output directory supplied on the command line. Prefer a new directory such as `artifacts/results` for new runs.

## Data sources

The raw dataset format is provided by the upstream [MUSAE](https://github.com/benedekrozemberczki/musae) and [FEATHER](https://github.com/benedekrozemberczki/FEATHER) repositories. Please review and comply with their respective dataset terms before downloading or redistributing raw data.
