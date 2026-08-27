# Lost in Moderation

Code for **[Lost in Moderation: How Commercial Content Moderation APIs Over- and Under-Moderate Group-Targeted Hate Speech and Linguistic Variations](https://doi.org/10.1145/3706598.3713998)** (CHI '25), by David Hartmann, Amin Oueslati, Dimitri Staufer, Lena Pohlmann, Simon Munzert, and Hendrik Heuer.

The paper audits five commercial content moderation APIs — **Google Natural Language API**, **Amazon Comprehend**, **Microsoft Azure Content Moderator**, **Perspective API**, and **OpenAI's Moderation API** — for over- and under-moderation of group-targeted hate speech, using three experiments across four datasets (ToxiGen, Civil Comments, HateXplain, SBIC) and over five million API queries. See the paper for the full methodology (§4–5) and findings.

This is a snapshot of the `API-Pipeline` folder from the project's working repository, published for transparency alongside the paper. It has been reorganized for readability and had hardcoded API keys removed (see [Security note](#security-note) below) — the analysis logic is otherwise unchanged.

## Layout

```
code/
├── pipeline/          API clients + the two experiment drivers that use them
├── preprocessing/      dataset loading/cleaning per benchmark
├── metrics/             performance & PSA metric computation
├── shap_analysis/     Experiment 3: SHAP explainability + qualitative coding support
└── topic_model/        exploratory structural-topic-model analysis (not in the paper)
input/                   preprocessed benchmark datasets, PSA token pairs
output/                  API responses, computed metrics, figures (populated by running the pipeline)
archive/                 superseded drafts, kept for provenance — not maintained, may not run
```

### `code/pipeline/`

- `apiGoogle.py`, `apiAmazon.py`, `apiGPT.py`, `apiMicrosoft.py`, `apiPerspective.py` — one thin client per API, each exposing `APIx(prompt)` for a single call and `hate_evaluation_x(text_list, labels, ids, max_workers=...)` for threaded batch scoring.
- `classification.py` — **Experiment 1** driver (aggregate + group-level performance, §5.1). CLI:
  ```
  python classification.py --dataset <name> --sample_frac <0-1> [--api <name>]
  ```
  `<name>` is one of `toxigen`, `jigsaw`, `megaspeech`, `hatexplain` — **`megaspeech` is this codebase's internal name for the paper's SBIC dataset** (it was built by combining several source hate-speech corpora, see `input/data/mega_speech_hate_source_partitioned/`); `jigsaw` corresponds to Civil Comments, which Jigsaw originally published. `--api` is one of `google`, `microsoft`, `gpt`, `amazon`, `perspective`, or omitted for all five. Writes results to `output/API_classification/<api>/<dataset>_<group|aggregate>/`.
- `psa.py` — **Experiment 2** driver (Perturbation Sensitivity Analysis / Counterfactual Token Fairness, §5.2). Same CLI shape as `classification.py`.
- `qualitative.py`, `qualitative_tests.py` — helpers for the qualitative SHAP coding pass (§5.3.1) and an ad-hoc smoke test of all five clients.
- `html_shap.py`, `shap_words.py` — generate per-token SHAP visualizations (Figures 4–5) and run global SHAP over already-classified data; both need the API clients live in this folder, which is why they live here rather than in `shap_analysis/`.
- `auswertung.py`, `robustness_check.py`, `id.py`, `lena_id.py` — supporting analysis scripts (result aggregation, robustness checks, ID/token bookkeeping for SHAP outputs).
- `experiments.bat` — Windows batch loop that runs `classification.py` over all dataset × API combinations sequentially, logging each to `logs/` (created at runtime).

### `code/preprocessing/`

One script per benchmark, turning raw downloads into the `<dataset>_aggregate.csv` / `<dataset>_group.csv` pairs `classification.py` expects in `input/data/`: `jigsaw_preprocessing.py`, `hatexplain_processing.py`, `toxigen_preprocessing.py`, `megaspeech_preprocessing.py`, `SBIC_preprocessing.ipynb`. `psa_real_data.py` / `psa_synthetic_data.py` build the counterfactual pairs for Experiment 2 (Tables 9–13 in the paper) from HateXplain and the Dixon et al. Identity Phrase Templates respectively. `descriptive_statistics.py` produces Table 8/9-style summaries.

### `code/metrics/`

`performance_metrics.py` computes the ACC/AUC/F1/FPR/FNR and pinned-ROC-AUC numbers behind Tables 2–3. `psa_metrics.py` computes the Counterfactual Token Fairness scores behind Table 10 and Figure 3.

### `code/shap_analysis/`

`global_shap.py` / `global_shap_dbscan.py` cluster per-token SHAP values into the global explanations in Table 4 (the DBSCAN variant is what's reported in the paper). `IOU_SHAP_hateXplain.py` and `t_SNE.ipynb` cross-check explanations across APIs. `shap_eval.py` renders individual token-attribution plots (Figures 4–5). `stacking.py`/`stacking2.py`/`stacking3.py` combine per-API SHAP outputs for comparison. The `falsepositives_*.csv` / `falsenegatives_*.csv` / `mislabeled_data.csv` files are intermediate samples from the qualitative coding pass (§5.3.1, Table 5).

### `code/topic_model/`

`create_datasets_for_stm.py` prepares data for a structural topic model over Google's false-moderation cases. This is exploratory work that isn't part of the published paper — included for completeness, not documented further here.

### `archive/`

Early drafts superseded by `code/pipeline/` (`header.py`, `evaluations.py`), an old results snapshot (`API_classification_220424.zip`), an unused SSH public key (`server-copresence.pub`), and a divergent early draft of the Google client (`apiGoogle_early_draft.py`). Kept for provenance; not guaranteed to run as-is (some use relative paths that assumed a different folder depth).

## Setup

```
conda env create -f environment.yml
conda activate NLP
```

(`experiments.bat` activates an environment named `AuditNLP` — that's a leftover naming mismatch from development; the actual spec in `environment.yml` is named `NLP`. Rename one or the other to match before using the batch script.)

### API credentials

Every hardcoded key from the original working copy has been removed. Each client now reads its credential from an environment variable — set the ones you need before running anything in `code/pipeline/`:

| Client | Environment variable | Notes |
|---|---|---|
| `apiPerspective.py` | `PERSPECTIVE_API_KEY` | [perspectiveapi.com](https://perspectiveapi.com/) |
| `apiGoogle.py` | `GOOGLE_API_KEY` | Google Cloud Natural Language API key |
| `apiGPT.py` | `OPENAI_API_KEY` | OpenAI API key |
| `apiAmazon.py` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | AWS Comprehend, region `us-east-1` |
| `apiMicrosoft.py` | `AZURE_SUBSCRIPTION_KEY` | Azure Content Moderator; endpoint is hardcoded to a `germanywestcentral` resource — point it at your own resource's hostname if different |
| `code/pipeline/*` (Hugging Face auth) | `HF_TOKEN` | used for gated dataset/model downloads in a few scripts |

## Running the pipeline

1. Preprocess the datasets you need (`code/preprocessing/`) so `input/data/` has the `*_aggregate.csv` / `*_group.csv` pairs.
2. From `code/pipeline/`, run Experiment 1 and/or 2 per dataset/API, e.g.:
   ```
   cd code/pipeline
   python classification.py --dataset hatexplain --sample_frac 1 --api perspective
   python psa.py --dataset megaspeech --sample_frac 1
   ```
3. Compute metrics from the resulting `output/API_classification/` files with `code/metrics/performance_metrics.py` and `code/metrics/psa_metrics.py`.
4. For Experiment 3, run the SHAP scripts in `code/pipeline/` (`html_shap.py`, `shap_words.py`) against classified data, then the aggregation/clustering scripts in `code/shap_analysis/`.

All scripts assume they're run from their own directory — paths are relative (`../../input/...`, `../../output/...`).

## Security note

This snapshot has been sanitized before publishing: hardcoded Google/Perspective, OpenAI, AWS, Hugging Face, and Azure credentials were removed from the source and replaced with environment-variable reads; a stray SSH private key file was deleted; and log files that had leaked a key in plaintext request URLs were dropped. If you're reusing this code, use your own fresh credentials — do not attempt to recover or reuse anything from the pre-sanitization history.

## Citation

```bibtex
@inproceedings{hartmann2025lost,
  title     = {Lost in Moderation: How Commercial Content Moderation APIs Over- and Under-Moderate Group-Targeted Hate Speech and Linguistic Variations},
  author    = {Hartmann, David and Oueslati, Amin and Staufer, Dimitri and Pohlmann, Lena and Munzert, Simon and Heuer, Hendrik},
  booktitle = {Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3706598.3713998}
}
```
