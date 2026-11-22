# mongolia-climate-mining-returns

Exploratory analysis of local climate anomalies and mining stock returns in Mongolia-linked firms, 2018–2025.

Code companion to *Local Climate Anomalies and Mining Stock Returns: Exploratory Evidence from Two Mongolia-Linked Firms, 2018–2025*. The repository contains the original analysis and manuscript build code. Inputs and generated outputs belong in a private local directory.

| Code | Purpose |
|---|---|
| [Data Preparation](code/data_preparation/) | Audit and construct the master dataset |
| [Exploratory Analysis](code/exploratory_analysis/) | Descriptive statistics, correlations, and controlled regressions |
| [Robustness and Diagnostics](code/robustness_and_diagnostics/) | Baseline replication, influence checks, and diagnostics |
| [Manuscript Generation](code/manuscript_generation/) | Render a private manuscript source and its statistical tables |

## Replicate the analysis locally

Use Python 3.12 with the [pinned dependencies](requirements.txt). Run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python code/reproduce.py --master data/Master_Dataset_2018_2025.xlsx
```

To start from raw inputs, place the archived files `0975.HK.csv`, `STGO.TO.csv`, `^HSI.csv`, `^GSPTSE.csv`, `UHG.csv`, `ATO.csv`, and `CMO-Historical-Data-Monthly.xlsx` in `data/Raw/` and run:

```sh
python code/reproduce.py --raw-dir data/Raw
```

The sources are [Yahoo Finance](https://finance.yahoo.com/), the [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets), and the [Open-Meteo historical archive](https://open-meteo.com/en/docs/historical-weather-api). Exact reproduction requires the original data vintages.

The runner creates a fresh `replication/run-*/` directory for each execution. The raw-input route writes the master to `data_preparation/Final/` within that run. Both routes write `exploratory_analysis_results.json` in `exploratory_analysis/` and `robustness_and_diagnostics_results.json` in `robustness_and_diagnostics/` within the same run. Git ignores the private inputs and generated outputs.

## Generate the manuscript locally

Pass the private source directory containing `manuscript.md`, `bibliography.json`, and `Evidence/`. Place the two results JSON files from the same replication run in `Evidence/`. The builder also accepts the complete pair from the earlier private archive under its original filenames.

```sh
python code/manuscript_generation/build_manuscript.py --source-dir PATH_TO_PRIVATE_MANUSCRIPT_SOURCE
```

## AI assistance

OpenAI Codex assisted with literature discovery, manuscript drafting and editing, and development of the analysis, verification, and document-production code. The estimates were computed from the documented data with the reproducible Python workflow. The human author is responsible for critically reviewing the evidence and approving the final submission.

The [MIT License](LICENSE) applies to the code in `code/`.
