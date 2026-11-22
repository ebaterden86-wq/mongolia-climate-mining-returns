# mongolia-climate-mining-returns
Exploratory analysis of local climate anomalies and mining stock returns in Mongolia-linked firms, 2018–2025.

Code companion to *Local Climate Anomalies and Mining Stock Returns: Exploratory Evidence from Two Mongolia-Linked Firms, 2018–2025*. The repository contains the original analysis and manuscript build code. Inputs and generated outputs belong in a private local directory.

| Code | Purpose |
|---|---|
| [Mission 1](code/mission1/) | Audit and construct the master dataset |
| [Mission 2](code/mission2/) | Descriptive statistics, correlations, and controlled regressions |
| [Mission 3](code/mission3/) | Baseline replication, robustness tests, and diagnostics |
| [Mission 4](code/mission4/) | Render a private manuscript source and its statistical tables |

## Run locally

Use Python 3.12 with the [pinned dependencies](requirements.txt).

```sh
python -m pip install -r requirements.txt
python code/reproduce.py --master data/Master_Dataset_2018_2025.xlsx
```

To go from archived raw inputs to the analysis, place the original files `0975.HK.csv`, `STGO.TO.csv`, `^HSI.csv`, `^GSPTSE.csv`, `UHG.csv`, `ATO.csv`, and `CMO-Historical-Data-Monthly.xlsx` in `data/Raw/` and run:

```sh
python code/reproduce.py --raw-dir data/Raw
```

The sources are [Yahoo Finance](https://finance.yahoo.com/), the [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets), and the [Open-Meteo historical archive](https://open-meteo.com/en/docs/historical-weather-api). Exact reproduction requires the original data vintages. Both routes write privately generated output under the ignored `replication/` directory.

If you have the private Mission 4 source, pass its directory with `--source-dir`. It must contain `manuscript.md`, `bibliography.json`, and `Evidence/` with the Mission 2 and 3 results JSON files.

```sh
python code/mission4/build_manuscript.py --source-dir PATH_TO_PRIVATE_MANUSCRIPT_SOURCE
```

The [MIT License](LICENSE) applies to the code in `code/`.
