"""Run data preparation locally with Python 3.11+ and pandas, NumPy and openpyxl.

From the project directory: python Code/run_all.py
The runner resolves paths from its own location and needs no network access.
"""

import hashlib
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import openpyxl
import pandas as pd


STAGES = (
    "01_raw_data_audit", "02_financial_cleaning", "03_climate_processing",
    "04_commodity_processing", "05_dataset_merging", "06_quality_assurance",
)
DELIVERABLES = {
    "Raw": ("0975.HK.csv", "STGO.TO.csv", "^HSI.csv", "^GSPTSE.csv", "UHG.csv", "ATO.csv",
            "CMO-Historical-Data-Monthly.xlsx"),
    "Cleaned": ("MMC_monthly_clean.csv", "SteppeGold_monthly_clean.csv", "HSI_monthly_clean.csv",
                "TSX_monthly_clean.csv", "UHG_monthly_climate.csv", "ATO_monthly_climate.csv",
                "Commodity_monthly_clean.csv"),
    "Final": ("Master_Dataset_2018_2025.csv", "Master_Dataset_2018_2025.xlsx", "Data_Dictionary.csv",
              "Missingness_Report.csv", "Quality_Control_Report.txt"),
    "Code": (*[f"{stage}.py" for stage in STAGES], "run_all.py"),
}

RESEARCH_NOTES = """INTERPRETATION AND OPEN ITEMS
The master is ready for exploratory data analysis. Model specifications should
address availability, proxy and exposure changes, and endpoint timing. Treat the
return sample as unbalanced and document the analysis-specific inclusion rule.

Financial prices are local-currency observations (HKD or CAD); no FX conversion
was applied. Commodity controls are nominal USD and are monthly price averages.
Stock and index returns compare closing levels. A calendar-month join aligns
labels, while the closing-price, average-price and climate exposure windows
retain their original timing. The resulting same-month columns are descriptive
inputs and may need lags in models that assume information available at the
start of a month. Index series are price indices and do not establish dividend
reinvestment. Identical Close and Adj Close fields do not independently verify
Yahoo's corporate-action treatment.

Steppe Gold's TSX trading commenced May 22, 2018, and the supplied file starts
May 23. January–April are pre-listing and May has no adjacent prior close. The
December 2025 return uses the December 30 supplied endpoint despite a scheduled
December 31 session. A December 2018 endpoint has zero reported volume. Verify
these endpoints against an exchange-level source before imposing a strict
month-end trading convention. Internal daily gaps and zero-volume values alone
do not identify exchange holidays, suspensions, no trades or vendor omissions.

MMC's 10-to-1 share consolidation took effect August 26, 2019. The three adjacent
monthly closes show no mechanical tenfold discontinuity, so the supplied price
basis was retained. Independently confirm the vendor adjustment history before
using corporate-action-sensitive specifications. Steppe Gold closed the Boroo
Gold acquisition August 1, 2024. ATO-only weather may represent a changing share
of its company-wide operational exposure. Assess exposure changes before pooling
the full sample under a fixed exposure relationship.

The climate brief identifies the Open-Meteo archive product as ERA5 (0.25° native
resolution, approximately 25 km). The CSVs preserve returned coordinates,
elevation and GMT timezone metadata. API request strings, extraction timestamps,
explicit model selection, cell-selection and downscaling settings were not
supplied. These grid-based values are reanalysis estimates. Reproduce the API
extraction and assess local-day boundaries and spatial sensitivity if needed.
The fixed 1991–2020 reference includes 2018–2020 and is suitable for descriptive
anomalies. Real-time forecasting backtests would need an information-set review.

Australian thermal coal is an imperfect proxy for Mongolian coking coal.
Investigate a metallurgical or coking-coal benchmark for the modeling stage.
The workbook's spot-to-futures coal change in 2022-02 and fixing-to-spot gold
change in 2025-06 span adjacent price ratios. Keep these observations, flag
their comparability, and consider sensitivity checks on source definitions.
The supplied November 2026 workbook is a later vintage than the study period.
The master uses that vintage as delivered. Real-time availability and benchmark
revisions require separate vintage verification.

External documentation (used for interpretation only)
  Open-Meteo Historical Weather API
    https://open-meteo.com/en/docs/historical-weather-api
  World Bank Commodity Markets and Pink Sheet publication
    https://www.worldbank.org/en/research/commodity-markets
  Steppe Gold listing announcement
    https://steppegold.com/917
  TSX December 2025 trade and settlement schedule
    https://www.tsx.com/en/resource/3243/
  MMC extraordinary meeting notice and annual report documenting consolidation
    https://www1.hkexnews.hk/listedco/listconews/sehk/2019/0806/ltn20190806523.pdf
    https://www.hkexnews.hk/listedco/listconews/sehk/2022/0426/2022042600547.pdf
  Steppe Gold confirming acquisition closing date
    https://steppegold.com/17105/"""


def fingerprints(raw):
    return {name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
            for name in DELIVERABLES["Raw"]}


def run():
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    original_hashes = fingerprints(root / "Raw")
    for folder in ("Cleaned", "Final"):
        (root / folder).mkdir(exist_ok=True)
    sections = [
        "SSRN MINING CLIMATE RESEARCH\nData Preparation | Audit and completion report",
        "Study period: 2018-01 to 2025-12 inclusive. Climate reference period: 1991–2020.\n"
        "Missing cells are blank. All financial and master commodity returns are decimals (0.10 = 10%).\n"
        "Climate precipitation percentage anomalies and the explicit cleaned commodity percentage "
        "returns are percentage values (10 = 10%).",
    ]
    for stage in STAGES:
        result = import_module(stage).run(root)
        if stage == "05_dataset_merging":
            sections.append(
                "E. MASTER ASSEMBLY\n"
                "Constructed the calendar independently of source availability. Validated one-to-one "
                "YYYY-MM keys before each left join. The left joins preserve every study month. "
                "Study-window selection applies to the master. The cleaned climate and World Bank files "
                "retain their broader coverage. No complete-case filter was applied. Original Yahoo "
                "price/volume columns and source date labels are retained in the cleaned financial files.\n"
                f"Master dimensions: {len(result)} rows × {len(result.columns)} columns, including month."
            )
        else:
            sections.append(result)
        print(f"PASS {stage}.py", flush=True)
    if original_hashes != fingerprints(root / "Raw"):
        raise RuntimeError("Raw fingerprints changed during execution")
    manifest = ["REPRODUCTION AND FILE LOCATIONS", f"Project root: {root}",
                "From that directory, run: python Code/run_all.py",
                f"Verified environment: Python {sys.version.split()[0]}, pandas {pd.__version__}, "
                f"NumPy {np.__version__}, openpyxl {openpyxl.__version__}.",
                "All six stages completed successfully. Raw SHA-256 hashes were unchanged throughout. "
                "At packaging, each of the seven Raw files was also SHA-256 checked against its supplied original.",
                "The scripts read only Raw files. Rerunning regenerates the Cleaned and Final files.", ""]
    manifest.extend(f"  {directory}/{name}" for directory, names in DELIVERABLES.items() for name in names)
    sections.extend([RESEARCH_NOTES, "\n".join(manifest)])
    report = root / "Final" / "Quality_Control_Report.txt"
    report.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"PASS run_all.py | Wrote {report}", flush=True)


if __name__ == "__main__":
    run()
