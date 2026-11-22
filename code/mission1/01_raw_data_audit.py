"""Inventory and audit the seven immutable inputs before processing."""

from collections import Counter
import csv
import hashlib
from pathlib import Path
import re

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import numpy as np
import pandas as pd


FINANCIAL = {
    "0975.HK.csv": ("monthly", "HKD per share", "reported share volume"),
    "STGO.TO.csv": ("daily market observations", "CAD per share", "reported share volume"),
    "^HSI.csv": ("monthly", "index points", "vendor-reported index volume"),
    "^GSPTSE.csv": ("monthly", "index points", "vendor-reported index volume"),
}
FILES = (*FINANCIAL, "UHG.csv", "ATO.csv", "CMO-Historical-Data-Monthly.xlsx")
MISSING = {"", "…", "...", "–", "—", "n.a.", "n/a", "na", "null", "nan"}
PERIOD = re.compile(r"[0-9]{4}M(0[1-9]|1[0-2])")


def missing(value):
    return value is None or str(value).strip().casefold() in MISSING


def csv_audit(path, climate=False):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    start = next(i for i, row in enumerate(rows) if row[:1] == ["time"]) if climate else 0
    columns = rows[start]
    observations = [row for row in rows[start + 1:] if row]
    if any(len(row) != len(columns) for row in observations):
        raise ValueError(f"{path.name}: inconsistent CSV record width")
    data = pd.DataFrame(observations, columns=columns)
    missing_mask = data.apply(lambda field: field.map(missing))
    dates = pd.to_datetime(data.iloc[:, 0], format="%Y-%m-%d", errors="coerce")
    numeric = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    invalid = (numeric.isna() | ~np.isfinite(numeric)) & ~missing_mask.iloc[:, 1:]
    invalid_counts = dict(invalid.sum())
    invalid_dates = int((dates.isna() & ~missing_mask.iloc[:, 0]).sum())
    frequency = "daily calendar" if climate else FINANCIAL[path.name][0]
    units = ("temperature °C, precipitation mm/day total" if climate else
             f"OHLC/Adj Close {FINANCIAL[path.name][1]}, Volume {FINANCIAL[path.name][2]}")
    lines = [
        f"{path.name} | {len(data):,} rows × {len(columns)} columns | {frequency}",
        f"  Fields: {', '.join(columns)}.",
        f"  CSV storage: text. Parsed dates: calendar date. Units: {units}.",
        f"  Parsed numeric types: {', '.join(f'{name}={kind}' for name, kind in numeric.dtypes.items())}.",
        f"  Date range: {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}. "
        f"Invalid dates: {invalid_dates}. Duplicate valid date rows beyond first: "
        f"{dates.dropna().duplicated().sum()}. Sorted: {dates.dropna().is_monotonic_increasing}.",
        "  Missing per field: " + ", ".join(f"{k}={v}" for k, v in missing_mask.sum().items()) + ".",
        "  Non-numeric or nonfinite cells excluding missing: "
        + ", ".join(f"{k}={v}" for k, v in invalid_counts.items()) + ".",
    ]
    if climate:
        expected = pd.date_range("1991-01-01", "2025-12-31", freq="D")
        lines.extend([
            f"  Observation header at CSV line {start + 1}. Metadata: "
            + ", ".join(f"{k}={v}" for k, v in zip(rows[0], rows[1])) + ".",
            f"  Expected dates: {len(expected):,}. Missing within range: {len(expected.difference(dates))}. "
            f"Outside range: {len(pd.DatetimeIndex(dates.dropna()).difference(expected))}. "
            f"Negative precipitation: {(numeric.iloc[:, 1] < 0).sum()}.",
        ])
    else:
        bad_ohlc = ((numeric["High"] < numeric[["Open", "Close", "Low"]].max(axis=1)) |
                    (numeric["Low"] > numeric[["Open", "Close", "High"]].min(axis=1)))
        numeric_prices = numeric[["Open", "High", "Low", "Close", "Adj Close"]]
        lines.append(f"  OHLC bounds violations: {bad_ohlc.sum()}. Nonpositive prices: "
                     f"{(numeric_prices <= 0).sum().sum()}. Negative volume: "
                     f"{(numeric['Volume'] < 0).sum()}. Zero volume: {(numeric['Volume'] == 0).sum()}.")
    return "\n".join(lines)


def workbook_audit(path):
    with path.open("rb") as source:
        workbook = load_workbook(source, data_only=True, read_only=True)
        sheet = workbook["Monthly Prices"]
        rows = list(sheet.values)
        dimensions = "; ".join(f"{s.title} {s.max_row}×{s.max_column}" for s in workbook)
        comparisons = list(workbook["Mismatch Details"].values)[4:] if "Mismatch Details" in workbook else []
        workbook.close()
    header, units = rows[4:6]
    fields = [i for i, value in enumerate(header) if value is not None]
    records = [row for row in rows[6:] if any(value is not None for value in row)]
    tokens = [str(row[0]).strip() for row in records]
    valid_periods = [token.replace("M", "-") for token in tokens if PERIOD.fullmatch(token)]
    observed = pd.PeriodIndex(valid_periods, freq="M")
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    unexpected_fields = [get_column_letter(i + 1) for i, value in enumerate(header)
                         if i and value is None and any(row[i] is not None for row in records)]
    lines = [
        f"{path.name} | worksheet dimensions (rows × columns): {dimensions}.",
        f"  Monthly Prices contains {len(records):,} observation rows and {len(fields) + 1} named or period fields. "
        "The header is row 5, units row 6. The unnamed column A stores period tokens such as 2018M01.",
        f"  Vintage as labeled by the attached worksheet: {rows[3][0]}.",
        f"  Monthly range: {observed.min()} to {observed.max()}. Invalid period tokens: "
        f"{len(tokens) - len(observed)}. Duplicate periods beyond first: {observed.duplicated().sum()}. "
        f"Missing periods inside range: {len(expected.difference(observed))}.",
        f"  Data found in columns without a row-5 label other than period column A: "
        f"{', '.join(unexpected_fields) or 'none'}.",
        f"  Mismatch Details has {len(comparisons)} comparison records. Neither target appears among "
        f"the comparison labels (target matches: {sum(str(r[1]).strip() in {'Gold', 'Coal, Australian'} for r in comparisons)}). "
        "The comparison sheet's workbook-vintage pairing is unverified.",
        "  Field-level inventory | source unit | value types | missing incl. placeholders | other invalid | nonpositive numeric",
        f"    A  period | YYYYMmm | text | {sum(missing(row[0]) for row in records)} | "
        f"{len(tokens) - len(observed) - sum(missing(row[0]) for row in records)} | n/a",
    ]
    for index in fields:
        values = [row[index] for row in records]
        absent = Counter(str(v) if v is not None else "(blank)" for v in values if missing(v))
        types = Counter(type(v).__name__ for v in values if not missing(v))
        numbers = [v for v in values if type(v) in {float, int} and np.isfinite(v)]
        bad = len(values) - sum(absent.values()) - len(numbers)
        lines.append(f"    {get_column_letter(index + 1)}  {header[index]} | {units[index]} | {dict(types)} | "
                     f"{sum(absent.values())} {dict(absent) if absent else ''} | {bad} | "
                     f"{sum(v <= 0 for v in numbers)}")
    return "\n".join(lines)


def run(root):
    raw = root / "Raw"
    lines = ["A. RAW DATA AUDIT", "File fingerprints (SHA-256 of the immutable copies)"]
    for filename in FILES:
        path = raw / filename
        lines.append(f"  {filename} | {path.stat().st_size:,} bytes | {hashlib.sha256(path.read_bytes()).hexdigest()}")
    lines.append("")
    for filename in FILES:
        path = raw / filename
        if path.suffix == ".xlsx":
            lines.append(workbook_audit(path))
        else:
            lines.append(csv_audit(path, climate=filename in {"UHG.csv", "ATO.csv"}))
        lines.append("")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
