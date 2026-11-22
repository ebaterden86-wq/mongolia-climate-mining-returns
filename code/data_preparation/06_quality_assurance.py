"""Independently reconcile the final CSV and XLSX to the raw observations."""

from calendar import monthrange
from collections import defaultdict
import csv
from datetime import date
from math import fsum, isclose, log
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


MASTER = "Master_Dataset_2018_2025"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        yield from csv.DictReader(source)


def same_number(actual, expected, label):
    if expected is None:
        require(pd.isna(actual), f"{label}: expected a blank, obtained {actual}")
    else:
        require(pd.notna(actual) and isclose(float(actual), expected, rel_tol=1e-11, abs_tol=1e-11),
                f"{label}: {actual} differs from {expected}")


def verify_price_family(master, prefix, price_field, prices):
    for month in master.index:
        current = prices.get(month)
        prior_key = str(pd.Period(month, freq="M") - 1)
        previous = prices.get(prior_key)
        label = f"{prefix} {month}"
        same_number(master.at[month, price_field], current, label + " price")
        ratio = current / previous if current is not None and previous is not None else None
        same_number(master.at[month, f"{prefix}_simple_return"],
                    None if ratio is None else ratio - 1, label + " simple return")
        same_number(master.at[month, f"{prefix}_log_return"],
                    None if ratio is None else log(ratio), label + " log return")


def verify_prices(master, raw):
    for prefix, filename in (("mmc", "0975.HK.csv"), ("steppe_gold", "STGO.TO.csv"),
                             ("hsi", "^HSI.csv"), ("tsx", "^GSPTSE.csv")):
        endpoints = {}
        for row in csv_rows(raw / filename):
            day = date.fromisoformat(row["Date"])
            month = day.isoformat()[:7]
            observed = endpoints.get(month)
            if observed is None or observed[0] < day:
                endpoints[month] = day, float(row["Adj Close"])
        prices = {month: observation[1] for month, observation in endpoints.items()}
        verify_price_family(master, prefix, f"{prefix}_adjusted_close", prices)
        if prefix == "steppe_gold":
            for month in master.index:
                actual = master.at[month, "steppe_gold_observation_date"]
                expected = endpoints[month][0].isoformat() if month in endpoints else ""
                require(("" if pd.isna(actual) else actual) == expected,
                        f"Steppe Gold endpoint does not match raw date for {month}")
    book = load_workbook(raw / "CMO-Historical-Data-Monthly.xlsx", data_only=True, read_only=True)
    rows = list(book["Monthly Prices"].values)
    book.close()
    for prefix, label, field in (
        ("gold", "Gold", "gold_price_usd_per_troy_oz"),
        ("coal_australian", "Coal, Australian", "coal_australian_price_usd_per_metric_ton"),
    ):
        index = rows[4].index(label)
        prices = {row[0].replace("M", "-"): float(row[index]) for row in rows[6:]
                  if isinstance(row[0], str) and type(row[index]) in {int, float}}
        verify_price_family(master, prefix, field, prices)


def verify_climate(master, raw):
    for name in ("UHG", "ATO"):
        groups = defaultdict(list)
        with (raw / f"{name}.csv").open(newline="", encoding="utf-8-sig") as source:
            for line in source:
                if line.startswith("time,"):
                    break
            for raw_day, temperature, precipitation in csv.reader(source):
                day = date.fromisoformat(raw_day)
                groups[day.year, day.month].append((day.day, float(temperature), float(precipitation)))
        monthly = {}
        for (year, month), days in groups.items():
            count = monthrange(year, month)[1]
            require(sorted(day for day, _, _ in days) == list(range(1, count + 1)),
                    f"{name} {year}-{month:02}: invalid or incomplete daily coverage")
            monthly[year, month] = (fsum(temperature for _, temperature, _ in days) / count,
                                     fsum(precipitation for _, _, precipitation in days))
        baseline = {}
        for month in range(1, 13):
            samples = [monthly[year, month] for year in range(1991, 2021)]
            baseline[month] = tuple(fsum(values) / len(samples) for values in zip(*samples))
        prefix = name.lower()
        for label in master.index:
            year, month = (int(piece) for piece in label.split("-"))
            temp, precip = monthly[year, month]
            base_temp, base_precip = baseline[month]
            expected = {
                "temperature_c": temp,
                "precipitation_mm": precip,
                "temperature_anomaly_c": temp - base_temp,
                "precipitation_anomaly_mm": precip - base_precip,
                "precipitation_anomaly_pct": 100 * (precip - base_precip) / base_precip if base_precip else None,
            }
            for field, value in expected.items():
                same_number(master.at[label, f"{prefix}_{field}"], value, f"{name} {label} {field}")


def verify_workbook(final, master, dictionary, missingness):
    workbook = load_workbook(final / f"{MASTER}.xlsx", read_only=True, data_only=True)
    for title, frame in (("Master", master.reset_index()), ("Data Dictionary", dictionary),
                         ("Missingness", missingness)):
        rows = list(workbook[title].values)
        require(rows[0] == tuple(frame.columns), f"{title}: workbook header mismatch")
        require(len(rows[1:]) == len(frame), f"{title}: workbook row mismatch")
        for row_number, (row, expected) in enumerate(zip(rows[1:], frame.itertuples(index=False, name=None)), 2):
            for field, actual, value in zip(frame.columns, row, expected):
                label = f"{title}!{field}, row {row_number}"
                if pd.isna(value):
                    require(actual in {None, ""}, label + " should be blank")
                elif isinstance(value, str):
                    require(actual == value, label + " text mismatch")
                else:
                    same_number(actual, float(value), label)
    workbook.close()


def run(root):
    final = root / "Final"
    master = pd.read_csv(final / f"{MASTER}.csv",
                         dtype={"month": "string", "steppe_gold_observation_date": "string"}).set_index("month")
    expected = pd.period_range("2018-01", "2025-12", freq="M").astype(str).tolist()
    require(master.index.is_unique and master.index.tolist() == expected,
            "Master keys must be exactly 96 unique, ordered study months")
    numeric = master.select_dtypes(include="number")
    require(not np.isinf(numeric.to_numpy(dtype=float)).any(), "Infinite values in master")
    dictionary = pd.read_csv(final / "Data_Dictionary.csv")
    require(dictionary["variable"].tolist() == ["month", *master.columns], "Data dictionary key mismatch")
    missingness = pd.read_csv(final / "Missingness_Report.csv")
    require(missingness["variable"].tolist() == ["month", *master.columns], "Missingness key mismatch")
    counts = []
    for row in missingness.itertuples(index=False):
        values = master.index.to_series() if row.variable == "month" else master[row.variable]
        months = values.index[values.isna()].tolist()
        exported = "" if pd.isna(row.missing_months) else row.missing_months
        require(row.total_months == 96 and row.usable_months == 96 - len(months)
                and row.missing_month_count == len(months) and exported == ", ".join(months),
                f"Missingness counts or months differ for {row.variable}")
        same_number(row.missing_pct, 100 * len(months) / 96, f"{row.variable} missing per cent")
        counts.append(f"  {row.variable:<46} {row.usable_months:>3} / 96")
    verify_prices(master, root / "Raw")
    verify_climate(master, root / "Raw")
    verify_workbook(final, master, dictionary, missingness)
    all_variables = int(master.notna().all(axis=1).sum())
    return "\n".join([
        "F. INDEPENDENT QUALITY ASSURANCE",
        f"PASS | Master has {len(master)} rows × {len(master.columns) + 1} columns, including month.",
        "PASS | Calendar keys are unique, ordered and exactly 2018-01 to 2025-12.",
        "PASS | All six price series and all available adjacent-month returns reconcile to original values.",
        "PASS | All Steppe Gold date selections match the last supplied raw date in the month.",
        "PASS | Every master climate cell reconciles to independently aggregated raw calendar days "
        "and independently computed 1991–2020 climatologies.",
        "PASS | No infinite values. Every master column has a dictionary entry and reconciled missingness record.",
        "PASS | All three XLSX sheets match the exported CSVs cell by cell, within floating-point tolerance.",
        f"Usable observations (nonmissing cells) by field. All fields are populated together in {all_variables} months.",
        *counts,
        "The complete-row count is descriptive. The export retains all 96 months.",
    ])


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
