"""Extract the two requested World Bank monthly commodity series."""

from collections import Counter
import math
from pathlib import Path
import re

import numpy as np
from openpyxl import load_workbook
import pandas as pd


WORKBOOK = "CMO-Historical-Data-Monthly.xlsx"
SHEET = "Monthly Prices"
HEADER_ROW = 5
UNITS_ROW = 6
PERIOD_PATTERN = re.compile(r"\d{4}M(?:0[1-9]|1[0-2])")
MISSING_MARKERS = {"", "…", "..."}
RESEARCH_MONTHS = pd.period_range("2018-01", "2025-12", freq="M")
CHECK_MONTHS = ("2018-01", "2022-02", "2025-06", "2025-12")
SERIES = (
    ("Gold", "($/troy oz)", "gold", "gold_price_usd_per_troy_oz"),
    ("Coal, Australian", "($/mt)", "coal_australian",
     "coal_australian_price_usd_per_metric_ton"),
)


def find_price_column(headers, units, label, expected_unit):
    matching = [i for i, header in enumerate(headers) if header == label]
    if len(matching) != 1:
        raise ValueError(f"Expected one {label!r} header on row {HEADER_ROW}")
    column = matching[0]
    if units[column] != expected_unit:
        raise ValueError(f"{label} row {UNITS_ROW} has unexpected units {units[column]!r}")
    return column


def parse_price(values):
    """Explicitly recognize source missing markers and identify invalid price cells."""
    missing = values.map(
        lambda value: value is None
        or isinstance(value, str) and value.strip() in MISSING_MARKERS
    )
    numeric = pd.to_numeric(values.mask(missing), errors="coerce")
    invalid = ~missing & (~np.isfinite(numeric) | numeric.le(0))
    return numeric.mask(invalid).astype(float), missing, invalid


def verify_source_returns(rows, row_for_month, data, source_columns):
    """Independently verify selected study returns directly from worksheet cells."""
    details = []
    for month_text in CHECK_MONTHS:
        month = pd.Period(month_text, freq="M")
        for label, _, prefix, price_field in SERIES:
            column = source_columns[label]
            prior_row = row_for_month.get(month - 1)
            current_row = row_for_month.get(month)
            if prior_row is None or current_row is None:
                details.append(f"  {month} {label} check skipped because a source month is absent.")
                continue
            prior = rows[prior_row - 1][column]
            current = rows[current_row - 1][column]
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       and value > 0 for value in (prior, current)):
                details.append(f"  {month} {label} check skipped because an input price is unavailable.")
                continue

            actual = data.loc[month]
            expected_simple = (current - prior) / prior
            expected_log = math.log(current) - math.log(prior)
            comparisons = (
                (actual[price_field], current),
                (actual[f"{prefix}_simple_return"], expected_simple),
                (actual[f"{prefix}_percentage_return"], 100 * expected_simple),
                (actual[f"{prefix}_log_return"], expected_log),
            )
            if not all(math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)
                       for got, expected in comparisons):
                raise ValueError(f"Independent commodity return check failed for {label} {month}")
            details.append(
                f"  {month} {label} Excel rows {prior_row:,} and {current_row:,}, "
                f"{prior:g} to {current:g}, simple {expected_simple:.10f}, "
                f"log {expected_log:.10f} (verified)."
            )
    return details


def summarize_price(label, prefix, field, data, raw_values, missing, invalid,
                    row_for_month):
    price = data[field]
    observed = price.dropna()
    study = data.reindex(RESEARCH_MONTHS)
    study_price = study[field].dropna()
    study_returns = study[f"{prefix}_simple_return"].dropna()
    types = Counter(type(value).__name__ for value in raw_values)
    markers = Counter(repr(value) for value in raw_values[missing])
    details = [
        f"{label}",
        f"  Source cell types {dict(types)}. Missing markers {dict(markers)}. "
        f"Invalid or nonpositive cells {int(invalid.sum())}.",
        f"  {len(observed):,} valid historical monthly prices, "
        f"{observed.index.min()} to {observed.index.max()}."
        if len(observed) else "  No valid historical monthly prices.",
        f"  Study coverage {len(study_price)}/{len(RESEARCH_MONTHS)} prices, "
        f"{len(study_returns)}/{len(RESEARCH_MONTHS)} simple and percentage returns, "
        f"{study[f'{prefix}_log_return'].count()}/{len(RESEARCH_MONTHS)} log returns.",
    ]
    if missing.any():
        missing_periods = raw_values.index[missing]
        details.append(f"  Source missing-marker span {missing_periods.min()} to "
                       f"{missing_periods.max()}.")
    for period, value in raw_values[invalid].items():
        details.append(f"  Unusable price in Excel row {row_for_month[period]}, "
                       f"period {period}, raw value {value!r}. Cleaned cell left blank.")
    if len(study_price):
        details.append(
            f"  Study price minimum {study_price.min():g} ({study_price.idxmin()}), "
            f"maximum {study_price.max():g} ({study_price.idxmax()})."
        )
    if len(study_returns):
        extreme_months = study_returns.abs().nlargest(5).index
        extreme_moves = ", ".join(
            f"{month} {study_returns.loc[month]:+.2%}" for month in extreme_months
        )
        details.append(f"  Five largest absolute monthly movements (signed) were {extreme_moves}.")
    return details


def run(root):
    """Write the cleaned commodity history and return the commodity QA narrative."""
    root = Path(root)
    workbook = load_workbook(root / "Raw" / WORKBOOK, read_only=True, data_only=True)
    try:
        sheet = workbook[SHEET]
        rows = list(sheet.values)
        description = workbook["Description"]
        gold_definition = description["B103"].value
        coal_definition = description["B4"].value
        mismatch_sheet = workbook["Mismatch Details"]
        compared_commodities = {
            row[1] for row in mismatch_sheet.iter_rows(min_row=5, values_only=True)
        }
    finally:
        workbook.close()

    headers = rows[HEADER_ROW - 1]
    units = rows[UNITS_ROW - 1]
    source_columns = {
        label: find_price_column(headers, units, label, expected_unit)
        for label, expected_unit, _, _ in SERIES
    }

    source_rows = []
    for row_number, row in enumerate(rows[UNITS_ROW:], start=UNITS_ROW + 1):
        if all(value is None for value in row):
            continue
        period_label = row[0]
        if not isinstance(period_label, str) or not PERIOD_PATTERN.fullmatch(period_label):
            raise ValueError(f"Invalid commodity period label in Excel row {row_number}: "
                             f"{period_label!r}")
        period = pd.Period(period_label.replace("M", "-"), freq="M")
        source_rows.append((period, row_number, period_label))

    periods = pd.PeriodIndex([period for period, _, _ in source_rows], freq="M")
    if periods.has_duplicates:
        raise ValueError(f"Duplicate World Bank periods: {periods[periods.duplicated()].tolist()}")
    if periods.empty:
        raise ValueError("No World Bank monthly observations were found")
    row_for_month = {period: row_number for period, row_number, _ in source_rows}
    data = pd.DataFrame(
        {"world_bank_period": [label for _, _, label in source_rows]}, index=periods
    )
    data.index.name = "month"
    data = data.sort_index()
    full_calendar = pd.period_range(periods.min(), periods.max(), freq="M")
    absent_months = full_calendar.difference(periods)
    data = data.reindex(full_calendar).rename_axis("month")
    details = [
        "COMMODITY TRANSFORMATION AND QUALITY CHECKS",
        f"Source is the supplied World Bank Pink Sheet workbook, {SHEET!r}. "
        f"{rows[3][0]}. Rows 5 and 6 provide headers and units. "
        "Gold is in column BR and Coal, Australian is in column F in this vintage.",
        f"Parsed {len(periods):,} original monthly labels such as 2018M01.",
        f"Monthly source coverage {periods.min()} through {periods.max()}. "
        f"Duplicate months 0. Missing calendar months {len(absent_months)}"
        + (f" ({', '.join(map(str, absent_months))})." if len(absent_months) else "."),
        "Retained supplied monthly prices at source precision and retained the original period "
        "text alongside the YYYY-MM join key. Units are the workbook labels for nominal USD per "
        "troy ounce of gold and nominal USD per metric ton of coal.",
        "Ellipsis markers and source blanks produce blank cleaned values. The price series were "
        "uninterpolated and all observed extreme changes were retained. Simple and log returns "
        "are decimals. The cleaned file also includes simple returns multiplied by 100.",
        "Each return requires positive prices in the current and immediately preceding calendar "
        "months. December 2017 supplies the base for January 2018. The first priced month in "
        "each full historical series has a blank return if its previous month has no price.",
    ]

    for label, unit, prefix, field in SERIES:
        raw_values = pd.Series(
            [rows[row_number - 1][source_columns[label]] for _, row_number, _ in source_rows],
            index=periods,
            dtype=object,
        )
        prices, missing, invalid = parse_price(raw_values)
        data[f"{prefix}_world_bank_unit"] = unit
        data[field] = prices
        price_ratio = data[field] / data[field].shift()
        data[f"{prefix}_simple_return"] = price_ratio - 1
        data[f"{prefix}_percentage_return"] = 100 * data[f"{prefix}_simple_return"]
        data[f"{prefix}_log_return"] = np.log(price_ratio)
        details.extend(summarize_price(label, prefix, field, data, raw_values, missing,
                                       invalid, row_for_month))

    details.append("Independent return checks from worksheet values")
    details.extend(verify_source_returns(rows, row_for_month, data, source_columns))
    details.extend((
        "Commodity benchmarks and limitations",
        f"  Australian coal definition from workbook Description B4. {coal_definition}",
        f"  Gold definition from workbook Description B103. {gold_definition}",
        "  The February 2022 coal spot-to-futures transition and June 2025 gold definition "
        "transition both occur within the study period. Returns across those months may include "
        "effects of the benchmark changes. The supplied series were retained as published.",
        "  Australian Newcastle thermal coal is a proxy for MMC's commodity exposure. It differs "
        "from Mongolian coking coal by grade and market. Investigate a coking-coal benchmark for "
        "the econometric stage.",
        "  The workbook includes a Mismatch Details tab listing differences in other series."
        if not {spec[0] for spec in SERIES} & compared_commodities
        else "  The workbook Mismatch Details tab flags a selected series. Review source vintage.",
        "  This post-study workbook vintage may contain historical revisions. The files do not "
        "establish the historical publication vintage or release-time availability of each value.",
    ))

    cleaned = root / "Cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    data.to_csv(cleaned / "Commodity_monthly_clean.csv", na_rep="")
    return "\n".join(details)


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[1]))
