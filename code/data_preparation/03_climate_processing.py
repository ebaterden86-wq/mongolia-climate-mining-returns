"""Aggregate daily site climate and calculate 1991–2020 monthly anomalies."""

import argparse
import calendar
from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


OBSERVATION_HEADER = ["time", "temperature_2m_mean (°C)", "precipitation_sum (mm)"]
FIRST_DAY = "1991-01-01"
LAST_DAY = "2025-12-31"
BASELINE_START = 1991
BASELINE_END = 2020
SITES = {"UHG": "Ukhaa Khudag", "ATO": "Altan Tsagaan Ovoo"}


def load_daily(path):
    """Read metadata and daily observations without treating the preamble as data."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        records = list(csv.reader(source))
    header = records.index(OBSERVATION_HEADER)
    metadata_rows = [row for row in records[:header] if row]
    if len(metadata_rows) != 2:
        raise ValueError(f"{path.name} has an unexpected climate metadata section")
    metadata = dict(zip(*metadata_rows, strict=True))
    rows = records[header + 1 :]
    bad_width = [line for line, row in enumerate(rows, header + 2) if len(row) != 3]
    if bad_width:
        raise ValueError(f"{path.name} has malformed observations on lines {bad_width}")

    raw = pd.DataFrame(rows, columns=OBSERVATION_HEADER, dtype=str)
    raw_dates = raw[OBSERVATION_HEADER[0]]
    dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    bad_dates = dates.isna() | ~raw_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    duplicate_dates = dates[~bad_dates].duplicated()
    if bad_dates.any() or duplicate_dates.any():
        raise ValueError(
            f"{path.name} has {int(bad_dates.sum())} invalid date rows and "
            f"{int(duplicate_dates.sum())} extra rows with duplicate dates"
        )

    expected = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    observed = pd.DatetimeIndex(dates, name="date")
    if len(observed.difference(expected)):
        raise ValueError(f"{path.name} has dates outside {FIRST_DAY} through {LAST_DAY}")
    missing_days = expected.difference(observed)
    daily = pd.DataFrame(index=observed)
    audit = [
        f"{path.name} — {len(rows):,} observation rows × 3 columns, {len(records):,} physical CSV records.",
        f"Observation header on line {header + 1}. Fields are {', '.join(OBSERVATION_HEADER)}.",
        f"Source dates {observed.min():%Y-%m-%d} through {observed.max():%Y-%m-%d}. "
        f"Daily frequency with {len(expected):,} expected dates and {len(missing_days):,} missing dates. "
        f"Input in ascending order: {observed.is_monotonic_increasing}.",
        f"Date parsed as date-only ISO text into datetime. Duplicate dates: {int(duplicate_dates.sum())}. "
        f"Invalid date rows: {int(bad_dates.sum())}.",
        "Returned metadata  " + ", ".join(f"{key}={value}" for key, value in metadata.items()),
    ]
    if len(missing_days):
        audit.append("Missing raw dates  " + ", ".join(missing_days.strftime("%Y-%m-%d")))

    for column, name, unit, lower in [
        (OBSERVATION_HEADER[1], "temperature_c", "°C", -273.15),
        (OBSERVATION_HEADER[2], "precipitation_mm", "mm", 0.0),
    ]:
        numeric = pd.to_numeric(raw[column], errors="coerce").astype(float)
        missing = raw[column].str.strip().eq("")
        invalid = ~missing & (~np.isfinite(numeric) | numeric.lt(lower))
        daily[name] = numeric.mask(invalid).to_numpy()
        series = daily[name]
        audit.append(
            f"{column} parsed as float ({unit}). Blank fields {int(missing.sum())}, "
            f"invalid numeric fields {int(invalid.sum())}."
        )
        if invalid.any():
            audit.append("Invalid numeric dates  " + ", ".join(observed[invalid].strftime("%Y-%m-%d")))
        if series.notna().any():
            audit.append(
                f"Daily range {series.min():.2f} {unit} ({series.idxmin():%Y-%m-%d}) "
                f"through {series.max():.2f} {unit} ({series.idxmax():%Y-%m-%d}) "
                "(first source date shown in a tie)."
            )
    audit.append(f"Daily precipitation zeros retained  {int(daily['precipitation_mm'].eq(0).sum()):,}.")
    return daily.sort_index(), rows, audit


def aggregate(daily):
    """Require complete variable-specific daily coverage and a 30-year baseline."""
    months = pd.period_range(FIRST_DAY, LAST_DAY, freq="M", name="month")
    groups = daily.groupby(daily.index.to_period("M"), sort=True)
    counts = groups.count().reindex(months, fill_value=0)
    output = pd.DataFrame(index=months)
    output["days_expected"] = months.days_in_month
    output["days_observed"] = groups.size().reindex(months, fill_value=0)
    output["temperature_days_valid"] = counts["temperature_c"]
    output["precipitation_days_valid"] = counts["precipitation_mm"]
    output["temperature_c"] = groups["temperature_c"].mean().reindex(months).where(
        output["temperature_days_valid"].eq(output["days_expected"])
    )
    output["precipitation_mm"] = groups["precipitation_mm"].sum(min_count=1).reindex(months).where(
        output["precipitation_days_valid"].eq(output["days_expected"])
    )
    reference = output.loc[f"{BASELINE_START}-01":f"{BASELINE_END}-12"]
    reference_years = BASELINE_END - BASELINE_START + 1
    for variable, unit in [("temperature", "c"), ("precipitation", "mm")]:
        observed = f"{variable}_{unit}"
        grouped = reference[observed].groupby(reference.index.month)
        baseline = grouped.mean().where(grouped.count().eq(reference_years))
        output[f"{variable}_baseline_n"] = months.month.map(grouped.count())
        output[f"{variable}_baseline_{unit}"] = months.month.map(baseline)
        output[f"{variable}_anomaly_{unit}"] = (
            output[observed] - output[f"{variable}_baseline_{unit}"]
        )
    rain_baseline = output["precipitation_baseline_mm"]
    output["precipitation_anomaly_pct"] = (
        100 * output["precipitation_anomaly_mm"] / rain_baseline.where(rain_baseline.ne(0))
    )
    return output


def independent_check(rows, monthly):
    """Cross-check all aggregations and baselines using the original CSV tokens."""
    raw_months = defaultdict(lambda: [[], []])
    for day, *fields in rows:
        values = raw_months[day[:7]]
        for position, (field, minimum) in enumerate(zip(fields, (-273.15, 0.0))):
            try:
                value = float(field)
            except ValueError:
                continue
            if math.isfinite(value) and value >= minimum:
                values[position].append(value)

    def complete_values(period):
        temperature, precipitation = raw_months[str(period)]
        days = calendar.monthrange(period.year, period.month)[1]
        return [
            math.fsum(temperature) / days if len(temperature) == days else math.nan,
            math.fsum(precipitation) if len(precipitation) == days else math.nan,
        ]

    raw_baseline = {month: [[], []] for month in range(1, 13)}
    manual = {}
    for period in monthly.index:
        values = complete_values(period)
        manual[period] = values
        if BASELINE_START <= period.year <= BASELINE_END:
            for position, value in enumerate(values):
                raw_baseline[period.month][position].append(value)

    def check(value, independent, name, period):
        if math.isnan(value) and math.isnan(independent):
            return
        if not math.isclose(value, independent, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"{period} {name} failed the independent climate comparison")

    notes = []
    examples = {"1991-01", "2000-02", "2018-01", "2018-07", "2020-02", "2025-12"}
    for period, (temperature, precipitation) in manual.items():
        base_temperature, base_precipitation = [
            math.fsum(values) / len(values) for values in raw_baseline[period.month]
        ]
        temperature_anomaly = temperature - base_temperature
        precipitation_anomaly = precipitation - base_precipitation
        precipitation_pct = (
            100 * precipitation_anomaly / base_precipitation if base_precipitation != 0 else math.nan
        )
        row = monthly.loc[period]
        for field, value in [
            ("temperature_c", temperature),
            ("precipitation_mm", precipitation),
            ("temperature_baseline_c", base_temperature),
            ("precipitation_baseline_mm", base_precipitation),
            ("temperature_anomaly_c", temperature_anomaly),
            ("precipitation_anomaly_mm", precipitation_anomaly),
            ("precipitation_anomaly_pct", precipitation_pct),
        ]:
            check(row[field], value, field, period)
        for field, values in zip(
            ("temperature_baseline_n", "precipitation_baseline_n"), raw_baseline[period.month]
        ):
            check(row[field], sum(math.isfinite(value) for value in values), field, period)

        if str(period) in examples:
            temps, rain = raw_months[str(period)]
            temp_days, rain_days = len(temps), len(rain)
            notes.append(
                f"{period} temperature sum {math.fsum(temps):.6f} / {temp_days} = {temperature:.6f} °C "
                f"and total of {rain_days} precipitation values = {precipitation:.6f} mm. "
                f"Baselines {base_temperature:.6f} °C and {base_precipitation:.6f} mm. "
                f"Anomalies {temperature_anomaly:+.6f} °C and {precipitation_anomaly:+.6f} mm "
                f"({precipitation_pct:+.6f}%)."
            )
    return notes


def run(root):
    """Write the two cleaned climate files and return auditable report text."""
    notes = [
        "CLIMATE AGGREGATION AND ANOMALIES",
        "The project brief identifies Open-Meteo historical reanalysis as ERA5. Open-Meteo "
        "documents ERA5 at 0.25° (~25 km). The CSVs record the returned grid locations, elevation "
        "and GMT day convention in a daily CSV export. Source request coordinates, endpoint parameters and selected model "
        "cannot be independently recovered from the CSVs. The supplied daily values are site-linked "
        "model estimates. Days were retained as supplied in GMT.",
        "Source documentation  https://open-meteo.com/en/docs/historical-weather-api",
        "For each site and calendar month, average daily 2 m mean temperature and sum daily precipitation. "
        "Each variable needs all expected valid days in its month. Retain a monthly row if a variable "
        "is incomplete, and leave that measurement missing. A climatology uses the arithmetic mean "
        "of the 30 same-calendar-month aggregates from 1991 through 2020 for each variable and site. "
        "This fixed baseline includes research months in 2018–2020 under the supplied design. "
        "Subtract that baseline for absolute anomalies. Percentage precipitation anomaly equals "
        "100 × (monthly total - baseline) / baseline where the baseline is nonzero.",
    ]
    cleaned = Path(root) / "Cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    for site, site_name in SITES.items():
        daily, rows, audit = load_daily(Path(root) / "Raw" / f"{site}.csv")
        monthly = aggregate(daily)
        examples = independent_check(rows, monthly)
        notes.extend(["", f"{site} — {site_name}", *audit])
        notes.append(
            f"{len(monthly)} unique calendar-month rows, {monthly.index.min()} through "
            f"{monthly.index.max()}. Missing monthly temperature {monthly['temperature_c'].isna().sum()}, "
            f"missing monthly precipitation {monthly['precipitation_mm'].isna().sum()}, "
            f"zero precipitation baselines {monthly['precipitation_baseline_mm'].eq(0).sum()}, "
            f"missing percentage anomalies {monthly['precipitation_anomaly_pct'].isna().sum()}."
        )
        baseline_table = monthly.loc["1991-01":"1991-12"]
        notes.append("Calendar month | Temperature baseline °C (n) | Precipitation baseline mm (n)")
        for period, row in baseline_table.iterrows():
            notes.append(
                f"{period.month:02} | {row['temperature_baseline_c']:.6f} "
                f"({int(row['temperature_baseline_n'])}) | "
                f"{row['precipitation_baseline_mm']:.6f} ({int(row['precipitation_baseline_n'])})"
            )
        notes.append(
            "Independent recomputation from CSV tokens passed for all monthly measurements, baselines "
            "and anomalies. Selected arithmetic checks follow."
        )
        notes.extend(examples)
        notes.append("2018–2025 observed climate extremes (first month reported in a tie).")
        research = monthly.loc["2018-01":"2025-12"]
        for field, unit in [
            ("temperature_c", "°C"),
            ("precipitation_mm", "mm"),
            ("temperature_anomaly_c", "°C"),
            ("precipitation_anomaly_mm", "mm"),
            ("precipitation_anomaly_pct", "%"),
        ]:
            series = research[field]
            if series.notna().any():
                notes.append(
                    f"{field}  {series.min():.6f} {unit} ({series.idxmin()}) to "
                    f"{series.max():.6f} {unit} ({series.idxmax()})."
                )
        notes.append(
            "All validated values and extremes were retained. Coverage checks assess continuity "
            "of the supplied dates. Method changes or climate regime shifts cannot be established "
            "from continuity alone. Small precipitation baselines can generate large percentage anomalies."
        )
        monthly.add_prefix(f"{site.lower()}_").to_csv(
            cleaned / f"{site}_monthly_climate.csv", encoding="utf-8", lineterminator="\n"
        )
    return "\n".join(notes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    print(run(args.root))
