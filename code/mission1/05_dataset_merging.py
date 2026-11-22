"""Left-join validated monthly series and export the master and its metadata."""

from pathlib import Path
from textwrap import wrap

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


MONTHS = pd.period_range("2018-01", "2025-12", freq="M").astype(str).tolist()
FINANCE = (
    ("mmc", "Mongolian Mining Corporation", "MMC_monthly_clean.csv", "0975.HK.csv", "HKD/share"),
    ("steppe_gold", "Steppe Gold", "SteppeGold_monthly_clean.csv", "STGO.TO.csv", "CAD/share"),
    ("hsi", "Hang Seng Index", "HSI_monthly_clean.csv", "^HSI.csv", "index points"),
    ("tsx", "S&P/TSX Composite", "TSX_monthly_clean.csv", "^GSPTSE.csv", "index points"),
)
CLIMATE = (("uhg", "Ukhaa Khudag", "UHG"), ("ato", "Altan Tsagaan Ovoo", "ATO"))
COMMODITIES = (
    ("gold", "Gold", "usd_per_troy_oz", "USD/troy oz"),
    ("coal_australian", "Australian thermal coal", "usd_per_metric_ton", "USD/metric ton"),
)
MASTER = "Master_Dataset_2018_2025"


def add_field(dictionary, variable, datatype, unit, source, definition, notes=""):
    dictionary.append({
        "variable": variable, "data_type": datatype, "unit": unit,
        "source": source, "definition": definition, "notes": notes,
    })


def add_returns(dictionary, prefix, name, source, caveat):
    add_field(dictionary, f"{prefix}_simple_return", "float", "decimal", source,
              f"{name}: current price divided by prior adjacent calendar month's price, minus one.", caveat)
    add_field(dictionary, f"{prefix}_log_return", "float", "decimal", source,
              f"{name}: natural logarithm of current price divided by prior adjacent calendar month's price.", caveat)


def join_monthly(master, path, fields):
    frame = pd.read_csv(path, dtype={"month": "string", "observation_date": "string"})
    if not frame["month"].str.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])").fillna(False).all():
        raise ValueError(f"Invalid month in {path.name}")
    return master.merge(frame[["month", *fields]].rename(columns=fields),
                        on="month", how="left", validate="one_to_one", sort=False)


def build_master(cleaned):
    master = pd.DataFrame({"month": MONTHS}, dtype="string")
    dictionary = []
    add_field(dictionary, "month", "text", "YYYY-MM", "Research calendar",
              "Calendar month, inclusive from 2018-01 through 2025-12.", "Primary key.")
    for prefix, name, filename, source, unit in FINANCE:
        fields = {kind: f"{prefix}_{kind}" for kind in ("adjusted_close", "simple_return", "log_return")}
        is_steppe = prefix == "steppe_gold"
        if is_steppe:
            fields["observation_date"] = f"{prefix}_observation_date"
        master = join_monthly(master, cleaned / filename, fields)
        endpoint = ("Adjusted close on the last date supplied within the month." if is_steppe else
                    "Vendor monthly adjusted close. Original Date is a calendar-month label.")
        caveat = "Index price return. Dividends are not necessarily included." if prefix in {"hsi", "tsx"} else ""
        add_field(dictionary, f"{prefix}_adjusted_close", "float", unit, source, f"{name}. {endpoint}", caveat)
        add_returns(dictionary, prefix, name, source, caveat)
        if is_steppe:
            add_field(dictionary, fields["observation_date"], "text", "YYYY-MM-DD", source,
                      "Date of the selected last supplied Steppe Gold observation within a calendar month.",
                      "A quoted price may have zero reported volume. December 2025 ends one scheduled TSX session early.")
    for prefix, name, code in CLIMATE:
        source = f"{code}.csv"
        fields = {
            "temperature_c": ("°C", "Mean of the daily mean near-surface (2 m) air temperatures."),
            "precipitation_mm": ("mm", "Sum of daily total precipitation across the calendar month."),
            "temperature_anomaly_c": ("°C", "Monthly mean temperature minus its 1991–2020 same-month climatology."),
            "precipitation_anomaly_mm": ("mm", "Monthly total precipitation minus its 1991–2020 same-month climatology."),
            "precipitation_anomaly_pct": ("%", "100 × precipitation anomaly divided by the same-month precipitation climatology."),
        }
        names = {f"{prefix}_{field}": f"{prefix}_{field}" for field in fields}
        master = join_monthly(master, cleaned / f"{code}_monthly_climate.csv", names)
        for field, (unit, definition) in fields.items():
            note = "Open-Meteo grid-cell estimate. Months use GMT day boundaries."
            if field.endswith("_pct"):
                note += " A value of 10 represents 10%. Undefined at a zero baseline."
            add_field(dictionary, f"{prefix}_{field}", "float", unit, source,
                      f"{name}. {definition}", note)
    filename = "Commodity_monthly_clean.csv"
    source = "CMO-Historical-Data-Monthly.xlsx, Monthly Prices"
    for prefix, name, suffix, unit in COMMODITIES:
        price = f"{prefix}_price_{suffix}"
        fields = {k: k for k in (price, f"{prefix}_simple_return", f"{prefix}_log_return")}
        master = join_monthly(master, cleaned / filename, fields)
        caveat = ("World Bank Australian thermal coal proxy. Spot-to-futures benchmark change in 2022-02."
                  if prefix == "coal_australian" else
                  "World Bank gold benchmark changes from London afternoon fixing to spot average in 2025-06.")
        add_field(dictionary, price, "float", unit, source,
                  f"{name}, nominal monthly average price as supplied.", caveat)
        add_returns(dictionary, prefix, name, source, caveat)
    return master, pd.DataFrame(dictionary)


def missing_reason(variable, month):
    if variable.startswith("steppe_gold_"):
        if month < "2018-05":
            return "Pre-listing month."
        if variable.endswith("_return") and month == "2018-05":
            return "First supplied Steppe Gold month has no April price."
    if variable.endswith("_return") and month == "2018-01" and variable.startswith(("mmc_", "hsi_", "tsx_")):
        return "December 2017 price absent from the supplied Yahoo Finance file."
    return "Source observation, valid adjacent price, or valid baseline unavailable. See source audit."


def missingness_report(master):
    records = []
    for variable in master:
        months = master.loc[master[variable].isna(), "month"].tolist()
        reasons = list(dict.fromkeys(missing_reason(variable, month) for month in months))
        records.append({
            "variable": variable,
            "total_months": len(master),
            "usable_months": len(master) - len(months),
            "missing_month_count": len(months),
            "missing_pct": 100 * len(months) / len(master),
            "missing_months": ", ".join(months),
            "reason": " ".join(reasons),
        })
    return pd.DataFrame(records)


def write_sheet(book, name, frame, widths):
    sheet = book.create_sheet(name)
    sheet.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(value) else value for value in row])
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 43 if name == "Master" else 28
    header_fill = PatternFill("solid", fgColor="17354A")
    alternate_fill = PatternFill("solid", fgColor="F0F5F7")
    sheet.sheet_view.zoomScale = 90 if name == "Master" else 85
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for cell in sheet[1]:
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        cell.fill = header_fill
    for row in sheet.iter_rows(min_row=2):
        max_lines = 1
        for index, cell in enumerate(row):
            if cell.row % 2 == 0:
                cell.fill = alternate_fill
            cell.font = Font(name="Calibri", size=11, color="1D303B")
            cell.alignment = Alignment(vertical="center", wrap_text=name != "Master")
            if name != "Master" and isinstance(cell.value, str):
                max_lines = max(max_lines, len(wrap(cell.value, width=max(8, widths[index] - 5))))
            if cell.data_type == "n" and cell.value is not None:
                if name == "Master":
                    header = frame.columns[index]
                    if header.endswith("_return"):
                        cell.number_format = "0.000000"
                    elif header.endswith("_pct"):
                        cell.number_format = "0.00"
                    else:
                        cell.number_format = "#,##0.0000"
                elif frame.columns[index] == "missing_pct":
                    cell.number_format = "0.00"
        sheet.row_dimensions[row[0].row].height = max(17, max_lines * 15 + 4)


def write_workbook(path, master, dictionary, missingness):
    book = Workbook()
    book.remove(book.active)
    book.properties.title = "SSRN Mining Climate Research | Master Dataset"
    book.properties.subject = "Monthly series, 2018–2025"
    master_widths = [12] + [max(22, min(51, len(name) + 4)) for name in master.columns[1:]]
    write_sheet(book, "Master", master, master_widths)
    write_sheet(book, "Data Dictionary", dictionary, [39, 14, 18, 34, 74, 68])
    write_sheet(book, "Missingness", missingness, [39, 16, 17, 23, 15, 37, 62])
    book.save(path)


def run(root):
    master, dictionary = build_master(root / "Cleaned")
    missingness = missingness_report(master)
    final = root / "Final"
    final.mkdir(exist_ok=True)
    master.to_csv(final / f"{MASTER}.csv", index=False)
    dictionary.to_csv(final / "Data_Dictionary.csv", index=False)
    missingness.to_csv(final / "Missingness_Report.csv", index=False)
    write_workbook(final / f"{MASTER}.xlsx", master, dictionary, missingness)
    return master


if __name__ == "__main__":
    frame = run(Path(__file__).resolve().parents[1])
    print(f"Master exported: {len(frame)} rows × {len(frame.columns)} columns.")
