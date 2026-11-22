"""Reproduce the final controlled models, influence checks and diagnostic tables."""

import argparse
import hashlib
from html import escape
from importlib.metadata import version
import json
from math import erfc, exp, pi, sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle


TERMS = ("intercept", "temperature", "precipitation", "market", "commodity")
CLIMATE = TERMS[1:3]
FIRMS = {
    "MMC": (("mmc", "uhg", "hsi", "coal_australian"), 95, "2018-02", "2020-12",
            (-2.23, 0.20, 0.57, 0.33, 0.100, 0.159)),
    "Steppe Gold": (("steppe_gold", "ato", "tsx", "gold"), 91, "2018-06", "2020-08",
                   (0.25, -0.06, 0.86, 1.73, 0.256, 0.918)),
}
LABELS = dict(zip(TERMS, ("Intercept", "Temperature anomaly (per 1 °C)",
                         "Precipitation anomaly (per 10 mm)", "Market log return (×100)",
                         "Commodity log return (×100)")))
STUDY = pd.period_range("2018-01", "2025-12", freq="M")
Z95 = 1.959963984540054
HAC_LAGS = 3
TOLERANCE = 1e-10
NEAR_ZERO = 0.25


def chi_square_sf(statistic, df):
    """Chi-square survival probabilities for the integer degrees of freedom used here."""
    if df not in {1, 2, 3, 4} or not np.isfinite(statistic) or statistic < -1e-12:
        raise ValueError(f"Invalid chi-square statistic or degrees of freedom: {statistic}, {df}")
    half = max(0.0, statistic) / 2
    base = exp(-half)
    if df == 1:
        return erfc(sqrt(half))
    if df == 2:
        return base
    if df == 3:
        return erfc(sqrt(half)) + 2 * sqrt(half / pi) * base
    return (1 + half) * base


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_samples(path):
    source = pd.read_excel(path, sheet_name="Master", engine="openpyxl", dtype={"month": "string"})
    if source["month"].tolist() != STUDY.astype(str).tolist():
        raise ValueError("Master must have 96 unique chronological months from 2018-01 through 2025-12.")
    source.index = STUDY
    samples = {}
    for firm, ((stock, site, market, commodity), *_) in FIRMS.items():
        values = pd.DataFrame({
            "return": 100 * source[f"{stock}_log_return"],
            "temperature": source[f"{site}_temperature_anomaly_c"],
            "precipitation": source[f"{site}_precipitation_anomaly_mm"] / 10,
            "market": 100 * source[f"{market}_log_return"],
            "commodity": 100 * source[f"{commodity}_log_return"],
        })
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"{firm} contains infinite analysis values.")
        samples[firm] = values.dropna()
    return samples


def calendar_check(months):
    expected = pd.period_range(months[0], months[-1], freq="M")
    return {"chronological": bool(months.is_monotonic_increasing), "unique": bool(months.is_unique),
            "missing_inside": expected.difference(months).astype(str).tolist()}


def fit_controlled(frame):
    """Fit OLS by reduced QR and construct the Bartlett covariance in score space."""
    n = len(frame)
    y = frame["return"].to_numpy(dtype=float)
    x = np.column_stack((np.ones(n), frame[list(TERMS[1:])].to_numpy(dtype=float)))
    q, r = np.linalg.qr(x)
    k = x.shape[1]
    if n <= k or np.linalg.matrix_rank(r) < k:
        raise ValueError("OLS design lacks rank or residual degrees of freedom.")
    beta = np.linalg.solve(r, q.T @ y)
    residual = y - x @ beta
    sse = float(residual @ residual)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst
    positions = frame.index.asi8 - frame.index.asi8[0]
    # Zero an omitted calendar month's score, preserving distances between observed months.
    score = np.zeros((int(positions[-1]) + 1, k))
    score[positions] = q * residual[:, None]
    meat = score.T @ score
    for lag in range(1, HAC_LAGS + 1):
        cross = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (HAC_LAGS + 1)) * (cross + cross.T)
    r_inverse = np.linalg.solve(r, np.eye(k))
    cov = (n / (n - k)) * r_inverse @ meat @ r_inverse.T
    se = np.sqrt(np.diag(cov))
    if not np.isfinite(se).all() or (se <= 0).any():
        raise ValueError("Invalid HAC standard error.")
    terms = {}
    for name, coefficient, error in zip(TERMS, beta.tolist(), se.tolist()):
        terms[name] = {"estimate": coefficient, "hac_se": error,
                       "ci95_low": coefficient - Z95 * error, "ci95_high": coefficient + Z95 * error,
                       "p_normal": erfc(abs(coefficient / error) / sqrt(2))}
    wald = float(beta[1:3] @ np.linalg.solve(cov[1:3, 1:3], beta[1:3]))
    leverage = np.sum(q ** 2, axis=1)
    cook = residual ** 2 / (k * sse / (n - k)) * leverage / (1 - leverage) ** 2
    index = int(cook.argmax())
    months = frame.index.astype(str).tolist()
    fit = {"n": n, "first": months[0], "last": months[-1], "included_months": months,
           "calendar": calendar_check(frame.index), "hac_lags": HAC_LAGS, "hac_finite_sample_factor": n / (n - k),
           "terms": terms, "hac_covariance": cov.tolist(), "r2": r2,
           "adjusted_r2": 1 - (1 - r2) * (n - 1) / (n - k),
           "climate_joint_wald_chi2_df2": wald, "climate_joint_p": chi_square_sf(wald, 2),
           "max_cooks_distance": float(cook[index]), "max_cooks_month": months[index]}
    return fit, (x, residual, leverage, cook)


def replication_check(firm, fit, prior):
    _, n, first, max_month, rounded = FIRMS[firm]
    if (fit["n"], fit["first"], fit["last"], fit["max_cooks_month"]) != (n, first, "2025-12", max_month):
        raise ValueError(f"{firm} sample or most influential month differs from Mission 2: {fit}")
    if fit["calendar"] != {"chronological": True, "unique": True, "missing_inside": []}:
        raise ValueError(f"{firm} primary calendar differs from Mission 2.")
    metrics = [fit["terms"][name]["estimate"] for name in TERMS[1:]] + [fit["r2"], fit["climate_joint_p"]]
    rendered = tuple(round(value, digits) for value, digits in zip(metrics, (2, 2, 2, 2, 3, 3)))
    if rendered != rounded:
        raise ValueError(f"{firm} brief targets {rounded} differ from replicated values {rendered}.")
    if fit["included_months"] != prior["included_months"] or prior["n"] != n:
        raise ValueError(f"{firm} exact Mission 2 sample month membership differs.")
    fields = ("r2", "adjusted_r2", "climate_joint_wald_chi2_df2", "climate_joint_p", "max_cooks_distance")
    deltas = {key: abs(fit[key] - prior[key]) for key in fields}
    for term in TERMS:
        for field, value in fit["terms"][term].items():
            deltas[f"{term}.{field}"] = abs(value - prior["terms"][term][field])
    if not all(value <= TOLERANCE for value in deltas.values()):
        raise ValueError(f"{firm} Mission 2 numeric discrepancy: {deltas}")
    return {"status": "PASS", "rounded_brief_targets_match": True,
            "exact_json_tolerance": TOLERANCE, "max_abs_difference": max(deltas.values()),
            "numeric_comparisons": len(deltas)}


def auxiliary_fit(y, design):
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Diagnostic auxiliary design lacks full rank.")
    fitted = design @ beta
    total = float(np.sum((y - y.mean()) ** 2))
    return fitted, 1 - float(np.sum((y - fitted) ** 2)) / total


def lm_record(statistic, df):
    return {"lm": float(statistic), "df": df, "p_value": chi_square_sf(statistic, df)}


def diagnostics(frame, arrays):
    x, residual, leverage, cook = arrays
    n, k = x.shape
    threshold = 4 / n
    bg = {}
    for lags in (1, 3):
        # BG initialization uses zero only for unavailable presample residual lags.
        padded = np.r_[np.zeros(lags), residual]
        lagged = np.column_stack([padded[lags - lag:lags - lag + n] for lag in range(1, lags + 1)])
        _, r2 = auxiliary_fit(residual, np.column_stack((x, lagged)))
        bg[str(lags)] = lm_record(n * r2, lags)
    squares = residual ** 2
    normalized = squares / squares.mean()
    fitted, bp_r2 = auxiliary_fit(normalized, x)
    bp_classical = lm_record(float(np.sum((fitted - normalized.mean()) ** 2)) / 2, k - 1)
    bp_koenker = lm_record(n * bp_r2, k - 1)
    vifs = {}
    for index, term in enumerate(TERMS[1:], 1):
        _, r2 = auxiliary_fit(x[:, index], np.delete(x, index, axis=1))
        vifs[term] = 1 / (1 - r2)
    zs = (frame - frame.mean()) / frame.std(ddof=1)
    flags = []
    for variable in zs:
        for month, z in zs[variable].items():
            if abs(z) > 3:
                flags.append({"variable": variable, "month": str(month),
                              "value": float(frame.at[month, variable]), "sample_z": float(z)})
    return {
        "durbin_watson": float(np.sum(np.diff(residual) ** 2) / (residual @ residual)),
        "breusch_godfrey": bg, "breusch_pagan_classical": bp_classical, "breusch_pagan_koenker": bp_koenker,
        "vif": vifs, "cook_4_over_n": threshold, "cook_count_above_4_over_n": int(np.count_nonzero(cook > threshold)),
        "cook_by_month": [{"month": str(month), "cook_distance": float(distance), "leverage": float(hat),
                          "residual": float(error), "above_4_over_n": bool(distance > threshold)}
                         for month, distance, hat, error in zip(frame.index, cook, leverage, residual)],
        "unusual_observations_abs_z_gt_3": flags,
    }


def coefficient_changes(base, reduced):
    changes = {}
    for term in CLIMATE:
        before = base["terms"][term]["estimate"]
        after = reduced["terms"][term]["estimate"]
        changes[term] = {
            "signed_change": after - before, "absolute_change": abs(after - before),
            "magnitude_pct_change": (100 * (abs(after) - abs(before)) / abs(before)
                                     if abs(before) >= NEAR_ZERO else None),
            "sign_changed": before * after < 0,
        }
    return changes


def analyze(workbook, reference, note, script):
    inputs = (workbook, reference, note, script)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {path.name: sha256(path) for path in inputs}
    original = json.loads(reference.read_text(encoding="utf-8"))
    if hashes[workbook.name] != original["input_sha256"]:
        raise ValueError("Workbook SHA-256 differs from the Mission 2 input recorded in its JSON.")
    samples = read_samples(workbook)
    firms = {}
    arrays = {}
    # Verify both primary models before estimating the exclusion checks or running diagnostics.
    for firm, data in samples.items():
        base, arrays[firm] = fit_controlled(data)
        check = replication_check(firm, base, original["firms"][firm]["controlled"])
        firms[firm] = {"baseline": base, "replication": check}
    for firm, data in samples.items():
        result = firms[firm]
        base = result["baseline"]
        excluded = base["max_cooks_month"]
        reduced, _ = fit_controlled(data.loc[data.index != pd.Period(excluded, freq="M")])
        if reduced["n"] != base["n"] - 1 or set(base["included_months"]) - set(reduced["included_months"]) != {excluded}:
            raise ValueError(f"{firm} influence sensitivity must omit exactly {excluded}.")
        result.update({"exclude_max_cook": reduced, "excluded_month": excluded,
                       "changes": coefficient_changes(base, reduced),
                       "diagnostics": diagnostics(data, arrays[firm])})
    if hashes != {path.name: sha256(path) for path in inputs}:
        raise RuntimeError("An input file changed during analysis.")
    return {
        "inputs_sha256": hashes, "software": {"python": sys.version.split()[0],
            **{package: version(package) for package in ("numpy", "pandas", "openpyxl", "reportlab")}},
        "method": {
            "outcome": "Monthly stock log return multiplied by 100",
            "predictors": LABELS, "hac": "Newey-West Bartlett, 3 calendar-month lags, n/(n-k) correction",
            "inference": "Asymptotic normal intervals and coefficient p-values; chi-square(2) climate Wald test",
            "changes": f"Percent change in coefficient magnitude when absolute baseline is at least {NEAR_ZERO:.2f}",
            "diagnostics": "BG full-sample nR² with zero presample residual lags; classical BP and Koenker studentized LM",
        },
        "firms": firms,
        "tables": publication_tables(firms),
    }


def table(name, title, headers, rows, notes, formats, widths, subtitle="", row_formats=None):
    return {"name": name, "title": title, "subtitle": subtitle, "headers": headers, "rows": rows,
            "notes": notes, "formats": formats, "widths": widths, "rowFormats": row_formats or {}}


def publication_tables(firms):
    mmc, steppe = firms.values()
    models = [m for firm in (mmc, steppe) for m in (firm["baseline"], firm["exclude_max_cook"])]
    primary = [models[0], models[2]]
    diag = [mmc["diagnostics"], steppe["diagnostics"]]
    baseline = [
        [LABELS[term], *[model["terms"][term][key] for model in primary
                          for key in ("estimate", "hac_se", "ci95_low", "ci95_high")]]
        for term in TERMS
    ]
    summaries = (("Observations", "n", "0"), ("R²", "r2", "0.000"),
                 ("Adjusted R²", "adjusted_r2", "0.000"), ("Joint climate p", "climate_joint_p", "0.000"))
    baseline.extend([label, primary[0][key], None, None, None, primary[1][key], None, None, None]
                    for label, key, _ in summaries)
    baseline_formats = {str(i + len(TERMS)): fmt for i, (_, _, fmt) in enumerate(summaries)}
    influence = [["Excluded month", None, mmc["excluded_month"], None, steppe["excluded_month"]],
                 ["Observations", *[m["n"] for m in models]]]
    for term in CLIMATE:
        for label, key in ((LABELS[term], "estimate"), ("95% HAC CI lower", "ci95_low"), ("95% HAC CI upper", "ci95_high")):
            influence.append([label, *[m["terms"][term][key] for m in models]])
        changes = [firm["changes"][term] for firm in (mmc, steppe)]
        for label, key in (("Change in coefficient magnitude (%)", "magnitude_pct_change"),
                           ("Absolute coefficient change (if near zero)", "absolute_change")):
            influence.append([label, None, changes[0][key] if (key != "absolute_change" or changes[0]["magnitude_pct_change"] is None) else None,
                              None, changes[1][key] if (key != "absolute_change" or changes[1]["magnitude_pct_change"] is None) else None])
    influence.append(["Joint climate p", *[m["climate_joint_p"] for m in models]])
    diagnostic_rows = [["Durbin–Watson", diag[0]["durbin_watson"], None, diag[1]["durbin_watson"], None]]
    for label, tests in (
        ("Breusch–Godfrey LM (1 lag)", [d["breusch_godfrey"]["1"] for d in diag]),
        ("Breusch–Godfrey LM (3 lags)", [d["breusch_godfrey"]["3"] for d in diag]),
        ("Breusch–Pagan LM (classical)", [d["breusch_pagan_classical"] for d in diag]),
        ("Koenker LM (studentized BP)", [d["breusch_pagan_koenker"] for d in diag]),
    ):
        diagnostic_rows.append([label, *[test[key] for test in tests for key in ("lm", "p_value")]])
    diagnostic_rows.extend([
        ["Maximum Cook’s distance", primary[0]["max_cooks_distance"], None, primary[1]["max_cooks_distance"], None],
        ["Month of maximum", primary[0]["max_cooks_month"], None, primary[1]["max_cooks_month"], None],
        ["Cook’s threshold (4/N)", diag[0]["cook_4_over_n"], None, diag[1]["cook_4_over_n"], None],
        ["Months above 4/N", diag[0]["cook_count_above_4_over_n"], None, diag[1]["cook_count_above_4_over_n"], None],
        ["Missing internal calendar months", len(primary[0]["calendar"]["missing_inside"]), None,
         len(primary[1]["calendar"]["missing_inside"]), None],
    ])
    vif = [[LABELS[term], diag[0]["vif"][term], diag[1]["vif"][term]] for term in TERMS[1:]]
    cook_rows = [[firm, row["month"], row["cook_distance"], row["leverage"], row["residual"],
                  result["diagnostics"]["cook_4_over_n"], "Yes" if row["above_4_over_n"] else "No"]
                 for firm, result in firms.items() for row in result["diagnostics"]["cook_by_month"]]
    flags = [[firm, item["month"], LABELS.get(item["variable"], "Stock log return (×100)"),
              item["value"], item["sample_z"]]
             for firm, result in firms.items() for item in result["diagnostics"]["unusual_observations_abs_z_gt_3"]]
    return [
        table("Baseline", "Table 1. Baseline replication",
              ["Term / statistic", "MMC estimate", "MMC HAC SE", "MMC CI low", "MMC CI high",
               "Steppe estimate", "Steppe HAC SE", "Steppe CI low", "Steppe CI high"], baseline,
              ["Coefficients use 100 × monthly stock log return. Newey–West HAC uses three monthly lags.",
               "Confidence intervals are 95%. Joint test uses a chi-square(2) reference."],
              [None, *["0.00"] * 8], [38, *[16] * 8], "Controlled models | MMC and Steppe Gold", baseline_formats),
        table("Influence", "Table 2. Single-month influence sensitivity",
              ["Term / statistic", "MMC primary", "MMC exclusion", "Steppe primary", "Steppe exclusion"], influence,
              ["Exclusions remove only the maximum Cook’s distance month for each firm. HAC uses actual month distances.",
               f"Percentage changes compare magnitudes. For absolute baseline estimates below {NEAR_ZERO:.2f}, absolute changes are shown."],
              [None, *["0.00"] * 4], [45, 17, 18, 17, 19], "Primary regressions remain full sample",
              {"1": "0", str(len(influence) - 1): "0.000"}),
        table("Diagnostics", "Table 3. Primary-model diagnostics",
              ["Diagnostic", "MMC statistic", "MMC p", "Steppe statistic", "Steppe p"], diagnostic_rows,
              ["Breusch–Godfrey LM tests use chi-square df 1 and 3. Both variance LM tests use df 4.",
               "Cook’s distance uses the standard OLS formula; 4/N is a screening heuristic."],
              [None, "0.000", "0.000", "0.000", "0.000"], [43, 18, 14, 19, 14], "Full estimation samples",
              {"8": "0", "9": "0"}),
        table("VIF", "Table 4. Variance inflation factors",
              ["Regressor", "MMC", "Steppe Gold"], vif,
              ["Auxiliary regressions include an intercept; the intercept itself has no reported VIF."],
              [None, "0.00", "0.00"], [39, 17, 18], "Full estimation samples"),
        table("Cook distance", "Appendix table. Monthly influence",
              ["Firm", "Month", "Cook’s distance", "Leverage", "Residual", "4/N threshold", "Above 4/N"], cook_rows,
              ["Residuals are in 100 × monthly stock log-return units."],
              [None, None, "0.000000", "0.000000", "0.00", "0.000000", None],
              [20, 13, 19, 16, 16, 18, 17], "One row per observation in the primary regressions"),
        table("Unusual observations", "Appendix table. Unusual primary-sample values",
              ["Firm", "Month", "Variable", "Observed value", "Sample z-score"], flags,
              ["A flag means an absolute within-firm sample z-score above 3. No observation is removed by this screen."],
              [None, None, None, "0.00", "0.00"], [20, 13, 38, 18, 18], "Descriptive flag | no outlier treatment"),
    ]


def write_workbook(results, path):
    """Regenerate the manuscript tables with Excel-readable numeric types."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = "Mission 3 | Final robustness and diagnostics"
    for panel in results["tables"]:
        sheet = workbook.create_sheet(panel["name"])
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "B6"
        width = len(panel["headers"])
        for row, line in enumerate((panel["title"], panel["subtitle"]), 1):
            sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=width)
            cell = sheet.cell(row, 1, line)
            cell.font = Font(name="Calibri", size=16 if row == 1 else 10, bold=row == 1, color="173447")
            sheet.row_dimensions[row].height = 26 if row == 1 else 17
        for column, column_width in enumerate(panel["widths"], 1):
            sheet.column_dimensions[get_column_letter(column)].width = column_width
        for column, header in enumerate(panel["headers"], 1):
            cell = sheet.cell(5, column, header)
            cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="215661")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.row_dimensions[5].height = 26
        for index, values in enumerate(panel["rows"]):
            row = index + 6
            for column, value in enumerate(values, 1):
                cell = sheet.cell(row, column, value)
                cell.font = Font(name="Calibri", size=11, color="173447")
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                if index % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="EEF5F6")
                if isinstance(value, (float, int)):
                    cell.number_format = panel["rowFormats"].get(str(index)) or panel["formats"][column - 1] or "General"
                if index == len(panel["rows"]) - 1:
                    cell.border = Border(bottom=Side(style="thin", color="B8CDD2"))
            sheet.row_dimensions[row].height = 19
        end = 5 + len(panel["rows"])
        if panel["name"] in {"Cook distance", "Unusual observations"}:
            sheet.auto_filter.ref = f"A5:{get_column_letter(width)}{end}"
        for index, note in enumerate(panel["notes"], end + 2):
            sheet.merge_cells(start_row=index, start_column=1, end_row=index, end_column=width)
            cell = sheet.cell(index, 1, note)
            cell.font = Font(name="Calibri", size=9, italic=True, color="506670")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            sheet.row_dimensions[index].height = 25
    workbook.save(path)


def validate_workbook(path, results):
    """Verify all Excel table headers and values against the exact JSON data."""
    from math import isclose
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for panel in results["tables"]:
            sheet = workbook[panel["name"]]
            expected = [panel["headers"], *panel["rows"]]
            actual = list(sheet.iter_rows(min_row=5, max_row=4 + len(expected),
                                          max_col=len(expected[0]), values_only=True))
            if len(actual) != len(expected):
                raise ValueError(f"{sheet.title} row count mismatch")
            for index, (actual_row, expected_row) in enumerate(zip(actual, expected), 5):
                for column, (value, reference) in enumerate(zip(actual_row, expected_row), 1):
                    agrees = (isinstance(value, (int, float)) and isclose(value, reference, rel_tol=1e-13, abs_tol=1e-13)
                              if isinstance(reference, (float, int)) else value == reference)
                    if not agrees:
                        raise ValueError(f"{sheet.title} row {index}, column {column}: {value!r} != {reference!r}")
    finally:
        workbook.close()


PDF_INK = colors.HexColor("#192D3C")
PDF_MUTED = colors.HexColor("#4D616B")
PDF_ACCENT = colors.HexColor("#1B565D")
PDF_LINE = colors.HexColor("#B8CCCF")
PDF_FILL = colors.HexColor("#EDF5F4")
PDF_PUNCTUATION = str.maketrans({"–": "-", "—": "-", "’": "'"})
PDF_STYLES = {
    "title": ParagraphStyle("Mission 3 Title", fontName="Times-Bold", fontSize=26,
                            leading=27, spaceAfter=8, textColor=PDF_INK),
    "sub": ParagraphStyle("Mission 3 Subtitle", fontName="Helvetica", fontSize=9.6,
                          leading=12.5, spaceAfter=12, textColor=PDF_MUTED),
    "section": ParagraphStyle("Mission 3 Section", fontName="Helvetica-Bold", fontSize=11.5,
                              leading=14, spaceBefore=9, spaceAfter=4,
                              keepWithNext=True, textColor=PDF_ACCENT),
    "body": ParagraphStyle("Mission 3 Body", fontName="Helvetica", fontSize=9.7,
                           leading=12.8, spaceAfter=5.5, textColor=PDF_INK),
    "note": ParagraphStyle("Mission 3 Table Note", fontName="Helvetica", fontSize=8.2,
                           leading=10.8, spaceBefore=5, spaceAfter=4, textColor=PDF_MUTED),
    "table_text": ParagraphStyle("Mission 3 Table Text", fontName="Helvetica", fontSize=8.7,
                                 leading=10.8, textColor=PDF_INK, alignment=TA_LEFT),
    "table_value": ParagraphStyle("Mission 3 Table Numeric", fontName="Helvetica", fontSize=8.7,
                                  leading=10.8, textColor=PDF_INK, alignment=TA_RIGHT),
}


def _pdf_paragraph(text, style="body"):
    return Paragraph(text, PDF_STYLES[style])


def _pdf_table(headers, rows, widths, left_columns=(0,), dividers=()):
    data = []
    for row_index, row in enumerate([headers, *rows]):
        entries = []
        for col_index, item in enumerate(row):
            value = escape(str(item).translate(PDF_PUNCTUATION)).replace("\n", "<br/>")
            if row_index == 0:
                value = "<b>" + value + "</b>"
            style = "table_text" if col_index in left_columns else "table_value"
            entries.append(_pdf_paragraph(value, style))
        data.append(entries)
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 2.7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.7),
        ("BACKGROUND", (0, 0), (-1, 0), PDF_FILL),
        ("LINEABOVE", (0, 0), (-1, 0), 0.85, PDF_ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, PDF_LINE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.65, PDF_LINE),
    ]
    for row in dividers:
        commands.append(("LINEABOVE", (0, row), (-1, row), 0.45, PDF_LINE))
    table.setStyle(TableStyle(commands))
    return table


def _pdf_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(PDF_LINE)
    canvas.line(56, 38, 556, 38)
    canvas.setFillColor(PDF_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(56, 25, "Local climate anomalies and mining stock returns")
    canvas.drawRightString(556, 25, str(document.page))
    canvas.restoreState()


def _pdf_baseline_table(firms):
    models = [firm["baseline"] for firm in firms.values()]
    headers = ["Parameter", "MMC", "Steppe Gold"]
    labels = (
        ("Temperature anomaly, per 1°C", "temperature"),
        ("Precipitation anomaly, per 10 mm", "precipitation"),
        ("Market log return", "market"),
        ("Commodity log return", "commodity"),
        ("Intercept", "intercept"),
    )
    rows = [[label, *[f"{model['terms'][term]['estimate']:.2f}" for model in models]]
            for label, term in labels]
    rows.extend([
        ["Estimation months", *[str(model["n"]) for model in models]],
        ["R²", *[f"{model['r2']:.3f}" for model in models]],
        ["Joint climate p-value", *[f"{model['climate_joint_p']:.3f}" for model in models]],
    ])
    return _pdf_table(headers, rows, [248, 125, 127], dividers=(6,))


def _pdf_with_ci(model, term):
    result = model["terms"][term]
    return (f"{result['estimate']:.2f}\n"
            f"[{result['ci95_low']:.2f}, {result['ci95_high']:.2f}]")


def _pdf_influence_table(firms):
    headers = ["Firm / sample", "N", "Temp. per 1°C\n95% CI", "Precip. per 10 mm\n95% CI", "Joint p"]
    rows = []
    for key, fits in firms.items():
        baseline = fits["baseline"]
        excluded = fits["exclude_max_cook"]
        rows.append([f"{key} / full", baseline["n"], _pdf_with_ci(baseline, "temperature"),
                     _pdf_with_ci(baseline, "precipitation"), f"{baseline['climate_joint_p']:.3f}"])
        rows.append([f"{key} / omit {baseline['max_cooks_month']}", excluded["n"],
                     _pdf_with_ci(excluded, "temperature"), _pdf_with_ci(excluded, "precipitation"),
                     f"{excluded['climate_joint_p']:.3f}"])
    return _pdf_table(headers, rows, [151, 36, 116, 126, 71], dividers=(3,))


def _pdf_format_cell(value, spec):
    if value is None:
        return "-"
    if isinstance(value, (float, int)) and isinstance(spec, str):
        if spec == "0":
            return f"{value:.0f}"
        if "%" in spec:
            decimals = len(spec.split(".")[-1].strip("%")) if "." in spec else 0
            return f"{value:.{decimals}%}"
        if "." in spec:
            return f"{value:.{len(spec.split('.')[-1])}f}"
    return str(value)


def _pdf_read_display(table):
    formats = table.get("formats", [])
    row_formats = table.get("rowFormats", {})
    rows = []
    for row_index, row in enumerate(table["rows"]):
        formatted = []
        for idx, cell in enumerate(row):
            spec = formats[idx] if isinstance(formats, list) and idx < len(formats) else None
            if idx:
                spec = row_formats.get(str(row_index), spec)
            formatted.append(_pdf_format_cell(cell, spec))
        rows.append(formatted)
    headers = table["headers"]
    if len(headers) == 3:
        return _pdf_table(headers, rows, [250, 125, 125])
    return _pdf_table(headers, rows, [216, 81, 54, 95, 54])


def write_pdf(results, path):
    """Render the analytical results without refitting or rounding stored values."""
    firms = results["firms"]
    mmc = firms["MMC"]
    steppe = firms["Steppe Gold"]
    mt = mmc["changes"]["temperature"]["magnitude_pct_change"]
    mp = mmc["changes"]["precipitation"]["signed_change"]
    st = steppe["changes"]["temperature"]["magnitude_pct_change"]
    sp = steppe["changes"]["precipitation"]["signed_change"]
    tables = {table["name"]: table for table in results["tables"]}
    mmc_unusual = next(record for record in mmc["diagnostics"]["unusual_observations_abs_z_gt_3"]
                       if record["month"] == mmc["excluded_month"] and record["variable"] == "return")
    steppe_unusual = next(record for record in steppe["diagnostics"]["unusual_observations_abs_z_gt_3"]
                          if record["month"] == steppe["excluded_month"] and record["variable"] == "precipitation")
    story = [
        _pdf_paragraph("Final robustness and regression diagnostics", "title"),
        _pdf_paragraph("Local Climate Anomalies and Mining Stock Returns | Mongolia-linked mining firms, 2018 to 2025", "sub"),
        _pdf_paragraph(
            "The original controlled regressions replicate and remain the primary models. "
            "Removing the highest-influence month changes some climate coefficient magnitudes, while every climate "
            "interval still includes zero and all joint climate p-values remain above 0.05."),
        _pdf_paragraph("Baseline specification and replication", "section"),
        _pdf_paragraph(
            "Separate firm-level OLS models with an intercept use log stock returns multiplied by 100, temperature "
            "anomalies in degrees Celsius, precipitation anomalies per 10 mm, and same-month market and commodity "
            "log returns multiplied by 100. Mongolian Mining Corporation (MMC) pairs Ukhaa Khudag with the Hang "
            "Seng Index and Australian thermal coal. Steppe Gold pairs Altan Tsagaan Ovoo (ATO) with the S&amp;P/TSX "
            "Composite and gold."),
        _pdf_paragraph(
            "Complete-case samples run February 2018 to December 2025 for MMC and June 2018 to December 2025 for "
            "Steppe Gold. Inference uses Newey-West HAC with a Bartlett kernel, three monthly lags, and the "
            "Mission 2 scale correction n/(n - k), where k includes the intercept."),
        _pdf_paragraph("Table 1. Replicated controlled regressions", "section"),
        _pdf_baseline_table(firms),
        _pdf_paragraph("Both models match the numerical Mission 2 output to within 1e-10. The joint Wald test "
                       "sets both climate coefficients to zero and uses a chi-square reference with two degrees of freedom.", "note"),
        _pdf_paragraph("Table 2. One-month influence sensitivity", "section"),
        _pdf_influence_table(firms),
        _pdf_paragraph("Coefficients are in log-return points. Brackets show 95% asymptotic normal HAC intervals. "
                       "The omitted-month models measure lags by actual calendar distance.", "note"),
        _pdf_paragraph(
            f"MMC's temperature coefficient magnitude falls {abs(mt):.1f}%. Steppe Gold's rises {st:.1f}%. "
            f"Precipitation coefficients move by {mp:+.2f} (MMC) and {sp:+.2f} (Steppe Gold) log-return "
            "points per 10 mm. Steppe Gold's point estimate changes sign. Precipitation changes are in coefficient "
            "units because both baseline magnitudes are below 0.25."),
        PageBreak(),
        _pdf_paragraph("Diagnostic evidence", "title"),
        _pdf_paragraph(
            "Each primary sample has one observation per month, in chronological order with no internal gaps. "
            "All diagnostics use the full samples. HAC inference is retained throughout."),
        _pdf_paragraph("Table 3. Primary-model residual and influence diagnostics", "section"),
        _pdf_read_display(tables["Diagnostics"]),
        _pdf_paragraph("Breusch-Godfrey LM uses zero-padded presample residual lags and chi-square df 1 and 3. "
                       "Both variance tests use df 4. Cook's 4/N cutoff is a screening heuristic.", "note"),
        _pdf_paragraph("Table 4. Variance inflation factors", "section"),
        _pdf_read_display(tables["VIF"]),
        _pdf_paragraph("VIF auxiliary regressions include an intercept. The table reports VIFs for the four predictors.", "note"),
        _pdf_paragraph(
            "Neither Breusch-Godfrey test rejects at 5% for either firm. MMC shows evidence of heteroskedasticity "
            "under the classical Breusch-Pagan test, with weaker evidence under Koenker's test. Steppe Gold's "
            "variance tests do not reject at 5%. VIFs are close to one for both firms."),
        _pdf_paragraph(
            f"Three-standard-deviation flags include MMC's {mmc_unusual['value']:+.2f} log-return points in "
            f"{mmc_unusual['month']} and Steppe Gold's {10 * steppe_unusual['value']:+.2f} mm ATO precipitation "
            f"anomaly in {steppe_unusual['month']}. The primary models retain every flagged month."),
        _pdf_paragraph("Specification assessment", "section"),
        _pdf_paragraph(
            "The short samples leave substantial uncertainty. Read individual p-values alongside confidence "
            "intervals, joint tests and Mission 2's Holm adjustment across firms. Diagnostics have limited power."),
        _pdf_paragraph(
            "The same-month models estimate conditional associations. Australian thermal coal is an imperfect "
            "proxy for MMC's coking-coal exposure. The coal series switched from spot to futures in February 2022 "
            "and gold switched from the London afternoon fixing to an average spot series in June 2025. "
            "Steppe Gold's August 2024 Boroo acquisition expanded its operating footprint beyond ATO."),
        _pdf_paragraph(
            "Retain both full-sample models and report the one-month exclusions alongside them. "
            "The exclusions change some effect sizes, with no change in the joint-test conclusions. "
            "The evidence supports an exploratory manuscript about conditional associations. Identifying causal "
            "effects or establishing climate risk pricing would require a different research design."),
        _pdf_paragraph(
            "Inputs are the Mission 1 master workbook and Mission 2 model outputs. "
            "Inference reference: Newey, W. K. and West, K. D. (1987), <i>Econometrica</i> 55(3), 703-708. "
            '<link href="https://doi.org/10.2307/1913610" color="#1B565D">doi:10.2307/1913610</link>.', "note"),
    ]
    document = SimpleDocTemplate(str(path), pagesize=letter, topMargin=43, bottomMargin=50,
                                 leftMargin=50, rightMargin=50,
                                 title="Final robustness and regression diagnostics",
                                 subject="Local climate anomalies and mining stock returns")
    document.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=directory.parent / "Final" / "Master_Dataset_2018_2025.xlsx")
    parser.add_argument("--mission2-dir", type=Path, default=directory.parent / "Mission_2")
    parser.add_argument("--output-dir", type=Path, default=directory)
    args = parser.parse_args()
    result = analyze(args.workbook, args.mission2_dir / "Mission_2_Results.json",
                     args.mission2_dir / "Mission_2_Research_Note.pdf", args.mission2_dir / "run_mission_2.py")
    print("PASS | Both Mission 2 controlled models replicate to within the signed-off tolerance.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mission3_results.json"
    pdf_path = args.output_dir / "Mission_3_Research_Note.pdf"
    excel_path = args.output_dir / "Mission_3_Final_Tables.xlsx"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_pdf(result, pdf_path)
    write_workbook(result, excel_path)
    validate_workbook(excel_path, result)
    print(f"PASS | Exact results, PDF and validated Excel tables written in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
