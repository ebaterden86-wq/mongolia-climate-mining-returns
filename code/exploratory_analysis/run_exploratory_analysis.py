"""Reproduce exploratory analysis from the data preparation master workbook.

Usage: python run_exploratory_analysis.py [--input PATH] [--output-dir PATH]
Packages: NumPy, pandas, openpyxl, ReportLab. No network access needed.
"""

import argparse
import hashlib
import json
from math import erfc, exp, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FIRMS = {
    "MMC": ("mmc", "uhg", "hsi", "coal_australian"),
    "Steppe Gold": ("steppe_gold", "ato", "tsx", "gold"),
}
FIELDS = ("return", "temperature", "precipitation", "market", "commodity")
CLIMATE = ("temperature", "precipitation")
CONTROLLED = (*CLIMATE, "market", "commodity")
BANDWIDTH = 3
Z95 = 1.959963984540054
MONTHS = pd.period_range("2018-01", "2025-12", freq="M")


def read_master(path):
    frame = pd.read_excel(path, sheet_name="Master", dtype={"month": "string"},
                          engine="openpyxl")
    if frame["month"].tolist() != MONTHS.astype(str).tolist():
        raise ValueError("Expected exactly 96 unique, ordered study months.")
    return frame.set_index(MONTHS).drop(columns="month")


def firm_sample(frame, firm):
    stock, site, market, commodity = FIRMS[firm]
    series = {
        "return": frame[f"{stock}_log_return"] * 100,
        "temperature": frame[f"{site}_temperature_anomaly_c"],
        "precipitation": frame[f"{site}_precipitation_anomaly_mm"] / 10,
        "market": frame[f"{market}_log_return"] * 100,
        "commodity": frame[f"{commodity}_log_return"] * 100,
    }
    data = pd.DataFrame(series)
    if np.isinf(data.to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite analytical value for {firm}")
    return data.dropna()


def describe(data):
    statistics = data[list(FIELDS)].describe(percentiles=[0.5]).T
    statistics.rename(columns={"count": "n", "std": "sd", "50%": "median"}, inplace=True)
    return statistics.to_dict(orient="index")


def correlations(data):
    return {
        driver: {
            "n": len(data),
            "pearson": float(data["return"].corr(data[driver])),
            "spearman": float(data["return"].rank().corr(data[driver].rank())),
        }
        for driver in CLIMATE
    }


def hac_model(data, regressors, bandwidth):
    """OLS with Newey-West Bartlett HAC, using actual calendar-month distances."""
    n = len(data)
    x = np.column_stack((np.ones(n), data[list(regressors)].to_numpy(dtype=float)))
    y = data["return"].to_numpy(dtype=float)
    k = x.shape[1]
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != k or n <= k:
        raise ValueError("Regression requires a full-rank design and residual degrees of freedom")
    u = y - x @ beta
    xx_inv = np.linalg.inv(x.T @ x)
    # Omitted months contribute no scores. The HAC kernel still uses calendar distance.
    positions = data.index.asi8 - data.index.asi8[0]
    score = np.zeros((positions[-1] + 1, k))
    score[positions] = x * u[:, None]
    meat = score.T @ score
    for lag in range(1, bandwidth + 1):
        product = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (bandwidth + 1)) * (product + product.T)
    cov = (n / (n - k)) * xx_inv @ meat @ xx_inv
    se = np.sqrt(np.diag(cov))
    if not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError("Invalid HAC standard error")
    terms = {}
    for term, coef, error in zip(("intercept", *regressors), beta, se):
        terms[term] = {
            "estimate": float(coef), "hac_se": float(error),
            "p_normal": erfc(abs(float(coef / error)) / sqrt(2)),
            "ci95_low": float(coef - Z95 * error),
            "ci95_high": float(coef + Z95 * error),
        }
    climate_cov = cov[1:3, 1:3]
    climate_beta = beta[1:3]
    wald = float(climate_beta @ np.linalg.solve(climate_cov, climate_beta))
    sse = float(u @ u)
    r2 = 1 - sse / float((y - y.mean()) @ (y - y.mean()))
    leverage = np.einsum("ij,jk,ik->i", x, xx_inv, x)
    cooks = (u ** 2 / (k * sse / (n - k))) * leverage / (1 - leverage) ** 2
    max_cook = int(np.argmax(cooks))
    return {
        "terms": terms, "n": n, "first": str(data.index[0]), "last": str(data.index[-1]),
        "included_months": data.index.astype(str).tolist(), "hac_lags": bandwidth,
        "r2": r2, "adjusted_r2": 1 - (1 - r2) * (n - 1) / (n - k),
        "climate_joint_wald_chi2_df2": wald, "climate_joint_p": exp(-wald / 2),
        "max_cooks_distance": float(cooks[max_cook]), "max_cooks_month": str(data.index[max_cook]),
    }


def collect(path):
    frame = read_master(path)
    output = {
        "input_file": path.name,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "study_period": ["2018-01", "2025-12"],
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
        "units": {"return": "100 × log return", "temperature": "°C anomaly",
                  "precipitation": "10 mm anomaly", "market": "100 × log return",
                  "commodity": "100 × log return"},
        "inference": "Newey-West Bartlett HAC with n/(n-k) scale correction. "
                     "Coefficient intervals and p-values use asymptotic normal reference. "
                     "Joint climate Wald tests use chi-square(2).",
        "firms": {},
    }
    for firm in FIRMS:
        data = firm_sample(frame, firm)
        baseline = hac_model(data, CLIMATE, BANDWIDTH)
        adjusted = hac_model(data, CONTROLLED, BANDWIDTH)
        if baseline["included_months"] != adjusted["included_months"]:
            raise AssertionError("Nested model samples must match")
        excluded = ["2022-02"] if firm == "MMC" else ["2025-06", "2025-12"]
        trimmed = data.loc[~data.index.astype(str).isin(excluded)]
        sensitivity = hac_model(trimmed, CONTROLLED, BANDWIDTH)
        output["firms"][firm] = {
            "descriptive": describe(data),
            "correlations": correlations(data),
            "climate_only": baseline,
            "controlled": adjusted,
            "sensitivity_source_months": {
                "excluded": excluded, "model": sensitivity,
            },
            "sensitivity_bandwidth_6": hac_model(data, CONTROLLED, 6),
        }
    # Holm adjustment for the two firm-level primary joint climate tests.
    ordered = sorted(output["firms"].values(), key=lambda f: f["controlled"]["climate_joint_p"])
    previous = 0.0
    for index, firm in enumerate(ordered):
        model = firm["controlled"]
        previous = min(1.0, max(previous, (len(ordered) - index) * model["climate_joint_p"]))
        model["climate_joint_p_holm"] = previous
    return output


INK = colors.HexColor("#152938")
MUTED = colors.HexColor("#4C626E")
ACCENT = colors.HexColor("#1B565D")
RULE = colors.HexColor("#C7D4D8")
PALE = colors.HexColor("#EFF5F5")
STYLES = {
    "title": ParagraphStyle("Title", fontName="Times-Bold", fontSize=27, leading=29,
                            textColor=INK, spaceAfter=9),
    "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10, leading=13,
                               textColor=MUTED, spaceAfter=13),
    "heading": ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=11, leading=14,
                              textColor=ACCENT, spaceBefore=11, spaceAfter=5, keepWithNext=True),
    "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.6, leading=13,
                           textColor=INK, spaceAfter=6),
    "note": ParagraphStyle("Note", fontName="Helvetica", fontSize=8, leading=10.5,
                           textColor=MUTED, spaceBefore=5, spaceAfter=4),
    "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.5, leading=10.5, textColor=INK),
    "num": ParagraphStyle("Numeric", fontName="Helvetica", fontSize=8.5, leading=10.5,
                          alignment=TA_RIGHT, textColor=INK),
    "group": ParagraphStyle("Group", fontName="Helvetica", fontSize=8.5, leading=10.5,
                            alignment=TA_CENTER, textColor=INK),
}


def p(content, style="body"):
    return Paragraph(content, STYLES[style])


def result_table(rows, widths, panel_rows=(), spans=()):
    rendered = []
    centered = {start for start, _ in spans}
    for row_number, row in enumerate(rows):
        cells = []
        for index, value in enumerate(row):
            if row_number == 0 or row_number in panel_rows:
                value = f"<b>{value}</b>"
            style = "group" if (index, row_number) in centered else ("cell" if index == 0 else "num")
            cells.append(p(value, style))
        rendered.append(cells)
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (1, 0), (-1, -1), 4),
        ("RIGHTPADDING", (1, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.75, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
    ]
    for row in panel_rows:
        commands.append(("LINEABOVE", (0, row), (-1, row), 0.5, RULE))
    for start, end in spans:
        commands.append(("SPAN", start, end))
    table.setStyle(TableStyle(commands))
    return table


def summary_table(results):
    labels = {
        "return": "Stock log return (×100)", "temperature": "Temperature anomaly (°C)",
        "precipitation": "Precipitation anomaly (10 mm)", "market": "Market log return (×100)",
        "commodity": "Commodity log return (×100)",
    }
    descriptive = [data["descriptive"] for data in results["firms"].values()]
    rows = [["", f"MMC (N = {int(descriptive[0]['return']['n'])})", "", "", "", "",
             f"Steppe Gold (N = {int(descriptive[1]['return']['n'])})", "", "", "", ""],
            ["Variable", *(["Mean", "SD", "Median", "Min", "Max"] * 2)]]
    for field, label in labels.items():
        values = [f"{data[field][stat]:.2f}" for data in descriptive
                  for stat in ("mean", "sd", "median", "min", "max")]
        rows.append([label, *values])
    return result_table(rows, [124, *([36, 36, 41, 39, 34] * 2)], (1,),
                        (((1, 0), (5, 0)), ((6, 0), (10, 0))))


def correlation_table(results):
    rows = [["Stock return and site anomaly", "N", "Pearson r", "Spearman rho"]]
    for firm, data in results["firms"].items():
        for term, values in data["correlations"].items():
            rows.append([f"{firm} / {term}", str(values["n"]),
                         f"{values['pearson']:.3f}", f"{values['spearman']:.3f}"])
    return result_table(rows, [237, 49, 102, 108])


def regression_table(results):
    specs = ("climate_only", "controlled")
    models = [data[spec] for data in results["firms"].values() for spec in specs]
    rows = [["Regressor", "MMC<br/>Climate", "MMC<br/>Controlled",
             "Steppe Gold<br/>Climate", "Steppe Gold<br/>Controlled"]]
    for label, term in (("Temperature anomaly<br/>per 1°C", "temperature"),
                        ("Precipitation anomaly<br/>per 10 mm", "precipitation"),
                        ("Market return", "market"), ("Commodity return", "commodity"),
                        ("Intercept", "intercept")):
        values = []
        for model in models:
            if term in model["terms"]:
                t = model["terms"][term]
                values.append(f"<b>{t['estimate']:.2f}</b><br/>[{t['ci95_low']:.2f}, {t['ci95_high']:.2f}]")
            else:
                values.append("-")
        rows.append([label, *values])
    rows.extend([
        ["Months", *[str(m["n"]) for m in models]],
        ["R²", *[f"{m['r2']:.3f}" for m in models]],
        ["Joint climate p", *[f"{m['climate_joint_p']:.3f}" for m in models]],
        ["Holm-adjusted joint p", *[f"{m['climate_joint_p_holm']:.3f}" if "climate_joint_p_holm" in m else "-"
                                   for m in models]],
    ])
    return result_table(rows, [140, 89, 89, 89, 89], (6,))


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(58, 40, 554, 40)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(58, 27, "Mining stock returns and site-level climate anomalies")
    canvas.drawRightString(554, 27, str(document.page))
    canvas.restoreState()


def create_report(path, results):
    firms = results["firms"]
    mmc = firms["MMC"]["controlled"]
    steppe = firms["Steppe Gold"]["controlled"]
    mt = mmc["terms"]["temperature"]
    st = steppe["terms"]["temperature"]
    s_mmc = firms["MMC"]["sensitivity_source_months"]["model"]
    s_steppe = firms["Steppe Gold"]["sensitivity_source_months"]["model"]
    lag6_mmc = firms["MMC"]["sensitivity_bandwidth_6"]
    lag6_steppe = firms["Steppe Gold"]["sensitivity_bandwidth_6"]
    story = [
        p("Monthly climate anomalies<br/>and mining stock returns", "title"),
        p("Mongolian Mining Corporation and Steppe Gold | January 2018 to December 2025", "subtitle"),
        p(f"The largest climate-return correlation is between warmer-than-usual UHG months and lower "
          f"MMC returns (Pearson r = {firms['MMC']['correlations']['temperature']['pearson']:.2f}). "
          f"The controlled temperature coefficient is {mt['estimate']:.2f} for MMC and "
          f"{st['estimate']:+.2f} for Steppe Gold, in monthly log-return points per 1°C. "
          "Both 95% confidence intervals span zero. Neither joint climate test rejects at 5%."),
        p("Data and estimation samples", "heading"),
        p("The 96-month data preparation master supplies local-currency stock returns, index price returns, "
          "nominal USD commodity returns, and Open-Meteo grid-based reanalysis anomalies. "
          "MMC is paired with Ukhaa Khudag (UHG), the Hang Seng Index, and Australian thermal coal. "
          "Steppe Gold is paired with Altan Tsagaan Ovoo (ATO), the S&amp;P/TSX Composite, and gold. "
          "Temperature and precipitation departures use a fixed calendar-month baseline from 1991 to 2020."),
        p("A common complete-case sample within each firm is used across its tables and models. "
          "MMC has 95 months (Feb 2018 to Dec 2025) and Steppe Gold has 91 (Jun 2018 to Dec 2025). "
          "The first available price for each firm lacks a prior-month price. "
          "Stock, market and commodity log returns are multiplied by 100."),
        p("Table 1. Descriptive statistics", "heading"),
        summary_table(results),
        p("SD is the sample standard deviation.", "note"),
        p("Table 2. Same-month climate-return correlations", "heading"),
        correlation_table(results),
        p("Pearson and Spearman coefficients are descriptive. Spearman uses average ranks for ties.", "note"),
        PageBreak(),
        p("Initial regressions", "title"),
        p("For each firm, monthly stock log returns are regressed on both contemporaneous climate anomalies. "
          "The controlled model adds the paired market and commodity log returns. Each specification has "
          "an intercept. Newey-West Bartlett standard errors use three monthly lags and the n/(n - k) "
          "scale correction, with asymptotic normal intervals and chi-square joint tests with two degrees of freedom. "
          "Holm adjustment covers the two controlled joint climate tests."),
        p("Table 3. OLS coefficients and 95% HAC confidence intervals", "heading"),
        regression_table(results),
        p("Outcome is 100 × the monthly stock log return. The joint test sets both climate coefficients to zero. "
          "Adjusted R² for the controlled specifications is "
          f"{mmc['adjusted_r2']:.3f} for MMC and {steppe['adjusted_r2']:.3f} for Steppe Gold.", "note"),
        p("Sensitivity and interpretation", "heading"),
        p(f"Using six HAC lags, the controlled joint p-values are "
          f"{lag6_mmc['climate_joint_p']:.3f} (MMC) and {lag6_steppe['climate_joint_p']:.3f} (Steppe Gold). "
          "Omitting the coal benchmark transition (Feb 2022) gives an MMC temperature estimate of "
          f"{s_mmc['terms']['temperature']['estimate']:.2f}, joint p = {s_mmc['climate_joint_p']:.3f} "
          f"(N = {s_mmc['n']}). Omitting the gold transition (Jun 2025) and Steppe Gold's early "
          "month-end close (Dec 2025) gives a temperature estimate of "
          f"{s_steppe['terms']['temperature']['estimate']:+.2f}, joint p = {s_steppe['climate_joint_p']:.3f} "
          f"(N = {s_steppe['n']}). HAC weights use actual month distances."),
        p("The controlled MMC temperature association is imprecise and weakens in the source-month check. "
          f"The largest Cook's distance occurs in {mmc['max_cooks_month']} for MMC and "
          f"{steppe['max_cooks_month']} for Steppe Gold. "
          "Primary models retain these observations. With two firms, short time series and "
          "contemporaneous controls, the estimates describe conditional associations. "
          "Australian thermal coal is an imperfect proxy for coking coal, "
          "and ATO represents one site in Steppe Gold's expanded portfolio after the 2024 Boroo acquisition."),
        p("Source and reproducibility", "heading"),
        p("data preparation master workbook, Master sheet. The accompanying Python script and JSON provide "
          "input fingerprint, exact estimates, standard errors, p-values, diagnostics, and month membership. "
          "Inference reference: Newey, W. K. and West, K. D. (1987). "
          "<i>Econometrica</i> 55(3), 703-708. "
          '<link href="https://doi.org/10.2307/1913610" color="#1B565D">doi:10.2307/1913610</link>.', "note"),
    ]
    document = SimpleDocTemplate(str(path), pagesize=letter, topMargin=48, bottomMargin=53,
                                 leftMargin=52, rightMargin=52,
                                 title="Monthly climate anomalies and mining stock returns",
                                 subject="exploratory analysis exploratory results")
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=directory.parents[1] / "data" / "Master_Dataset_2018_2025.xlsx")
    parser.add_argument("--output-dir", type=Path, default=directory)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    results = collect(arguments.input)
    path = arguments.output_dir / "exploratory_analysis_results.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report = arguments.output_dir / "exploratory_analysis_research_note.pdf"
    create_report(report, results)
    print(f"Validated workbook and wrote {path.name} and {report.name} in {arguments.output_dir}")


if __name__ == "__main__":
    main()
