"""Format manuscript tables directly from the archived Mission 2 and 3 results."""


FIRMS = ("MMC", "Steppe Gold")
TERMS = (
    ("temperature", "Temperature anomaly, per 1 °C"),
    ("precipitation", "Precipitation anomaly, per 10 mm"),
    ("market", "Paired market log return × 100"),
    ("commodity", "Paired commodity log return × 100"),
    ("intercept", "Intercept"),
)


def _fmt(value, digits):
    """Use a typographic minus without changing numeric rounding."""
    return f"{value:.{digits}f}".replace("-", "−")


def _coefficient(term):
    return (
        f"{_fmt(term['estimate'], 2)} "
        f"[{_fmt(term['ci95_low'], 2)}, {_fmt(term['ci95_high'], 2)}]"
    )


def _descriptive(m2):
    rows = []
    statistics = (
        ("return", "Stock log return × 100"),
        ("temperature", "Temperature anomaly (°C)"),
        ("precipitation", "Precipitation anomaly (10 mm)"),
    )
    for key, label in statistics:
        values = [m2["firms"][name]["descriptive"][key] for name in FIRMS]
        rows.append([label, *[
            f"{_fmt(stat['mean'], 2)} ({_fmt(stat['sd'], 2)})" for stat in values
        ]])
    for term, label in TERMS[:2]:
        values = [m2["firms"][name]["correlations"][term] for name in FIRMS]
        variable = label.split(",")[0]
        rows.append([f"Return vs. {variable.lower()}", *[
            f"{_fmt(stat['pearson'], 3)} / {_fmt(stat['spearman'], 3)}" for stat in values
        ]])
    return {
        "title": "Table 1. Descriptive statistics and climate–return correlations",
        "headers": [
            "Statistic", *[f"{name} (N = {m2['firms'][name]['controlled']['n']})" for name in FIRMS]
        ],
        "rows": rows,
        "note": (
            "The first three rows report means (sample standard deviations). The last two report "
            "Pearson r / Spearman rank correlations on each firm’s estimation sample."
        ),
        "widths": [0.46, 0.27, 0.27],
    }


def _primary(m2, m3):
    models = [m3["firms"][name]["baseline"] for name in FIRMS]
    rows = [[label, *[_coefficient(model["terms"][term]) for model in models]]
            for term, label in TERMS]
    rows.extend([
        ["Observations", *[str(model["n"]) for model in models]],
        ["R²", *[_fmt(model["r2"], 3) for model in models]],
        ["Joint climate Wald p", *[_fmt(model["climate_joint_p"], 3) for model in models]],
        ["Holm-adjusted joint p", *[
            _fmt(m2["firms"][name]["controlled"]["climate_joint_p_holm"], 3) for name in FIRMS
        ]],
    ])
    return {
        "title": "Table 2. Primary controlled regressions",
        "headers": ["Regressor or statistic", *FIRMS],
        "rows": rows,
        "note": (
            "Dependent variable is monthly stock log return × 100. Brackets show 95% normal-reference "
            "HAC confidence intervals using three Bartlett lags and the N/(N−k) correction. "
            "The joint Wald test sets both climate slopes to zero. Holm adjustment covers the two firm-level tests."
        ),
        "widths": [0.46, 0.27, 0.27],
    }


def _influence(m3):
    rows = []
    for name in FIRMS:
        firm = m3["firms"][name]
        for label, model in (("full sample", firm["baseline"]),
                             (f"omit {firm['excluded_month']}", firm["exclude_max_cook"])):
            rows.append([
                f"{name}, {label}", str(model["n"]),
                *[_coefficient(model["terms"][term]) for term, _ in TERMS[:2]],
                _fmt(model["climate_joint_p"], 3),
            ])
    return {
        "title": "Table 3. Sensitivity to the most influential month",
        "headers": ["Firm and sample", "N", "Temperature per 1 °C", "Precipitation per 10 mm", "Joint p"],
        "rows": rows,
        "note": (
            "Exclusions remove only the month with the largest Cook’s distance for each firm. All models include the "
            "paired market and commodity controls. Brackets show 95% normal-reference HAC confidence intervals. "
            "HAC uses actual calendar-month distances across the omission."
        ),
        "widths": [0.27, 0.06, 0.27, 0.27, 0.13],
    }


def _diagnostics(m3):
    specs = [
        ("Durbin–Watson statistic", lambda d, b: _fmt(d["durbin_watson"], 3)),
        ("Breusch–Godfrey LM (1 lag)", lambda d, b: _lm(d["breusch_godfrey"]["1"])),
        ("Breusch–Godfrey LM (3 lags)", lambda d, b: _lm(d["breusch_godfrey"]["3"])),
        ("Breusch–Pagan LM", lambda d, b: _lm(d["breusch_pagan_classical"])),
        ("Koenker LM", lambda d, b: _lm(d["breusch_pagan_koenker"])),
        ("Maximum Cook’s distance", lambda d, b: _fmt(b["max_cooks_distance"], 3)),
        ("Cook’s 4/N threshold", lambda d, b: _fmt(d["cook_4_over_n"], 3)),
        ("Months above 4/N", lambda d, b: str(d["cook_count_above_4_over_n"])),
        ("VIF range", lambda d, b: f"{_fmt(min(d['vif'].values()), 3)}–{_fmt(max(d['vif'].values()), 3)}"),
    ]
    rows = [[label, *[
        value(m3["firms"][name]["diagnostics"], m3["firms"][name]["baseline"])
        for name in FIRMS
    ]] for label, value in specs]
    return {
        "title": "Table 4. Selected primary-model diagnostics",
        "headers": [
            "Diagnostic", *[f"{name} (N = {m3['firms'][name]['baseline']['n']})" for name in FIRMS]
        ],
        "rows": rows,
        "note": (
            "LM entries report statistic (p-value). Breusch–Godfrey uses one and three degrees of freedom, "
            "respectively. The two variance tests each use four. Cook’s 4/N threshold is a screening heuristic. "
            "VIF ranges exclude the intercept."
        ),
        "widths": [0.46, 0.27, 0.27],
    }


def _lm(record):
    return f"{_fmt(record['lm'], 3)} ({_fmt(record['p_value'], 3)})"


def make_tables(m2, m3):
    """Return ordered table specifications with relative column widths."""
    return {
        "descriptive": _descriptive(m2),
        "primary": _primary(m2, m3),
        "influence": _influence(m3),
        "diagnostics": _diagnostics(m3),
    }
