"""Prepare monthly Yahoo Finance closes and adjacent-month returns."""

import argparse
from decimal import Decimal
import math
from pathlib import Path

import numpy as np
import pandas as pd


START = "2018-01"
END = "2025-12"
YAHOO_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLUMNS = YAHOO_COLUMNS[1:-1]
SERIES = (
    ("0975.HK.csv", "MMC_monthly_clean.csv", "monthly", "HKD/share"),
    ("STGO.TO.csv", "SteppeGold_monthly_clean.csv", "daily", "CAD/share"),
    ("^HSI.csv", "HSI_monthly_clean.csv", "monthly", "index points"),
    ("^GSPTSE.csv", "TSX_monthly_clean.csv", "monthly", "index points"),
)


def _load_yahoo(path):
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(raw.columns) != YAHOO_COLUMNS:
        raise ValueError(f"{path.name}: unexpected Yahoo Finance header")
    source_dates = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    canonical_dates = source_dates.dt.strftime("%Y-%m-%d").astype("string")
    malformed = source_dates.isna() | raw["Date"].ne(canonical_dates)
    if malformed.any():
        raise ValueError(f"{path.name}: malformed dates {raw.loc[malformed, 'Date'].tolist()}")
    if source_dates.duplicated().any():
        duplicates = raw.loc[source_dates.duplicated(keep=False), "Date"].tolist()
        raise ValueError(f"{path.name}: ambiguous duplicate dates {duplicates}")
    return raw.assign(
        date_parsed=source_dates,
        month=source_dates.dt.to_period("M"),
    )


def _price_diagnostics(raw, filename, unit, frequency):
    numeric = raw[YAHOO_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    prices = numeric[PRICE_COLUMNS]
    valid_prices = prices.gt(0) & np.isfinite(prices)
    volume = numeric["Volume"]
    valid_volume = volume.ge(0) & np.isfinite(volume) & volume.mod(1).eq(0)
    invalid = ~pd.concat([valid_prices, valid_volume.rename("Volume")], axis=1)
    missing = raw[YAHOO_COLUMNS].apply(lambda column: column.str.strip().eq(""))
    invalid = invalid & ~missing[invalid.columns]
    ohlc = prices[["Open", "High", "Low", "Close"]]
    bad_range = (
        ohlc["High"].lt(ohlc[["Low", "Open", "Close"]].max(axis=1))
        | ohlc["Low"].gt(ohlc[["High", "Open", "Close"]].min(axis=1))
    ) & valid_prices[ohlc.columns].all(axis=1)
    valid_pair = valid_prices[["Close", "Adj Close"]].all(axis=1)
    different = valid_pair & prices["Close"].ne(prices["Adj Close"])
    date_parsed = raw["date_parsed"]
    diagnostics = [
        f"{filename}: {len(raw):,} rows x {len(YAHOO_COLUMNS)} source columns; "
        f"{date_parsed.min():%Y-%m-%d} through {date_parsed.max():%Y-%m-%d}; {frequency}.",
        "  Source fields: " + ", ".join(YAHOO_COLUMNS) + ".",
        f"  Types/units: Date ISO calendar date; Open, High, Low, Close and Adj Close "
        f"numeric ({unit}); Volume nonnegative integer (shares for stocks, Yahoo series "
        "volume field for indices).",
        "  Blank fields: " + ", ".join(f"{name}={count}" for name, count in missing.sum().items()) + ".",
        "  Invalid nonblank numerics: "
        + ", ".join(f"{name}={count}" for name, count in invalid.sum().items()) + ".",
        f"  Duplicate dates: 0. Source dates strictly ascending: "
        f"{date_parsed.is_monotonic_increasing}. Violated OHLC bounds: {bad_range.sum()}.",
        f"  Close and Adj Close differ on {different.sum()} of {valid_pair.sum()} usable pairs; "
        "equality in this export alone cannot establish dividend or corporate-action treatment.",
    ]
    if invalid.any(axis=1).any() or bad_range.any():
        invalid_dates = raw.loc[invalid.any(axis=1) | bad_range, "Date"]
        diagnostics.append("  Invalid field or OHLC dates: " + ", ".join(invalid_dates) + ".")
    if missing.any(axis=1).any():
        diagnostics.append("  Source rows with blanks: " + ", ".join(raw.loc[missing.any(axis=1), "Date"]) + ".")
    zero_volume = raw.loc[volume.eq(0), "Date"]
    diagnostics.append(
        f"  Zero-volume source rows: {len(zero_volume)}."
        + (" Dates: " + ", ".join(zero_volume) + "." if len(zero_volume) else "")
    )
    return numeric["Adj Close"].where(valid_prices["Adj Close"]), diagnostics


def _endpoint_check(source, row, frequency):
    if frequency == "monthly":
        assert row.Date[:7] == str(row.month), f"Unaligned monthly source label: {row.Date}"
        return
    month_rows = source.loc[source["month"].eq(row.month)]
    usable_rows = month_rows.loc[month_rows["adjusted_close"].notna()]
    expected_date = (usable_rows if not usable_rows.empty else month_rows)["Date"].max()
    assert row.observation_date == expected_date, f"Wrong daily endpoint in {row.month}"


def _return_checks(cleaned, filename):
    checkable = cleaned.loc[cleaned["simple_return"].notna()]
    if checkable.empty:
        return ["  No adjacent positive-price observations are available for return verification."]
    check_months = checkable["month"].iloc[[0, len(checkable) // 2, -1]].unique()
    indexed = cleaned.set_index("month")
    result = []
    for month in check_months:
        previous = indexed.loc[month - 1, "Adj Close"]
        current = indexed.loc[month, "Adj Close"]
        ratio = Decimal(current) / Decimal(previous)
        expected_simple = float(ratio - 1)
        expected_log = math.log(float(ratio))
        actual = indexed.loc[month, ["simple_return", "log_return"]].to_numpy(dtype=float)
        np.testing.assert_allclose(
            actual, [expected_simple, expected_log], rtol=1e-12, atol=1e-14,
            err_msg=f"{filename}: return calculation mismatch for {month}",
        )
        result.append(
            f"  {month} return verified from {month - 1} {previous} and {current}: "
            f"simple {expected_simple:+.9f}, log {expected_log:+.9f}."
        )
    return result


def _daily_details(source, endpoints):
    dates = source["date_parsed"]
    gaps = dates.diff().dt.days
    gap_days = int(gaps.max()) if gaps.notna().any() else 0
    longest = pd.DataFrame({"before": dates.shift(), "after": dates, "days": gaps})
    longest = longest.loc[longest["days"].eq(gap_days)]
    weekend = source.loc[dates.dt.dayofweek.ge(5), "Date"]
    zero_endpoint = endpoints.loc[pd.to_numeric(endpoints["Volume"], errors="coerce").eq(0), "Date"]
    details = [
        f"  Daily source weekend rows: {len(weekend)}."
        + (" Dates: " + ", ".join(weekend) + "." if len(weekend) else ""),
        f"  Largest internal gap between supplied dates: {gap_days} calendar days."
        + (
            " " + ", ".join(f"{row.before:%Y-%m-%d} to {row.after:%Y-%m-%d}" for row in longest.itertuples()) + "."
            if not longest.empty else ""
        ),
        f"  First/last supplied dates: {dates.iloc[0]:%Y-%m-%d}, {dates.iloc[-1]:%Y-%m-%d}. "
        "The first and last months may have partial coverage. A raw date gap alone is "
        "insufficient to identify a trading interruption.",
        f"  Zero-volume monthly endpoints: {len(zero_endpoint)}."
        + (" Dates: " + ", ".join(zero_endpoint) + ". Retained as supplied daily closes." if len(zero_endpoint) else ""),
    ]
    daily_return = source["adjusted_close"].div(source["adjusted_close"].shift()) - 1
    largest = daily_return.abs().dropna().nlargest(3).index
    details.append(
        "  Largest absolute price movements between consecutive supplied daily rows: "
        + ", ".join(f"{source.at[i, 'Date']} {daily_return.at[i]:+.2%}" for i in largest) + "."
    )
    return details


def _documented_events(filename, selected):
    indexed = selected.set_index("month")
    if filename == "0975.HK.csv":
        periods = pd.period_range("2019-07", "2019-09", freq="M")
        nearby = indexed["adjusted_close"].reindex(periods)
        return [
            "  MMC consolidated ten old shares into one new share effective 2019-08-26. "
            "Supplied Adj Close around the event (HKD/share): "
            + ", ".join(f"{month} {price:.2f}" for month, price in nearby.items() if pd.notna(price))
            + ". These levels show no mechanical tenfold jump. Vendor adjustment mechanics "
            "remain unverified. The supplied prices receive no additional split adjustment.",
            "  HKEX circular: https://www1.hkexnews.hk/listedco/listconews/sehk/2019/0806/ltn20190806523.pdf",
            "  MMC annual report: https://www.hkexnews.hk/listedco/listconews/sehk/2022/0426/2022042600547.pdf",
        ]
    if filename == "STGO.TO.csv":
        may = indexed["observation_date"].get(pd.Period("2018-05", freq="M"))
        december = indexed["observation_date"].get(pd.Period("2025-12", freq="M"))
        return [
            "  Steppe Gold announced that TSX trading would commence on 2018-05-22. "
            f"The supplied export omits that session. Its May 2018 endpoint is {may or 'unavailable'}.",
            "  Steppe Gold listing announcement: https://steppegold.com/917/",
            f"  The December 2025 endpoint is {december or 'unavailable'}. TSX scheduled a session "
            "on 2025-12-31. The supplied file ends before that session, so the December return "
            "uses a partial-month endpoint. The reason is unverified.",
            "  TSX year-end session schedule: https://www.tsx.com/en/resource/3243/",
            "  On the zero-volume 2018-12-31 endpoint, the supplied Adj Close of CAD 0.85 "
            "matches the previous positive-volume close on 2018-12-28. The price is retained.",
            "  Steppe Gold reported closing the Boroo Gold acquisition on 2024-08-01. "
            "From that point the ATO climate series represents one site within the expanded "
            "company's mine portfolio. This is a documented exposure change.",
            "  Steppe Gold corporate update: https://steppegold.com/17105/",
        ]
    return []


def _clean_series(root, series):
    filename, output, frequency, unit = series
    raw = _load_yahoo(root / "Raw" / filename)
    adjusted, lines = _price_diagnostics(raw, filename, unit, frequency)
    source = raw.assign(adjusted_close=adjusted).sort_values("date_parsed").reset_index(drop=True)
    if frequency == "monthly":
        if source["month"].duplicated().any():
            raise ValueError(f"{filename}: multiple source rows for a calendar month")
        weekends = source["date_parsed"].dt.dayofweek.ge(5).sum()
        first_days = source["date_parsed"].dt.day.eq(1).sum()
        lines.append(
            f"  Yahoo monthly source labels: {first_days}/{len(source)} on day 1, including "
            f"{weekends} weekend labels. Date is a month label and the actual close session "
            "date is unavailable. Raw OHLCV and source month labels are retained."
        )
        endpoints = source.assign(observation_date="")
    else:
        last_available = source.loc[source["adjusted_close"].notna()].drop_duplicates("month", keep="last")
        last_raw = source.drop_duplicates("month", keep="last")
        months_with_no_usable_price = ~last_raw["month"].isin(last_available["month"])
        endpoints = pd.concat([last_available, last_raw.loc[months_with_no_usable_price]]).sort_values("month")
        endpoints = endpoints.assign(observation_date=endpoints["Date"])
        older_endpoint = endpoints.loc[endpoints.index.difference(last_raw.index), "Date"]
        lines.append(
            "  Kept the latest supplied daily row with positive finite Adj Close per calendar month. "
            "If a month has no usable Adj Close, the last original row remains with blank analytical "
            "close and returns. The actual source date is retained in observation_date."
        )
        if not older_endpoint.empty:
            lines.append("  Endpoints preceding an unusable final row: " + ", ".join(older_endpoint) + ".")
        lines.extend(_daily_details(source, endpoints))

    # Calculate on consecutive calendar periods so a lag can never skip a missing month.
    full_index = pd.period_range(source["month"].min(), source["month"].max(), freq="M")
    prices = endpoints.set_index("month")["adjusted_close"].reindex(full_index)
    ratio = prices.div(prices.shift())
    returns = pd.DataFrame({"simple_return": ratio - 1, "log_return": np.log(ratio)})
    endpoints = endpoints.join(returns, on="month", validate="one_to_one")
    target = pd.period_range(START, END, freq="M")
    selected = endpoints.loc[endpoints["month"].isin(target)].copy()
    for row in selected.itertuples():
        _endpoint_check(source, row, frequency)
    lines.extend(_return_checks(endpoints, filename))
    np.testing.assert_array_equal(returns["simple_return"].isna(), ratio.isna())
    np.testing.assert_array_equal(returns["log_return"].isna(), ratio.isna())
    assert selected["month"].is_unique and selected["month"].is_monotonic_increasing
    missing_months = target.difference(selected["month"])
    unavailable = target.difference(selected.loc[selected["simple_return"].notna(), "month"])
    lines.extend([
        f"  Target months with records: {len(selected)}/{len(target)}; with usable Adj Close: "
        f"{selected['adjusted_close'].count()}; with adjacent-month returns: {selected['simple_return'].count()}.",
        "  Missing source months: " + (", ".join(missing_months.astype(str)) or "none") + ".",
        "  Months without returns: " + (", ".join(unavailable.astype(str)) or "none") + ".",
    ])
    valid_returns = selected.set_index("month")["simple_return"].dropna()
    extremes = valid_returns.loc[valid_returns.abs().nlargest(3).index]
    lines.append(
        "  Largest absolute monthly simple returns retained: "
        + (", ".join(f"{month} {change:+.2%}" for month, change in extremes.items()) or "none") + "."
    )
    if filename in {"^HSI.csv", "^GSPTSE.csv"}:
        lines.append("  Index returns are price-index changes. Distributions may be excluded.")
    lines.extend(_documented_events(filename, selected))
    columns = ["month", *YAHOO_COLUMNS, "observation_date", "adjusted_close", "simple_return", "log_return"]
    selected[columns].to_csv(root / "Cleaned" / output, index=False)
    return "\n".join(lines)


def run(root):
    """Write the four cleaned series and return verification notes for the QA report."""
    root = Path(root)
    (root / "Cleaned").mkdir(parents=True, exist_ok=True)
    details = [_clean_series(root, series) for series in SERIES]
    return "\n\n".join([
        "FINANCIAL DATA CLEANING AND VERIFICATION",
        "Yahoo Finance provided the four source exports. Prices use valid positive Adj Close "
        "in the source currency or index points. Monthly simple returns are P(t)/P(t-1) - 1 "
        "and log returns are ln(P(t)/P(t-1)); both are decimals. Returns require two prices "
        "in adjacent calendar months. Available pre-sample prices may supply a prior month; "
        "a missing price leaves the current and following return unavailable. Cleaned files "
        "retain the supplied source fields on selected rows, with missing source months "
        "reported below. Extreme moves and zero volumes are retained. Price and date evidence "
        "alone does not establish a structural break or confirm a trading halt.",
        *details,
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(run(parser.parse_args().root))
