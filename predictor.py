from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


VALID_PERIODS = {"6mo", "1y", "2y", "5y"}


@dataclass
class PredictionArtifacts:
    model: LinearRegression
    feature_columns: list[str]
    metrics: dict[str, float]
    confidence_band: tuple[float, float]


def build_stock_response(ticker: str, period: str, target_date: Optional[str] = None) -> dict:
    clean_ticker = ticker.strip().upper()
    clean_period = period.strip()

    if not clean_ticker:
        raise ValueError("Please enter a stock ticker, for example RELIANCE.NS or TCS.NS.")
    if clean_period not in VALID_PERIODS:
        raise ValueError(f"Period must be one of: {', '.join(sorted(VALID_PERIODS))}.")

    history, data_source = _download_history(clean_ticker, clean_period)

    history = history.reset_index()
    if "Date" not in history.columns:
        raise ValueError("The market data format was unexpected. Please try a different ticker.")

    prepared = _engineer_features(history)
    if len(prepared) < 25:
        raise ValueError("Not enough historical data to train a simple prediction model.")

    artifacts = _train_model(prepared)
    last_market_date = prepared.iloc[-1]["Date"]
    target_timestamp = _resolve_target_date(last_market_date, target_date)
    forecast = _forecast_to_date(prepared, artifacts, target_timestamp)

    last_row = prepared.iloc[-1]
    last_close = float(last_row["Close"])
    predicted_close = float(forecast["predictedClose"])
    change_pct = ((predicted_close - last_close) / last_close) * 100 if last_close else 0.0
    lower, upper = artifacts.confidence_band

    chart_points = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "ma7": round(float(row["MA_7"]), 2),
            "ma21": round(float(row["MA_21"]), 2),
        }
        for _, row in prepared.tail(120).iterrows()
    ]

    future_point = {
        "date": forecast["date"],
        "predictedClose": round(predicted_close, 2),
        "lowerBound": round(predicted_close + lower, 2),
        "upperBound": round(predicted_close + upper, 2),
        "sessionsAhead": forecast["sessionsAhead"],
    }

    return {
        "ticker": clean_ticker,
        "companyName": _extract_company_name(clean_ticker),
        "period": clean_period,
        "dataSource": data_source,
        "summary": {
            "lastClose": round(last_close, 2),
            "predictedNextClose": round(predicted_close, 2),
            "predictedChangePct": round(float(change_pct), 2),
            "ma7": round(float(last_row["MA_7"]), 2),
            "ma21": round(float(last_row["MA_21"]), 2),
            "rsi14": round(float(last_row["RSI_14"]), 2),
        },
        "metrics": artifacts.metrics,
        "chartData": chart_points,
        "predictionPoint": future_point,
        "forecastMeta": {
            "lastMarketDate": last_market_date.strftime("%Y-%m-%d"),
            "requestedTargetDate": target_timestamp.strftime("%Y-%m-%d"),
            "targetTradingDate": future_point["date"],
            "sessionsAhead": future_point["sessionsAhead"],
        },
        "insights": _build_learning_insights(change_pct, artifacts.metrics, last_row, future_point["sessionsAhead"]),
    }


def _download_history(ticker: str, period: str) -> tuple[pd.DataFrame, str]:
    try:
        history = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
    except Exception:
        history = pd.DataFrame()

    history = _normalize_history(history)
    if not history.empty:
        return history, "Yahoo Finance"

    try:
        history = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    except Exception:
        history = pd.DataFrame()

    history = _normalize_history(history)
    if not history.empty:
        return history, "Yahoo Finance"

    return _generate_demo_history(ticker, period), "Demo fallback"


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()

    normalized = history.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(set(normalized.columns)):
        return pd.DataFrame()

    return normalized


def _engineer_features(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df["Return_1D"] = df["Close"].pct_change()
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["Volatility_7"] = df["Return_1D"].rolling(window=7).std()
    df["Momentum_7"] = df["Close"] - df["Close"].shift(7)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)

    df["Target"] = df["Close"].shift(-1)
    return df.dropna().reset_index(drop=True)


def _train_model(df: pd.DataFrame) -> PredictionArtifacts:
    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return_1D",
        "MA_7",
        "MA_21",
        "Volatility_7",
        "Momentum_7",
        "RSI_14",
    ]

    x = df[feature_columns]
    y = df["Target"]

    split_index = max(int(len(df) * 0.8), 20)
    split_index = min(split_index, len(df) - 5)
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    residuals = y_test - predictions
    residual_std = float(np.std(residuals)) if len(residuals) else 0.0

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "trainSize": int(len(x_train)),
        "testSize": int(len(x_test)),
    }

    confidence_band = (-1.96 * residual_std, 1.96 * residual_std)

    return PredictionArtifacts(
        model=model,
        feature_columns=feature_columns,
        metrics=metrics,
        confidence_band=confidence_band,
    )


def _build_learning_insights(
    change_pct: float, metrics: dict[str, float], latest_row: pd.Series, sessions_ahead: int
) -> list[str]:
    direction = "upward" if change_pct >= 0 else "downward"
    strength = "strong" if abs(change_pct) >= 2 else "mild"
    reliability = "more reliable" if metrics["r2"] >= 0.6 else "less reliable"
    trend_note = (
        "Short-term trend is above the 21-day average."
        if latest_row["MA_7"] >= latest_row["MA_21"]
        else "Short-term trend is below the 21-day average."
    )

    return [
        f"The model suggests a {strength} {direction} move over the next {sessions_ahead} trading session(s).",
        f"Model fit looks {reliability} for this ticker and time period with R2 = {metrics['r2']}.",
        trend_note,
        "This forecast is a basic regression model for learning, not financial advice.",
    ]


def _extract_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def _generate_demo_history(ticker: str, period: str) -> pd.DataFrame:
    period_lengths = {
        "6mo": 132,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
    }
    rows = period_lengths.get(period, 252)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)

    seed = sum(ord(char) for char in ticker)
    rng = np.random.default_rng(seed)
    base_price = 150 + (seed % 2400)
    drift = rng.normal(0.0008, 0.0003, size=rows)
    noise = rng.normal(0, 0.018, size=rows)
    returns = drift + noise

    closes = [base_price]
    for daily_return in returns[1:]:
        closes.append(max(20, closes[-1] * (1 + daily_return)))

    closes = np.array(closes)
    opens = closes * (1 + rng.normal(0, 0.004, size=rows))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0.002, 0.018, size=rows))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0.002, 0.018, size=rows))
    volumes = rng.integers(300_000, 4_500_000, size=rows)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )


def _resolve_target_date(last_market_date: pd.Timestamp, target_date: Optional[str]) -> pd.Timestamp:
    next_session = _next_trading_day(last_market_date)
    if not target_date:
        return next_session

    try:
        parsed = pd.Timestamp(target_date).normalize()
    except Exception as exc:
        raise ValueError("Please choose a valid target date.") from exc

    if parsed <= last_market_date.normalize():
        return next_session

    target_trading = parsed
    while target_trading.weekday() >= 5:
        target_trading += pd.Timedelta(days=1)

    sessions_ahead = len(pd.bdate_range(next_session, target_trading))
    if sessions_ahead > 30:
        raise ValueError("Please choose a target date within the next 30 trading sessions.")

    return target_trading


def _forecast_to_date(df: pd.DataFrame, artifacts: PredictionArtifacts, target_date: pd.Timestamp) -> dict:
    simulated = df.copy()
    last_market_date = simulated.iloc[-1]["Date"]
    future_dates = pd.bdate_range(_next_trading_day(last_market_date), target_date)

    if len(future_dates) == 0:
        future_dates = pd.DatetimeIndex([_next_trading_day(last_market_date)])

    predicted_close = float(simulated.iloc[-1]["Close"])

    for future_date in future_dates:
        latest = simulated.iloc[-1]
        features = latest[artifacts.feature_columns].to_frame().T
        predicted_close = float(artifacts.model.predict(features)[0])
        next_row = _build_future_row(simulated, pd.Timestamp(future_date), predicted_close)
        simulated = pd.concat([simulated, pd.DataFrame([next_row])], ignore_index=True)

    return {
        "date": pd.Timestamp(future_dates[-1]).strftime("%Y-%m-%d"),
        "predictedClose": predicted_close,
        "sessionsAhead": int(len(future_dates)),
    }


def _build_future_row(df: pd.DataFrame, future_date: pd.Timestamp, predicted_close: float) -> dict:
    closes = pd.concat([df["Close"], pd.Series([predicted_close])], ignore_index=True)
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan)
    delta = closes.diff()
    gains = delta.clip(lower=0).rolling(window=14).mean()
    losses = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50)

    previous_close = float(df.iloc[-1]["Close"])
    last_volume = float(df.iloc[-1]["Volume"])
    volatility_window = returns.tail(7).dropna()

    return {
        "Date": future_date,
        "Open": previous_close,
        "High": max(previous_close, predicted_close),
        "Low": min(previous_close, predicted_close),
        "Close": predicted_close,
        "Volume": last_volume,
        "Return_1D": float(returns.iloc[-1]) if not np.isnan(returns.iloc[-1]) else 0.0,
        "MA_7": float(closes.tail(7).mean()),
        "MA_21": float(closes.tail(21).mean()),
        "Volatility_7": float(volatility_window.std()) if len(volatility_window) else 0.0,
        "Momentum_7": float(predicted_close - closes.iloc[-8]) if len(closes) >= 8 else 0.0,
        "RSI_14": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
        "Target": np.nan,
    }


def _next_trading_day(last_date: pd.Timestamp) -> pd.Timestamp:
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)
    return next_date.normalize()
