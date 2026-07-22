"""Reusable, side-effect-free utilities for exploratory market-data analysis.

All tabular inputs follow the pandas convention: rows are observations ordered
through time and columns are instruments. Public functions never mutate inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DataLike: TypeAlias = pd.Series | pd.DataFrame | np.ndarray
ReturnMethod: TypeAlias = Literal["simple", "log", "difference"]


def _as_frame(data: DataLike, *, name: str = "value") -> pd.DataFrame:
    """Return a numeric, finite, non-empty copy of data as a DataFrame."""
    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or name)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        array = np.asarray(data)
        if array.ndim == 1:
            frame = pd.DataFrame({name: array})
        elif array.ndim == 2:
            frame = pd.DataFrame(array)
        else:
            raise ValueError("data must be one- or two-dimensional")

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("data must contain at least one observation and column")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("data must contain only numeric values") from exc
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("data must not contain infinite values")
    return frame


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _restore_type(frame: pd.DataFrame, original: DataLike) -> pd.Series | pd.DataFrame:
    return frame.iloc[:, 0].rename(frame.columns[0]) if isinstance(original, pd.Series) else frame


def instrument_name(data: DataLike, index: int) -> object:
    """Return the instrument column name at a zero-based position."""
    frame = _as_frame(data)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer")
    index = int(index)
    if index < 0 or index >= frame.shape[1]:
        raise IndexError(f"instrument index {index} is outside [0, {frame.shape[1] - 1}]")
    return frame.columns[index]


def instrument_index(data: DataLike, name: object) -> int:
    """Return the zero-based position of an instrument column name."""
    frame = _as_frame(data)
    if not frame.columns.is_unique:
        raise ValueError("instrument names must be unique")
    try:
        location = frame.columns.get_loc(name)
    except KeyError as exc:
        raise KeyError(f"unknown instrument name: {name!r}") from exc
    if not isinstance(location, (int, np.integer)):
        raise ValueError(f"instrument name {name!r} does not resolve to one column")
    return int(location)


def select_instruments(
    data: DataLike,
    instruments: Sequence[object],
) -> pd.DataFrame:
    """Select instruments by zero-based position, column name, or a mixture."""
    frame = _as_frame(data)
    if len(instruments) == 0:
        raise ValueError("instruments must not be empty")
    names = [
        instrument_name(frame, value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        else value
        for value in instruments
    ]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"unknown instrument name(s): {missing}")
    return frame.loc[:, names].copy()


def calculate_returns(
    prices: DataLike,
    *,
    method: ReturnMethod = "simple",
    periods: int = 1,
    dropna: bool = True,
) -> pd.Series | pd.DataFrame:
    """Calculate simple, logarithmic, or absolute-difference returns."""
    frame = _as_frame(prices, name="price")
    periods = _positive_int(periods, name="periods")

    if method == "simple":
        result = frame.pct_change(periods=periods, fill_method=None)
    elif method == "log":
        if (frame <= 0).any().any():
            raise ValueError("log returns require strictly positive prices")
        result = np.log(frame).diff(periods=periods)
    elif method == "difference":
        result = frame.diff(periods=periods)
    else:
        raise ValueError("method must be 'simple', 'log', or 'difference'")

    if dropna:
        result = result.dropna(how="all")
    return _restore_type(result, prices)


def rolling_statistics(
    returns: DataLike,
    *,
    window: int = 20,
    annualization: int = 250,
    risk_free_rate: float = 0.0,
    min_periods: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate rolling mean, volatility, and annualized Sharpe ratio."""
    frame = _as_frame(returns, name="return")
    window = _positive_int(window, name="window")
    annualization = _positive_int(annualization, name="annualization")
    min_periods = window if min_periods is None else _positive_int(min_periods, name="min_periods")
    if min_periods > window:
        raise ValueError("min_periods cannot exceed window")

    rolling = frame.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    volatility = rolling.std(ddof=1)
    daily_risk_free = float(risk_free_rate) / annualization
    sharpe = np.sqrt(annualization) * (mean - daily_risk_free) / volatility
    sharpe = sharpe.where(volatility > np.finfo(float).eps)
    return {"mean": mean, "volatility": volatility, "sharpe": sharpe}


def summary_statistics(
    data: DataLike,
    *,
    input_type: Literal["prices", "returns"] = "prices",
    return_method: ReturnMethod = "simple",
    annualization: int = 250,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Return common distribution and risk statistics for each instrument."""
    annualization = _positive_int(annualization, name="annualization")
    if input_type == "prices":
        returns = _as_frame(calculate_returns(data, method=return_method), name="return")
    elif input_type == "returns":
        returns = _as_frame(data, name="return")
    else:
        raise ValueError("input_type must be 'prices' or 'returns'")

    mean = returns.mean()
    volatility = returns.std(ddof=1)
    daily_risk_free = float(risk_free_rate) / annualization
    sharpe = np.sqrt(annualization) * (mean - daily_risk_free) / volatility
    sharpe = sharpe.where(volatility > np.finfo(float).eps)

    result = pd.DataFrame(
        {
            "count": returns.count(),
            "mean": mean,
            "annualized_return": mean * annualization,
            "annualized_volatility": volatility * np.sqrt(annualization),
            "sharpe": sharpe,
            "skew": returns.skew(),
            "kurtosis": returns.kurt(),
            "minimum": returns.min(),
            "median": returns.median(),
            "maximum": returns.max(),
        }
    )
    result.insert(0, "instrument_index", np.arange(len(result), dtype=int))
    result.insert(1, "instrument_name", result.index.astype(str))
    result.index.name = "instrument"
    return result


def correlation_matrix(
    data: DataLike,
    *,
    input_type: Literal["prices", "returns"] = "returns",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """Calculate an instrument correlation matrix."""
    if input_type == "prices":
        frame = _as_frame(calculate_returns(data), name="return")
    elif input_type == "returns":
        frame = _as_frame(data, name="return")
    else:
        raise ValueError("input_type must be 'prices' or 'returns'")
    return frame.corr(method=method)


def autocorrelation(
    data: DataLike,
    *,
    lags: int | Sequence[int] = 20,
    squared: bool = False,
) -> pd.DataFrame:
    """Calculate per-instrument autocorrelation for one or more positive lags."""
    frame = _as_frame(data, name="value")
    lag_values = range(1, _positive_int(lags, name="lags") + 1) if isinstance(lags, int) else lags
    validated = [_positive_int(lag, name="lag") for lag in lag_values]
    if not validated:
        raise ValueError("lags must not be empty")
    values = frame.pow(2) if squared else frame
    return pd.DataFrame(
        {column: [values[column].autocorr(lag=lag) for lag in validated] for column in values},
        index=pd.Index(validated, name="lag"),
    )


def compare_periods(
    data: DataLike,
    periods: Mapping[str, tuple[object, object]],
    *,
    input_type: Literal["prices", "returns"] = "prices",
    annualization: int = 250,
) -> pd.DataFrame:
    """Compare summary statistics across inclusive index-label periods."""
    frame = _as_frame(data)
    if not periods:
        raise ValueError("periods must not be empty")

    results: list[pd.DataFrame] = []
    for label, bounds in periods.items():
        if len(bounds) != 2:
            raise ValueError(f"period {label!r} must contain (start, end)")
        sample = frame.loc[bounds[0] : bounds[1]]
        if sample.empty:
            raise ValueError(f"period {label!r} contains no observations")
        stats = summary_statistics(sample, input_type=input_type, annualization=annualization)
        stats.insert(0, "period", label)
        stats.insert(1, "instrument", stats.index)
        results.append(stats.reset_index(drop=True))
    return pd.concat(results, ignore_index=True).set_index(["period", "instrument"])


def plot_series(
    data: DataLike,
    *,
    normalize: bool = False,
    log_scale: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one or more time series on a shared axis."""
    frame = _as_frame(data)
    if normalize:
        first = frame.apply(lambda column: column.dropna().iloc[0] if column.notna().any() else np.nan)
        if (first == 0).any():
            raise ValueError("cannot normalize a series whose first valid value is zero")
        frame = frame.divide(first).multiply(100)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure
    frame.plot(ax=ax)
    ax.set_title(title or "Time Series")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Normalized value (base 100)" if normalize else "Value")
    ax.set_yscale("log" if log_scale else "linear")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_series_indicators(
    prices: DataLike,
    *,
    ema_windows: Sequence[int] = (10, 30),
    volatility_window: int = 20,
    sharpe_window: int | None = 60,
    annualization: int = 250,
    normalize: bool = False,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot prices, EMAs, rolling volatility, and optionally rolling Sharpe."""
    frame = _as_frame(prices, name="price")
    windows = [_positive_int(window, name="ema_window") for window in ema_windows]
    volatility_window = _positive_int(volatility_window, name="volatility_window")
    if sharpe_window is not None:
        sharpe_window = _positive_int(sharpe_window, name="sharpe_window")

    display = frame.copy()
    if normalize:
        first = display.apply(lambda column: column.dropna().iloc[0] if column.notna().any() else np.nan)
        if (first == 0).any():
            raise ValueError("cannot normalize a series whose first valid value is zero")
        display = display.divide(first).multiply(100)

    panel_count = 3 if sharpe_window is not None else 2
    fig, axes = plt.subplots(panel_count, 1, figsize=(12, 3.3 * panel_count), sharex=True)
    axes = np.atleast_1d(axes)
    display.plot(ax=axes[0], linewidth=1.5)
    for column in display:
        for window in windows:
            axes[0].plot(
                display.index,
                display[column].ewm(span=window, adjust=False, min_periods=window).mean(),
                label=f"{column} EMA {window}",
                alpha=0.75,
            )
    axes[0].set_title(title or "Prices and indicators")
    axes[0].set_ylabel("Normalized price" if normalize else "Price")
    axes[0].legend(ncol=2, fontsize="small")

    returns = _as_frame(calculate_returns(frame), name="return")
    stats = rolling_statistics(returns, window=volatility_window, annualization=annualization)
    (stats["volatility"] * np.sqrt(annualization)).plot(ax=axes[1])
    axes[1].set_ylabel("Annualized volatility")

    if sharpe_window is not None:
        sharpe = rolling_statistics(returns, window=sharpe_window, annualization=annualization)["sharpe"]
        sharpe.plot(ax=axes[2])
        axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axes[2].set_ylabel("Rolling Sharpe")

    for axis in axes:
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Observation")
    fig.tight_layout()
    return fig, axes


def plot_correlation_matrix(
    data: DataLike,
    *,
    input_type: Literal["prices", "returns"] = "returns",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    annotate: bool = False,
    ax: Axes | None = None,
) -> tuple[pd.DataFrame, Figure, Axes]:
    """Calculate and plot an instrument correlation matrix."""
    matrix = correlation_matrix(data, input_type=input_type, method=method)
    if ax is None:
        size = max(6.0, min(14.0, 0.45 * len(matrix)))
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure

    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    positions = np.arange(len(matrix))
    ax.set_xticks(positions, matrix.columns, rotation=90)
    ax.set_yticks(positions, matrix.index)
    ax.set_title(f"{method.title()} correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        for row, column in np.ndindex(matrix.shape):
            ax.text(column, row, f"{matrix.iat[row, column]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    return matrix, fig, ax


__all__ = [
    "autocorrelation",
    "calculate_returns",
    "compare_periods",
    "correlation_matrix",
    "instrument_index",
    "instrument_name",
    "plot_correlation_matrix",
    "plot_series",
    "plot_series_indicators",
    "rolling_statistics",
    "select_instruments",
    "summary_statistics",
]
