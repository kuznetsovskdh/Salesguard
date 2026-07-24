"""Прогноз выручки и детекция аномалий по товарным продажам.

Ключевое архитектурное требование (см. отчёт сессии): перед анализом
временной ряд ОБЯЗАТЕЛЬНО реиндексируется на полный календарный диапазон
с зафиксированными нулями на пропущенных датах. WB/Ozon не создают строку
с нулевым значением при отсутствии продаж (стокаут) - день просто
отсутствует в выгрузке. Наивный groupby('date') пропустит такие дни
полностью, и провал продаж (аномалия) останется незамеченным."""
import pandas as pd
import numpy as np


def _build_daily_series(df: pd.DataFrame, product: str = None) -> pd.Series:
    """Строит ежедневный ряд quantity (или revenue, если quantity нет),
    реиндексированный на полный календарный диапазон с zero-fill пропусков."""
    work = df.copy()
    if 'is_return' in work.columns:
        work = work[~work['is_return']]
    if product is not None:
        work = work[work['product'] == product]
    if work.empty or 'sale_date' not in work.columns:
        return pd.Series(dtype=float)

    value_col = 'quantity' if 'quantity' in work.columns else 'revenue'
    daily = work.groupby(work['sale_date'].dt.date)[value_col].sum()
    daily.index = pd.to_datetime(daily.index)

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(full_range, fill_value=0)
    return daily


def detect_anomalies(df: pd.DataFrame, product: str = None, z_threshold: float = 3.0) -> pd.DataFrame:
    """Детектирует аномальные дни в продажах через z-score на скользящем окне.

    Сравнивает каждый день с ЛОКАЛЬНЫМ трендом (rolling mean/std за 14 дней),
    а не с общим средним по всему периоду - иначе всплеск поверх растущего
    тренда (например FC-GROWING) не будет пойман, т.к. общее среднее уже
    "смазано" ростом.
    """
    daily = _build_daily_series(df, product)
    if daily.empty or len(daily) < 14:
        return pd.DataFrame()

    rolling_mean = daily.rolling(window=14, center=True, min_periods=7).mean()
    rolling_std = daily.rolling(window=14, center=True, min_periods=7).std()

    z_scores = (daily - rolling_mean) / rolling_std.replace(0, np.nan)

    anomalies = pd.DataFrame({
        'date': daily.index,
        'value': daily.values,
        'local_trend': rolling_mean.values,
        'z_score': z_scores.values,
    })
    anomalies = anomalies[anomalies['z_score'].abs() >= z_threshold].copy()
    anomalies['type'] = anomalies['z_score'].apply(
        lambda z: 'spike' if z > 0 else 'dip'
    )
    return anomalies.reset_index(drop=True)


def forecast_revenue(df: pd.DataFrame, product: str = None, horizon_days: int = 30) -> pd.DataFrame:
    """Простой бейзлайн-прогноз через экспоненциальное сглаживание.

    Использует линейную экстраполяцию тренда (не просто сглаживание к
    среднему) - иначе растущие/падающие SKU (например FC-GROWING) дали бы
    плоский прогноз вместо продолжения тренда.
    """
    daily = _build_daily_series(df, product)
    if daily.empty or len(daily) < 14:
        return pd.DataFrame()

    # Экспоненциально взвешенное среднее для сглаживания шума (используется
    # только как стартовая точка экстраполяции)
    smoothed = daily.ewm(span=14, adjust=False).mean()

    # Наклон тренда считаем на СЫРЫХ данных, не сглаженных - EWM запаздывает
    # за трендом на конце ряда и занижает slope для растущих/падающих SKU
    trend_window = min(30, len(daily))
    recent_raw = daily.iloc[-trend_window:]
    x = np.arange(len(recent_raw))
    slope, _ = np.polyfit(x, recent_raw.values, 1)

    last_value = smoothed.iloc[-1]
    future_dates = pd.date_range(
        daily.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D'
    )
    forecast_values = [
        max(0, last_value + slope * (i + 1)) for i in range(horizon_days)
    ]

    return pd.DataFrame({
        'date': future_dates,
        'forecast': forecast_values,
    })
