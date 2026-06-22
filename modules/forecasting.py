"""Forecasting Engine Module.

Provides time-series forecasting using Prophet as the primary engine,
with Holt-Winters ExponentialSmoothing as a fallback.

The module is dataset-agnostic and works with any uploaded dataset
containing at least one datetime column and one numeric target column.
"""
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


def _detect_frequency(dt_series: pd.Series) -> str:
    """Infer the frequency of a datetime series.
    
    Returns one of 'D' (daily), 'W' (weekly), 'MS' (monthly), or 'D' as default.
    """
    try:
        freq = pd.infer_freq(dt_series.sort_values().dropna().unique()[:100])
        if freq:
            if 'M' in freq:
                return 'MS'
            if 'W' in freq:
                return 'W'
            return 'D'
    except Exception:
        pass
    
    # Fallback: estimate from median gap
    try:
        sorted_dt = dt_series.sort_values().dropna()
        gaps = sorted_dt.diff().dropna()
        median_gap = gaps.median().days
        if median_gap >= 25:
            return 'MS'
        elif median_gap >= 5:
            return 'W'
        return 'D'
    except Exception:
        return 'D'


def prepare_time_series(
    df: pd.DataFrame, 
    date_column: str, 
    target_column: str,
    freq: str = None,
    agg_method: str = "mean"
) -> pd.DataFrame:
    """Prepare a time-series DataFrame suitable for forecasting.
    
    Args:
        df: The input DataFrame.
        date_column: Name of the datetime column.
        target_column: Name of the numeric target column.
        freq: Frequency string ('D', 'W', 'MS'). Auto-detected if None.
        agg_method: Aggregation method (currently 'mean' for all frequencies).
        
    Returns:
        DataFrame with columns ['ds', 'y'] sorted by date.
    """
    ts_df = df[[date_column, target_column]].copy()
    ts_df.columns = ['ds', 'y']
    ts_df['ds'] = pd.to_datetime(ts_df['ds'], errors='coerce')
    ts_df = ts_df.dropna(subset=['ds', 'y'])
    ts_df['y'] = pd.to_numeric(ts_df['y'], errors='coerce')
    ts_df = ts_df.dropna(subset=['y'])
    
    if freq is None:
        freq = _detect_frequency(ts_df['ds'])
    
    # Aggregate by the specified frequency using the configured method
    ts_df = ts_df.set_index('ds').resample(freq).agg({'y': agg_method}).dropna().reset_index()
    ts_df = ts_df.sort_values('ds').reset_index(drop=True)
    
    return ts_df


def train_and_forecast(
    ts_df: pd.DataFrame,
    horizon: int = 30,
    freq: str = 'D'
) -> Dict[str, Any]:
    """Train a forecasting model and generate future predictions.
    
    Uses Prophet as the primary engine. Falls back to Holt-Winters
    ExponentialSmoothing if Prophet import or fitting fails.
    
    Args:
        ts_df: DataFrame with columns ['ds', 'y'].
        horizon: Number of future periods to forecast.
        freq: Frequency string ('D', 'W', 'MS').
        
    Returns:
        Dictionary containing forecast DataFrame, model name, and metadata.
    """
    engine_used = None
    forecast_df = None
    
    # --- Attempt 1: Prophet ---
    try:
        from prophet import Prophet
        
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto' if freq == 'D' else False,
            daily_seasonality=False,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(ts_df)
        
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        raw_forecast = model.predict(future)
        
        forecast_df = raw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        engine_used = "Prophet"
        
    except Exception as prophet_error:
        print(f"Prophet failed: {prophet_error}. Falling back to Holt-Winters.")
        
        # --- Attempt 2: Holt-Winters ---
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            ts_indexed = ts_df.set_index('ds')['y']
            ts_indexed = ts_indexed.asfreq(freq, method='ffill')
            
            # Determine seasonal periods
            if freq == 'D':
                seasonal_periods = 7
            elif freq == 'W':
                seasonal_periods = 52
            else:
                seasonal_periods = 12
            
            # Only use seasonal if enough data
            use_seasonal = len(ts_indexed) >= 2 * seasonal_periods
            
            if use_seasonal:
                model = ExponentialSmoothing(
                    ts_indexed, trend='add', seasonal='add',
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(ts_indexed, trend='add', seasonal=None)
            
            fitted = model.fit(optimized=True)
            hw_forecast = fitted.forecast(horizon)
            
            # Build forecast DataFrame
            future_dates = pd.date_range(
                start=ts_indexed.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=horizon,
                freq=freq
            )
            
            # Simple confidence interval approximation
            residuals = ts_indexed - fitted.fittedvalues
            std_resid = residuals.std()
            
            hist_df = pd.DataFrame({
                'Date': ts_indexed.index,
                'Forecast': fitted.fittedvalues.values,
                'Lower Bound': (fitted.fittedvalues - 1.96 * std_resid).values,
                'Upper Bound': (fitted.fittedvalues + 1.96 * std_resid).values,
            })
            
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': hw_forecast.values,
                'Lower Bound': (hw_forecast - 1.96 * std_resid).values,
                'Upper Bound': (hw_forecast + 1.96 * std_resid).values,
            })
            
            forecast_df = pd.concat([hist_df, future_df], ignore_index=True)
            engine_used = "Holt-Winters ExponentialSmoothing"
            
        except Exception as hw_error:
            raise RuntimeError(
                f"Both Prophet and Holt-Winters failed.\n"
                f"Prophet error: {prophet_error}\n"
                f"Holt-Winters error: {hw_error}"
            )
    
    # Compute insights
    future_only = forecast_df.tail(horizon)
    avg_forecast = future_only['Forecast'].mean()
    min_forecast = future_only['Forecast'].min()
    max_forecast = future_only['Forecast'].max()
    
    # Trend direction and strength
    first_half = future_only['Forecast'].iloc[:len(future_only)//2].mean()
    second_half = future_only['Forecast'].iloc[len(future_only)//2:].mean()
    
    pct_change = abs(second_half - first_half) / (abs(first_half) + 1e-9) * 100
    if pct_change < 2:
        trend = "Stable Trend"
        trend_strength = "None"
    elif pct_change < 5:
        direction = "Upward" if second_half > first_half else "Downward"
        trend = f"Weak {direction} Trend"
        trend_strength = "Weak"
    elif pct_change < 15:
        direction = "Upward" if second_half > first_half else "Downward"
        trend = f"Moderate {direction} Trend"
        trend_strength = "Moderate"
    else:
        direction = "Upward" if second_half > first_half else "Downward"
        trend = f"Strong {direction} Trend"
        trend_strength = "Strong"
    
    # Confidence Interval Width
    ci_widths = future_only['Upper Bound'] - future_only['Lower Bound']
    avg_ci_width = round(float(ci_widths.mean()), 4)
    
    return {
        "forecast_df": forecast_df,
        "future_df": future_only,
        "engine": engine_used,
        "horizon": horizon,
        "freq": freq,
        "insights": {
            "average_forecast": round(avg_forecast, 4),
            "min_forecast": round(min_forecast, 4),
            "max_forecast": round(max_forecast, 4),
            "forecast_range": f"{round(min_forecast, 4)} — {round(max_forecast, 4)}",
            "trend_direction": trend,
            "trend_strength": trend_strength,
            "avg_confidence_interval_width": avg_ci_width
        }
    }


def generate_forecast_plot(
    ts_df: pd.DataFrame,
    forecast_result: Dict[str, Any],
    output_path: Path,
    target_column: str = "Target"
) -> str:
    """Generate and save a Plotly forecast chart.
    
    Args:
        ts_df: The original time-series DataFrame with ['ds', 'y'].
        forecast_result: The dictionary returned by train_and_forecast().
        output_path: Path to save the forecast_plot.png.
        target_column: Name of the target column for chart labels.
        
    Returns:
        String path of the saved image.
    """
    import plotly.graph_objects as go
    
    forecast_df = forecast_result["forecast_df"]
    future_df = forecast_result["future_df"]
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=ts_df['ds'], y=ts_df['y'],
        mode='lines', name='Historical',
        line=dict(color='#2196F3')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_df['Date'], y=future_df['Forecast'],
        mode='lines', name='Forecast',
        line=dict(color='#FF5722', dash='dash')
    ))
    
    # Confidence bounds
    fig.add_trace(go.Scatter(
        x=pd.concat([future_df['Date'], future_df['Date'][::-1]]),
        y=pd.concat([future_df['Upper Bound'], future_df['Lower Bound'][::-1]]),
        fill='toself', fillcolor='rgba(255,87,34,0.15)',
        line=dict(color='rgba(255,87,34,0)'),
        showlegend=True, name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=f"Forecast: {target_column}",
        xaxis_title="Date",
        yaxis_title=target_column,
        template="plotly_white",
        height=500
    )
    
    filepath = output_path / "forecast_plot.png"
    fig.write_image(str(filepath))
    
    # Free memory
    fig.data = []
    del fig
    
    return str(filepath)
