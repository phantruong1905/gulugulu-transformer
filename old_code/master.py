import sqlalchemy
from sqlalchemy import create_engine, text
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import ast
import json

from src.load_data import fetch_stock_data
from src.feature_engineering import *
from model.dual_transformer import *
from scripts.train_test_dual_transformer import *
from scripts.test_backtest import SimpleTradingAlgorithm

import sqlalchemy
from datetime import datetime, timedelta


def update_features_for_new_day(engine, ticker: str, new_raw_data: pd.DataFrame):
    """
    Fetch existing features, compute new features for the latest day, and append to database

    Args:
        engine: SQLAlchemy engine
        ticker: Stock ticker symbol
        new_raw_data: DataFrame with new raw data (Date, Open, High, Low, Close, Adj Close, Volume)

    Returns:
        tuple: (new_feature_row, existing_features) - both as DataFrames
    """
    try:
        print(f"[INFO] Updating features for {ticker}")

        # Step 1: Fetch the last 199 rows of features for this ticker (need for MA200)
        query = """
        SELECT * FROM stock_features 
        WHERE "Symbol" = %s 
        ORDER BY "Date" DESC 
        LIMIT 199
        """

        existing_features = pd.read_sql(query, engine, params=(ticker,))

        if existing_features.empty:
            print(f"[ERROR] No existing features found for {ticker}")
            return None, None

        # Sort by date ascending
        existing_features = existing_features.sort_values('Date').reset_index(drop=True)
        existing_features['Date'] = pd.to_datetime(existing_features['Date'])

        # Step 2: Combine with new raw data
        new_raw_data['Date'] = pd.to_datetime(new_raw_data['Date'])

        # Create new row with raw data
        new_row = {
            'Symbol': ticker,
            'Date': new_raw_data['Date'].iloc[0],
            'Open': new_raw_data['Open'].iloc[0],
            'High': new_raw_data['High'].iloc[0],
            'Low': new_raw_data['Low'].iloc[0],
            'Adj Close': new_raw_data['Adj Close'].iloc[0],
            'Volume': new_raw_data['Volume'].iloc[0]
        }

        # Combine existing data with new row for calculations
        combined_data = pd.concat([existing_features, pd.DataFrame([new_row])], ignore_index=True)
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)

        # Step 3: Calculate OBV (On-Balance Volume)
        print("[INFO] Calculating OBV...")

        # Get the last known OBV value (existing data already has OBV)
        last_obv = existing_features['OBV'].iloc[-1]

        # Calculate OBV for the new day
        last_close = existing_features['Adj Close'].iloc[-1]
        current_close = new_row['Adj Close']
        current_volume = new_row['Volume']

        if current_close > last_close:
            new_obv = last_obv + current_volume
        elif current_close < last_close:
            new_obv = last_obv - current_volume
        else:
            new_obv = last_obv

        new_row['OBV'] = new_obv

        # Step 4: Calculate Moving Averages
        print("[INFO] Calculating Moving Averages...")

        # Get recent close prices including new day (existing data already has close prices)
        recent_closes = list(existing_features['Adj Close']) + [current_close]

        for window in [10, 20, 50, 100, 200]:
            if len(recent_closes) >= window:
                ma_value = np.mean(recent_closes[-window:])
                new_row[f'MA{window}'] = ma_value
            else:
                # If not enough data, use whatever we have
                ma_value = np.mean(recent_closes)
                new_row[f'MA{window}'] = ma_value

        # Step 5: Calculate MACD, RSI, Signal, Histogram
        print("[INFO] Calculating MACD and RSI...")

        # For MACD, we need EMA calculations
        if len(recent_closes) >= 26:
            # Calculate EMA12 and EMA26
            ema12_data = pd.Series(recent_closes).ewm(span=12, adjust=False).mean()
            ema26_data = pd.Series(recent_closes).ewm(span=26, adjust=False).mean()

            macd = ema12_data.iloc[-1] - ema26_data.iloc[-1]
            new_row['MACD'] = macd

            # For Signal line, we need MACD history (existing data already has MACD)
            if len(existing_features) >= 9:
                recent_macd = list(existing_features['MACD'].tail(8)) + [macd]
                signal = pd.Series(recent_macd).ewm(span=9, adjust=False).mean().iloc[-1]
            else:
                signal = macd  # fallback

            new_row['Signal'] = signal
            new_row['Histogram'] = macd - signal
        else:
            new_row['MACD'] = 0
            new_row['Signal'] = 0
            new_row['Histogram'] = 0

        # Calculate RSI
        if len(recent_closes) >= 15:  # need at least 15 for 14-period RSI
            deltas = pd.Series(recent_closes).diff().dropna()
            gains = deltas.where(deltas > 0, 0).rolling(window=14).mean()
            losses = (-deltas.where(deltas < 0, 0)).rolling(window=14).mean()
            rs = gains.iloc[-1] / (losses.iloc[-1] + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            new_row['RSI'] = rsi
        else:
            new_row['RSI'] = 50  # neutral RSI

        # Step 6: Calculate Pivot Points
        print("[INFO] Calculating Pivot Points...")

        current_date = new_row['Date']
        current_week = current_date.to_period('W').start_time

        # Check if we need new pivot points for this week
        if not existing_features.empty:
            last_date = existing_features['Date'].iloc[-1]
            last_week = last_date.to_period('W').start_time

            if current_week == last_week:
                # Same week, use existing pivot distances
                pivot_cols = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                              'dist_to_S1', 'dist_to_S2', 'dist_to_S3']

                # Get last week's pivot levels (we need to reverse-calculate them)
                last_row = existing_features.iloc[-1]
                if 'dist_to_P' in last_row:
                    # Reverse calculate pivot from distance
                    pivot = last_row['Adj Close'] / (1 + last_row['dist_to_P'])

                    # Calculate new distances
                    new_row['dist_to_P'] = (current_close - pivot) / pivot

                    # For R1, R2, R3, S1, S2, S3, we need to reverse-calculate them too
                    # This is approximate since we don't store the actual levels
                    for col in pivot_cols[1:]:  # skip dist_to_P
                        if col in last_row:
                            old_level = last_row['Adj Close'] / (1 + last_row[col]) if last_row[col] != -1 else pivot
                            new_row[col] = (current_close - old_level) / pivot
                        else:
                            new_row[col] = 0
                else:
                    # No pivot data, set to 0
                    for col in pivot_cols:
                        new_row[col] = 0
            else:
                # New week, calculate new pivot points
                # Get this week's data (might just be this one day)
                week_high = new_row['High']
                week_low = new_row['Low']
                week_close = new_row['Adj Close']

                # If we have previous week data, use it for pivot calculation
                if not existing_features.empty:
                    prev_week_data = existing_features[
                        existing_features['Date'] >= last_week
                        ]

                    if not prev_week_data.empty:
                        week_high = prev_week_data['High'].max()
                        week_low = prev_week_data['Low'].min()
                        week_close = prev_week_data['Adj Close'].iloc[-1]

                # Calculate pivot points
                pivot = (week_high + week_low + week_close) / 3
                s1 = 2 * pivot - week_high
                r1 = 2 * pivot - week_low
                s2 = pivot - (week_high - week_low)
                r2 = pivot + (week_high - week_low)
                s3 = pivot - 2 * (week_high - week_low)
                r3 = pivot + 2 * (week_high - week_low)

                # Calculate distances
                new_row['dist_to_P'] = (current_close - pivot) / pivot
                new_row['dist_to_R1'] = (current_close - r1) / pivot
                new_row['dist_to_R2'] = (current_close - r2) / pivot
                new_row['dist_to_R3'] = (current_close - r3) / pivot
                new_row['dist_to_S1'] = (current_close - s1) / pivot
                new_row['dist_to_S2'] = (current_close - s2) / pivot
                new_row['dist_to_S3'] = (current_close - s3) / pivot
        else:
            # No existing data, set pivot distances to 0
            pivot_cols = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                          'dist_to_S1', 'dist_to_S2', 'dist_to_S3']
            for col in pivot_cols:
                new_row[col] = 0

        # Step 7: Calculate Foundation Days
        print("[INFO] Calculating Foundation Days...")

        if not existing_features.empty:
            # Short-term foundation (5% range)
            last_short_foundation = existing_features['Short_Term_Foundation_Days'].iloc[-1]

            # Get recent prices for foundation calculation
            recent_short_data = existing_features.tail(int(last_short_foundation) if last_short_foundation > 0 else 1)
            short_low = min(recent_short_data['Adj Close'].min(), current_close)
            short_high = max(recent_short_data['Adj Close'].max(), current_close)
            short_range = (short_high - short_low) / short_low * 100 if short_low > 0 else 0

            if short_range <= 5:
                new_row['Short_Term_Foundation_Days'] = last_short_foundation + 1
            else:
                new_row['Short_Term_Foundation_Days'] = 1

            # Long-term foundation (25% range)
            last_long_foundation = existing_features['Long_Term_Foundation_Days'].iloc[-1]

            recent_long_data = existing_features.tail(int(last_long_foundation) if last_long_foundation > 0 else 1)
            long_low = min(recent_long_data['Adj Close'].min(), current_close)
            long_high = max(recent_long_data['Adj Close'].max(), current_close)
            long_range = (long_high - long_low) / long_low * 100 if long_low > 0 else 0

            if long_range <= 25:
                new_row['Long_Term_Foundation_Days'] = last_long_foundation + 1
            else:
                new_row['Long_Term_Foundation_Days'] = 1
        else:
            # No existing data
            new_row['Short_Term_Foundation_Days'] = 1
            new_row['Long_Term_Foundation_Days'] = 1

        # Step 8: Create final DataFrame and save to database
        print("[INFO] Saving to database...")

        # Ensure all required columns exist
        required_columns = [
            'Symbol', 'Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume',
            'OBV', 'MACD', 'Signal', 'Histogram', 'RSI',
            'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
            'dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
            'dist_to_S1', 'dist_to_S2', 'dist_to_S3',
            'Short_Term_Foundation_Days', 'Long_Term_Foundation_Days'
        ]

        # Fill missing columns with 0
        for col in required_columns:
            if col not in new_row:
                new_row[col] = 0

        # Create DataFrame with new row
        new_feature_row = pd.DataFrame([new_row])

        # Append to database
        new_feature_row.to_sql("stock_features", engine, if_exists="append", index=False)

        print(f"[SUCCESS] Features updated for {ticker} on {new_row['Date'].strftime('%Y-%m-%d')}")
        print(f"New features: OBV={new_row['OBV']:.2f}, RSI={new_row['RSI']:.2f}, "
              f"Short_Foundation={new_row['Short_Term_Foundation_Days']}, "
              f"Long_Foundation={new_row['Long_Term_Foundation_Days']}")

        return new_feature_row, existing_features

    except Exception as e:
        print(f"[ERROR] Failed to update features for {ticker}: {str(e)}")
        raise e


def prepare_inference_sequence(engine, existing_features, new_feature_row, ticker, seq_len, level):
    """
    FIXED: Prepare the latest sequence for model inference using EXACT same preprocessing as training
    """
    try:
        print(f"[INFO] Preparing inference sequence for {ticker}")

        # Step 1: Take the last seq_len rows from existing features
        if len(existing_features) >= seq_len:
            recent_existing = existing_features.tail(seq_len).reset_index(drop=True)
        else:
            recent_existing = existing_features.reset_index(drop=True)

        # Step 2: Add the new feature row to get seq_len + 1 total
        recent_features = pd.concat([recent_existing, new_feature_row], ignore_index=True)
        recent_features = recent_features.sort_values('Date').reset_index(drop=True)
        recent_features['Date'] = pd.to_datetime(recent_features['Date'])

        if len(recent_features) < seq_len + 1:
            print(f"[ERROR] Not enough data for {ticker}. Need {seq_len + 1}, got {len(recent_features)}")
            return None, False

        # Step 3: Define feature columns (MUST match training exactly)
        base_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        technical_features = ['OBV', 'MACD', 'Signal', 'Histogram', 'RSI'] + \
                             [f'MA{w}' for w in [10, 20, 50, 100, 200]]
        pivot_features = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                          'dist_to_S1', 'dist_to_S2', 'dist_to_S3']
        foundation_features = ['Short_Term_Foundation_Days', 'Long_Term_Foundation_Days']

        # Step 4: Apply EXACT same preprocessing as training
        print("[INFO] Applying preprocessing (must match training exactly)...")

        # CRITICAL: Apply preprocessing in the SAME order as training

        # 4a. Convert base features to percent changes (FIRST)
        base_data = recent_features[base_features].copy()
        base_data = base_data.pct_change()
        base_data = base_data.replace([np.inf, -np.inf], np.nan)

        # 4b. Technical indicators (use as-is, no transformation)
        technical_data = recent_features[technical_features + pivot_features + foundation_features].copy()

        # 4c. Combine all features
        all_features = pd.concat([base_data, technical_data], axis=1)

        # 4d. Clean data (fill NaN, handle outliers)
        for col in all_features.columns:
            all_features[col] = all_features[col].ffill().bfill().fillna(0)

            # Apply outlier clipping ONLY to base features (percent changes)
            if col in base_features:
                lower = all_features[col].quantile(0.01)
                upper = all_features[col].quantile(0.99)
                all_features[col] = all_features[col].clip(lower, upper)

        # 4e. Remove first row (NaN from pct_change) - gives us exactly seq_len rows
        features = all_features.iloc[1:].reset_index(drop=True)

        if len(features) != seq_len:
            print(f"[ERROR] Sequence length mismatch for {ticker}. Expected {seq_len}, got {len(features)}")
            return None, False

        print(f"[INFO] Preprocessed features shape: {features.shape}")
        print(f"[INFO] Feature columns: {list(features.columns)}")

        # Step 5: Fetch and parse normalization stats
        print("[INFO] Fetching normalization stats...")
        norm_query = "SELECT * FROM normalization_stats WHERE symbol = %s"
        norm_stats = pd.read_sql(norm_query, engine, params=(ticker,))

        x_mean = np.array(ast.literal_eval(norm_stats['x_mean'].iloc[0]))
        x_std = np.array(ast.literal_eval(norm_stats['x_std'].iloc[0]))

        print(x_mean)
        print(x_std)

        # Step 6: Validate dimensions
        X_sequence = features.values

        if X_sequence.shape[1] != len(x_mean):
            print(f"[ERROR] Feature dimension mismatch!")
            print(f"  Current features: {X_sequence.shape[1]}")
            print(f"  Expected features: {len(x_mean)}")
            print(f"  Feature columns: {list(features.columns)}")
            return None, False

        # Step 7: Apply normalization
        print("[INFO] Applying normalization...")
        print(f"  Before normalization - Shape: {X_sequence.shape}")
        print(f"  Before normalization - Range: [{X_sequence.min():.6f}, {X_sequence.max():.6f}]")

        # Normalize using training stats
        X_sequence = (X_sequence - x_mean) / (x_std)

        print(f"  After normalization - Range: [{X_sequence.min():.6f}, {X_sequence.max():.6f}]")

        # Step 8: Add batch dimension and apply wavelet decomposition
        X_sequence = X_sequence.reshape(1, X_sequence.shape[0], X_sequence.shape[1])

        print (X_sequence)

        # Apply MODWT decomposition
        print("[INFO] Applying wavelet decomposition...")
        X_sequence = modwt_decompose(X_sequence, level=level)

        print(f"[SUCCESS] Created inference sequence for {ticker}")
        print(f"  Final sequence shape: {X_sequence.shape}")

        return X_sequence, True

    except Exception as e:
        print(f"[ERROR] Failed to prepare inference sequence for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False


def calculate_signal_strength(y_pred):
    """Calculate signal strength with weighted prediction"""

    # Ensure input is a flattened list
    if isinstance(y_pred, (int, float)):
        y_pred = [y_pred] * 5
    elif isinstance(y_pred, np.ndarray):
        y_pred = y_pred.flatten().tolist()
    elif not isinstance(y_pred, list):
        try:
            y_pred = list(y_pred)
        except (TypeError, ValueError):
            y_pred = [float(y_pred)] * 5

    if len(y_pred) < 2:
        return 0.0

    # Use only first 5 predictions and compute weighted signal
    y_pred_truncated = y_pred[:5]
    weights = np.array([0.2, 0.5, 0.2, 0.05, 0.05])
    weights = weights[:len(y_pred_truncated)]
    weights /= weights.sum()
    weighted_signal = np.sum(np.array(y_pred_truncated) * weights)

    return weighted_signal


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


# Setup once
engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "../trained_wavelet_model.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Checkpoint not found!")

logging.basicConfig(level=logging.INFO)


# Updated function to integrate with your existing pipeline
def fetch_and_infer_latest_signal(ticker: str, current_date: str):
    """
    Complete pipeline for fetching new data, processing features, and generating trading signals
    """
    try:
        config = Config()
        seq_len = config.seq_len
        level = 2

        print(f"[INFO] Starting pipeline for {ticker} on {current_date}")

        # Step 1: Fetch latest raw stock data for current_date
        print(f"[INFO] Fetching new raw data for {current_date}")
        fetch_stock_data('../data', [ticker], current_date, current_date, "latest_day.csv")
        df_new = pd.read_csv("../data/latest_day.csv")
        df_new = df_new[df_new['Symbol'] == ticker]
        df_new = df_new[df_new['Date'] == current_date]

        if df_new.empty:
            print(f"[ERROR] No data fetched for {ticker} on {current_date}")
            return

        # Step 2: Store new raw data in database
        print(f"[INFO] Storing raw data to database")
        df_new.to_sql("raw_stock_data", engine, if_exists="append", index=False)

        # Step 3: Update features in database and get both new and existing data
        print(f"[INFO] Updating features in database")
        new_feature_row, existing_features = update_features_for_new_day(engine, ticker, df_new)

        if new_feature_row is None or existing_features is None:
            print(f"[ERROR] Failed to update features for {ticker}")
            return

        # Step 4: Prepare inference sequence using existing data
        print(f"[INFO] Preparing inference sequence")
        X_sequence, success = prepare_inference_sequence(engine, existing_features, new_feature_row, ticker, seq_len, level)


        if not success:
            print(f"[ERROR] Failed to prepare inference sequence for {ticker}")
            return

        # Step 5: Load model and make prediction
        print(f"[INFO] Loading model and making prediction")

        input_dim = X_sequence.shape[2]
        wavelet_levels = X_sequence.shape[-1]

        model = CleanWaveletTransformer(
            input_dim=input_dim,
            wavelet_levels=wavelet_levels,
            d_model=64,
            nhead=2,
            num_layers=3,
            drop_out=0.5,
            pred_len=config.pred_len
        )

        # Load model weights
        model.to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequence).to(device)
            prediction = model(X_tensor).cpu().numpy()

        # Step 6: Calculate signal strength
        signal_strength = calculate_signal_strength(prediction[0])

        print(f"[SUCCESS] Pipeline completed for {ticker} on {current_date}")
        print(f"Raw prediction: {prediction[0]}")
        print(f"Signal strength: {signal_strength:.4f}")

        return {
            'ticker': ticker,
            'date': current_date,
            'prediction': prediction[0].tolist(),
            'signal_strength': signal_strength,
            'features_updated': True
        }

    except Exception as e:
        print(f"[ERROR] Pipeline failed for {ticker}: {str(e)}")
        raise e


def main():
    """Main execution function"""
    tickers = ["FTS", "HAH"]
    current_date = "2025-07-01"  # The new day we want to fetch and process

    print("=" * 80)
    print("REAL-TIME TRADING SIGNAL GENERATION")
    print("=" * 80)
    print(f"Processing date: {current_date}")
    print(f"Tickers: {tickers}")
    print("=" * 80)

    for ticker in tickers:
        print(f"\n{'=' * 20} Processing {ticker} {'=' * 20}")
        fetch_and_infer_latest_signal(ticker, current_date)
        print(f"{'=' * 50}")

    print(f"\n[✅] Pipeline completed for {current_date}")


if __name__ == "__main__":
    main()