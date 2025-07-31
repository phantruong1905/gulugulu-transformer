import sqlalchemy
from sqlalchemy import create_engine
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text

from src.load_data import fetch_stock_data
from src.feature_engineering import *
from model.dual_transformer import *
from train_test_scripts.train_test_dual_transformer import *


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


# Setup once
engine = create_engine(
    "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "../trained_wavelet_model.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Checkpoint not found!")


def run_and_upload_stock(ticker: str, current_date: str):
    try:
        config = Config()
        data_path = '../data'
        os.makedirs(data_path, exist_ok=True)

        pooled_path = os.path.join(data_path, "test_stock_data.csv")
        if os.path.exists(pooled_path):
            os.remove(pooled_path)

        # Fetch data from 2017 (more history for better features)
        fetch_stock_data(data_path, [ticker], "2017-01-01", current_date, "test_stock_data.csv")
        df_all_stocks = pd.read_csv(pooled_path)

        # Upload raw data to database (basic OHLCV only)
        raw_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        df_raw = df_all_stocks[raw_columns].copy()
        df_raw.to_sql("raw_stock_data", engine, if_exists="append", index=False, method='multi', chunksize=500)

        # Filter data for the specific ticker and add technical indicators
        df_filtered = df_all_stocks[df_all_stocks['Symbol'] == ticker].copy()
        df_filtered = add_technical_indicators(df_filtered)
        df_filtered = add_pivot_features(df_filtered)
        df_filtered = calculate_foundation(df_filtered)

        # Convert dates and sort
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
        df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)

        # Load model checkpoint first to get dimensions
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Initialize model outside the loop
        model = None
        model_initialized = False

        # Sequential processing - process ALL dates from 2023-01-01 onwards
        start_date = pd.to_datetime('2023-01-01')
        end_date = pd.to_datetime(current_date)

        # Get all available trading dates from the data (skip weekends/holidays)
        available_dates = df_filtered['Date'].unique()
        available_dates = pd.to_datetime(available_dates)
        available_dates = available_dates[(available_dates >= start_date) & (available_dates <= end_date)]
        available_dates = sorted(available_dates)

        all_feature_rows = []  # Store all feature rows for ALL dates

        print(f"Processing {len(available_dates)} dates for {ticker}")

        # Process ALL available trading dates
        for i, current_test_date in enumerate(available_dates):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(available_dates)} dates processed")

            # Get data only up to current test date
            data_up_to_date = df_filtered[df_filtered['Date'] <= current_test_date].copy()

            if len(data_up_to_date) < config.seq_len + 10:  # Need minimum data
                continue

            try:
                # Create single sequence ending at current_test_date
                if len(data_up_to_date) < config.seq_len:
                    continue

                # Apply same preprocessing as in original script
                df_work = data_up_to_date.copy()
                df_work['Date'] = pd.to_datetime(df_work['Date'])
                df_work = df_work.set_index('Date').sort_index()

                # Define features
                base_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
                technical_features = ['OBV', 'MACD', 'Signal', 'Histogram', 'RSI'] + \
                                     [f'MA{w}' for w in [10, 20, 50, 100, 200]]
                pivot_features = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                                  'dist_to_S1', 'dist_to_S2', 'dist_to_S3']
                foundation_features = ['Short_Term_Foundation_Days', 'Long_Term_Foundation_Days']

                # Apply preprocessing
                features = df_work[base_features].copy()
                features = features.pct_change().replace([np.inf, -np.inf], np.nan)

                # Add technical indicators
                for feature in technical_features + pivot_features + foundation_features:
                    if feature in df_work.columns:
                        features[feature] = df_work[feature]

                # Create target
                returns = df_work['Adj Close'].pct_change().replace([np.inf, -np.inf], np.nan)
                trend = returns.rolling(window=10, min_periods=1).mean()
                target = returns - trend

                # Combine and clean
                data = features.join(pd.DataFrame({'target': target}), how='inner').dropna()

                # Clean and clip features
                for col in data.columns:
                    if col != 'target':
                        data[col] = data[col].ffill().bfill().fillna(0)
                        if col in base_features:
                            lower = data[col].quantile(0.01)
                            upper = data[col].quantile(0.99)
                            data[col] = data[col].clip(lower, upper)

                if len(data) < config.seq_len:
                    continue

                # Create single sequence (last seq_len days)
                feat_cols = [col for col in data.columns if col != 'target']
                x = data[feat_cols].values
                y = data['target'].values

                # Take the last sequence
                X_seq = x[-config.seq_len:].reshape(1, config.seq_len, -1)  # [1, seq_len, features]
                y_seq = y[-1:].reshape(1, 1)  # [1, 1] - just current target
                seq_date = data.index[-1]  # Current date

                # Extract original unscaled features (before normalization - for later inference)
                original_features = data.iloc[-1][feat_cols].to_dict()  # Last day's features

                # Compute normalization from this sequence
                X_mean = np.mean(X_seq, axis=1, keepdims=True)  # [1, 1, features]
                X_std = np.std(X_seq, axis=1, keepdims=True) + 1e-8
                y_mean = np.mean(y_seq)
                y_std = np.std(y_seq) + 1e-8

                # Normalize
                X_seq = (X_seq - X_mean) / X_std

                # Apply wavelet decomposition
                X_seq = modwt_decompose(X_seq, level=2)  # [1, seq_len, features, levels]

                # Initialize model only once when we have the first sequence
                if not model_initialized:
                    input_dim = X_seq.shape[2]
                    wavelet_levels = X_seq.shape[-1]
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
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    model_initialized = True

                # Make prediction
                with torch.no_grad():
                    # Convert to tensor and move to device
                    X_tensor = torch.FloatTensor(X_seq).to(device)

                    # Make prediction
                    prediction = model(X_tensor)

                    # Move prediction back to CPU for numpy conversion
                    prediction = prediction.cpu().numpy()

                # Keep predictions as they are (normalized/unscaled)
                prediction_unscaled = prediction

                # Create prediction sequence for this date
                pred_array = prediction_unscaled[0]
                if isinstance(pred_array, np.ndarray):
                    pred_list = pred_array.flatten().tolist()
                else:
                    pred_list = [pred_array]

                # Combine original features + normalized predictions for this date
                feature_row = {
                    "date": str(seq_date.date()),
                    "symbol": ticker,
                    **original_features,  # Original UNSCALED features
                    **{f"y_pred_{i + 1}": val for i, val in enumerate(pred_list)}  # Keep predictions normalized
                }
                all_feature_rows.append(feature_row)

            except Exception as e:
                print(f"Skipping date {current_test_date}: {str(e)}")

        # Upload ALL feature rows to database
        if all_feature_rows:
            df_all_features = pd.DataFrame(all_feature_rows)
            df_all_features.to_sql("stock_features", engine, if_exists="append", index=False, method='multi',
                                   chunksize=500)

            # Save to CSV for inspection
            output_path = f"../data/all_features_{ticker}_{current_date.replace('-', '')}.csv"
            df_all_features.to_csv(output_path, index=False)
            print(f"\n💾 Saved {len(all_feature_rows)} feature rows to database and CSV: {output_path}")

            # Show summary stats for original features only (not predictions)
            non_pred_cols = [col for col in df_all_features.columns if
                             not col.startswith('y_pred') and col not in ['date', 'symbol']]
            print(f"\n📈 Features Summary for {ticker}:")
            print(f"   📅 Date range: {df_all_features['date'].min()} to {df_all_features['date'].max()}")
            print(f"   📊 Total rows: {len(df_all_features)}")
            print(df_all_features[non_pred_cols].describe())

        print(f"[✅] {ticker}: {len(all_feature_rows)} feature rows processed and uploaded")

    except Exception as e:
        print(f"[❌] {ticker}: Error - {str(e)}")


def run_batch_feature_generation(tickers: list, current_date: str = None):
    """Run feature generation for a list of stocks"""
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')

    successful = 0
    failed = 0

    print(f"🚀 Starting feature generation for {len(tickers)} stocks up to {current_date}")
    print("=" * 70)

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            run_and_upload_stock(ticker, current_date)
            successful += 1

        except Exception as e:
            print(f"❌ {ticker} failed: {str(e)}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"📊 Batch Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {successful / len(tickers) * 100:.1f}%")


# Example usage
if __name__ == "__main__":
    # Single stock - generates features for ALL dates from 2023-01-01 to current_date
    # ticker = "AAPL"
    # current_date = "2024-12-31"
    # run_and_upload_stock(ticker, current_date)

    # Multiple stocks
    stock_list = [
        # Ngân hàng – Tài chính
        'CTG', 'MBB', 'TCB', 'MSB', 'BID', 'EIB', 'LPB', 'OCB', 'NAB', 'VAB', 'SHB', 'VPB', 'ABB', 'STB', 'ACB', 'KLB',

        # Bất động sản – KCN – Hạ tầng
        'DXG', 'KDH', 'HDG', 'NLG', 'IDC', 'KBC', 'DPG',
        'SZC', 'BCM', 'NTC', 'SIP', 'KHG', 'NTL', 'HHS', 'VCG', 'HDC', 'TCH',

        # Bán lẻ – Tiêu dùng
        'MWG', 'DGW', 'PNJ', 'FRT', 'VHC', 'ANV', 'NKG', 'MCH', 'MSN',
        'MSH',

        # Công nghiệp – Hóa chất – VLXD
        'DGC', 'GVR', 'HSG', 'HPG', 'DPM', 'DCM', 'BFC', 'CSV', 'GEX', 'REE', 'GEG', 'NT2', 'BMP',

        # Dầu khí
        'PVS', 'GEE', 'PLC',

        # Công nghệ – Viễn thông
        'CTR', 'ELC', 'VGI', 'VTP', 'TLG', 'FOX',

        # Nông nghiệp – Thực phẩm
        'PAN', 'DBC', 'QNS',

        # Chứng khoán
        'VIX', 'VCI', 'HCM', 'MBS', 'VDS', 'TVS', 'BSI', 'FTS', 'SSI', 'CTS', 'SHS',

        # Khác – Lẻ tiềm năngL
        'HAH', 'PHP', 'TNG', 'TDT', 'AST', 'VSC', 'KOS', 'NAF', 'DGC', 'NTP', 'CMG', 'VGS', 'FCN', 'VOS',

        # Dệt may
        'TCM', 'MSH', 'GIL', 'TNG',

        # Cao su
        'PHR', 'DRC', 'TRC', 'DPR',

        # Tải
        'GMD',

        # VN30
        'HPG', 'VRE', 'VIC', 'VHM',

        # Thêm bừa không biết ngành gì
        'AGG', 'EVG', 'IJC', 'HAG', 'DXS', 'EVF', 'VTO', 'CTD', 'CTI', 'HHV', 'DDV', 'HNG', 'MCH', 'HVN'
    ]

    current_date = "2025-07-28"  # or use None for today
    run_batch_feature_generation(stock_list, current_date)