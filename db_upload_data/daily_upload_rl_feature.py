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


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


class ProductionStockInference:
    def __init__(self, checkpoint_path: str, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.config = Config()
        self.model = None
        self.model_initialized = False

        # Validate checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def _initialize_model(self, input_dim: int, wavelet_levels: int):
        """Initialize model only once when we have the first sequence"""
        if not self.model_initialized:
            self.model = CleanWaveletTransformer(
                input_dim=input_dim,
                wavelet_levels=wavelet_levels,
                d_model=64,
                nhead=2,
                num_layers=3,
                drop_out=0.5,
                pred_len=self.config.pred_len
            )

            # Load model weights
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model_initialized = True

    def _fetch_and_prepare_data(self, ticker: str, current_date: str) -> pd.DataFrame:
        """Fetch stock data from 2017 to current date and apply feature engineering"""
        data_path = '../data'
        os.makedirs(data_path, exist_ok=True)

        # Create temporary file for this ticker
        temp_file = f"temp_stock_data_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pooled_path = os.path.join(data_path, temp_file)

        try:
            # Fetch data from 2017 to current date
            fetch_stock_data(data_path, [ticker], "2017-01-01", current_date, temp_file)
            df_all_stocks = pd.read_csv(pooled_path)

            # Filter data for the specific ticker and add technical indicators
            df_filtered = df_all_stocks[df_all_stocks['Symbol'] == ticker].copy()
            df_filtered = add_technical_indicators(df_filtered)
            df_filtered = add_pivot_features(df_filtered)
            df_filtered = calculate_foundation(df_filtered)

            # Convert dates and sort
            df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
            df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)

            return df_filtered

        finally:
            # Clean up temporary file
            if os.path.exists(pooled_path):
                os.remove(pooled_path)

    def _upload_current_raw_data(self, df: pd.DataFrame, ticker: str, current_date: str):
        """Upload only current date raw data to database (basic OHLCV only)"""
        # Get only current date data
        current_data = df[df['Date'] == current_date].copy()

        if not current_data.empty:
            # Select only basic OHLCV columns for raw_stock_data table
            raw_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
            raw_data = current_data[raw_columns].copy()

            # Upload to raw_stock_data table (append mode)
            raw_data.to_sql("raw_stock_data", self.engine, if_exists="append",
                            index=False, method='multi', chunksize=100)
            print(f"✅ Uploaded raw data for {ticker} on {current_date}")

    def _prepare_features_for_inference(self, df_filtered: pd.DataFrame, current_date: str) -> dict:
        """Prepare features for inference using the exact same logic as training"""
        # Get data only up to current date
        data_up_to_date = df_filtered[df_filtered['Date'] <= current_date].copy()

        if len(data_up_to_date) < self.config.seq_len + 10:
            raise ValueError(f"Insufficient data: need at least {self.config.seq_len + 10} days")

        # Apply same preprocessing as in training
        df_work = data_up_to_date.copy()
        df_work['Date'] = pd.to_datetime(df_work['Date'])
        df_work = df_work.set_index('Date').sort_index()

        # Define features (same as training)
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

        # Create target (needed for consistency, but won't be used in inference)
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

        if len(data) < self.config.seq_len:
            raise ValueError(f"Insufficient clean data: need at least {self.config.seq_len} days")

        return data

    def _make_prediction(self, data: pd.DataFrame, ticker: str, current_date: str) -> dict:
        """Make prediction for the current date"""
        # Create single sequence (last seq_len days)
        feat_cols = [col for col in data.columns if col != 'target']
        x = data[feat_cols].values

        # Take the last sequence
        X_seq = x[-self.config.seq_len:].reshape(1, self.config.seq_len, -1)
        seq_date = data.index[-1]  # Current date

        # Extract original unscaled features (for database storage)
        original_features = data.iloc[-1][feat_cols].to_dict()

        # Compute normalization from this sequence
        X_mean = np.mean(X_seq, axis=1, keepdims=True)
        X_std = np.std(X_seq, axis=1, keepdims=True) + 1e-8

        # Normalize
        X_seq = (X_seq - X_mean) / X_std

        # Apply wavelet decomposition
        X_seq = modwt_decompose(X_seq, level=2)

        # Initialize model if needed
        if not self.model_initialized:
            input_dim = X_seq.shape[2]
            wavelet_levels = X_seq.shape[-1]
            self._initialize_model(input_dim, wavelet_levels)

        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            prediction = self.model(X_tensor)
            prediction = prediction.cpu().numpy()

        # Process prediction
        pred_array = prediction[0].flatten()
        pred_list = pred_array.tolist()

        # Combine original features + predictions
        feature_row = {
            "date": str(seq_date.date()),
            "symbol": ticker,
            **original_features,  # Original UNSCALED features
            **{f"y_pred_{i + 1}": val for i, val in enumerate(pred_list)}
        }

        return feature_row

    def _upload_features(self, feature_row: dict):
        """Upload feature row to database"""
        df_features = pd.DataFrame([feature_row])
        df_features.to_sql("stock_features", self.engine, if_exists="append",
                           index=False, method='multi', chunksize=100)
        print(f"✅ Uploaded features for {feature_row['symbol']} on {feature_row['date']}")

    def run_inference(self, ticker: str, current_date: str):
        """Main inference pipeline for a single stock"""
        try:
            print(f"🚀 Starting inference for {ticker} on {current_date}")

            # 1. Fetch and prepare data (2017 to current date)
            print(f"📊 Fetching data from 2017-01-01 to {current_date}...")
            df_filtered = self._fetch_and_prepare_data(ticker, current_date)

            # 2. Upload current date raw data to database
            print(f"💾 Uploading raw data for {current_date}...")
            self._upload_current_raw_data(df_filtered, ticker, current_date)

            # 3. Prepare features for inference
            print(f"🔧 Preparing features...")
            data = self._prepare_features_for_inference(df_filtered, current_date)

            # 4. Make prediction
            print(f"🤖 Making prediction...")
            feature_row = self._make_prediction(data, ticker, current_date)

            # 5. Upload features to database
            print(f"💾 Uploading features...")
            self._upload_features(feature_row)

            print(f"✅ Successfully completed inference for {ticker}")
            return feature_row

        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            raise e


# Usage Functions
def run_production_inference(ticker: str, current_date: str = None):
    """Run production inference for a single stock"""
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')

    # Database connection
    db_connection = "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres"
    checkpoint_path = "../trained_wavelet_model.pt"

    # Initialize inference pipeline
    inference = ProductionStockInference(checkpoint_path, db_connection)

    # Run inference
    result = inference.run_inference(ticker, current_date)

    return result


def run_production_inference_batch(tickers: list, current_date: str = None):
    """Run production inference for a list of stocks"""
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')

    # Database connection
    db_connection = "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres"
    checkpoint_path = "../trained_wavelet_model.pt"

    # Initialize inference pipeline once for all stocks
    inference = ProductionStockInference(checkpoint_path, db_connection)

    results = {}
    successful = 0
    failed = 0

    print(f"🚀 Starting batch inference for {len(tickers)} stocks on {current_date}")
    print("=" * 60)

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            result = inference.run_inference(ticker, current_date)
            results[ticker] = {
                'status': 'success',
                'data': result,
                'predictions': [result[f'y_pred_{j + 1}'] for j in range(5)]
            }
            successful += 1
            print(f"✅ {ticker} completed successfully")

        except Exception as e:
            results[ticker] = {
                'status': 'failed',
                'error': str(e),
                'data': None,
                'predictions': None
            }
            failed += 1
            print(f"❌ {ticker} failed: {str(e)}")

    print("\n" + "=" * 60)
    print(f"📊 Batch Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {successful / len(tickers) * 100:.1f}%")

    return results


def print_batch_results(results: dict):
    """Print a summary of batch results"""
    print("\n🔍 Detailed Results:")
    print("-" * 60)

    for ticker, result in results.items():
        if result['status'] == 'success':
            preds = result['predictions']
            pred_str = ', '.join([f"{p:.4f}" for p in preds])
            print(f"✅ {ticker:6s}: [{pred_str}]")
        else:
            print(f"❌ {ticker:6s}: {result['error']}")


# Example usage
if __name__ == "__main__":

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
        'HAH', 'SCS', 'PHP', 'TNG', 'TDT', 'AST', 'VSC', 'KOS', 'NAF', 'DGC', 'NTP', 'CMG', 'VGS', 'FCN', 'VOS',

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

    current_date = "2025-07-21"  # or use None for today

    # Run batch inference
    results = run_production_inference_batch(stock_list, current_date)

    # Print summary
    print_batch_results(results)

    # Access individual results
    for ticker in stock_list:
        if results[ticker]['status'] == 'success':
            predictions = results[ticker]['predictions']
            print(f"\n{ticker} predictions: {predictions}")
        else:
            print(f"\n{ticker} failed: {results[ticker]['error']}")

