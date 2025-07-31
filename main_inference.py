import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text, Date
import warnings
import pickle
import torch

from train_test_scripts.inference_script import ActualPriceFetcher, transform_dataframe
from model.ddqn_rl import DDQNTradingAgent
from db_upload_data.daily_upload_rl_feature import run_production_inference_batch, print_batch_results



warnings.filterwarnings('ignore')


class RLDataPreprocessor:
    def __init__(self, engine):
        self.engine = engine
        self.feature_columns = [
            'Open', 'High', 'Low', 'Adj Close', 'Volume', 'OBV', 'MACD', 'Signal',
            'Histogram', 'RSI', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
            'dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
            'dist_to_S1', 'dist_to_S2', 'dist_to_S3',
            'Short_Term_Foundation_Days', 'Long_Term_Foundation_Days'
        ]
        self.prediction_columns = ['y_pred_1', 'y_pred_2', 'y_pred_3', 'y_pred_4', 'y_pred_5']
        self.scaling_stats = {}

    def fetch_test_data(self, table_name='stock_features', start_date='2024-07-01', end_date=None):
        """Fetch test data with proper date filtering to prevent future data leakage"""

        # Build query with proper date filtering
        if end_date:
            query = f"""
            SELECT * FROM {table_name}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY symbol, date
            """
            print(f"Fetching test data from {start_date} to {end_date}...")
        else:
            query = f"""
            SELECT * FROM {table_name}
            WHERE date >= '{start_date}'
            ORDER BY symbol, date
            """
            print(f"Fetching test data from {start_date}...")

        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} rows for {df['symbol'].nunique()} symbols")
        return df

    def load_scaling_stats(self, stats_path="C:/Users/PC/PycharmProjects/GILIGILI_RL/data/scaling_stats.pkl"):
        """Load pre-computed scaling stats from training"""
        try:
            with open(stats_path, "rb") as f:
                self.scaling_stats = pickle.load(f)
            print(f"Loaded scaling stats for {len(self.scaling_stats)} features")
        except FileNotFoundError:
            print(f"Warning: Scaling stats file not found at {stats_path}")
            print("Will compute stats from test data (not recommended)")

    def normalize_with_stats(self, df):
        df_normalized = df.copy()
        for col in self.feature_columns:
            if col in df.columns and col in self.scaling_stats:
                mean_val, std_val = self.scaling_stats[col]
                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
                else:
                    df_normalized[col] = 0
        return df_normalized

    def create_rl_samples(self, df, cutoff_date=None, is_inference_mode=False):
        """
        Create RL samples with proper temporal constraints to prevent data leakage

        Args:
            df: DataFrame with stock data
            cutoff_date: Maximum date to create samples for (prevents future data leakage)
            is_inference_mode: If True, creates samples for pure inference (no reward calculation)
        """
        all_samples = []

        # Convert cutoff_date to pandas datetime if provided
        if cutoff_date:
            cutoff_date = pd.to_datetime(cutoff_date)
            print(f"Creating samples up to cutoff date: {cutoff_date}")

        for symbol in df['symbol'].unique():
            print(f"Processing {symbol}...")
            symbol_data = df[df['symbol'] == symbol].copy().reset_index(drop=True)

            if len(symbol_data) < 2:
                print(f"Skipping {symbol}: insufficient data ({len(symbol_data)} rows)")
                continue

            # Filter symbol data by cutoff_date if provided
            if cutoff_date:
                symbol_data = symbol_data[symbol_data['date'] <= cutoff_date].reset_index(drop=True)
                if len(symbol_data) < 1:
                    print(f"Skipping {symbol}: no data before cutoff date")
                    continue

            # CRITICAL FIX: In inference mode, we can only create samples where we have
            # enough future data OR we're at the last available date
            if is_inference_mode:
                # For inference, we process all available dates but handle "next_state" carefully
                max_sample_idx = len(symbol_data) - 1
            else:
                # For training, we need future returns so leave room
                max_sample_idx = max(0, len(symbol_data) - 6)

            for i in range(max_sample_idx + 1):
                current_row = symbol_data.iloc[i]

                # Skip if current row is beyond cutoff (additional safety check)
                if cutoff_date and current_row['date'] > cutoff_date:
                    continue

                # State features from current row
                state_features = [current_row[col] for col in self.feature_columns if col in current_row]
                state_features += [current_row[col] for col in self.prediction_columns if col in current_row]

                # Next state handling with temporal constraints
                if i + 1 < len(symbol_data):
                    next_row = symbol_data.iloc[i + 1]

                    # CRITICAL: Check if next_row violates cutoff date
                    if cutoff_date and next_row['date'] > cutoff_date:
                        # Use current row as next state to avoid future data leakage
                        next_state_features = state_features.copy()
                        next_date = current_row['date']
                        next_price = current_row['Adj Close']
                        done = True
                        print(f"  Date {current_row['date']}: Using current state as next_state (cutoff constraint)")
                    else:
                        # Normal case: use actual next row
                        next_state_features = [next_row[col] for col in self.feature_columns if col in next_row]
                        next_state_features += [next_row[col] for col in self.prediction_columns if col in next_row]
                        next_date = next_row['date']
                        next_price = next_row['Adj Close']
                        done = False
                else:
                    # Last available row case
                    next_state_features = state_features.copy()
                    next_date = current_row['date']
                    next_price = current_row['Adj Close']
                    done = True

                # Future returns calculation with cutoff constraints
                if is_inference_mode or cutoff_date:
                    # In inference mode or with cutoff, use available data only
                    future_end_idx = min(i + 6, len(symbol_data))

                    # Filter future data by cutoff_date
                    future_prices = []
                    for future_i in range(i + 1, future_end_idx):
                        future_row = symbol_data.iloc[future_i]
                        if cutoff_date and future_row['date'] > cutoff_date:
                            break
                        future_prices.append(future_row['Adj Close'])

                    # Pad with current price if insufficient future data
                    while len(future_prices) < 5:
                        future_prices.append(current_row['Adj Close'])

                    next_5_returns = future_prices
                else:
                    # Training mode: use all available future returns
                    future_end_idx = min(i + 6, len(symbol_data))
                    next_5_returns = symbol_data['Adj Close'].iloc[i + 1:future_end_idx].tolist()
                    while len(next_5_returns) < 5:
                        next_5_returns.append(next_5_returns[-1] if next_5_returns else current_row['Adj Close'])

                sample = {
                    'state': np.array(state_features, dtype=np.float32),
                    'action': 0,  # Placeholder
                    'reward': 0.0,  # Placeholder (not used in inference mode)
                    'next_state': np.array(next_state_features, dtype=np.float32),
                    'done': done,
                    'symbol': symbol,
                    'date': current_row['date'],
                    'next_date': next_date,
                    'current_price': current_row['Adj Close'],
                    'next_price': next_price,
                    'y_pred': [current_row[col] for col in self.prediction_columns if col in current_row],
                    'future_returns': next_5_returns,
                    'info': {
                        'volume': current_row['Volume'] if 'Volume' in current_row else 0,
                        'rsi': current_row['RSI'] if 'RSI' in current_row else 50,
                        'cutoff_applied': cutoff_date is not None
                    }
                }
                all_samples.append(sample)

        print(f"Created {len(all_samples)} RL samples")
        return all_samples

    def get_data_stats(self, samples):
        if not samples:
            return
        symbols = set(sample['symbol'] for sample in samples)
        state_dim = len(samples[0]['state'])
        print(f"\n=== Dataset Statistics ===")
        print(f"Total samples: {len(samples)}")
        print(f"Unique symbols: {len(symbols)}")
        print(f"State dimension: {state_dim}")

        # Show date ranges per symbol
        print(f"\nDate ranges per symbol:")
        for symbol in sorted(symbols):
            symbol_samples = [s for s in samples if s['symbol'] == symbol]
            if symbol_samples:
                dates = [s['date'] for s in symbol_samples]
                min_date = min(dates)
                max_date = max(dates)
                print(f"  {symbol}: {len(symbol_samples)} samples from {min_date.date()} to {max_date.date()}")


def prepare_test_data_with_cutoff(engine, table_name='stock_features', start_date='2024-07-01', cutoff_date=None):
    """
    Prepare test data with proper temporal cutoff to prevent data leakage

    Args:
        engine: Database engine
        table_name: Table name
        start_date: Start date for data
        cutoff_date: Cutoff date - no data beyond this date will be used
    """
    preprocessor = RLDataPreprocessor(engine)

    # Load pre-computed scaling stats from training
    preprocessor.load_scaling_stats()

    # Fetch data with proper end_date to prevent loading future data
    test_df = preprocessor.fetch_test_data(table_name, start_date, cutoff_date)

    if test_df.empty:
        print("No data available for the specified date range")
        return []

    # Normalize using training stats
    test_df_normalized = preprocessor.normalize_with_stats(test_df)

    # Create RL samples with cutoff date constraint
    test_samples = preprocessor.create_rl_samples(
        test_df_normalized,
        cutoff_date=cutoff_date,
        is_inference_mode=True
    )

    # Show normalization check
    print("\n🔍 Normalization check per feature (test set):")
    for col in preprocessor.feature_columns:
        if col in test_df_normalized.columns:
            values = test_df_normalized[col].astype(float)
            print(
                f"{col:30} | min: {values.min():7.4f} | max: {values.max():7.4f} | mean: {values.mean():7.4f} | std: {values.std():7.4f}")

    # Show stats
    preprocessor.get_data_stats(test_samples)

    return test_samples


# Modified main function with proper temporal constraints
def prepare_and_test(current_date=None):
    """
    Main function that prepares test data with proper temporal constraints

    Args:
        current_date: String in format 'YYYY-MM-DD' - NO DATA BEYOND THIS DATE WILL BE USED
    """
    # Database connection
    engine = create_engine(
        "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres"
    )

    # Prepare test data with proper cutoff
    test_samples = prepare_test_data_with_cutoff(
        engine,
        table_name='stock_features',
        start_date='2024-07-01',
        cutoff_date=current_date  # This prevents any future data leakage
    )

    if not test_samples:
        print("No test samples available")
        return None, None, None

    # Model path
    model_path = "C:/Users/PC/PycharmProjects/GILIGILI_RL/ddqn_trading_agent.pth"

    # Get state size from data
    state_size = len(test_samples[0]['state'])

    # Load trained model
    def load_trained_agent(model_path, state_size):
        """Load trained agent with proper checkpoint structure"""
        try:
            # Create agent with same parameters as training
            agent = DDQNTradingAgent(
                state_size=state_size,
                action_size=3,  # 0=SELL, 1=HOLD, 2=BUY
                learning_rate=5e-5,
                gamma=0.95,
                epsilon_start=0.0,  # Set to 0.0 for fully deterministic evaluation
                epsilon_end=0.0,
                epsilon_decay=0.9995,
                memory_size=50000,
                batch_size=32,
                target_update=500
            )

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=agent.device)

            # Load state dictionaries
            agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            agent.step_count = checkpoint['step_count']

            return agent

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    agent = load_trained_agent(model_path, state_size)
    if agent is None:
        print("Failed to load trained agent")
        return None, None, None

    # Initialize actual price fetcher
    try:
        price_fetcher = ActualPriceFetcher(engine)
    except Exception as e:
        print(f"Error initializing price fetcher: {e}")
        return None, None, None

    # CRITICAL: Use the fixed agent evaluation with proper position reset
    results = agent.evaluate(test_samples, calculate_rewards=False, reset_position=True)

    # Verify no future data leakage in results
    if current_date:
        cutoff_date = pd.to_datetime(current_date)
        future_data_count = sum(
            1 for log_entry in results['action_log']
            if pd.to_datetime(log_entry['date']) > cutoff_date
        )
        if future_data_count > 0:
            print(f"🚨 WARNING: Found {future_data_count} entries beyond cutoff date!")
        else:
            print(f"✅ No future data leakage detected - all entries are <= {current_date}")

    # Add actual prices
    action_log_df = price_fetcher.add_actual_prices_to_log(results['action_log'])

    # Transform DataFrame according to specifications
    transformed_df = transform_dataframe(action_log_df)

    # Print successfully processed stocks
    processed_symbols = transformed_df['symbol'].unique()
    for symbol in processed_symbols:
        symbol_data = transformed_df[transformed_df['symbol'] == symbol]
        valid_prices = symbol_data['price'].notna().sum()
        total_rows = len(symbol_data)
        date_range = f"{symbol_data['date'].min()} to {symbol_data['date'].max()}"
        print(f"Successfully processed {symbol}: {valid_prices}/{total_rows} records with valid prices ({date_range})")


    return agent, results, transformed_df


if __name__ == "__main__":

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

    current_date = "2025-07-31" # Điền ngày cho bố mau lên

    # Run batch inference
    results = run_production_inference_batch(stock_list, current_date)

    # Print summary
    print_batch_results(results)

    agent, results, df = prepare_and_test(current_date=current_date)

    df = df.reset_index()  # If 'date' is the index, this will make it a column
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.drop(columns=['index'])

    # Keep only the last row per symbol
    df_latest = df.sort_values('date').groupby('symbol', as_index=False).tail(1)

    # Setup engine
    engine = create_engine(
        "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres")

    with engine.begin() as conn:
        for _, row in df_latest.iterrows():
            # Delete existing row if exists (same date + symbol)
            conn.execute(
                text("DELETE FROM stock_trades WHERE date = :date AND symbol = :symbol"),
                {"date": row["date"], "symbol": row["symbol"]}
            )

        # Upload all filtered rows (clean append)
        df_latest.to_sql(
            'stock_trades',
            con=engine,
            if_exists='append',
            index=False,
            dtype={'date': Date()}
        )
