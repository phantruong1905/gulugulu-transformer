import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import warnings
import pickle

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

    def fetch_data(self, table_name='your_table_name'):
        query = f"""
        SELECT * FROM {table_name}
        WHERE date >= '2023-01-01'
        ORDER BY symbol, date
        """
        print("Fetching data from database...")
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} rows for {df['symbol'].nunique()} symbols")
        return df

    def compute_scaling_stats(self, df):
        for col in self.feature_columns:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                self.scaling_stats[col] = (mean_val, std_val)

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


    def create_rl_samples(self, df):
        all_samples = []
        for symbol in df['symbol'].unique():
            print(f"Processing {symbol}...")
            symbol_data = df[df['symbol'] == symbol].copy().reset_index(drop=True)
            if len(symbol_data) < 10:
                print(f"Skipping {symbol}: insufficient data ({len(symbol_data)} rows)")
                continue
            for i in range(len(symbol_data) - 6):
                current_row = symbol_data.iloc[i]
                next_row = symbol_data.iloc[i + 1]
                state_features = [current_row[col] for col in self.feature_columns if col in current_row]
                state_features += [current_row[col] for col in self.prediction_columns if col in current_row]
                next_state_features = [next_row[col] for col in self.feature_columns if col in next_row]
                next_state_features += [next_row[col] for col in self.prediction_columns if col in next_row]
                done = (i == len(symbol_data) - 2)
                next_5_returns = symbol_data['Adj Close'].iloc[i + 1:i + 6].tolist()
                sample = {
                    'state': np.array(state_features, dtype=np.float32),
                    'action': 0,
                    'reward': 0.0,
                    'next_state': np.array(next_state_features, dtype=np.float32),
                    'done': done,
                    'symbol': symbol,
                    'date': current_row['date'],
                    'next_date': next_row['date'],
                    'current_price': current_row['Adj Close'],
                    'next_price': next_row['Adj Close'],
                    'y_pred': [current_row[col] for col in self.prediction_columns if col in current_row],
                    'future_returns': next_5_returns,
                    'info': {
                        'volume': current_row['Volume'] if 'Volume' in current_row else 0,
                        'rsi': current_row['RSI'] if 'RSI' in current_row else 50
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
        print(f"Symbols: {sorted(symbols)}")
        symbol_counts = {}
        for sample in samples:
            symbol_counts[sample['symbol']] = symbol_counts.get(sample['symbol'], 0) + 1
        print(f"\nSamples per symbol:")
        for symbol, count in sorted(symbol_counts.items()):
            print(f"  {symbol}: {count}")


def prepare_rl_data(engine, table_name='stock_features'):
    preprocessor = RLDataPreprocessor(engine)
    df = preprocessor.fetch_data(table_name)

    # Split into train and test first
    cutoff_date = pd.to_datetime("2024-07-01")
    train_df = df[df['date'] < cutoff_date].copy()
    test_df = df[df['date'] >= cutoff_date].copy()

    # Compute global normalization on training set
    preprocessor.compute_scaling_stats(train_df)

    # Normalize both train and test using training stats
    train_df_normalized = preprocessor.normalize_with_stats(train_df)
    test_df_normalized = preprocessor.normalize_with_stats(test_df)

    # Create RL samples
    train_samples = preprocessor.create_rl_samples(train_df_normalized)
    test_samples = preprocessor.create_rl_samples(test_df_normalized)

    # Save normalization stats for future use
    with open("C:/Users/PC/PycharmProjects/GILIGILI_RL/data/scaling_stats.pkl", "wb") as f:
        pickle.dump(preprocessor.scaling_stats, f)

    # Show stats
    print("\n🔍 Normalization check per feature (train set):")
    for col in preprocessor.feature_columns:
        if col in train_df_normalized.columns:
            values = train_df_normalized[col].astype(float)
            print(f"{col:30} | min: {values.min():7.4f} | max: {values.max():7.4f} | mean: {values.mean():7.4f} | std: {values.std():7.4f}")

    preprocessor.get_data_stats(train_samples)
    preprocessor.get_data_stats(test_samples)

    return train_samples, test_samples


if __name__ == "__main__":
    engine = create_engine(...)

    train_data, test_data = prepare_rl_data(engine, table_name='stock_features')

    with open("C:/Users/PC/PycharmProjects/GILIGILI_RL/data/train_rl_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open("C:/Users/PC/PycharmProjects/GILIGILI_RL/data/test_rl_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print(f"✅ Train: {len(train_data)}, Test: {len(test_data)}")

    if train_data:
        print(f"\nFirst sample:")
        print(f"State shape: {train_data[0]['state'].shape}")
        print(f"Symbol: {train_data[0]['symbol']}")
        print(f"Date: {train_data[0]['date']}")
        print(f"Current price: {train_data[0]['current_price']:.4f}")
        print(f"Next price: {train_data[0]['next_price']:.4f}")
        print(f"Predictions: {train_data[0]['y_pred']}")
