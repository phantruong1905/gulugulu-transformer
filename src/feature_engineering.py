import pandas as pd
import numpy as np
import pywt
from datetime import timedelta



def add_pivot_features(df):
    """Add pivot point technical indicators"""
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    pivot_features = []

    for symbol, group in df.groupby('Symbol'):
        group = group.copy().reset_index(drop=True)
        group['Week'] = group['Date'].dt.to_period('W').apply(lambda r: r.start_time)

        weekly = group.groupby('Week').agg({
            'High': 'max',
            'Low': 'min',
            'Adj Close': 'last'
        }).reset_index()

        weekly['Pivot'] = (weekly['High'] + weekly['Low'] + weekly['Adj Close']) / 3
        weekly['S1'] = 2 * weekly['Pivot'] - weekly['High']
        weekly['R1'] = 2 * weekly['Pivot'] - weekly['Low']
        weekly['S2'] = weekly['Pivot'] - (weekly['High'] - weekly['Low'])
        weekly['R2'] = weekly['Pivot'] + (weekly['High'] - weekly['Low'])
        weekly['S3'] = weekly['Pivot'] - 2 * (weekly['High'] - weekly['Low'])
        weekly['R3'] = weekly['Pivot'] + 2 * (weekly['High'] - weekly['Low'])

        merged = group.merge(weekly[['Week', 'Pivot', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']], on='Week', how='left')

        # Handle any missing pivot values by forward fill
        pivot_cols = ['Pivot', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']
        for col in pivot_cols:
            merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')

        merged['dist_to_P'] = (merged['Adj Close'] - merged['Pivot']) / merged['Pivot']
        merged['dist_to_R1'] = (merged['Adj Close'] - merged['R1']) / merged['Pivot']
        merged['dist_to_R2'] = (merged['Adj Close'] - merged['R2']) / merged['Pivot']
        merged['dist_to_R3'] = (merged['Adj Close'] - merged['R3']) / merged['Pivot']
        merged['dist_to_S1'] = (merged['Adj Close'] - merged['S1']) / merged['Pivot']
        merged['dist_to_S2'] = (merged['Adj Close'] - merged['S2']) / merged['Pivot']
        merged['dist_to_S3'] = (merged['Adj Close'] - merged['S3']) / merged['Pivot']

        # Drop intermediate columns but keep the distance features
        merged = merged.drop(columns=['R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'Pivot', 'Week'])
        pivot_features.append(merged)

    result = pd.concat(pivot_features, ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)
    return result


def calculate_foundation(df):
    """Calculate foundation days indicators"""
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    updated_dfs = []

    for symbol, group in df.groupby('Symbol'):
        group = group.copy().reset_index(drop=True)
        group = group.sort_values('Date').reset_index(drop=True)

        # Initialize for this stock
        short_count = 0
        long_count = 0
        short_low = group['Adj Close'].iloc[0]
        short_high = group['Adj Close'].iloc[0]
        long_low = group['Adj Close'].iloc[0]
        long_high = group['Adj Close'].iloc[0]

        short_term_days = []
        long_term_days = []

        for idx, row in group.iterrows():
            current_close = row['Adj Close']

            # Short-term range (5%)
            short_low = min(short_low, current_close)
            short_high = max(short_high, current_close)
            short_range = (short_high - short_low) / short_low * 100 if short_low > 0 else 0

            if short_range <= 5:
                short_count += 1
            else:
                short_count = 1
                short_low = current_close
                short_high = current_close

            # Long-term range (25%)
            long_low = min(long_low, current_close)
            long_high = max(long_high, current_close)
            long_range = (long_high - long_low) / long_low * 100 if long_low > 0 else 0

            if long_range <= 25:
                long_count += 1
            else:
                long_count = 1
                long_low = current_close
                long_high = current_close

            short_term_days.append(short_count)
            long_term_days.append(long_count)

        group['Short_Term_Foundation_Days'] = short_term_days
        group['Long_Term_Foundation_Days'] = long_term_days

        updated_dfs.append(group)

    return pd.concat(updated_dfs, ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)


def add_technical_indicators(df):
    """Add technical indicators per stock"""
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    result_dfs = []

    for symbol, group in df.groupby('Symbol'):
        indicator_df = group.copy().reset_index(drop=True)

        # Ensure Volume column exists
        if 'Volume' not in indicator_df.columns:
            indicator_df['Volume'] = 0

        # On-Balance Volume (OBV) - needs to be calculated per stock
        price_change = indicator_df['Adj Close'].diff()
        obv_values = []
        obv_cumsum = 0
        for i, change in enumerate(price_change):
            if pd.isna(change) or change == 0:
                obv_values.append(obv_cumsum)
            elif change > 0:
                obv_cumsum += indicator_df['Volume'].iloc[i]
                obv_values.append(obv_cumsum)
            else:  # change < 0
                obv_cumsum -= indicator_df['Volume'].iloc[i]
                obv_values.append(obv_cumsum)
        indicator_df['OBV'] = obv_values

        # MACD - per stock
        exp1 = indicator_df['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = indicator_df['Adj Close'].ewm(span=26, adjust=False).mean()
        indicator_df['MACD'] = exp1 - exp2
        indicator_df['Signal'] = indicator_df['MACD'].ewm(span=9, adjust=False).mean()
        indicator_df['Histogram'] = indicator_df['MACD'] - indicator_df['Signal']

        # RSI (14) - per stock
        delta = indicator_df['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        indicator_df['RSI'] = 100 - (100 / (1 + rs))

        # Moving Averages - per stock
        for window in [10, 20, 50, 100, 200]:
            indicator_df[f'MA{window}'] = indicator_df['Adj Close'].rolling(window=window).mean()

        result_dfs.append(indicator_df)

    return pd.concat(result_dfs, ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)


def create_sequences(x, y, seq_len, pred_len, stride=1):
    """Create sequences for direct prediction without masking."""
    X_seq, y_seq = [], []
    for i in range(0, len(x) - seq_len - pred_len + 1, stride):
        x_window = x[i:i + seq_len]
        y_window = y[i + seq_len:i + seq_len + pred_len]

        X_seq.append(x_window)
        y_seq.append(y_window)

    return np.array(X_seq), np.array(y_seq)


def modwt_decompose(X, wavelet='db4', level=None):
    """Decompose using MODWT (Maximal Overlap Discrete Wavelet Transform)."""
    N, T, F = X.shape
    coeffs_all = []
    for i in range(N):
        sample_coeffs = []
        for j in range(F):
            coeffs = pywt.swt(X[i, :, j], wavelet=wavelet, level=level, norm=True)
            cA_last = coeffs[-1][0]  # approximation
            cDs = [cD for _, cD in coeffs]  # all details
            sample_coeffs.append(np.stack([cA_last] + cDs, axis=-1))  # shape [T, level+1]
        sample_coeffs = np.stack(sample_coeffs, axis=-2)  # [T, F, level+1]
        coeffs_all.append(sample_coeffs)
    return np.stack(coeffs_all, axis=0)  # [N, T, F, level+1]


def load_data_pooled(df, seq_len, pred_len, stride, level,
                          train_stocks, val_stocks, test_stocks,
                          clip_value):
    """
    Load and prepare data using stock-based splitting strategy.
    Args:
        df: DataFrame with stock data
        seq_len: sequence length
        pred_len: prediction length
        stride: stride for sequence creation
        level: wavelet decomposition level
        train_stocks: number of stocks for training (default 80)
        val_stocks: number of stocks for validation (default 10)
        test_stocks: number of stocks for testing (default 10)
        clip_value: clipping value for features
    """
    # Add all technical features
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    df = add_pivot_features(df)
    df = calculate_foundation(df)

    df['Date'] = pd.to_datetime(df['Date'])

    # Get unique symbols and shuffle them for random assignment
    symbols = df['Symbol'].unique()
    np.random.shuffle(symbols)

    # Split symbols into train/val/test
    train_symbols = symbols[:train_stocks]
    val_symbols = symbols[train_stocks:train_stocks + val_stocks]
    test_symbols = symbols[train_stocks + val_stocks:train_stocks + val_stocks + test_stocks]

    print(f"Stock distribution:")
    print(f"Train: {len(train_symbols)} stocks")
    print(f"Val: {len(val_symbols)} stocks")
    print(f"Test: {len(test_symbols)} stocks")

    train_dfs, val_dfs, test_dfs = [], [], []

    # Define feature columns (all technical indicators)
    base_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
    technical_features = ['OBV', 'MACD', 'Signal', 'Histogram', 'RSI'] + \
                         [f'MA{w}' for w in [10, 20, 50, 100, 200]]
    pivot_features = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                      'dist_to_S1', 'dist_to_S2', 'dist_to_S3']
    foundation_features = ['Short_Term_Foundation_Days', 'Long_Term_Foundation_Days']

    all_features = base_features + technical_features + pivot_features + foundation_features

    def process_symbol_group(symbols_list, split_name):
        """Process a group of symbols for a specific split"""
        processed_dfs = []

        for symbol in symbols_list:
            group = df[df['Symbol'] == symbol].copy()
            group = group.set_index('Date').sort_index()

            if len(group) < (seq_len + pred_len):
                continue

            # Use percent change for base features
            features = group[base_features].copy()
            features = features.pct_change().replace([np.inf, -np.inf], np.nan)

            # Add technical indicators (already calculated)
            for feature in technical_features + pivot_features + foundation_features:
                if feature in group.columns:
                    features[feature] = group[feature]

            # Create target (detrended returns)
            returns = group['Adj Close'].pct_change().replace([np.inf, -np.inf], np.nan)
            trend = returns.rolling(window=10, min_periods=1).mean()
            target = returns - trend

            # Combine features and target
            data = features.join(pd.DataFrame({'target': target}), how='inner').dropna()

            # Clean and clip features
            for col in data.columns:
                if col != 'target':
                    data[col] = data[col].ffill().bfill().fillna(0)
                    if col in base_features:  # Only clip base features (percent changes)
                        lower = data[col].quantile(0.01)
                        upper = data[col].quantile(0.99)
                        data[col] = data[col].clip(lower, upper)

            if len(data) >= seq_len + pred_len:
                processed_dfs.append(data)

        return processed_dfs

    # Process each split
    atrain_dfs = process_symbol_group(train_symbols, 'train')
    aval_dfs = process_symbol_group(val_symbols, 'val')
    atest_dfs = process_symbol_group(test_symbols, 'test')

    def time_cut(df_list, train_end_date, val_end_date):
        train, val, test = [], [], []
        for df in df_list:
            df = df.sort_index()
            df_train = df[df.index <= train_end_date]
            df_val = df[(df.index > train_end_date) & (df.index <= val_end_date)]
            df_test = df[df.index > val_end_date]

            if len(df_train) >= seq_len + pred_len:
                train.append(df_train)
            if len(df_val) >= seq_len + pred_len:
                val.append(df_val)
            if len(df_test) >= seq_len + pred_len:
                test.append(df_test)
        return train, val, test

    train_dfs, val_dfs, test_dfs = time_cut(atrain_dfs + aval_dfs + atest_dfs,
                                            train_end_date='2022-12-31',
                                            val_end_date='2023-12-31')

    def make_sequences(df_list):
        """Create sequences from dataframe list."""
        X_set, y_set = [], []

        if not df_list:
            return np.array([]), np.array([])

        feat_cols = [col for col in df_list[0].columns if col != 'target']

        for df in df_list:
            x = df[feat_cols].values
            y = df['target'].values
            X_seq, y_seq = create_sequences(x, y, seq_len, pred_len, stride)
            if X_seq.shape[0] > 0:
                X_set.append(X_seq)
                y_set.append(y_seq)

        if not X_set:
            return np.array([]), np.array([])

        return np.concatenate(X_set, axis=0), np.concatenate(y_set, axis=0)

    # Create sequences for ALL splits first
    all_dfs = atrain_dfs + aval_dfs + atest_dfs
    X_all, y_all = make_sequences(all_dfs)

    # Check if we have any data
    if X_all.size == 0:
        raise ValueError("No training data created. Check your data and parameters.")

    # Normalize using ALL data statistics
    X_mean = np.mean(X_all, axis=(0, 1), keepdims=True)
    X_std = np.std(X_all, axis=(0, 1), keepdims=True) + 1e-8
    X_all_normalized = (np.clip(X_all, -clip_value, clip_value) - X_mean) / X_std

    # Normalize targets using ALL data
    y_mean = np.mean(y_all)
    y_std = np.std(y_all) + 1e-8
    y_all_normalized = (y_all - y_mean) / y_std

    # Now create individual splits from normalized data
    X_train, y_train = make_sequences(train_dfs)
    X_valid, y_valid = make_sequences(val_dfs)
    X_test, y_test = make_sequences(test_dfs)

    # Apply same normalization to each split
    X_train = (np.clip(X_train, -clip_value, clip_value) - X_mean) / X_std
    X_valid = (np.clip(X_valid, -clip_value, clip_value) - X_mean) / X_std
    X_test = (np.clip(X_test, -clip_value, clip_value) - X_mean) / X_std

    y_train = (y_train - y_mean) / y_std
    y_valid = (y_valid - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Apply wavelet decomposition
    X_train = modwt_decompose(X_train, level=level)
    X_valid = modwt_decompose(X_valid, level=level)
    X_test = modwt_decompose(X_test, level=level)

    print("Data shapes after processing:")
    print(f"X_train: {X_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_valid: {y_valid.shape}")
    print(f"y_test: {y_test.shape}")

    print(f"Feature count: {X_train.shape[2]} features")

    return (X_train, y_train, X_valid, y_valid, X_test, y_test,
            y_mean, y_std, X_mean, X_std, train_symbols, val_symbols, test_symbols)


def prepare_backtest_inference_data(df, seq_len, pred_len, level,
                                    start_date='2024-01-01'):
    """
    Prepare data for backtesting and inference from a single stock.

    Args:
        df: DataFrame with single stock data (assumes already has technical indicators)
        seq_len: sequence length for model input
        pred_len: prediction length
        level: wavelet decomposition level
        start_date: start date for backtesting/inference (default '2024-01-01')

    Returns:
        X_sequences: input sequences for model [N, seq_len, features, level+1]
        y_sequences: target sequences (padded with 0 for final sequence) [N, pred_len]
        dates: corresponding dates for each sequence
        is_inference: boolean array indicating which sequences are for inference (no true targets)
        X_mean, X_std: computed normalization parameters for features
        y_mean, y_std: computed normalization parameters for targets
    """

    # Ensure we have all required columns and apply same preprocessing as training
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # DON'T filter by start_date yet - need full history for proper calculations

    if len(df) < seq_len:
        raise ValueError(f"Not enough data points. Need at least {seq_len}, got {len(df)}")

    # Define same feature columns as in training
    base_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
    technical_features = ['OBV', 'MACD', 'Signal', 'Histogram', 'RSI'] + \
                         [f'MA{w}' for w in [10, 20, 50, 100, 200]]
    pivot_features = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                      'dist_to_S1', 'dist_to_S2', 'dist_to_S3']
    foundation_features = ['Short_Term_Foundation_Days', 'Long_Term_Foundation_Days']

    all_features = base_features + technical_features + pivot_features + foundation_features

    # Apply same preprocessing as training
    features = df[base_features].copy()
    features = features.pct_change().replace([np.inf, -np.inf], np.nan)

    # Add technical indicators
    for feature in technical_features + pivot_features + foundation_features:
        if feature in df.columns:
            features[feature] = df[feature]

    # Create target (same as training)
    returns = df['Adj Close'].pct_change().replace([np.inf, -np.inf], np.nan)
    trend = returns.rolling(window=10, min_periods=1).mean()
    target = returns - trend

    # Combine and clean
    data = features.join(pd.DataFrame({'target': target}), how='inner').dropna()

    # Clean and clip features (same as training)
    for col in data.columns:
        if col != 'target':
            data[col] = data[col].ffill().bfill().fillna(0)
            if col in base_features:  # Only clip base features (percent changes)
                lower = data[col].quantile(0.01)
                upper = data[col].quantile(0.99)
                data[col] = data[col].clip(lower, upper)

    # NOW filter from start_date onwards - after all preprocessing is done
    start_date = pd.to_datetime(start_date)
    data = data[data.index >= start_date]

    if len(data) < seq_len:
        raise ValueError(f"Not enough data after filtering by start_date. Need at least {seq_len}, got {len(data)}")

    # Create sequences
    feat_cols = [col for col in data.columns if col != 'target']
    x = data[feat_cols].values
    y = data['target'].values
    dates = data.index

    X_sequences = []
    y_sequences = []
    sequence_dates = []
    is_inference = []

    # Create sequences with stride=1 for backtesting
    for i in range(len(x) - seq_len + 1):
        # Input sequence
        x_seq = x[i:i + seq_len]
        X_sequences.append(x_seq)

        # Target sequence
        if i + seq_len + pred_len <= len(y):
            # Normal case: we have future targets
            y_seq = y[i + seq_len:i + seq_len + pred_len]
            is_inference.append(False)
        else:
            # Inference case: pad with zeros
            available_targets = len(y) - (i + seq_len)
            if available_targets > 0:
                y_seq = np.concatenate([
                    y[i + seq_len:],
                    np.zeros(pred_len - available_targets)
                ])
            else:
                y_seq = np.zeros(pred_len)
            is_inference.append(True)

        y_sequences.append(y_seq)

        # Date for this sequence (end of input sequence)
        sequence_dates.append(dates[i + seq_len - 1])

    # Convert to arrays
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    sequence_dates = np.array(sequence_dates)
    is_inference = np.array(is_inference)

    # Compute normalization parameters from this data
    X_mean = np.mean(X_sequences, axis=(0, 1), keepdims=True)
    X_std = np.std(X_sequences, axis=(0, 1), keepdims=True) + 1e-8

    # For targets, only use non-inference sequences for computing stats
    y_backtest = y_sequences[~is_inference] if np.any(~is_inference) else y_sequences
    y_mean = np.mean(y_backtest)
    y_std = np.std(y_backtest) + 1e-8

    # Apply normalization (no additional clipping needed since already done per feature)
    X_sequences = (X_sequences - X_mean) / X_std
    y_sequences = (y_sequences - y_mean) / y_std

    # Apply wavelet decomposition
    X_sequences = modwt_decompose(X_sequences, level=level)

    # print(f"Created {len(X_sequences)} sequences:")
    # print(f"- Backtest sequences (with true targets): {np.sum(~is_inference)}")
    # print(f"- Inference sequences (padded targets): {np.sum(is_inference)}")
    # print(f"X_sequences shape: {X_sequences.shape}")
    # print(f"y_sequences shape: {y_sequences.shape}")
    # print(f"Applied quantile clipping (0.01, 0.99) to base features")
    # print(f"Computed normalization stats from input data")

    return X_sequences, y_sequences, sequence_dates, is_inference, X_mean, X_std, y_mean, y_std