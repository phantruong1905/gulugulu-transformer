import streamlit as st
import torch
import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import your modules (make sure these paths are correct)
from src.load_data import fetch_stock_data
from src.feature_engineering import *
from model.dual_transformer import *
from scripts.train_test_dual_transformer import *
from scripts.test_backtest import SimpleTradingAlgorithm

# Disable matplotlib GUI backend for Streamlit
plt.switch_backend('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


# Streamlit app
st.title("📈 Rô Bốt GULUGULU")

# Sidebar for inputs
ticker = st.sidebar.text_input("Nhập mã để backtest và dự đoán:").strip().upper()
current_date = st.sidebar.date_input("Ngày hôm nay", pd.to_datetime("today")).strftime("%Y-%m-%d")


def run_backtest(ticker, current_date):
    config = Config()

    # Step 1: Define stock list and fetch/load pooled data
    data_path = './data'
    os.makedirs(data_path, exist_ok=True)

    stocks_to_test = [ticker]
    pooled_data_filename = "test_stock_data.csv"
    pooled_data_path = os.path.join(data_path, pooled_data_filename)

    # Force re-fetching for fresh data
    if os.path.exists(pooled_data_path):
        os.remove(pooled_data_path)

    # Fetch data from 2017 (more history for better features)
    fetch_stock_data(
        data_path=data_path,
        stocks=stocks_to_test,
        start_date="2017-01-01",  # Changed to 2017
        end_date=current_date,
        output_filename=pooled_data_filename
    )

    df_all_stocks = pd.read_csv(pooled_data_path)

    # Filter data for the specific ticker and add technical indicators
    df_filtered = df_all_stocks[df_all_stocks['Symbol'] == ticker].copy()
    df_filtered = add_technical_indicators(df_filtered)
    df_filtered = add_pivot_features(df_filtered)
    df_filtered = calculate_foundation(df_filtered)

    # Convert dates and sort
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)

    # Step 2: Sequential daily processing
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime(current_date)

    # Initialize device and model ONCE at the beginning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "trained_wavelet_model.pt"
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint '{checkpoint_path}' not found. Please train the model first.")
        return None

    # Load model checkpoint first to get dimensions
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Initialize model outside the loop
    model = None
    model_initialized = False

    # Initialize trader
    trader = SimpleTradingAlgorithm(
        min_hold_days=2,
        max_hold_days=10,
        strong_signal_threshold=0.04,
        stop_loss=-0.05
    )

    # Results storage
    all_predictions = []
    all_targets = []
    all_dates = []
    prediction_sequences = []

    # Process each day sequentially
    current_test_date = start_date
    while current_test_date <= end_date:
        # Get data only up to current test date
        data_up_to_date = df_filtered[df_filtered['Date'] <= current_test_date].copy()

        if len(data_up_to_date) < config.seq_len + 10:  # Need minimum data
            current_test_date += pd.Timedelta(days=1)
            continue

        try:
            # Create single sequence ending at current_test_date
            # Check if we have enough data for a sequence
            if len(data_up_to_date) < config.seq_len:
                current_test_date += pd.Timedelta(days=1)
                continue

            # Manually create single sequence for efficiency
            # Apply same preprocessing as in prepare_backtest_inference_data
            df_work = data_up_to_date.copy()
            df_work['Date'] = pd.to_datetime(df_work['Date'])
            df_work = df_work.set_index('Date').sort_index()

            # Define features (same as in prepare_backtest_inference_data)
            base_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
            technical_features = ['OBV', 'MACD', 'Signal', 'Histogram', 'RSI'] + \
                                 [f'MA{w}' for w in [10, 20, 50, 100, 200]]
            pivot_features = ['dist_to_P', 'dist_to_R1', 'dist_to_R2', 'dist_to_R3',
                              'dist_to_S1', 'dist_to_S2', 'dist_to_S3']
            foundation_features = ['Short_Term_Foundation_Days', 'Long_Term_Foundation_Days']

            # Apply same preprocessing
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
                current_test_date += pd.Timedelta(days=1)
                continue

            # Create single sequence (last seq_len days)
            feat_cols = [col for col in data.columns if col != 'target']
            x = data[feat_cols].values
            y = data['target'].values

            # Take the last sequence
            X_seq = x[-config.seq_len:].reshape(1, config.seq_len, -1)  # [1, seq_len, features]
            y_seq = y[-1:].reshape(1, 1)  # [1, 1] - just current target
            seq_date = data.index[-1]  # Current date

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
                model.to(device)  # Move model to device
                model.eval()
                model_initialized = True

            # Make prediction - ENSURE ALL TENSORS ARE ON THE SAME DEVICE
            with torch.no_grad():
                # Convert to tensor and move to device
                X_tensor = torch.FloatTensor(X_seq).to(device)

                # Make prediction
                prediction = model(X_tensor)

                # Move prediction back to CPU for numpy conversion
                prediction = prediction.cpu().numpy()

            # Unscale prediction
            prediction_unscaled = prediction * y_std + y_mean
            y_unscaled = y_seq * y_std + y_mean

            # Store results
            all_predictions.append(prediction_unscaled[0])
            all_targets.append(y_unscaled[0])
            all_dates.append(seq_date)

            # Create prediction sequence for trading
            pred_array = prediction_unscaled[0]
            if isinstance(pred_array, np.ndarray):
                pred_list = pred_array.flatten().tolist()
            else:
                pred_list = [pred_array]

            prediction_sequences.append({
                "date": str(seq_date.date()) if hasattr(seq_date, 'date') else str(seq_date),
                "symbol": ticker,
                "y_pred": pred_list
            })

            # Process trading decision for this day
            trader.process_prediction(prediction_sequences[-1], data_up_to_date)

        except Exception as e:
            st.warning(f"Skipping date {current_test_date}: {str(e)}")
            logging.error(f"Error processing date {current_test_date}: {str(e)}")

        current_test_date += pd.Timedelta(days=1)

    # Convert results to arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_dates = np.array(all_dates)

    # Create final summary
    summary = trader.get_performance_summary()
    trades_df = pd.DataFrame(trader.trade_history)

    # Separate backtest and inference (last few predictions without true targets)
    # For now, treat all as backtest since we're doing sequential processing
    backtest_predictions = all_predictions
    backtest_targets = all_targets
    backtest_dates = all_dates

    # Inference would be predictions beyond current_date (none in this case)
    inference_predictions = np.array([])
    inference_dates = np.array([])

    return {
        'summary': summary,
        'trades': trades_df,
        'backtest_results': {
            'y_pred': backtest_predictions,
            'y_true': backtest_targets,
            'dates': backtest_dates
        },
        'inference_results': {
            'predictions': inference_predictions,
            'dates': inference_dates
        },
        'trader': trader,
        'df_filtered': df_filtered,
        'test_start_date': backtest_dates[0] if len(backtest_dates) > 0 else start_date
    }


# Run button
if st.sidebar.button("Chạy cho tôi", type="primary"):
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu!")
    else:
        with st.spinner("Đang chạy backtest... Chờ mình một chút nhé..."):
            try:
                # Run the backtest
                results = run_backtest(ticker, current_date)

                if results is None:
                    st.error("Failed to run backtest. Please check the logs above.")
                else:
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Trading Summary", "Predictions", "Trades", "Chart"])

                    with tab1:
                        results['trader'].streamlit_trading_summary(results['trades'])

                    with tab2:
                        if len(results['inference_results']['predictions']) > 0:
                            inf_preds = results['inference_results']['predictions']

                            # TAKE ONLY THE LAST SEQUENCE (most recent prediction)
                            latest_inf_pred = inf_preds[-1]  # Last inference sequence

                            # Convert to list and round
                            pred_list = latest_inf_pred.flatten() if hasattr(latest_inf_pred,
                                                                             'flatten') else latest_inf_pred
                            pred_list = [round(float(x), 4) for x in pred_list]

                            # Display as a simple list
                            st.write(f"**{ticker} - Dự đoán % thay đổi so với MA10 trong 5 ngày tới:**")
                            st.write(pred_list)
                        else:
                            st.write("No future predictions available")

                    with tab3:
                        if not results['trades'].empty:
                            st.dataframe(results['trades'])
                        else:
                            st.write("No trades executed")

                    with tab4:
                        try:
                            # Get the plotting function from trader
                            if len(results['backtest_results']['dates']) > 0:
                                plot_end_date = results['backtest_results']['dates'][-1]
                            else:
                                plot_end_date = pd.to_datetime(current_date)

                            # Use backtest predictions for plotting
                            plot_predictions = results['backtest_results']['y_pred'] if len(
                                results['backtest_results']['y_pred']) > 0 else []

                            # The plot_trading_results method should handle Streamlit display internally
                            results['trader'].plot_trading_results(
                                results['df_filtered'],
                                symbol=ticker,
                                start_date=results['test_start_date'],
                                end_date=plot_end_date,
                                y_pred=plot_predictions
                            )

                        except Exception as e:
                            st.error(f"Error plotting chart: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
                st.exception(e)