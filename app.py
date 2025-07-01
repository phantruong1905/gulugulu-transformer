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

    # Fetch data
    fetch_stock_data(
        data_path=data_path,
        stocks=stocks_to_test,
        start_date="2015-01-01",
        end_date=current_date,
        output_filename=pooled_data_filename
    )

    df_all_stocks = pd.read_csv(pooled_data_path)

    # Filter data for the specific ticker and add technical indicators
    df_filtered = df_all_stocks[df_all_stocks['Symbol'] == ticker].copy()

    # Add technical indicators to the filtered data
    df_filtered = add_technical_indicators(df_filtered)
    df_filtered = add_pivot_features(df_filtered)
    df_filtered = calculate_foundation(df_filtered)

    # Step 2: Prepare data for backtesting and inference using the new function
    try:
        X_sequences, y_sequences, sequence_dates, is_inference, X_mean, X_std, y_mean, y_std = prepare_backtest_inference_data(
            df_filtered,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            level=2,
            start_date='2024-01-01'
        )
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None

    # Step 3: Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_sequences.shape[2]
    wavelet_levels = X_sequences.shape[-1]
    model = CleanWaveletTransformer(
        input_dim=input_dim,
        wavelet_levels=wavelet_levels,
        d_model=64,
        nhead=2,
        num_layers=3,
        drop_out=0.5,
        pred_len=config.pred_len
    )

    # Step 4: Use the existing test function to get predictions
    checkpoint_path = "trained_wavelet_model.pt"
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint '{checkpoint_path}' not found. Please train the model first.")
        return None

    try:
        # Use the existing test_wavelet_model function
        all_preds, all_targets = test_wavelet_model(
            df_filtered, model, X_sequences, y_sequences, config, device, checkpoint_path
        )

        # Convert to numpy if they're tensors
        if hasattr(all_preds, 'numpy'):
            predictions = all_preds.numpy()
        else:
            predictions = all_preds

        if hasattr(all_targets, 'numpy'):
            targets = all_targets.numpy()
        else:
            targets = all_targets

        # Unscale predictions and targets
        predictions_unscaled = predictions * y_std + y_mean
        y_sequences_unscaled = targets * y_std + y_mean

    except Exception as e:
        st.error(f"Error running test function: {str(e)}")
        return None

    # Step 5: Prepare prediction sequences for trading
    prediction_sequences = []

    for i in range(len(predictions_unscaled)):
        pred_array = predictions_unscaled[i]
        if isinstance(pred_array, np.ndarray):
            pred_list = pred_array.flatten().tolist()
        elif isinstance(pred_array, (int, float)):
            pred_list = [pred_array]
        else:
            pred_list = list(pred_array)

        prediction_sequences.append({
            "date": str(sequence_dates[i].date()) if hasattr(sequence_dates[i], 'date') else str(sequence_dates[i]),
            "symbol": ticker,
            "y_pred": pred_list
        })

    # Step 6: Run trading simulation
    trader = SimpleTradingAlgorithm(
        min_hold_days=5,
        max_hold_days=10,
        strong_signal_threshold=0.04,
        stop_loss=-0.05
    )

    summary = trader.run_trading(prediction_sequences, df_filtered)
    trades_df = pd.DataFrame(trader.trade_history)

    # Separate backtest and inference results
    backtest_mask = ~is_inference
    inference_mask = is_inference

    backtest_predictions = predictions_unscaled[backtest_mask] if np.any(backtest_mask) else np.array([])
    backtest_targets = y_sequences_unscaled[backtest_mask] if np.any(backtest_mask) else np.array([])
    backtest_dates = sequence_dates[backtest_mask] if np.any(backtest_mask) else np.array([])

    inference_predictions = predictions_unscaled[inference_mask] if np.any(inference_mask) else np.array([])
    inference_dates = sequence_dates[inference_mask] if np.any(inference_mask) else np.array([])

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
        'test_start_date': backtest_dates[0] if len(backtest_dates) > 0 else sequence_dates[0]
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