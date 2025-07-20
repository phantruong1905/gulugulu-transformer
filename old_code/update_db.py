import streamlit as st
import torch
import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy
from datetime import datetime
from sqlalchemy import create_engine, text
import pickle
import base64

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

# Database connection
DATABASE_URL = "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres"


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


def get_db_engine():
    """Create database engine"""
    return create_engine(DATABASE_URL)


def save_results_to_db(ticker, run_date, results):
    """Save all results to PostgreSQL database using pandas to_sql"""
    engine = get_db_engine()

    # Prepare trading summary DataFrame
    summary = results['summary']
    summary_df = pd.DataFrame([{
        'symbol': ticker,
        'run_date': run_date,
        'total_trades': summary.get('total_trades', 0),
        'winning_trades': len([t for t in results['trades'].to_dict('records') if t.get('profit_loss', 0) > 0]) if not
        results['trades'].empty else 0,
        'losing_trades': len([t for t in results['trades'].to_dict('records') if t.get('profit_loss', 0) <= 0]) if not
        results['trades'].empty else 0,
        'win_rate': summary.get('win_rate', 0),
        'total_return': summary.get('total_return', 0),
        'max_drawdown': summary.get('max_drawdown', 0),
        'sharpe_ratio': summary.get('sharpe_ratio', 0),
        'created_at': datetime.now()
    }])

    # Save trading summary
    summary_df.to_sql('trading_summary', engine, if_exists='append', index=False)

    # Prepare and save trades DataFrame
    trades_df = results['trades'].copy()
    if not trades_df.empty:
        trades_df['symbol'] = ticker
        trades_df['run_date'] = run_date
        trades_df['created_at'] = datetime.now()

        # Rename columns to match database schema
        trades_df = trades_df.rename(columns={
            'date': 'trade_date',
            'profit_loss': 'return_amount'
        })

        # Add missing columns with default values
        if 'buy_date' not in trades_df.columns:
            trades_df['buy_date'] = trades_df['trade_date']
        if 'sell_date' not in trades_df.columns:
            trades_df['sell_date'] = trades_df['trade_date']
        if 'buy_price' not in trades_df.columns:
            trades_df['buy_price'] = trades_df['price']
        if 'sell_price' not in trades_df.columns:
            trades_df['sell_price'] = trades_df['price']
        if 'quantity' not in trades_df.columns:
            trades_df['quantity'] = 1
        if 'return_pct' not in trades_df.columns:
            trades_df['return_pct'] = 0
        if 'signal_strength' not in trades_df.columns:
            trades_df['signal_strength'] = 0

        # Select only the columns we need
        trades_columns = ['symbol', 'run_date', 'buy_date', 'sell_date', 'buy_price',
                          'sell_price', 'quantity', 'return_pct', 'signal_strength', 'created_at']
        trades_df = trades_df[[col for col in trades_columns if col in trades_df.columns]]

        trades_df.to_sql('trades', engine, if_exists='append', index=False)

    # Prepare and save backtest results
    backtest = results['backtest_results']
    if len(backtest['dates']) > 0:
        backtest_data = []
        for i, date in enumerate(backtest['dates']):
            pred_data = backtest['y_pred'][i] if i < len(backtest['y_pred']) else []
            target_data = backtest['y_true'][i] if i < len(backtest['y_true']) else []

            backtest_data.append({
                'symbol': ticker,
                'run_date': run_date,
                'prediction_date': date,
                'predictions': str(pred_data.tolist() if hasattr(pred_data, 'tolist') else pred_data),
                'targets': str(target_data.tolist() if hasattr(target_data, 'tolist') else target_data),
                'created_at': datetime.now()
            })

        backtest_df = pd.DataFrame(backtest_data)
        backtest_df.to_sql('backtest_results', engine, if_exists='append', index=False)

    # Prepare and save inference results
    inference = results['inference_results']
    if len(inference['dates']) > 0:
        inference_data = []
        for i, date in enumerate(inference['dates']):
            pred_data = inference['predictions'][i] if i < len(inference['predictions']) else []

            inference_data.append({
                'symbol': ticker,
                'run_date': run_date,
                'prediction_for_date': date,
                'predictions': str(pred_data.tolist() if hasattr(pred_data, 'tolist') else pred_data),
                'created_at': datetime.now()
            })

        inference_df = pd.DataFrame(inference_data)
        inference_df.to_sql('inference_results', engine, if_exists='append', index=False)

    # Prepare and save stock data
    df_filtered = results['df_filtered'].copy()
    df_filtered['symbol'] = ticker
    df_filtered['run_date'] = run_date
    df_filtered['created_at'] = datetime.now()

    # Rename columns to match database schema
    df_filtered = df_filtered.rename(columns={
        'Date': 'date',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Volume': 'volume'
    })

    # Create data_blob (serialized DataFrame for first row only to save space)
    df_serialized = base64.b64encode(pickle.dumps(df_filtered)).decode()
    df_filtered['data_blob'] = ''
    df_filtered.iloc[0, df_filtered.columns.get_loc('data_blob')] = df_serialized

    # Select only the columns we need for the database
    stock_columns = ['symbol', 'run_date', 'date', 'open_price', 'high_price',
                     'low_price', 'close_price', 'volume', 'data_blob', 'created_at']
    df_filtered = df_filtered[[col for col in stock_columns if col in df_filtered.columns]]

    df_filtered.to_sql('stock_data', engine, if_exists='append', index=False)


def run_single_stock_backtest(ticker, current_date):
    """Run backtest for a single stock"""
    config = Config()

    # Step 1: Define stock list and fetch/load pooled data
    data_path = '../data'
    os.makedirs(data_path, exist_ok=True)

    stocks_to_test = [ticker]
    pooled_data_filename = f"{ticker}_stock_data.csv"
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

    # Step 2: Prepare data for backtesting and inference
    X_sequences, y_sequences, sequence_dates, is_inference, X_mean, X_std, y_mean, y_std = prepare_backtest_inference_data(
        df_filtered,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        level=2,
        start_date='2024-01-01'
    )

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
    checkpoint_path = "../trained_wavelet_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint '{checkpoint_path}' not found. Please train the model first.")

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


def run_multi_stock_backtest(stock_list, current_date):
    """Run backtest for multiple stocks and save to database"""

    run_date = pd.to_datetime(current_date).date()
    results_summary = []

    for i, ticker in enumerate(stock_list):
        try:
            st.write(f"Processing {ticker} ({i + 1}/{len(stock_list)})...")

            # Run backtest for single stock
            results = run_single_stock_backtest(ticker, current_date)

            # Save results to database using to_sql
            save_results_to_db(ticker, run_date, results)

            # Add to summary
            results_summary.append({
                'ticker': ticker,
                'status': 'Success',
                'total_trades': results['summary'].get('total_trades', 0),
                'total_return': results['summary'].get('total_return', 0),
                'win_rate': results['summary'].get('win_rate', 0)
            })

            st.success(f"✅ {ticker} completed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing {ticker}: {str(e)}")
            results_summary.append({
                'ticker': ticker,
                'status': f'Error: {str(e)}',
                'total_trades': 0,
                'total_return': 0,
                'win_rate': 0
            })

    return results_summary


# Streamlit App
st.title("📈 Multi-Stock Backtest Processor")
st.write("Process multiple stocks and save results to PostgreSQL database")

# Input section
st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    stock_input = st.text_area(
        "Enter stock symbols (one per line):",
        value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN",
        height=150
    )

with col2:
    current_date = st.date_input(
        "Current Date:",
        pd.to_datetime("today")
    ).strftime("%Y-%m-%d")

# Process stocks
if st.button("🚀 Run Multi-Stock Backtest", type="primary"):
    if not stock_input.strip():
        st.error("Please enter at least one stock symbol!")
    else:
        # Parse stock list
        stock_list = [ticker.strip().upper() for ticker in stock_input.split('\n') if ticker.strip()]

        st.write(f"Processing {len(stock_list)} stocks: {', '.join(stock_list)}")

        with st.spinner("Running multi-stock backtest..."):
            results_summary = run_multi_stock_backtest(stock_list, current_date)

        # Display results summary
        st.header("📊 Processing Summary")
        summary_df = pd.DataFrame(results_summary)
        st.dataframe(summary_df)

        # Show success/failure counts
        success_count = len([r for r in results_summary if r['status'] == 'Success'])
        failure_count = len(results_summary) - success_count

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(stock_list))
        with col2:
            st.metric("Successful", success_count)
        with col3:
            st.metric("Failed", failure_count)

        st.success(f"✅ Multi-stock backtest completed! Results saved to database.")

# Database status
st.sidebar.header("Database Status")
try:
    engine = get_db_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        st.sidebar.success("🟢 Database Connected")
except Exception as e:
    st.sidebar.error(f"🔴 Database Error: {str(e)}")

# Show recent runs
st.sidebar.header("Recent Runs")
try:
    engine = get_db_engine()
    with engine.connect() as conn:
        recent_runs = conn.execute(text("""
            SELECT symbol, run_date, total_trades, total_return, win_rate
            FROM trading_summary 
            ORDER BY created_at DESC 
            LIMIT 10
        """)).fetchall()

        if recent_runs:
            for run in recent_runs:
                st.sidebar.write(f"**{run[0]}** ({run[1]}) - {run[2]} trades, {run[3]:.2%} return")
        else:
            st.sidebar.write("No recent runs")
except Exception as e:
    st.sidebar.error(f"Error loading recent runs: {str(e)}")

run_single_stock_backtest('FTS', '2025-07-01')