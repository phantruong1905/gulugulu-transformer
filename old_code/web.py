import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import torch
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px


# Disable matplotlib GUI backend for Streamlit
plt.switch_backend('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page config
st.set_page_config(page_title="Gulugulu Rô Bốt", layout="wide")

# Styling
st.markdown(
    """
    <h1 style='text-align: left; color: #FF4B4B; margin-bottom: 0;'>🤖 Robot Gulugulu</h1>
    <hr style='margin-top: 0;'>
    """,
    unsafe_allow_html=True
)


# DB engine
engine = create_engine(
    "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres")


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


def load_latest_buys():
    query = """
        SELECT DISTINCT ON (symbol) *
        FROM stock_trades
        ORDER BY symbol, date DESC
    """
    df = pd.read_sql(query, engine)
    return df




def display_trading_summary_db(symbol, results):
    tab1, tab2, tab3, tab4 = st.tabs(["Trading Summary", "Predictions", "Trades", "Chart"])

    with tab1:
        col = st.columns([1, 6, 1])[1]
        with col:
            st.subheader(f" Trading Summary for {symbol}")
            trades = results['trades']
            buy_trades = trades[trades['action'].str.lower() == 'buy']
            sell_trades = trades[trades['action'].str.lower() == 'sell']
            win_trades = sell_trades[sell_trades['profit_loss'] > 0]

            # Metrics in separate boxes
            metrics = [
                ("Số giao dịch", len(buy_trades)),
                ("Tỷ lệ thắng", f"{len(win_trades) / len(sell_trades) * 100:.1f}%" if len(sell_trades) else "0%"),
                ("Lợi nhuận trung bình", f"{sell_trades['return_pct'].mean() * 100:.2f}%" if len(sell_trades) else "0%"),
                ("Số ngày nắm giữ trung bình", f"{sell_trades['days_held'].mean():.1f} ngày" if len(sell_trades) else "0 ngày")
            ]

            for col, (label, value) in zip(st.columns(4), metrics):
                col.markdown(f"""
                    <div style="background-color:#1c1c1c;padding:20px;border-radius:10px;text-align:center;border:1px solid #444;">
                        <div style="color:#aaa;font-size:16px;">{label}</div>
                        <div style="color:white;font-size:28px;font-weight:bold;margin-top:5px;">{value}</div>
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        col = st.columns([1, 6, 1])[1]
        with col:
            st.subheader(f"Tín hiệu hiện tại {symbol}")
            preds = results['predictions']
            if not preds.empty:
                latest = preds.iloc[-1]
                pred_list = [round(float(latest[f"y_pred_{i}"]), 4) for i in range(1, 6)]
                st.write(pred_list)
            else:
                st.write("No predictions found.")

    with tab3:
        col = st.columns([1, 10, 1])[1]
        with col:
            st.subheader(f"Trade History for {symbol}")
            st.dataframe(results['trades'], use_container_width=True)

    with tab4:
        col = st.columns([1, 10, 1])[1]
        with col:

            prices = results['prices']
            trades = results['trades']
            prices['Date'] = pd.to_datetime(prices['Date'])
            trades['date'] = pd.to_datetime(trades['date'])

            # Pair trades: assume buys and sells alternate correctly
            buy_trades = trades[trades['action'].str.lower() == 'buy'].reset_index(drop=True)
            sell_trades = trades[trades['action'].str.lower() == 'sell'].reset_index(drop=True)

            # Label only buys: T1, T2,...
            buy_trades = buy_trades.copy()
            buy_trades['label'] = ['T' + str(i + 1) for i in range(len(buy_trades))]

            fig = px.line(prices, x='Date', y='Adj Close', title=f'{symbol} Price Chart')

            if not buy_trades.empty:
                fig.add_scatter(
                    x=buy_trades['date'], y=buy_trades['price'], mode='markers+text',
                    marker=dict(color='green', symbol='triangle-up', size=10),
                    name='BUY', text=buy_trades['label'], textposition='top center'
                )

            if not sell_trades.empty:
                fig.add_scatter(
                    x=sell_trades['date'], y=sell_trades['price'], mode='markers',
                    marker=dict(color='red', symbol='triangle-down', size=10),
                    name='SELL'
                )

            st.plotly_chart(fig, use_container_width=True)

            # 📉 Bar chart with return_pct
            if not sell_trades.empty:

                # Attach T-labels to sell trades for x-axis
                labeled_sells = sell_trades.copy()
                labeled_sells['label'] = buy_trades['label'][:len(sell_trades)]

                # Bar chart with return % label
                labeled_sells['color'] = labeled_sells['return_pct'].apply(lambda x: 'green' if x > 0 else 'red')
                labeled_sells['text'] = labeled_sells['return_pct'].apply(lambda x: f"{x * 100:.2f}%")

                fig_bar = px.bar(
                    labeled_sells,
                    x='label',
                    y='return_pct',
                    title="Lợi nhuận từng lệnh",
                    labels={'return_pct': 'Return (%)'},
                    text='text'
                )

                fig_bar.update_traces(marker_color=labeled_sells['color'], texttemplate='%{text}',
                                      textposition='outside')
                fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

                # 🔧 Add this below
                fig_bar.update_layout(margin=dict(t=50))  # prevent top label being cut

                st.plotly_chart(fig_bar, use_container_width=True)


def fetch_backtest_results(symbol):
    trades = pd.read_sql(
        f'SELECT * FROM stock_trades WHERE symbol = %s ORDER BY date',
        engine, params=(symbol,)
    )
    predictions = pd.read_sql(
        f'SELECT * FROM daily_signals WHERE symbol = %s ORDER BY date',
        engine, params=(symbol,)
    )
    prices = pd.read_sql(
        'SELECT * FROM raw_stock_data WHERE "Symbol" = %s ORDER BY "Date"',
        engine, params=(symbol,)
    )

    return {
        'trades': trades,
        'predictions': predictions,
        'prices': prices
    }


def handle_symbol_click(symbol):
    with st.spinner(f"Đang tải data cho {symbol}... Chờ mình một chút nhé..."):
        try:
            results = fetch_backtest_results(symbol)

            if results['trades'].empty:
                st.warning("Không có dữ liệu giao dịch cho mã này.")
            else:
                display_trading_summary_db(symbol, results)

        except Exception as e:
            st.error(f"❌ Error during execution for {symbol}: {str(e)}")
            st.exception(e)



def main():
    df = load_latest_buys()
    buy_df = df[df['action'].str.lower() == 'buy'][['symbol', 'date', 'price', 'signal_strength']]
    buy_df = buy_df.sort_values(by='date', ascending=False)

    if not buy_df.empty:

        # 💡 Put the buttons in their own container so layout doesn’t mess with the rest
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            with col1:
                st.markdown(
                    "<div style='background-color:rgba(0, 255, 100, 0.2);padding:10px;border-radius:5px;color:white;font-weight:bold;text-align:left'>Symbol</div>",
                    unsafe_allow_html=True)
            with col2:
                st.markdown(
                    "<div style='background-color:rgba(0, 255, 100, 0.2);padding:10px;border-radius:5px;color:white;font-weight:bold;text-align:left'>Date</div>",
                    unsafe_allow_html=True)
            with col3:
                st.markdown(
                    "<div style='background-color:rgba(0, 255, 100, 0.2);padding:10px;border-radius:5px;color:white;font-weight:bold;text-align:left'>Price</div>",
                    unsafe_allow_html=True)
            with col4:
                st.markdown(
                    "<div style='background-color:rgba(0, 255, 100, 0.2);padding:10px;border-radius:5px;color:white;font-weight:bold;text-align:left'>Signal Strength</div>",
                    unsafe_allow_html=True)

            st.divider()

            for index, row in buy_df.iterrows():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
                with col1:
                    if st.button(row['symbol'], key=f"btn_{row['symbol']}_{index}"):
                        st.session_state['selected_symbol'] = row['symbol']
                with col2: st.write(str(row['date']))
                with col3: st.write(f"${row['price']:.2f}")
                with col4: st.write(f"{row['signal_strength']:.4f}")
                st.divider()

        # 🧠 Trigger the symbol selection
        selected_symbol = st.session_state.get('selected_symbol', None)
        if selected_symbol:
            handle_symbol_click(selected_symbol)

    else:
        st.warning("No active BUY signals at the moment.")


if __name__ == "__main__":
    main()