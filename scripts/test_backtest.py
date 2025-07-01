import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.express as px
from typing import Union, List
from scipy.stats import linregress
import plotly.graph_objects as go

class SimpleTradingAlgorithm:
    def __init__(self,
                 min_hold_days: int = None,
                 max_hold_days: int = None,
                 strong_signal_threshold: float = None,
                 stop_loss: float = None,
                 portfolio_size: float = 10000):

        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.strong_signal_threshold = strong_signal_threshold
        self.stop_loss = stop_loss

        # Trading state - only one position at a time
        self.current_position = None  # {symbol, entry_date, entry_price, quantity, days_held, max_profit}
        self.cash = portfolio_size
        self.trade_history = []
        self.profit_drawdown_threshold = 0.03  # 5% drawdown from peak profit

    def calculate_signal_strength(self, y_pred: Union[List[float], float, np.ndarray]) -> float:
        """Calculate signal strength if there's consistent upward trend and directional agreement"""

        # Ensure input is a flattened list of 10 elements
        if isinstance(y_pred, (int, float)):
            y_pred = [y_pred] * 10
        elif isinstance(y_pred, np.ndarray):
            y_pred = y_pred.flatten().tolist()
        elif not isinstance(y_pred, list):
            try:
                y_pred = list(y_pred)
            except (TypeError, ValueError):
                y_pred = [float(y_pred)] * 10

        if len(y_pred) < 2 or y_pred[0] <= 0 or y_pred[1] <= 0:
            return 0.0

        # 3. Compute weighted signal
        weights = np.array([0.2, 0.5, 0.2, 0, 0])
        y_pred_truncated = y_pred
        weights = weights[:len(y_pred_truncated)]
        weights /= weights.sum()
        weighted_signal = np.sum(np.array(y_pred_truncated) * weights)

        return weighted_signal

    def should_buy(self, signal_strength: float) -> bool:
        """Determine if we should buy based on signal strength"""
        return (self.current_position is None and  # No current position
                signal_strength > self.strong_signal_threshold)  # Strong positive signal

    def should_sell(self, current_price: float, current_date: str) -> tuple:
        """Determine if we should sell based on holding period, stop loss, and profit drawdown"""
        if self.current_position is None:
            return False, "no_position"

        # Calculate days held
        entry_date = datetime.strptime(self.current_position['entry_date'], '%Y-%m-%d')
        current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
        days_held = (current_date_obj - entry_date).days

        # Calculate current return
        entry_price = self.current_position['entry_price']
        current_return = (current_price - entry_price) / entry_price

        # Update maximum profit seen
        max_profit = self.current_position.get('max_profit', 0)
        if current_return > max_profit:
            max_profit = current_return
            self.current_position['max_profit'] = max_profit

        # Calculate profit drawdown from peak
        profit_drawdown = max_profit - current_return

        # Sell conditions
        if days_held < self.min_hold_days:
            return False, f"min_hold_not_met_{days_held}_days"

        # Stop loss hit
        if current_return <= self.stop_loss:
            return True, f"stop_loss_{current_return:.3f}"

        # Profit drawdown protection (sell if profit drops 5% from peak)
        if max_profit > 0 and profit_drawdown >= self.profit_drawdown_threshold:
            return True, f"profit_drawdown_{profit_drawdown:.3f}_from_peak_{max_profit:.3f}"

        # Maximum holding period reached
        if days_held >= self.max_hold_days:
            return True, f"max_hold_{days_held}_days"

        return False, f"holding_{days_held}_days_profit_{current_return:.3f}_peak_{max_profit:.3f}"

    def execute_buy(self, symbol: str, price: float, date: str, signal_strength: float):
        """Execute buy order"""
        # Use all available cash for the position
        quantity = self.cash / price
        cost = quantity * price

        self.current_position = {
            'symbol': symbol,
            'entry_date': date,
            'entry_price': price,
            'quantity': quantity,
            'signal_strength': signal_strength,
            'max_profit': 0  # Track maximum profit reached
        }

        self.cash = 0  # All cash invested

        self.trade_history.append({
            'date': date,
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'cost': cost,
            'signal_strength': signal_strength
        })

        print(f"BUY: {date} | {symbol} | Price: ${price:.2f} | Qty: {quantity:.0f} | Signal: {signal_strength:.3f}")

    def execute_sell(self, price: float, date: str, reason: str):
        """Execute sell order"""
        if self.current_position is None:
            return

        position = self.current_position
        proceeds = position['quantity'] * price
        profit_loss = proceeds - (position['quantity'] * position['entry_price'])
        return_pct = profit_loss / (position['quantity'] * position['entry_price'])

        # Calculate holding period
        entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
        sell_date = datetime.strptime(date, '%Y-%m-%d')
        days_held = (sell_date - entry_date).days

        self.cash = proceeds

        self.trade_history.append({
            'date': date,
            'action': 'SELL',
            'symbol': position['symbol'],
            'price': price,
            'quantity': position['quantity'],
            'proceeds': proceeds,
            'profit_loss': profit_loss,
            'return_pct': return_pct,
            'days_held': days_held,
            'reason': reason
        })

        print(
            f"SELL: {date} | {position['symbol']} | Price: ${price:.2f} | P&L: ${profit_loss:.0f} | Return: {return_pct:.2%} | Days: {days_held} | Reason: {reason}")

        # Clear position
        self.current_position = None

    def process_prediction(self, sequence_data: Dict, stock_data: pd.DataFrame):
        """Process a single prediction sequence and make trading decisions"""
        date = sequence_data['date']
        y_pred = sequence_data['y_pred']
        # Auto-detect symbol if not in sequence_data
        if 'symbol' in sequence_data:
            symbol = sequence_data['symbol']
        else:
            symbol = stock_data['Symbol'].iloc[0] if 'Symbol' in stock_data.columns else 'UNKNOWN'

        # Get current stock price
        current_stock = stock_data[(stock_data['Date'] == date) & (stock_data['Symbol'] == symbol)]
        if current_stock.empty:
            print(f"No stock data for {symbol} on {date}")
            return

        current_price = current_stock['Adj Close'].iloc[0]

        # Calculate signal strength
        signal_strength = self.calculate_signal_strength(y_pred)

        # Check if we should sell first (if we have a position)
        if self.current_position is not None:
            should_sell_flag, sell_reason = self.should_sell(current_price, date)
            if should_sell_flag:
                self.execute_sell(current_price, date, sell_reason)

        # Check if we should buy (only if no current position)
        if self.should_buy(signal_strength):
            self.execute_buy(symbol, current_price, date, signal_strength)

    def run_trading(self, prediction_sequences: List[Dict], df_all_stocks: pd.DataFrame):
        """Run the trading algorithm on prediction sequences"""
        print("Starting Simple Trading Algorithm")
        print("=" * 60)
        print(f"Strong Signal Threshold: {self.strong_signal_threshold:.3f}")
        print(f"Min Hold Days: {self.min_hold_days}")
        print(f"Max Hold Days: {self.max_hold_days}")
        print(f"Stop Loss: {self.stop_loss:.2%}")
        print(f"Profit Drawdown Threshold: {self.profit_drawdown_threshold:.2%}")
        print("=" * 60)

        for i, sequence in enumerate(prediction_sequences):
            print(f"\nProcessing Sequence {i + 1}: {sequence['date']}")
            print(f"Predictions: {sequence['y_pred']}")

            self.process_prediction(sequence, df_all_stocks)

            # Show current status
            if self.current_position:
                entry_date = datetime.strptime(self.current_position['entry_date'], '%Y-%m-%d')
                current_date = datetime.strptime(sequence['date'], '%Y-%m-%d')
                days_held = (current_date - entry_date).days
                max_profit = self.current_position.get('max_profit', 0)
                print(
                    f"Current Position: {self.current_position['symbol']} | Days Held: {days_held} | Peak Profit: {max_profit:.2%}")
            else:
                print(f"Current Position: None | Cash: ${self.cash:.0f}")

        return self.get_performance_summary()

    def get_performance_summary(self) -> Dict:
        """Get trading performance summary"""
        if not self.trade_history:
            return {'message': 'No trades executed'}

        trades_df = pd.DataFrame(self.trade_history)

        # Calculate current portfolio value
        current_value = self.cash
        if self.current_position:
            # Estimate current position value (would need current price in real scenario)
            current_value += self.current_position['quantity'] * self.current_position['entry_price']

        # Analyze completed trades (sell orders)
        sell_trades = trades_df[trades_df['action'] == 'SELL']

        if not sell_trades.empty:
            total_return = sell_trades['profit_loss'].sum()
            win_rate = len(sell_trades[sell_trades['profit_loss'] > 0]) / len(sell_trades)
            avg_return = sell_trades['return_pct'].mean()
            avg_holding_days = sell_trades['days_held'].mean()
            best_trade = sell_trades.loc[sell_trades['profit_loss'].idxmax()]
            worst_trade = sell_trades.loc[sell_trades['profit_loss'].idxmin()]
        else:
            total_return = 0
            win_rate = 0
            avg_return = 0
            avg_holding_days = 0
            best_trade = None
            worst_trade = None

        return {
            'total_trades': len(trades_df[trades_df['action'] == 'BUY']),
            'completed_trades': len(sell_trades),
            'current_position': self.current_position is not None,
            'current_cash': self.cash,
            'total_profit_loss': total_return,
            'win_rate': win_rate,
            'avg_return_pct': avg_return,
            'avg_holding_days': avg_holding_days,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'trade_history': trades_df
        }

    def plot_trading_results(self, df_stocks: pd.DataFrame, symbol: str = None,
                             figsize: tuple = (15, 8), start_date: pd.Timestamp = None,
                             end_date: pd.Timestamp = None, y_pred=None):

        if not self.trade_history:
            st.warning("⚠️ No trades to plot")
            return None

        trades_df = pd.DataFrame(self.trade_history)

        if symbol is None:
            symbol = trades_df['symbol'].value_counts().index[0]

        stock_data = df_stocks[df_stocks['Symbol'] == symbol].copy()
        if start_date is not None and end_date is not None:
            stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)].copy()
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data = stock_data.sort_values('Date')

        symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
        symbol_trades['date'] = pd.to_datetime(symbol_trades['date'])

        # Calculate height from figsize
        plot_height = int(figsize[1] * 60)  # Convert to pixels approximately

        # ========================
        # Plot 1: Price Chart with Buy/Sell markers
        # ========================

        # Create price line chart
        price_fig = px.line(stock_data, x='Date', y='Adj Close',
                            title=f'{symbol} Price Chart with Trading Points')
        price_fig.update_traces(name=f'{symbol} Price', line_color='blue', line_width=2)

        # Add inference predictions if available
        if hasattr(self, 'inference_predictions') and hasattr(self, 'inference_dates'):
            inf_dates = pd.to_datetime(self.inference_dates)
            inf_prices = self.inference_predictions.flatten()

            # Extend existing price line with inference
            last_price_date = stock_data['Date'].iloc[-1]
            last_price = stock_data['Adj Close'].iloc[-1]

            ext_dates = [last_price_date] + list(inf_dates)
            ext_prices = [last_price] + list(inf_prices)

            # Create extension dataframe
            ext_df = pd.DataFrame({'Date': ext_dates, 'Price': ext_prices})

            price_fig.add_scatter(x=ext_df['Date'], y=ext_df['Price'],
                                  mode='lines', name='Inference Extension',
                                  line=dict(color='orange', dash='dash', width=2))

        # FIXED: Sort all trades by date first, then separate buy/sell
        symbol_trades_sorted = symbol_trades.sort_values('date').reset_index(drop=True)

        buy_trades = symbol_trades_sorted[symbol_trades_sorted['action'] == 'BUY'].reset_index(drop=True)
        sell_trades = symbol_trades_sorted[symbol_trades_sorted['action'] == 'SELL'].reset_index(drop=True)

        # Add buy markers with correct sequential numbering
        if not buy_trades.empty:
            # Create trade numbers based on chronological order of ALL trades
            trade_numbers = []
            for _, buy_trade in buy_trades.iterrows():
                # Find the position of this buy trade in the sorted list
                trade_position = symbol_trades_sorted[symbol_trades_sorted['date'] == buy_trade['date']].index[0]
                # Count how many buy trades occurred before this one
                buy_count = len(symbol_trades_sorted[(symbol_trades_sorted['date'] <= buy_trade['date']) &
                                                     (symbol_trades_sorted['action'] == 'BUY')])
                trade_numbers.append(f'T{buy_count}')

            price_fig.add_scatter(x=buy_trades['date'], y=buy_trades['price'],
                                  mode='markers+text', name='BUY',
                                  marker=dict(color='green', symbol='triangle-up', size=12),
                                  text=trade_numbers,
                                  textposition='top center',
                                  textfont=dict(size=10, color='green'))

        # Add sell markers
        if not sell_trades.empty:
            price_fig.add_scatter(x=sell_trades['date'], y=sell_trades['price'],
                                  mode='markers', name='SELL',
                                  marker=dict(color='red', symbol='triangle-down', size=12))

        price_fig.update_layout(
            height=int(plot_height * 0.75),  # 3/4 of total height like matplotlib subplot
            xaxis_title='Date',
            yaxis_title='Price ($)',
            showlegend=True,
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )

        st.plotly_chart(price_fig, use_container_width=True)

        # ========================
        # Plot 2: Individual Trade Returns - FIXED
        # ========================

        if not sell_trades.empty:
            # Use the already sorted sell_trades
            # Create DataFrame for plotly ensuring proper chronological order
            returns_data = []
            for i, row in sell_trades.iterrows():
                returns_data.append({
                    'Trade': f'T{i + 1}',  # Sequential numbering based on chronological order
                    'Return_Pct': row['return_pct'] * 100,
                    'Date': row['date'],
                    'Color': 'Profit' if row['return_pct'] > 0 else 'Loss'
                })

            returns_df = pd.DataFrame(returns_data)

            # Create bar chart with proper color mapping
            returns_fig = px.bar(returns_df,
                                 x='Trade',
                                 y='Return_Pct',
                                 color='Color',
                                 color_discrete_map={'Profit': 'green', 'Loss': 'red'},
                                 title='Individual Trade Returns (%)',
                                 labels={'Return_Pct': 'Return (%)', 'Trade': 'Trade Number'},
                                 text='Return_Pct')  # Add text directly to bars



            # Format the text labels properly
            returns_fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont=dict(size=10, color='white'),
                cliponaxis=False  # Allow text to go outside plot area
            )

            # Update layout for dark theme and better spacing
            returns_fig.update_layout(
                height=450,
                xaxis_title='Trade Number',
                yaxis_title='Return (%)',
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',  # Dark theme
                paper_bgcolor='rgba(0,0,0,0)',  # Dark theme
                font=dict(color='white'),  # Dark theme
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    zeroline=False,
                    zerolinecolor='white',
                    zerolinewidth=1
                ),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    # Force the exact order we want
                    categoryorder='array',
                    categoryarray=[f'T{i + 1}' for i in range(len(returns_data))]
                ),
                # Auto-adjust margins for text
                margin=dict(t=60, b=60)
            )

            st.plotly_chart(returns_fig, use_container_width=True)

        else:
            st.info("No completed trades to show returns for.")

    def print_trading_summary(self, trades_df: pd.DataFrame):
        """Print detailed trading summary"""
        if trades_df.empty:
            return

        print("\n" + "=" * 80)
        print("DETAILED TRADING SUMMARY")
        print("=" * 80)

        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']

        print(f"Total Buy Orders: {len(buy_trades)}")
        print(f"Total Sell Orders: {len(sell_trades)}")

        if not sell_trades.empty:
            total_pnl = sell_trades['profit_loss'].sum()
            win_trades = sell_trades[sell_trades['profit_loss'] > 0]
            lose_trades = sell_trades[sell_trades['profit_loss'] <= 0]

            print(f"\nProfit & Loss:")
            total_return_pct = sell_trades['return_pct'].sum() * 100
            print(f"  Total Return: {total_return_pct:.2f}%")
            print(f"  Winning trades: {len(win_trades)} ({len(win_trades) / len(sell_trades) * 100:.1f}%)")
            print(f"  Losing trades: {len(lose_trades)} ({len(lose_trades) / len(sell_trades) * 100:.1f}%)")

            if not win_trades.empty:
                print(f"  Average win: {win_trades['return_pct'].mean() * 100:.2f}%")
                print(f"  Largest win: {win_trades['return_pct'].max() * 100:.2f}%")

            if not lose_trades.empty:
                print(f"  Average loss: {lose_trades['return_pct'].mean() * 100:.2f}%")
                print(f"  Largest loss: {lose_trades['return_pct'].min() * 100:.2f}%")

            print(f"\nHolding Periods:")
            print(f"  Average holding: {sell_trades['days_held'].mean():.1f} days")
            print(f"  Min holding: {sell_trades['days_held'].min()} days")
            print(f"  Max holding: {sell_trades['days_held'].max()} days")

            print(f"\nSell Reasons:")
            reason_counts = sell_trades['reason'].value_counts()
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count} trades")

        print("\n" + "=" * 80)

    def streamlit_trading_summary(self, trades_df: pd.DataFrame):
        if trades_df.empty:
            st.warning("⚠️ No trade data to display.")
            return

        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        win_trades = sell_trades[sell_trades['profit_loss'] > 0]
        lose_trades = sell_trades[sell_trades['profit_loss'] <= 0]

        # Precomputed metrics
        total_buys = len(buy_trades)
        win_rate = f"{len(win_trades) / len(sell_trades) * 100:.1f}%" if len(sell_trades) else "0%"
        avg_holding = f"{sell_trades['days_held'].mean():.1f}" if not sell_trades.empty else "0"
        avg_win = f"{win_trades['return_pct'].mean() * 100:.2f}%" if not win_trades.empty else "0%"
        avg_loss = f"{lose_trades['return_pct'].mean() * 100:.2f}%" if not lose_trades.empty else "0%"
        max_win = f"{win_trades['return_pct'].max() * 100:.2f}%" if not win_trades.empty else "0%"
        max_loss = f"{lose_trades['return_pct'].min() * 100:.2f}%" if not lose_trades.empty else "0%"

        # ========================
        # Section: Metric Cards
        # ========================
        st.markdown("### Hiệu suất")

        def metric_card(title, value):
            st.markdown(f"""
                <div style='background-color:#1f2937;padding:20px 10px;border-radius:12px;text-align:center;
                            box-shadow:0 2px 8px rgba(0,0,0,0.2);margin-bottom:10px'>
                    <div style='color:#bbb;font-size:14px'>{title}</div>
                    <div style='color:white;font-size:24px;font-weight:bold'>{value}</div>
                </div>
            """, unsafe_allow_html=True)

        cols = st.columns(4)
        with cols[0]:
            metric_card("Số giao dịch", total_buys)
        with cols[1]:
            metric_card("Tỷ lệ thắng", win_rate)
        with cols[2]:
            metric_card("Số ngày nắm giữ trung bình", avg_holding)
        with cols[3]:
            metric_card("Lãi trung bình", avg_win)

        cols = st.columns(3)
        with cols[0]:
            metric_card("Lỗ trung bình", avg_loss)
        with cols[1]:
            metric_card("Lãi lớn nhất", max_win)
        with cols[2]:
            metric_card("Lỗ lớn nhất", max_loss)

        # ========================
        # Section: Cumulative P&L
        # ========================
        st.markdown("### Tăng trưởng NAV")
        sell_trades = sell_trades.copy()
        sell_trades['date'] = pd.to_datetime(sell_trades['date'])
        sell_trades['cumulative_pnl'] = sell_trades['profit_loss'].cumsum()
        sell_trades['portfolio_value'] = 10000 + sell_trades['cumulative_pnl']

        pnl_fig = px.line(sell_trades, x='date', y='portfolio_value',
                          labels={'portfolio_value': 'Giá trị danh mục ($)', 'date': 'Ngày'})
        pnl_fig.update_traces(line_color='limegreen')

        pnl_fig.update_layout(height=400)
        st.plotly_chart(pnl_fig, use_container_width=True)


