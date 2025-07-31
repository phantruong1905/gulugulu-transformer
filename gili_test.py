import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from gili_train import DDQNTradingAgent  # Fixed import
import os
from datetime import datetime


class ActualPriceFetcher:
    """Helper class to fetch actual prices from raw_stock_data table"""

    def __init__(self, engine):
        self.engine = engine
        print("Initialized ActualPriceFetcher for raw_stock_data table")

    def fetch_prices_for_dates(self, symbol_date_pairs):
        """
        Fetch actual prices for specific symbol-date combinations

        Args:
            symbol_date_pairs: List of tuples [(symbol, date), ...]
                              or DataFrame with 'symbol' and 'date' columns

        Returns:
            Dictionary with (symbol, date) as key and adj_close as value
        """
        if isinstance(symbol_date_pairs, pd.DataFrame):
            # Convert DataFrame to list of tuples
            symbol_date_pairs = list(zip(symbol_date_pairs['symbol'], symbol_date_pairs['date']))

        if not symbol_date_pairs:
            print("No symbol-date pairs provided")
            return {}

        # Get unique symbols and date range
        symbols = list(set([pair[0] for pair in symbol_date_pairs]))
        dates = [pair[1] for pair in symbol_date_pairs]

        # Convert dates to string format for SQL
        date_strings = []
        for date in dates:
            if isinstance(date, str):
                date_strings.append(date)
            elif hasattr(date, 'strftime'):
                date_strings.append(date.strftime('%Y-%m-%d'))
            else:
                date_strings.append(str(date))

        min_date = min(date_strings)
        max_date = max(date_strings)

        # Build SQL query with date filtering from 2024-07-01 onward
        symbols_str = "','".join(symbols)

        # Ensure we only fetch data from 2024-07-01 onward
        start_date = max(min_date, '2024-07-01')

        query = f"""
        SELECT "Symbol" as symbol, "Date" as date, "Adj Close" as adj_close
        FROM raw_stock_data 
        WHERE "Symbol" IN ('{symbols_str}')
        AND "Date" BETWEEN '{start_date}' AND '{max_date}'
        AND "Date" >= '2024-07-01'
        ORDER BY "Symbol", "Date"
        """

        print(
            f"Fetching prices for {len(symbols)} symbols between {start_date} and {max_date} (from 2024-07-01 onward)")

        try:
            df = pd.read_sql(query, self.engine)
            print(f"Fetched {len(df)} price records from database")

            # Convert to dictionary for fast lookup
            price_dict = {}
            for _, row in df.iterrows():
                key = (row['symbol'], str(row['date']))
                price_dict[key] = float(row['adj_close'])

            return price_dict

        except Exception as e:
            print(f"Error fetching prices from database: {e}")
            return {}

    def add_actual_prices_to_log(self, action_log):
        """
        Add actual prices to action log by fetching from database

        Args:
            action_log: List of dictionaries with trading actions

        Returns:
            DataFrame with actual prices added
        """
        df = pd.DataFrame(action_log)

        if len(df) == 0:
            print("Empty action log")
            return df

        print(f"Processing {len(df)} action log entries")

        # Ensure we have symbol and date columns
        if 'symbol' not in df.columns:
            print("Warning: No 'symbol' column in action log")
            df['symbol'] = 'UNKNOWN'

        if 'date' not in df.columns:
            print("Warning: No 'date' column in action log")
            return df

        # Convert dates to consistent format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Remove rows with invalid dates

        if len(df) == 0:
            print("No valid dates found in action log")
            return df

        # Create symbol-date pairs for fetching
        symbol_date_pairs = []
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            symbol_date_pairs.append((row['symbol'], date_str))

        # Fetch actual prices
        price_dict = self.fetch_prices_for_dates(symbol_date_pairs)

        # Add actual prices to DataFrame
        actual_prices = []
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            key = (row['symbol'], date_str)
            actual_price = price_dict.get(key, None)
            actual_prices.append(actual_price)

        df['actual_price'] = actual_prices

        # Calculate accumulated return from latest buy action for each symbol
        df = self._calculate_accumulated_return_from_buy(df)

        # Remove ALL old price columns - no normalization/denormalization needed
        columns_to_remove = ['price', 'next_price', 'price_actual', 'next_price_actual']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Removed old price column: {col}")

        # Report success rate
        valid_prices = df['actual_price'].notna().sum()
        print(
            f"Successfully matched {valid_prices}/{len(df)} entries with actual prices ({valid_prices / len(df) * 100:.1f}%)")

        if valid_prices == 0:
            print("Warning: No prices were matched. Check if symbols and dates exist in raw_stock_data table")

        return df

    def _calculate_accumulated_return_from_buy(self, df):
        """
        Calculate percentage return only for SELL actions based on the latest BUY price for each symbol.

        Args:
            df: DataFrame with columns including 'symbol', 'action', 'actual_price', 'date'

        Returns:
            DataFrame with added 'accumulated_return_from_buy' column, containing returns only for SELL actions
        """
        if len(df) == 0 or 'actual_price' not in df.columns:
            df['accumulated_return_from_buy'] = None
            return df

        # Sort by symbol and date to ensure correct order
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Initialize the accumulated return column
        df['accumulated_return_from_buy'] = None

        # Process each symbol separately
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_data = df[symbol_mask].copy()

            # Track the latest buy price
            latest_buy_price = None
            accumulated_returns = []

            for _, row in symbol_data.iterrows():
                current_price = row['actual_price']
                action = row['action']

                # Action 2 = BUY, update the buy price
                if action == 2 and pd.notna(current_price):
                    latest_buy_price = current_price
                    accumulated_returns.append(None)  # No return for BUY

                # Action 0 = SELL, calculate return if there's a buy price
                elif action == 0:
                    if latest_buy_price is not None and pd.notna(current_price):
                        # Calculate return as percentage: (sell_price - buy_price) / buy_price * 100
                        return_pct = ((current_price - latest_buy_price) / latest_buy_price) * 100
                        accumulated_returns.append(return_pct)
                        latest_buy_price = None  # Reset after SELL
                    else:
                        accumulated_returns.append(None)  # No buy price or invalid price
                else:
                    # HOLD or other actions
                    accumulated_returns.append(None)  # No return for non-SELL actions

            # Update the main dataframe
            df.loc[symbol_mask, 'accumulated_return_from_buy'] = accumulated_returns

        print("Calculated percentage returns for SELL actions from latest BUY actions")
        return df


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
        agent.current_position = checkpoint.get('current_position', 0)

        print(f"Model loaded successfully from {model_path}")
        print(f"Epsilon: {agent.epsilon:.4f}, Step count: {agent.step_count}")
        print(f"Current position: {agent.current_position}")

        return agent

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_agent_fixed(agent, test_samples):
    """Fixed evaluation function matching the training script's evaluate method"""
    # Reset position for evaluation
    agent.current_position = 0

    # Use the agent's built-in evaluate method
    results = agent.evaluate(test_samples)

    return results


def plot_single_stock(df, symbol, use_dates=True, price_column='actual_price', save_dir="gili_actions_plots", save_fig=True):
    """Plot actions for a single stock (only price and actions, no cumulative rewards)"""
    symbol_data = df[df['symbol'] == symbol].copy()

    if len(symbol_data) == 0:
        print(f"No data for symbol {symbol}")
        return

    # Filter out rows without actual prices for plotting
    symbol_data = symbol_data[symbol_data[price_column].notna()]

    if len(symbol_data) == 0:
        print(f"No valid price data for symbol {symbol}")
        return

    # Create a single plot (no subplots)
    fig, ax = plt.subplots(figsize=(15, 6))  # Adjusted figure size for single plot

    # Plot 1: Actual Price and Actions
    ax.plot(symbol_data['date'], symbol_data[price_column],
            label='Actual Price', color='black', alpha=0.7, linewidth=1)

    # Plot actions - Updated mapping for DDQNTradingAgent
    buy_mask = symbol_data['action'] == 2  # BUY is action 2
    sell_mask = symbol_data['action'] == 0  # SELL is action 0
    hold_mask = symbol_data['action'] == 1  # HOLD is action 1

    if buy_mask.any():
        ax.scatter(symbol_data[buy_mask]['date'], symbol_data[buy_mask][price_column],
                   color='green', label='BUY', marker='^', s=50, alpha=0.8)

    if sell_mask.any():
        ax.scatter(symbol_data[sell_mask]['date'], symbol_data[sell_mask][price_column],
                   color='red', label='SELL', marker='v', s=50, alpha=0.8)

    if hold_mask.any():
        ax.scatter(symbol_data[hold_mask]['date'], symbol_data[hold_mask][price_column],
                   color='blue', label='HOLD', marker='o', s=20, alpha=0.3)

    ax.set_title(f"Agent Trading Actions - {symbol} ({len(symbol_data)} data points)")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if use_dates:
        ax.set_xlabel("Date")
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax.set_xlabel("Time Step")

    plt.tight_layout()
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{symbol}_actions.png")
        plt.savefig(save_path)
        print(f"Saved plot for {symbol} to {save_path}")
        plt.close()  # Don't display
    else:
        plt.show()

    # Print symbol-specific statistics - Updated action names
    print(f"\n=== {symbol} Statistics ===")
    action_counts = symbol_data['action'].value_counts().sort_index()
    action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    for action, count in action_counts.items():
        pct = count / len(symbol_data) * 100
        print(f"{action_names.get(action, action)}: {count} ({pct:.1f}%)")

    print(f"Average reward: {symbol_data['reward'].mean():.4f}")
    print(f"Total reward: {symbol_data['reward'].sum():.4f}")

    # Price statistics
    if price_column in symbol_data.columns:
        price_range = symbol_data[price_column].max() - symbol_data[price_column].min()
        print(
            f"Price range: {symbol_data[price_column].min():.2f} - {symbol_data[price_column].max():.2f} ({price_range:.2f})")


def plot_multiple_stocks_overview(df, symbols, price_column='actual_price', max_symbols=6):
    """Plot overview of multiple stocks in subplots"""
    if len(symbols) > max_symbols:
        print(f"Too many symbols ({len(symbols)}), showing first {max_symbols}")
        symbols = symbols[:max_symbols]

    # Calculate subplot layout
    n_symbols = len(symbols)
    n_cols = min(3, n_symbols)
    n_rows = (n_symbols + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_symbols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, symbol in enumerate(symbols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        symbol_data = df[df['symbol'] == symbol].copy()
        # Filter out rows without actual prices
        symbol_data = symbol_data[symbol_data[price_column].notna()]

        if len(symbol_data) == 0:
            ax.text(0.5, 0.5, f'No price data\nfor {symbol}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(symbol)
            continue

        # Plot price
        ax.plot(symbol_data['date'], symbol_data[price_column],
                color='black', alpha=0.7, linewidth=1)

        # Plot actions - Updated for DDQNTradingAgent
        buy_mask = symbol_data['action'] == 2  # BUY is action 2
        sell_mask = symbol_data['action'] == 0  # SELL is action 0

        if buy_mask.any():
            ax.scatter(symbol_data[buy_mask]['date'], symbol_data[buy_mask][price_column],
                       color='green', marker='^', s=20, alpha=0.8)

        if sell_mask.any():
            ax.scatter(symbol_data[sell_mask]['date'], symbol_data[sell_mask][price_column],
                       color='red', marker='v', s=20, alpha=0.8)

        ax.set_title(f"{symbol} ({len(symbol_data)} pts)")
        ax.grid(True, alpha=0.3)

        # Rotate dates
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Hide empty subplots
    for i in range(n_symbols, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.suptitle("Trading Actions Overview - Multiple Stocks", fontsize=16)
    plt.tight_layout()


def plot_actions_with_actual_prices(action_log, price_fetcher, symbols=None, max_points=1000,
                                    plot_individual=True, plot_overview=True):
    """Plot agent actions on actual price chart with multiple stock support"""

    # Add actual prices to action log
    print("Fetching actual prices from raw_stock_data table...")
    df = price_fetcher.add_actual_prices_to_log(action_log)

    if len(df) == 0:
        print("No data to plot after fetching actual prices")
        return

    # Handle missing dates
    if 'date' not in df.columns or df['date'].isna().all():
        print("Warning: No date information available, using index instead")
        df['date'] = range(len(df))
        use_dates = False
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        use_dates = True

    df = df.sort_values(['symbol', 'date'])

    # Get available symbols
    available_symbols = df['symbol'].unique()
    print(f"Available symbols: {available_symbols}")

    # Filter by symbols if specified
    if symbols is not None and len(symbols) > 0:
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [s for s in symbols if s in available_symbols]
        if len(symbols) == 0:
            print("No matching symbols found")
            return
        df = df[df['symbol'].isin(symbols)]
    else:
        symbols = available_symbols

    print(f"Plotting for symbols: {symbols}")

    # Use actual prices for plotting
    price_column = 'actual_price'

    # Plot overview of multiple stocks
    if plot_overview and len(symbols) > 1:
        print("Creating multi-stock overview...")
        plot_multiple_stocks_overview(df, symbols, price_column)

    # Plot individual stocks
    if plot_individual:
        print("Creating individual stock gili_actions_plots...")
        for symbol in symbols:
            plot_single_stock(df, symbol, use_dates, price_column)


def analyze_performance(results):
    """Analyze and print detailed performance metrics"""
    print(f"\n=== Performance Analysis ===")
    print(f"Total Reward: {results['total_reward']:.4f}")
    print(f"Average Reward: {results['average_reward']:.4f}")

    print(f"\n=== Action Distribution ===")
    total_actions = sum(results['action_distribution'].values())
    for action, count in results['action_distribution'].items():
        pct = count / total_actions * 100
        print(f"{action}: {count} ({pct:.1f}%)")

    # Analyze rewards by action
    action_log_df = pd.DataFrame(results['action_log'])
    if 'reward' in action_log_df.columns:
        print(f"\n=== Reward by Action Type ===")
        reward_by_action = action_log_df.groupby('action_name')['reward'].agg(['mean', 'std', 'count'])
        for action, stats in reward_by_action.iterrows():
            print(f"{action}: "
                  f"Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Count={stats['count']}")


def main():
    """Main function to load model, evaluate, and plot results with actual prices for all symbols"""

    # === File paths (adjust these to your actual paths) ===
    model_path = "C:/Users/PC/PycharmProjects/GILIGILI_RL/ddqn_trading_agent.pth"
    test_data_path = "C:/Users/PC/PycharmProjects/GILIGILI_RL/data/test_rl_data.pkl"

    # Database connection (for price fetching)
    engine = create_engine(...)

    # === Load test samples ===
    print("Loading test data...")
    try:
        with open(test_data_path, "rb") as f:
            test_samples = pickle.load(f)
        print(f"Loaded {len(test_samples)} test samples")
    except FileNotFoundError:
        print(f"Test data file not found: {test_data_path}")
        return
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # === Get state size from data ===
    if len(test_samples) == 0:
        print("No test samples found")
        return

    state_size = len(test_samples[0]['state'])
    print(f"State size: {state_size}")

    # === Filter by date only (no symbol filtering) ===
    cutoff_date = datetime.strptime("2025-07-18", "%Y-%m-%d")  # or any other date

    # Filter test samples by date only
    filtered_test_samples = [
        s for s in test_samples
        if 'date' in s and pd.to_datetime(s['date']) <= cutoff_date
    ]

    if len(filtered_test_samples) == 0:
        print(f"No samples found before {cutoff_date}")
        return
    else:
        print(f"Using {len(filtered_test_samples)} samples before {cutoff_date.date()}")

    # === Get all unique symbols ===
    unique_symbols = list(set([s['symbol'] for s in filtered_test_samples if 'symbol' in s]))
    print(f"Found {len(unique_symbols)} unique symbols: {unique_symbols}")

    # === Load trained model ===
    print("Loading trained model...")
    agent = load_trained_agent(model_path, state_size)
    if agent is None:
        return

    # === Initialize actual price fetcher ===
    print("Initializing actual price fetcher...")
    try:
        price_fetcher = ActualPriceFetcher(engine)
    except Exception as e:
        print(f"Error initializing price fetcher: {e}")
        return

    # === Evaluate agent ===
    print("Evaluating agent on test data...")
    results = evaluate_agent_fixed(agent, filtered_test_samples)

    # === Print results ===
    analyze_performance(results)

    # === Add actual prices and save to CSV ===
    print("Adding actual prices to action log...")
    action_log_df = price_fetcher.add_actual_prices_to_log(results['action_log'])

    # Save to CSV with actual prices only
    csv_filename = "action_log_all_symbols.csv"  # Updated filename to reflect all symbols
    action_log_df.to_csv(csv_filename, index=False)
    print(f"Saved action log with actual prices to {csv_filename}")

    # === Plot actions with actual prices ===
    print("Plotting results with actual prices...")

    if 'symbol' in action_log_df.columns and not action_log_df['symbol'].isna().all():
        # Use all unique symbols from the action log
        selected_symbols = list(action_log_df['symbol'].unique())
        print(f"Selected symbols for plotting: {selected_symbols}")

        plot_actions_with_actual_prices(
            results['action_log'],
            price_fetcher,
            symbols=selected_symbols,  # Pass all symbols
            plot_individual=True,      # Plot individual stock charts
            plot_overview=True         # Plot overview of all stocks
        )
    else:
        print("No symbol information available, plotting all data")
        plot_actions_with_actual_prices(results['action_log'], price_fetcher)

    return agent, results


if __name__ == "__main__":
    agent, results = main()