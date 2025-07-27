import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from gili_train import ImprovedDDQNTradingAgent  # Fixed import
import os


class PriceReconstructor:
    """Helper class to reconstruct actual prices from normalized percentage returns"""

    def __init__(self, scaling_stats_path, engine):
        self.engine = engine

        # Load scaling stats
        with open(scaling_stats_path, "rb") as f:
            self.scaling_stats = pickle.load(f)

        print("Loaded scaling stats for price reconstruction")

    def denormalize_adj_close(self, normalized_adj_close):
        """Convert normalized adj close back to percentage returns"""
        if 'Adj Close' not in self.scaling_stats:
            print("Warning: No scaling stats for Adj Close, returning as-is")
            return normalized_adj_close

        mean_val, std_val = self.scaling_stats['Adj Close']
        return normalized_adj_close * std_val + mean_val

    def fetch_initial_prices(self, symbols, start_date='2023-01-01'):
        """Fetch the first actual price for each symbol to use as base for reconstruction"""
        symbols_str = "','".join(symbols)
        query = f"""
        SELECT DISTINCT ON (symbol) symbol, date, "Adj Close" as adj_close_actual
        FROM stock_features 
        WHERE symbol IN ('{symbols_str}')
        AND date >= '{start_date}'
        ORDER BY symbol, date
        """

        df = pd.read_sql(query, self.engine)
        return dict(zip(df['symbol'], df['adj_close_actual']))

    def reconstruct_prices(self, action_log):
        """Reconstruct actual prices from normalized percentage returns"""
        df = pd.DataFrame(action_log)

        if len(df) == 0:
            return df

        # Debug: Print available columns
        print(f"Available columns in action_log: {list(df.columns)}")

        # Get unique symbols
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['unknown']

        # Fetch initial prices for each symbol
        try:
            initial_prices = self.fetch_initial_prices(symbols)
            print(f"Fetched initial prices for {len(initial_prices)} symbols")
        except Exception as e:
            print(f"Warning: Could not fetch initial prices: {e}")
            # Use dummy initial prices
            initial_prices = {symbol: 100.0 for symbol in symbols}

        df_reconstructed = df.copy()

        for symbol in symbols:
            symbol_mask = df['symbol'] == symbol if 'symbol' in df.columns else pd.Series([True] * len(df))
            symbol_data = df[symbol_mask].copy()

            if len(symbol_data) == 0:
                continue

            # Get initial price for this symbol
            initial_price = initial_prices.get(symbol, 100.0)

            # Denormalize the percentage returns for current price
            if 'price' in symbol_data.columns:
                denorm_current = self.denormalize_adj_close(symbol_data['price'].values)
                # Convert percentage returns back to actual prices
                reconstructed_prices = self._pct_returns_to_prices(denorm_current, initial_price)
                df_reconstructed.loc[symbol_mask, 'price_actual'] = reconstructed_prices
                print(
                    f"Reconstructed current prices for {symbol}: {initial_price:.2f} -> {reconstructed_prices[-1]:.2f}")

            # Handle next_price if it exists
            if 'next_price' in symbol_data.columns:
                denorm_next = self.denormalize_adj_close(symbol_data['next_price'].values)
                reconstructed_next_prices = self._pct_returns_to_prices(denorm_next, initial_price)
                df_reconstructed.loc[symbol_mask, 'next_price_actual'] = reconstructed_next_prices
                print(f"Reconstructed next prices for {symbol}")
            else:
                print(f"Warning: 'next_price' column not found for {symbol}, skipping next price reconstruction")

        return df_reconstructed

    def _pct_returns_to_prices(self, pct_returns, initial_price):
        """Convert percentage returns to actual prices using cumulative product"""
        # Convert percentage returns to price ratios (1 + return)
        price_ratios = 1 + (pct_returns / 100.0)

        # Calculate cumulative prices
        prices = [initial_price]
        for ratio in price_ratios:
            prices.append(prices[-1] * ratio)

        return np.array(prices[1:])  # Remove the initial price, return same length as input


def load_trained_agent(model_path, state_size):
    """Load trained agent with proper checkpoint structure"""
    try:
        # Create agent with same parameters as training
        agent = ImprovedDDQNTradingAgent(
            state_size=state_size,
            action_size=3,  # 0=SELL, 1=HOLD, 2=BUY
            learning_rate=5e-5,
            gamma=0.95,
            epsilon_start=0.01,  # Set low epsilon for evaluation
            epsilon_end=0.01,
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


def plot_single_stock(df, symbol, use_dates=True, price_column='price_actual', save_dir="plots", save_fig=True):
    """Plot actions for a single stock"""
    symbol_data = df[df['symbol'] == symbol].copy()

    if len(symbol_data) == 0:
        print(f"No data for symbol {symbol}")
        return

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Actual Price and Actions
    ax1.plot(symbol_data['date'], symbol_data[price_column],
             label='Actual Price', color='black', alpha=0.7, linewidth=1)

    # Plot actions - Updated mapping for ImprovedDDQNTradingAgent
    buy_mask = symbol_data['action'] == 2  # BUY is action 2
    sell_mask = symbol_data['action'] == 0  # SELL is action 0
    hold_mask = symbol_data['action'] == 1  # HOLD is action 1

    if buy_mask.any():
        ax1.scatter(symbol_data[buy_mask]['date'], symbol_data[buy_mask][price_column],
                    color='green', label='BUY', marker='^', s=50, alpha=0.8)

    if sell_mask.any():
        ax1.scatter(symbol_data[sell_mask]['date'], symbol_data[sell_mask][price_column],
                    color='red', label='SELL', marker='v', s=50, alpha=0.8)

    if hold_mask.any():
        ax1.scatter(symbol_data[hold_mask]['date'], symbol_data[hold_mask][price_column],
                    color='blue', label='HOLD', marker='o', s=20, alpha=0.3)

    ax1.set_title(f"Agent Trading Actions - {symbol} ({len(symbol_data)} data points)")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative Rewards
    symbol_data['cumulative_reward'] = symbol_data['reward'].cumsum()
    ax2.plot(symbol_data['date'], symbol_data['cumulative_reward'], color='purple', linewidth=2)
    ax2.set_title(f"Cumulative Rewards Over Time - {symbol}")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True, alpha=0.3)

    if use_dates:
        ax2.set_xlabel("Date")
        # Rotate x-axis labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.set_xlabel("Time Step")

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
            f"Price range: ${symbol_data[price_column].min():.2f} - ${symbol_data[price_column].max():.2f} (${price_range:.2f})")


def plot_multiple_stocks_overview(df, symbols, price_column='price_actual', max_symbols=6):
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

        if len(symbol_data) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {symbol}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(symbol)
            continue

        # Plot price
        ax.plot(symbol_data['date'], symbol_data[price_column],
                color='black', alpha=0.7, linewidth=1)

        # Plot actions - Updated for ImprovedDDQNTradingAgent
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
    plt.show()


def plot_actions_with_actual_prices(action_log, price_reconstructor, symbols=None, max_points=1000,
                                    plot_individual=True, plot_overview=True):
    """Plot agent actions on actual price chart with multiple stock support"""

    # Reconstruct actual prices
    print("Reconstructing actual prices from normalized percentage returns...")
    df = price_reconstructor.reconstruct_prices(action_log)

    if len(df) == 0:
        print("No data to plot after price reconstruction")
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

    # Use actual reconstructed prices for plotting
    price_column = 'price_actual' if 'price_actual' in df.columns else 'price'

    # Plot overview of multiple stocks
    if plot_overview and len(symbols) > 1:
        print("Creating multi-stock overview...")
        plot_multiple_stocks_overview(df, symbols, price_column)

    # Plot individual stocks
    if plot_individual:
        print("Creating individual stock plots...")
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
    """Main function to load model, evaluate, and plot results with actual prices"""

    # === File paths (adjust these to your actual paths) ===
    model_path = "C:/Users/PC/PycharmProjects/GILIGILI_RL/ddqn_trading_agent.pth"
    test_data_path = "C:/Users/PC/PycharmProjects/GILIGILI_RL/data/test_rl_data.pkl"
    scaling_stats_path = "C:/Users/PC/PycharmProjects/GILIGILI_RL/data/scaling_stats.pkl"

    # Database connection (for price reconstruction)
    engine = create_engine(
        "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres"
    )

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

    # === Load trained model ===
    print("Loading trained model...")
    agent = load_trained_agent(model_path, state_size)
    if agent is None:
        return

    # === Initialize price reconstructor ===
    print("Initializing price reconstructor...")
    try:
        price_reconstructor = PriceReconstructor(scaling_stats_path, engine)
    except Exception as e:
        print(f"Error initializing price reconstructor: {e}")
        return

    # === Evaluate agent ===
    print("Evaluating agent on test data...")
    results = evaluate_agent_fixed(agent, test_samples)

    # === Print results ===
    analyze_performance(results)

    # === Plot actions with actual prices ===
    print("Plotting results with actual prices...")

    action_log_df = pd.DataFrame(results['action_log'])
    if 'symbol' in action_log_df.columns and not action_log_df['symbol'].isna().all():
        unique_symbols = action_log_df['symbol'].dropna().unique()
        selected_symbols = unique_symbols[:10]
        print(f"Selected symbols for plotting: {selected_symbols}")

        plot_actions_with_actual_prices(
            results['action_log'],
            price_reconstructor,
            symbols=selected_symbols,  # First 10 symbols
            plot_individual=True,
            plot_overview=False  # Skip overview to save time
        )
    else:
        print("No symbol information available, plotting all data")
        plot_actions_with_actual_prices(results['action_log'], price_reconstructor)

    return agent, results


if __name__ == "__main__":
    agent, results = main()