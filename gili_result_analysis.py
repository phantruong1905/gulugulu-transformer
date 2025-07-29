import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# Load your trading log
df = pd.read_csv('action_log_all_symbols.csv')
df['q_values'] = df['q_values'].apply(ast.literal_eval)

# Expand list into separate columns
q_df = df['q_values'].apply(pd.Series)
q_df.columns = ['q_sell', 'q_hold', 'q_buy']

# Join back to original DataFrame
df = pd.concat([df, q_df], axis=1)
# Ensure numeric columns are properly cast
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
df['accumulated_return_from_buy'] = pd.to_numeric(df['accumulated_return_from_buy'], errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['q_buy'] = pd.to_numeric(df['q_buy'], errors='coerce')
df['q_sell'] = pd.to_numeric(df['q_sell'], errors='coerce')
df['q_hold'] = pd.to_numeric(df['q_hold'], errors='coerce')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by symbol and date to ensure proper sequence
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

def analyze_buy_sell_pairs(df):
    """
    Link each SELL action to its corresponding BUY action to analyze
    the correlation between Q-values at BUY and actual returns at SELL.
    """
    results = []

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()

        # Track open positions
        position_stack = []  # Stack to handle multiple positions

        for idx, row in symbol_df.iterrows():
            if row['action_name'] == 'BUY':
                # Store BUY information
                buy_info = {
                    'buy_date': row['date'],
                    'buy_idx': idx,
                    'buy_price': row['actual_price'],
                    'q_buy_at_buy': row['q_buy'],
                    'q_sell_at_buy': row['q_sell'],
                    'q_hold_at_buy': row['q_hold'],
                    'estimated_q_value_at_buy': row['estimated_q_value']
                }
                position_stack.append(buy_info)

            elif row['action_name'] == 'SELL' and position_stack:
                # Match with most recent BUY (LIFO - Last In First Out)
                buy_info = position_stack.pop()

                # Create matched pair
                pair = {
                    'symbol': symbol,
                    'buy_date': buy_info['buy_date'],
                    'sell_date': row['date'],
                    'buy_price': buy_info['buy_price'],
                    'sell_price': row['actual_price'],
                    'days_held': row['days_since_buy'],
                    'accumulated_return': row['accumulated_return_from_buy'],
                    'sell_reward': row['reward'],

                    # Q-values at BUY time
                    'q_buy_at_buy': buy_info['q_buy_at_buy'],
                    'q_sell_at_buy': buy_info['q_sell_at_buy'],
                    'q_hold_at_buy': buy_info['q_hold_at_buy'],
                    'estimated_q_value_at_buy': buy_info['estimated_q_value_at_buy'],

                    # Q-values at SELL time
                    'q_buy_at_sell': row['q_buy'],
                    'q_sell_at_sell': row['q_sell'],
                    'q_hold_at_sell': row['q_hold'],
                    'estimated_q_value_at_sell': row['estimated_q_value']
                }
                results.append(pair)

    return pd.DataFrame(results)

# Create buy-sell pairs
pairs_df = analyze_buy_sell_pairs(df)

print("🔍 Buy-Sell Pairs Analysis")
print(f"Total matched pairs: {len(pairs_df)}")
print(f"Symbols analyzed: {pairs_df['symbol'].nunique()}")

if not pairs_df.empty:
    # Basic statistics
    print(f"\n📈 Return Statistics:")
    print(f"Average return: {pairs_df['accumulated_return'].mean():.4f}")
    print(f"Median return: {pairs_df['accumulated_return'].median():.4f}")
    print(f"Std deviation: {pairs_df['accumulated_return'].std():.4f}")
    print(f"Min return: {pairs_df['accumulated_return'].min():.4f}")
    print(f"Max return: {pairs_df['accumulated_return'].max():.4f}")

    # Q-value statistics at BUY time
    print(f"\n🧠 Q-Value Statistics (at BUY time):")
    print(f"Average Q-BUY: {pairs_df['q_buy_at_buy'].mean():.4f}")
    print(f"Average Q-SELL: {pairs_df['q_sell_at_buy'].mean():.4f}")
    print(f"Average Q-HOLD: {pairs_df['q_hold_at_buy'].mean():.4f}")
    print(f"Average Estimated Q: {pairs_df['estimated_q_value_at_buy'].mean():.4f}")

    # Correlation analysis
    print(f"\n🔗 Correlation Analysis (Return vs Q-Values at BUY time):")
    correlations = {
        'Q-BUY vs Return': pairs_df['q_buy_at_buy'].corr(pairs_df['accumulated_return']),
        'Q-SELL vs Return': pairs_df['q_sell_at_buy'].corr(pairs_df['accumulated_return']),
        'Q-HOLD vs Return': pairs_df['q_hold_at_buy'].corr(pairs_df['accumulated_return']),
        'Estimated Q vs Return': pairs_df['estimated_q_value_at_buy'].corr(pairs_df['accumulated_return'])
    }

    for corr_name, corr_value in correlations.items():
        print(f"{corr_name}: {corr_value:.4f}")

    # Win/Loss analysis
    wins = pairs_df[pairs_df['accumulated_return'] > 0]
    losses = pairs_df[pairs_df['accumulated_return'] <= 0]

    print(f"\n📊 Win/Loss Analysis:")
    print(f"Win rate: {len(wins)/len(pairs_df):.2%}")
    print(f"Average win: {wins['accumulated_return'].mean():.4f}")
    print(f"Average loss: {losses['accumulated_return'].mean():.4f}")

    if len(wins) > 0 and len(losses) > 0:
        print(f"\n🧠 Q-Value Analysis by Outcome:")
        print(f"Winners - Avg Q-BUY: {wins['q_buy_at_buy'].mean():.4f}")
        print(f"Losers - Avg Q-BUY: {losses['q_buy_at_buy'].mean():.4f}")
        print(f"Winners - Avg Q-SELL: {wins['q_sell_at_buy'].mean():.4f}")
        print(f"Losers - Avg Q-SELL: {losses['q_sell_at_buy'].mean():.4f}")

    # Enhanced correlation matrix
    print(f"\n🔗 Detailed Correlation Matrix:")
    corr_cols = ['accumulated_return', 'q_buy_at_buy', 'q_sell_at_buy', 'q_hold_at_buy',
                 'estimated_q_value_at_buy', 'days_held', 'buy_price', 'sell_price']
    corr_matrix = pairs_df[corr_cols].corr()
    print(corr_matrix.round(3))

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Return vs Q-BUY at buy time
    axes[0,0].scatter(pairs_df['q_buy_at_buy'], pairs_df['accumulated_return'], alpha=0.6)
    axes[0,0].set_xlabel('Q-BUY Value (at BUY time)')
    axes[0,0].set_ylabel('Accumulated Return')
    axes[0,0].set_title(f'Return vs Q-BUY (r={correlations["Q-BUY vs Return"]:.3f})')
    axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 2. Return vs Q-SELL at buy time
    axes[0,1].scatter(pairs_df['q_sell_at_buy'], pairs_df['accumulated_return'], alpha=0.6, color='orange')
    axes[0,1].set_xlabel('Q-SELL Value (at BUY time)')
    axes[0,1].set_ylabel('Accumulated Return')
    axes[0,1].set_title(f'Return vs Q-SELL (r={correlations["Q-SELL vs Return"]:.3f})')
    axes[0,1].axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 3. Return vs Estimated Q-value at buy time
    axes[1,0].scatter(pairs_df['estimated_q_value_at_buy'], pairs_df['accumulated_return'], alpha=0.6, color='green')
    axes[1,0].set_xlabel('Estimated Q-Value (at BUY time)')
    axes[1,0].set_ylabel('Accumulated Return')
    axes[1,0].set_title(f'Return vs Estimated Q (r={correlations["Estimated Q vs Return"]:.3f})')
    axes[1,0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 4. Q-values distribution by outcome
    win_loss_data = []
    for outcome, color in [('Win', 'green'), ('Loss', 'red')]:
        subset = wins if outcome == 'Win' else losses
        if not subset.empty:
            win_loss_data.extend([
                {'Q-Value': 'Q-BUY', 'Value': val, 'Outcome': outcome}
                for val in subset['q_buy_at_buy']
            ])
            win_loss_data.extend([
                {'Q-Value': 'Q-SELL', 'Value': val, 'Outcome': outcome}
                for val in subset['q_sell_at_buy']
            ])

    if win_loss_data:
        win_loss_df = pd.DataFrame(win_loss_data)
        sns.boxplot(data=win_loss_df, x='Q-Value', y='Value', hue='Outcome', ax=axes[1,1])
        axes[1,1].set_title('Q-Values Distribution by Trade Outcome')
        axes[1,1].set_ylabel('Q-Value')

    plt.tight_layout()


    # Export results for further analysis
    print(f"\n💾 Buy-Sell pairs data shape: {pairs_df.shape}")
    print("Columns available for analysis:")
    print(pairs_df.columns.tolist())

else:
    print("❌ No buy-sell pairs found in the data")

pairs_df['holding_days'] = (pd.to_datetime(pairs_df['sell_date']) - pd.to_datetime(pairs_df['buy_date'])).dt.days
print(f"🕒 Overall Average Holding Days: {pairs_df['holding_days'].mean():.2f} days")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def enhanced_stock_selection_analysis(df, pairs_df):
    """
    Advanced stock selection strategies beyond simple Q-BUY correlation
    """

    print("🚀 Enhanced Stock Selection Analysis")
    print("=" * 50)

    # Strategy 1: Q-Value Confidence (Q-BUY advantage over others)
    print("\n📊 Strategy 1: Q-Value Confidence Analysis")

    # Calculate Q-value differences at buy time
    pairs_df['q_buy_vs_sell_diff'] = pairs_df['q_buy_at_buy'] - pairs_df['q_sell_at_buy']
    pairs_df['q_buy_vs_hold_diff'] = pairs_df['q_buy_at_buy'] - pairs_df['q_hold_at_buy']
    pairs_df['q_buy_vs_max_other'] = pairs_df['q_buy_at_buy'] - pairs_df[['q_sell_at_buy', 'q_hold_at_buy']].max(axis=1)

    # Calculate confidence metrics
    pairs_df['q_confidence'] = pairs_df['q_buy_vs_max_other']  # How much better is BUY vs best alternative
    pairs_df['q_dominance'] = (pairs_df['q_buy_at_buy'] > pairs_df['q_sell_at_buy']) & \
                              (pairs_df['q_buy_at_buy'] > pairs_df['q_hold_at_buy'])

    # Correlations with confidence metrics
    confidence_correlations = {
        'Q-BUY vs Return': pairs_df['q_buy_at_buy'].corr(pairs_df['accumulated_return']),
        'Q-Confidence vs Return': pairs_df['q_confidence'].corr(pairs_df['accumulated_return']),
        'Q-BUY vs SELL diff vs Return': pairs_df['q_buy_vs_sell_diff'].corr(pairs_df['accumulated_return']),
        'Q-BUY vs HOLD diff vs Return': pairs_df['q_buy_vs_hold_diff'].corr(pairs_df['accumulated_return'])
    }

    print("Correlation Analysis:")
    for name, corr in confidence_correlations.items():
        print(f"  {name}: {corr:.4f}")

    # Strategy 2: Threshold-based filtering
    print("\n🎯 Strategy 2: Threshold-based Selection")

    # Test different confidence thresholds
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    threshold_results = []

    for threshold in thresholds:
        high_confidence = pairs_df[pairs_df['q_confidence'] >= threshold]
        if len(high_confidence) > 0:
            avg_return = high_confidence['accumulated_return'].mean()
            win_rate = (high_confidence['accumulated_return'] > 0).mean()
            count = len(high_confidence)
            sharpe = avg_return / high_confidence['accumulated_return'].std() if high_confidence['accumulated_return'].std() > 0 else 0

            threshold_results.append({
                'threshold': threshold,
                'count': count,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'coverage': count / len(pairs_df)
            })

    threshold_df = pd.DataFrame(threshold_results)
    print("\nThreshold Analysis:")
    print(threshold_df.round(4))

    print("\nDetailed Percentile Bucket Analysis (Q-Confidence):")
    quantile_levels = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    quantile_stats = []

    for q in quantile_levels:
        lower_bound = pairs_df['q_confidence'].quantile(q)
        upper_df = pairs_df[pairs_df['q_confidence'] >= lower_bound]

        if len(upper_df) > 0:
            avg_conf = upper_df['q_confidence'].mean()
            avg_return = upper_df['accumulated_return'].mean()
            win_rate = (upper_df['accumulated_return'] > 0).mean()
            sharpe = avg_return / upper_df['accumulated_return'].std() if upper_df['accumulated_return'].std() > 0 else 0
            count = len(upper_df)

            quantile_stats.append({
                'Top X%': f"Top {int((1 - q) * 100)}%",
                'Count': count,
                'Mean Q-Confidence': avg_conf,
                'Avg Return': avg_return,
                'Win Rate': win_rate,
                'Sharpe': sharpe
            })

    quantile_df = pd.DataFrame(quantile_stats)
    print(quantile_df.round(4))


    return pairs_df, threshold_df

def multi_stock_selection_strategies(df):
    """
    Strategies for selecting the best stock when multiple signals are available
    """

    print("\n🎯 Multi-Stock Selection Strategies")
    print("=" * 50)

    # Group by date to find competing signals
    daily_signals = df[df['action_name'] == 'BUY'].groupby('date').agg({
        'symbol': list,
        'q_buy': list,
        'q_sell': list,
        'q_hold': list,
        'estimated_q_value': list,
        'actual_price': list
    }).reset_index()

    # Filter days with multiple buy signals
    multi_signal_days = daily_signals[daily_signals['symbol'].apply(len) > 1]

    if len(multi_signal_days) > 0:
        print(f"Found {len(multi_signal_days)} days with multiple BUY signals")
        print(f"Average signals per day: {daily_signals['symbol'].apply(len).mean():.2f}")

        # Strategy examples for multi-stock selection
        print("\n📋 Selection Strategy Examples:")

        strategies = {
            'Highest Q-BUY': 'Select stock with highest Q-BUY value',
            'Highest Q-Confidence': 'Select stock with highest Q-BUY - max(Q-SELL, Q-HOLD)',
            'Highest Estimated Q': 'Select stock with highest estimated Q-value',
            'Lowest Risk (Q-SELL)': 'Select stock with lowest Q-SELL (less likely to sell immediately)',
            'Portfolio Diversification': 'Avoid stocks already held, select from different sectors',
            'Price Momentum': 'Combine Q-values with recent price momentum',
            'Volume-weighted': 'Weight Q-values by trading volume if available'
        }

        for strategy, description in strategies.items():
            print(f"  • {strategy}: {description}")

    return multi_signal_days

def advanced_signal_combinations(pairs_df):
    """
    Advanced combinations of signals for better selection
    """

    print("\n🔬 Advanced Signal Combinations")
    print("=" * 50)

    # Combination 1: Weighted confidence score
    pairs_df['weighted_confidence'] = (
        0.4 * pairs_df['q_buy_at_buy'] +
        0.3 * pairs_df['q_confidence'] +
        0.2 * (pairs_df['q_buy_at_buy'] - pairs_df['q_hold_at_buy']) +
        0.1 * pairs_df['estimated_q_value_at_buy']
    )

    # Combination 2: Risk-adjusted score
    pairs_df['risk_adjusted_score'] = pairs_df['q_buy_at_buy'] / (1 + pairs_df['q_sell_at_buy'])

    # Combination 3: Momentum-adjusted (if you have previous day Q-values)
    # This would require historical Q-values - placeholder for concept
    pairs_df['momentum_score'] = pairs_df['q_buy_at_buy']  # Placeholder

    # Test combined metrics
    combination_correlations = {
        'Weighted Confidence': pairs_df['weighted_confidence'].corr(pairs_df['accumulated_return']),
        'Risk Adjusted': pairs_df['risk_adjusted_score'].corr(pairs_df['accumulated_return']),
        'Original Q-BUY': pairs_df['q_buy_at_buy'].corr(pairs_df['accumulated_return']),
        'Q-Confidence': pairs_df['q_confidence'].corr(pairs_df['accumulated_return'])
    }

    print("Combined Signal Correlations:")
    for name, corr in combination_correlations.items():
        print(f"  {name}: {corr:.4f}")

    # Find best performing combination
    best_metric = max(combination_correlations, key=combination_correlations.get)
    print(f"\n🏆 Best performing metric: {best_metric} ({combination_correlations[best_metric]:.4f})")

    return pairs_df

def implementation_recommendations():
    """
    Practical implementation recommendations
    """

    print("\n💡 Implementation Recommendations")
    print("=" * 50)

    recommendations = {
        "Immediate Improvements": [
            "Use Q-confidence (Q-BUY - max(Q-SELL, Q-HOLD)) instead of raw Q-BUY",
            "Set minimum confidence threshold (e.g., 0.05-0.1) to filter weak signals",
            "Focus on top 20-30% of confidence scores for better risk-reward",
            "Track and avoid recently sold stocks for a cooling period"
        ],

        "Multi-Stock Selection": [
            "Rank all daily BUY signals by confidence score",
            "Diversify: max 1-2 stocks per sector/correlation group",
            "Consider position sizing: higher confidence = larger position",
            "Implement portfolio balance: avoid overconcentration"
        ],

        "Advanced Enhancements": [
            "Add volatility adjustment: confidence / stock_volatility",
            "Include market regime detection (bull/bear market conditions)",
            "Use ensemble approach: combine multiple selection criteria",
            "Implement dynamic thresholds based on market conditions"
        ],

        "Risk Management": [
            "Set maximum daily positions (e.g., 3-5 stocks max)",
            "Use confidence score for position sizing",
            "Implement sector/correlation limits",
            "Add maximum drawdown stops per stock"
        ]
    }

    for category, items in recommendations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

# Example usage function
def run_complete_analysis(df, pairs_df):
    """
    Run the complete enhanced analysis
    """

    # Enhanced selection analysis
    pairs_df_enhanced, threshold_results = enhanced_stock_selection_analysis(df, pairs_df)

    # Multi-stock selection
    multi_signals = multi_stock_selection_strategies(df)

    # Advanced combinations
    pairs_df_final = advanced_signal_combinations(pairs_df_enhanced)

    # Implementation recommendations
    implementation_recommendations()

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Q-Confidence vs Return
    axes[0,0].scatter(pairs_df_final['q_confidence'], pairs_df_final['accumulated_return'], alpha=0.6)
    axes[0,0].set_xlabel('Q-Confidence (Q-BUY - max(Q-SELL, Q-HOLD))')
    axes[0,0].set_ylabel('Accumulated Return')
    axes[0,0].set_title(f'Return vs Q-Confidence (r={pairs_df_final["q_confidence"].corr(pairs_df_final["accumulated_return"]):.3f})')
    axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0,0].axvline(0, color='gray', linestyle='--', alpha=0.5)

    # 2. Threshold performance
    if len(threshold_results) > 0:
        axes[0,1].plot(threshold_results['threshold'], threshold_results['avg_return'], 'bo-', label='Avg Return')
        axes[0,1].set_xlabel('Confidence Threshold')
        axes[0,1].set_ylabel('Average Return')
        axes[0,1].set_title('Return vs Confidence Threshold')
        ax2 = axes[0,1].twinx()
        ax2.plot(threshold_results['threshold'], threshold_results['coverage'], 'ro-', label='Coverage')
        ax2.set_ylabel('Coverage %')
        axes[0,1].legend(loc='upper left')
        ax2.legend(loc='upper right')

    # 3. Weighted confidence vs return
    axes[1,0].scatter(pairs_df_final['weighted_confidence'], pairs_df_final['accumulated_return'], alpha=0.6, color='green')
    axes[1,0].set_xlabel('Weighted Confidence Score')
    axes[1,0].set_ylabel('Accumulated Return')
    axes[1,0].set_title(f'Return vs Weighted Confidence (r={pairs_df_final["weighted_confidence"].corr(pairs_df_final["accumulated_return"]):.3f})')
    axes[1,0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 4. Performance by confidence quintiles (dual bar chart)
    pairs_df_final['confidence_quintile'] = pd.qcut(
    pairs_df_final['q_confidence'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )

    # Group by quintile and compute both metrics
    quintile_stats = pairs_df_final.groupby('confidence_quintile').agg({
        'accumulated_return': 'mean',
        'accumulated_return': lambda x: (x > 0).mean()
    }).rename(columns={
        'accumulated_return': 'Avg Return',
        '<lambda_0>': 'Win Rate'  # Rename automatically named column
    })

    # Recalculate both properly (due to duplicate keys)
    quintile_stats = pd.DataFrame({
        'Avg Return': pairs_df_final.groupby('confidence_quintile')['accumulated_return'].mean(),
        'Win Rate': pairs_df_final.groupby('confidence_quintile')['accumulated_return'].apply(lambda x: (x > 0).mean())
    })

    # Plot side-by-side bars
    quintile_stats.plot(kind='bar', ax=axes[1,1], width=0.8)
    axes[1,1].set_title('Return & Win Rate by Confidence Quintile')
    axes[1,1].set_ylabel('Value')
    axes[1,1].tick_params(axis='x', rotation=0)
    axes[1,1].legend(loc='upper left')


    return pairs_df_final

# Usage example:
pairs_df_enhanced = run_complete_analysis(df, pairs_df)

def assign_signal_ranks(buy_signals_df):
    """
    Assign ranks based on Q-confidence percentiles
    S: Top 5% (95th percentile and above)
    A: Top 9% (91st-95th percentile)
    B: Top 20% (80th-91st percentile)
    C: Top 50% (50th-80th percentile)
    D: Below 50th percentile
    """

    # Work with the buy signals directly (already filtered)
    buy_signals = buy_signals_df.copy()

    # Ensure we have q_confidence column
    if 'q_confidence' not in buy_signals.columns:
        print("❌ q_confidence column missing!")
        return buy_signals

    # Calculate percentiles
    percentiles = buy_signals['q_confidence'].quantile([0.5, 0.8, 0.91, 0.95]).to_dict()

    def get_rank(q_conf):
        if pd.isna(q_conf):
            return 'D'  # Default for NaN values
        elif q_conf >= percentiles[0.95]:  # Top 5%
            return 'S'
        elif q_conf >= percentiles[0.91]:  # Top 9%
            return 'A'
        elif q_conf >= percentiles[0.8]:  # Top 20%
            return 'B'
        elif q_conf >= percentiles[0.5]:  # Top 50%
            return 'C'
        else:
            return 'D'

    buy_signals['signal_rank'] = buy_signals['q_confidence'].apply(get_rank)

    print("Signal Rank Distribution:")
    rank_dist = buy_signals['signal_rank'].value_counts().sort_index()
    for rank, count in rank_dist.items():
        pct = count / len(buy_signals) * 100
        print(f"  Rank {rank}: {count:4d} signals ({pct:5.1f}%)")

    print(f"\nPercentile Thresholds:")
    for pct, val in percentiles.items():
        print(f"  {pct * 100:2.0f}th percentile: {val:.4f}")

    return buy_signals


class PortfolioBacktester:
    def __init__(self, max_positions=5, min_hold_days=2):
        self.max_positions = max_positions
        self.min_hold_days = min_hold_days
        self.portfolio = {}  # {symbol: {'buy_date': date, 'buy_price': price, 'rank': rank, 'hold_days': int}}
        self.cash = 100000  # Starting cash
        self.position_size = self.cash / max_positions  # Equal position sizing
        self.trade_log = []
        self.portfolio_value_history = []

        # Rank priority (higher number = higher priority)
        self.rank_priority = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}

    def can_sell(self, symbol, current_date):
        """Check if we can sell a position (minimum holding period)"""
        if symbol not in self.portfolio:
            return False

        buy_date = pd.to_datetime(self.portfolio[symbol]['buy_date'])
        current_date = pd.to_datetime(current_date)
        hold_days = (current_date - buy_date).days

        return hold_days >= self.min_hold_days

    def get_weakest_position(self, current_date):
        """Get the weakest position that can be sold"""
        sellable_positions = []

        for symbol, pos_info in self.portfolio.items():
            if self.can_sell(symbol, current_date):
                sellable_positions.append((symbol, self.rank_priority[pos_info['rank']]))

        if sellable_positions:
            # Sort by rank priority (ascending - weakest first)
            sellable_positions.sort(key=lambda x: x[1])
            return sellable_positions[0][0]  # Return symbol with lowest rank

        return None

    def execute_buy(self, symbol, date, price, rank, q_confidence):
        """Execute a buy order"""
        shares = self.position_size / price
        cost = shares * price

        self.portfolio[symbol] = {
            'buy_date': date,
            'buy_price': price,
            'shares': shares,
            'rank': rank,
            'q_confidence': q_confidence,
            'cost': cost
        }

        self.cash -= cost

        self.trade_log.append({
            'date': date,
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'rank': rank,
            'q_confidence': q_confidence,
            'portfolio_size': len(self.portfolio)
        })

    def execute_sell(self, symbol, date, price, reason):
        """Execute a sell order"""
        if symbol not in self.portfolio:
            return

        pos_info = self.portfolio[symbol]
        shares = pos_info['shares']
        proceeds = shares * price

        buy_date = pd.to_datetime(pos_info['buy_date'])
        sell_date = pd.to_datetime(date)
        hold_days = (sell_date - buy_date).days

        return_pct = (price - pos_info['buy_price']) / pos_info['buy_price']

        self.cash += proceeds

        self.trade_log.append({
            'date': date,
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'buy_price': pos_info['buy_price'],
            'return_pct': return_pct,
            'hold_days': hold_days,
            'reason': reason,
            'rank': pos_info['rank'],
            'portfolio_size': len(self.portfolio) - 1
        })

        del self.portfolio[symbol]

    def process_signals(self, signals_df, prices_df):
        """Process all buy signals and execute trades"""

        # Convert date columns to datetime if they're strings
        signals_df = signals_df.copy()
        if signals_df['date'].dtype == 'object':
            signals_df['date'] = pd.to_datetime(signals_df['date'])

        # Sort signals by date and rank priority (higher q_confidence first within each date)
        signals_df = signals_df.sort_values(['date', 'q_confidence'], ascending=[True, False])

        print(f"Processing {len(signals_df)} signals from {signals_df['date'].min()} to {signals_df['date'].max()}")

        processed_count = 0
        for _, signal in signals_df.iterrows():
            date = signal['date']
            symbol = signal['symbol']
            price = signal['actual_price']
            rank = signal['signal_rank']
            q_confidence = signal['q_confidence']

            processed_count += 1
            if processed_count % 500 == 0:
                print(f"  Processed {processed_count} signals... Current portfolio size: {len(self.portfolio)}")

            # Skip if already holding this stock
            if symbol in self.portfolio:
                continue

            # Check if we have space in portfolio
            if len(self.portfolio) < self.max_positions:
                # Execute buy
                self.execute_buy(symbol, date, price, rank, q_confidence)
            else:
                # Portfolio is full - check if new signal is stronger than weakest position
                weakest_symbol = self.get_weakest_position(date)

                if weakest_symbol:
                    weakest_rank = self.portfolio[weakest_symbol]['rank']

                    # Only replace if new signal is significantly better
                    if self.rank_priority[rank] > self.rank_priority[weakest_rank]:
                        # Get current price for the stock we're selling
                        sell_price = self.get_current_price(weakest_symbol, date, prices_df)
                        if sell_price:
                            self.execute_sell(weakest_symbol, date, sell_price, f"Replaced by {rank} signal")
                            self.execute_buy(symbol, date, price, rank, q_confidence)

        print(f"Finished processing all signals. Final portfolio size: {len(self.portfolio)}")

        # Final portfolio liquidation at the end
        final_date = signals_df['date'].max()
        print(f"Liquidating portfolio on {final_date}")
        self.liquidate_portfolio(final_date, prices_df)

    def get_current_price(self, symbol, date, prices_df):
        """Get current price for a symbol on a given date"""
        try:
            # Convert date to datetime if needed
            if isinstance(date, str):
                date = pd.to_datetime(date)

            # Convert prices_df dates if needed
            prices_df_copy = prices_df.copy()
            if prices_df_copy['date'].dtype == 'object':
                prices_df_copy['date'] = pd.to_datetime(prices_df_copy['date'])

            # Try exact date match first
            price_data = prices_df_copy[(prices_df_copy['symbol'] == symbol) & (prices_df_copy['date'] == date)]
            if not price_data.empty:
                return price_data.iloc[0]['actual_price']

            # If exact date not found, get the closest previous date within 5 days
            date_window = pd.Timedelta(days=5)
            price_data = prices_df_copy[
                (prices_df_copy['symbol'] == symbol) &
                (prices_df_copy['date'] <= date) &
                (prices_df_copy['date'] >= date - date_window)
                ].sort_values('date')

            if not price_data.empty:
                return price_data.iloc[-1]['actual_price']

        except Exception as e:
            print(f"Error getting price for {symbol} on {date}: {e}")

        return None

    def liquidate_portfolio(self, final_date, prices_df):
        """Liquidate all remaining positions"""
        print(f"Liquidating {len(self.portfolio)} remaining positions...")

        for symbol in list(self.portfolio.keys()):
            sell_price = self.get_current_price(symbol, final_date, prices_df)
            if sell_price:
                self.execute_sell(symbol, final_date, sell_price, "Final liquidation")
            else:
                print(f"Warning: Could not find price for {symbol} on {final_date}")
                # Use the buy price as fallback (no gain/loss)
                buy_price = self.portfolio[symbol]['buy_price']
                self.execute_sell(symbol, final_date, buy_price, "Final liquidation (no price found)")


def run_backtest_strategy(buy_signals_df, price_data_df):
    """
    Run the complete backtesting strategy
    """

    print("🚀 Portfolio Strategy Backtesting")
    print("=" * 50)

    # Assign ranks to signals
    ranked_signals = assign_signal_ranks(buy_signals_df)

    # Initialize backtester
    backtester = PortfolioBacktester(max_positions=5, min_hold_days=2)

    # Run backtest
    print(f"\nRunning backtest with {len(ranked_signals)} buy signals...")
    backtester.process_signals(ranked_signals, price_data_df)

    # Analyze results
    trade_df = pd.DataFrame(backtester.trade_log)

    if len(trade_df) > 0:
        print(f"\n📊 Backtest Results")
        print("=" * 30)

        buy_trades = trade_df[trade_df['action'] == 'BUY']
        sell_trades = trade_df[trade_df['action'] == 'SELL']

        print(f"Total Trades: {len(buy_trades)} buys, {len(sell_trades)} sells")
        print(f"Portfolio Utilization: {buy_trades['portfolio_size'].mean():.1f} avg positions")

        if len(sell_trades) > 0:
            avg_return = sell_trades['return_pct'].mean()
            win_rate = (sell_trades['return_pct'] > 0).mean()
            avg_hold_days = sell_trades['hold_days'].mean()

            print(f"\nPerformance Metrics:")
            print(f"  Average Return per Trade: {avg_return:.3f} ({avg_return * 100:.1f}%)")
            print(f"  Win Rate: {win_rate:.3f} ({win_rate * 100:.1f}%)")
            print(f"  Average Holding Period: {avg_hold_days:.1f} days")

            # Performance by rank
            print(f"\nPerformance by Signal Rank:")
            rank_performance = sell_trades.groupby('rank').agg({
                'return_pct': ['count', 'mean', lambda x: (x > 0).mean()],
                'hold_days': 'mean'
            }).round(3)

            rank_performance.columns = ['Count', 'Avg Return', 'Win Rate', 'Avg Hold Days']
            print(rank_performance)

            # Replacement analysis
            replacements = sell_trades[sell_trades['reason'].str.contains('Replaced', na=False)]
            if len(replacements) > 0:
                print(f"\nReplacement Analysis:")
                print(f"  Stocks replaced: {len(replacements)}")
                print(f"  Avg return of replaced stocks: {replacements['return_pct'].mean():.3f}")

                replaced_ranks = replacements['rank'].value_counts()
                print(f"  Ranks replaced: {dict(replaced_ranks)}")

    # Create visualizations
    if len(sell_trades) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Returns by rank
        rank_returns = sell_trades.groupby('rank')['return_pct'].mean().sort_index()
        rank_returns.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Average Return by Signal Rank')
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].tick_params(axis='x', rotation=0)
        axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)

        # 2. Win rate by rank
        rank_winrate = sell_trades.groupby('rank')['return_pct'].apply(lambda x: (x > 0).mean())
        rank_winrate.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('Win Rate by Signal Rank')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].tick_params(axis='x', rotation=0)
        axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        # 3. Return distribution
        axes[1, 0].hist(sell_trades['return_pct'], bins=30, alpha=0.7, color='coral')
        axes[1, 0].set_title('Distribution of Trade Returns')
        axes[1, 0].set_xlabel('Return %')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)

        # 4. Cumulative returns over time
        sell_trades_sorted = sell_trades.sort_values('date')
        cumulative_returns = (1 + sell_trades_sorted['return_pct']).cumprod()
        axes[1, 1].plot(range(len(cumulative_returns)), cumulative_returns, 'b-', linewidth=2)
        axes[1, 1].set_title('Cumulative Return Curve')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].axhline(1, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()


    return backtester, trade_df


# CORRECTED APPROACH - Analyze Historical Performance Instead of Backtesting
print("Analyzing historical buy-sell pairs performance...")

# Use pairs_df_enhanced which contains the BUY-SELL pairs with all the enhanced metrics
if 'pairs_df_enhanced' not in globals():
    print("❌ pairs_df_enhanced not found! Please run the previous enhanced analysis first.")
    print("Looking for pairs_df instead...")
    if 'pairs_df' in globals():
        pairs_data = pairs_df.copy()
        print("✅ Using pairs_df")
    else:
        print("❌ No pairs data found! Cannot proceed.")
        pairs_data = None
else:
    pairs_data = pairs_df_enhanced.copy()
    print("✅ Using pairs_df_enhanced")


def analyze_historical_performance(pairs_df):
    """
    Analyze the performance of the Q-confidence ranking system
    using the actual historical trades
    """

    print(f"\n🔍 Historical Performance Analysis")
    print("=" * 50)

    # Assign ranks to historical pairs
    ranked_pairs = assign_signal_ranks(pairs_df)

    print(f"\nAnalyzing {len(ranked_pairs)} historical buy-sell pairs")
    print(f"Period: {ranked_pairs['buy_date'].min()} to {ranked_pairs['sell_date'].max()}")

    # Performance by rank
    print(f"\n📊 Performance by Signal Rank:")
    rank_performance = ranked_pairs.groupby('signal_rank').agg({
        'accumulated_return': ['count', 'mean', 'std', lambda x: (x > 0).mean()],
        'days_held': 'mean',
        'q_confidence': 'mean'
    }).round(4)

    rank_performance.columns = ['Count', 'Avg Return', 'Std Dev', 'Win Rate', 'Avg Days', 'Avg Q-Conf']
    print(rank_performance)

    # Statistical significance test
    print(f"\n📈 Rank Performance Analysis:")
    s_rank = ranked_pairs[ranked_pairs['signal_rank'] == 'S']['accumulated_return']
    d_rank = ranked_pairs[ranked_pairs['signal_rank'] == 'D']['accumulated_return']

    if len(s_rank) > 0 and len(d_rank) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(s_rank, d_rank)
        print(f"S-rank vs D-rank performance:")
        print(f"  S-rank mean return: {s_rank.mean():.4f} ({len(s_rank)} trades)")
        print(f"  D-rank mean return: {d_rank.mean():.4f} ({len(d_rank)} trades)")
        print(f"  Statistical significance: p-value = {p_value:.6f}")
        print(f"  {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} difference")

    # Portfolio simulation with ranking
    print(f"\n💼 Portfolio Simulation Results:")

    # Sort by date and q_confidence for portfolio simulation
    sorted_pairs = ranked_pairs.sort_values(['buy_date', 'q_confidence'], ascending=[True, False])

    # Simulate portfolio performance if we selected top 5 positions each day
    portfolio_results = simulate_ranking_portfolio(sorted_pairs, max_positions=5)

    return ranked_pairs, portfolio_results


def simulate_ranking_portfolio(pairs_df, max_positions=5):
    """
    Simulate what would happen if we used ranking to select positions
    """

    results = {
        'all_trades': {'returns': pairs_df['accumulated_return'].tolist(), 'count': len(pairs_df)},
        'rank_filtered': {}
    }

    # Test different ranking strategies
    strategies = {
        'Top_20%': pairs_df[pairs_df['signal_rank'].isin(['S', 'A'])],
        'S_rank_only': pairs_df[pairs_df['signal_rank'] == 'S'],
        'A_rank_and_above': pairs_df[pairs_df['signal_rank'].isin(['S', 'A'])],
        'B_rank_and_above': pairs_df[pairs_df['signal_rank'].isin(['S', 'A', 'B'])]
    }

    print(f"Strategy Performance Comparison:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Count':<8} {'Avg Ret':<10} {'Win Rate':<10} {'Sharpe':<8}")
    print("-" * 60)

    for strategy_name, strategy_df in strategies.items():
        if len(strategy_df) > 0:
            avg_return = strategy_df['accumulated_return'].mean()
            win_rate = (strategy_df['accumulated_return'] > 0).mean()
            sharpe = avg_return / strategy_df['accumulated_return'].std() if strategy_df[
                                                                                 'accumulated_return'].std() > 0 else 0

            results['rank_filtered'][strategy_name] = {
                'returns': strategy_df['accumulated_return'].tolist(),
                'count': len(strategy_df),
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe': sharpe
            }

            print(f"{strategy_name:<20} {len(strategy_df):<8} {avg_return:<10.4f} {win_rate:<10.3f} {sharpe:<8.3f}")

    # Baseline (all trades)
    all_avg = pairs_df['accumulated_return'].mean()
    all_win_rate = (pairs_df['accumulated_return'] > 0).mean()
    all_sharpe = all_avg / pairs_df['accumulated_return'].std() if pairs_df['accumulated_return'].std() > 0 else 0
    print(f"{'All_trades (baseline)':<20} {len(pairs_df):<8} {all_avg:<10.4f} {all_win_rate:<10.3f} {all_sharpe:<8.3f}")

    return results


if pairs_data is not None:
    # Use the q_confidence column that already exists
    if 'q_confidence' not in pairs_data.columns:
        print("❌ q_confidence column missing! Please run the enhanced analysis first.")
    else:
        # Run historical analysis instead of backtesting
        analyzed_pairs, portfolio_results = analyze_historical_performance(pairs_data)

        # Create summary visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Returns by rank
        rank_returns = analyzed_pairs.groupby('signal_rank')['accumulated_return'].mean().sort_index()
        rank_returns.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Average Return by Signal Rank')
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].tick_params(axis='x', rotation=0)
        axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)

        # 2. Win rate by rank
        rank_winrate = analyzed_pairs.groupby('signal_rank')['accumulated_return'].apply(lambda x: (x > 0).mean())
        rank_winrate.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('Win Rate by Signal Rank')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].tick_params(axis='x', rotation=0)
        axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        # 3. Q-confidence distribution by rank
        for rank in ['S', 'A', 'B', 'C', 'D']:
            rank_data = analyzed_pairs[analyzed_pairs['signal_rank'] == rank]['q_confidence']
            if len(rank_data) > 0:
                axes[1, 0].hist(rank_data, alpha=0.6, label=f'Rank {rank}', bins=15)
        axes[1, 0].set_title('Q-Confidence Distribution by Rank')
        axes[1, 0].set_xlabel('Q-Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # 4. Cumulative returns if we only took top ranks
        top_trades = analyzed_pairs[analyzed_pairs['signal_rank'].isin(['S', 'A'])].sort_values('buy_date')
        if len(top_trades) > 0:
            cumulative_returns = (1 + top_trades['accumulated_return']).cumprod()
            axes[1, 1].plot(range(len(cumulative_returns)), cumulative_returns, 'b-', linewidth=2,
                            label='Top Ranks (S+A)')

        all_trades = analyzed_pairs.sort_values('buy_date')
        cumulative_all = (1 + all_trades['accumulated_return']).cumprod()
        axes[1, 1].plot(range(len(cumulative_all)), cumulative_all, 'r--', alpha=0.7, label='All Trades')

        axes[1, 1].set_title('Cumulative Return: Top Ranks vs All')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].legend()
        axes[1, 1].axhline(1, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()


        # Set the results for compatibility
        trade_results = analyzed_pairs  # Use analyzed pairs as "trade results"
        backtester = None  # No backtester object since we're analyzing historical data

print("\n✅ Historical Analysis Complete!")
print(f"Total historical pairs analyzed: {len(trade_results) if 'trade_results' in locals() else 0}")

if 'analyzed_pairs' in locals():
    print(f"Signal rank distribution:")
    rank_counts = analyzed_pairs['signal_rank'].value_counts().sort_index()
    for rank, count in rank_counts.items():
        pct = count / len(analyzed_pairs) * 100
        print(f"  Rank {rank}: {count} trades ({pct:.1f}%)")

    print(f"\nKey Insights:")
    s_rank_trades = analyzed_pairs[analyzed_pairs['signal_rank'] == 'S']
    if len(s_rank_trades) > 0:
        print(f"  S-rank (top 5%) average return: {s_rank_trades['accumulated_return'].mean():.4f}")
        print(f"  S-rank win rate: {(s_rank_trades['accumulated_return'] > 0).mean():.3f}")

    all_avg = analyzed_pairs['accumulated_return'].mean()
    print(f"  Overall average return: {all_avg:.4f}")
    print(f"  Overall win rate: {(analyzed_pairs['accumulated_return'] > 0).mean():.3f}")

