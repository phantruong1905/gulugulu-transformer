import torch
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from gili_train import DDQNTradingAgent  # Fixed import
import os
from datetime import datetime


class ActualPriceFetcher:
    """Helper class to fetch actual prices from raw_stock_data table"""

    def __init__(self, engine):
        self.engine = engine

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

        try:
            df = pd.read_sql(query, self.engine)

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
            return df

        # Ensure we have symbol and date columns
        if 'symbol' not in df.columns:
            df['symbol'] = 'UNKNOWN'

        if 'date' not in df.columns:
            return df

        # Convert dates to consistent format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Remove rows with invalid dates

        if len(df) == 0:
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
                        # Calculate return as percentage: (sell_price - buy_price) / buy_price
                        return_pct = ((current_price - latest_buy_price) / latest_buy_price)
                        accumulated_returns.append(return_pct)
                        latest_buy_price = None  # Reset after SELL
                    else:
                        accumulated_returns.append(None)  # No buy price or invalid price
                else:
                    # HOLD or other actions
                    accumulated_returns.append(None)  # No return for non-SELL actions

            # Update the main dataframe
            df.loc[symbol_mask, 'accumulated_return_from_buy'] = accumulated_returns

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


def transform_dataframe(df):
    """
    Transform the DataFrame according to specifications:
    - Split q_values into q_sell, q_hold, q_buy columns
    - Rename accumulated_return_from_buy to return_pct
    - Drop reward column
    - Add signal_strength column
    - Rename actual_price to price
    - Add days_held column
    """
    if len(df) == 0:
        return df

    # Parse q_values string to list and create separate columns
    if 'q_values' in df.columns:
        # Convert string representation of list to actual list
        q_values_lists = []
        for q_val in df['q_values']:
            try:
                # Handle string representation of list
                if isinstance(q_val, str):
                    # Remove brackets and split by comma
                    q_val = q_val.strip('[]')
                    values = [float(x.strip()) for x in q_val.split(',')]
                elif isinstance(q_val, list):
                    values = [float(x) for x in q_val]
                else:
                    values = [0.0, 0.0, 0.0]  # Default values

                # Ensure we have exactly 3 values
                if len(values) == 3:
                    q_values_lists.append(values)
                else:
                    q_values_lists.append([0.0, 0.0, 0.0])
            except:
                q_values_lists.append([0.0, 0.0, 0.0])

        # Create separate columns for q_values
        df['q_sell'] = [q[0] for q in q_values_lists]
        df['q_hold'] = [q[1] for q in q_values_lists]
        df['q_buy'] = [q[2] for q in q_values_lists]

        # Calculate signal_strength
        signal_strengths = []
        for i, row in df.iterrows():
            action = row['action']
            q_sell = row['q_sell']
            q_hold = row['q_hold']
            q_buy = row['q_buy']

            if action == 0:  # SELL
                signal_strength = q_sell - max(q_hold, q_buy)
            elif action == 1:  # HOLD
                signal_strength = q_hold - max(q_sell, q_buy)
            elif action == 2:  # BUY
                signal_strength = q_buy - max(q_sell, q_hold)
            else:
                signal_strength = 0.0

            signal_strengths.append(signal_strength)

        df['signal_strength'] = signal_strengths

        # Drop the original q_values column
        df = df.drop(columns=['q_values'])

    # Rename columns
    if 'accumulated_return_from_buy' in df.columns:
        df = df.rename(columns={'accumulated_return_from_buy': 'return_pct'})

    if 'actual_price' in df.columns:
        df = df.rename(columns={'actual_price': 'price'})

    # Drop reward column
    if 'reward' in df.columns:
        df = df.drop(columns=['reward'])

    # Calculate days_held for each trade
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    days_held_list = []
    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_data = df[symbol_mask].copy()

        buy_date = None
        symbol_days_held = []

        for _, row in symbol_data.iterrows():
            action = row['action']
            current_date = row['date']

            if action == 2:  # BUY
                buy_date = current_date
                symbol_days_held.append(None)  # No days held for BUY action
            elif action == 0 and buy_date is not None:  # SELL
                days_held = (current_date - buy_date).days
                symbol_days_held.append(days_held)
                buy_date = None  # Reset after sell
            else:
                symbol_days_held.append(None)  # HOLD or no previous buy

        days_held_list.extend(symbol_days_held)

    df['days_held'] = days_held_list

    return df

