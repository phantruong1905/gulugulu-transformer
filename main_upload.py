import sqlalchemy
from sqlalchemy import create_engine
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import text

from src.load_data import fetch_stock_data
from src.feature_engineering import *
from model.dual_transformer import *
from scripts.train_test_dual_transformer import *
from scripts.test_backtest import SimpleTradingAlgorithm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Config:
    batch_size = 32
    num_epochs = 50
    patience = 10
    learning_rate = 1e-5
    min_learning_rate = 1e-6
    seq_len = 64
    pred_len = 5


class DailyTradingEngine:
    def __init__(self, checkpoint_path="trained_wavelet_model.pt"):
        self.engine = create_engine(
            "postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.config = Config()
        self.model = None
        self.model_initialized = False

        # Load model weights once
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found!")
        self.state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

    def fetch_recent_data(self, ticker: str, target_date: str, days_back: int = 500):
        """Fetch only the data needed for the target date prediction"""
        # Calculate start date (go back enough days + buffer for weekends/holidays)
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=days_back)
        start_date = start_dt.strftime("%Y-%m-%d")

        # Create temporary data directory
        data_path = './temp_data'
        os.makedirs(data_path, exist_ok=True)

        temp_file = os.path.join(data_path, f"temp_{ticker}_data.csv")
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Fetch data
        fetch_stock_data(data_path, [ticker], start_date, target_date, f"temp_{ticker}_data.csv")
        df = pd.read_csv(temp_file)
        #print(f"[DEBUG] Raw fetched data for {ticker}: {df.shape[0]} rows, date range: {df['Date'].min()} to {df['Date'].max()}")

        # Clean up temp file
        os.remove(temp_file)
        os.rmdir(data_path)

        return df

    def prepare_features_for_date(self, df: pd.DataFrame, ticker: str, target_date: str):
        """Prepare features for a specific target date"""
        # Filter for specific ticker
        df_filtered = df[df['Symbol'] == ticker].copy()

        # Add all technical indicators
        df_with_features = add_technical_indicators(df_filtered)
        df_with_features = add_pivot_features(df_with_features)
        df_with_features = calculate_foundation(df_with_features)

        # Convert dates and sort
        df_with_features['Date'] = pd.to_datetime(df_with_features['Date'])
        df_with_features = df_with_features.sort_values('Date').reset_index(drop=True)

        # Get data only up to target date
        target_dt = pd.to_datetime(target_date)
        data_up_to_date = df_with_features[df_with_features['Date'] <= target_dt].copy()
        #print(f"[DEBUG] Feature-engineered data for {ticker}: {df_with_features.shape[0]} rows")

        return data_up_to_date

    def create_model_input(self, df_features: pd.DataFrame):
        """Create model input from features dataframe"""
        if len(df_features) < self.config.seq_len:
            raise ValueError(f"Not enough data. Need {self.config.seq_len}, got {len(df_features)}")

        # Apply same preprocessing as training
        df_work = df_features.copy()
        df_work = df_work.set_index('Date').sort_index()

        # Define features (same as training)
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

        # Create target (needed for normalization, even though we don't use it)
        returns = df_work['Adj Close'].pct_change().replace([np.inf, -np.inf], np.nan)
        trend = returns.rolling(window=10, min_periods=1).mean()
        target = returns - trend

        # Combine and clean
        data = features.join(pd.DataFrame({'target': target}), how='inner').dropna()
        #print("[DEBUG] NaN ratio per column:")
        #print(features.isna().mean().sort_values(ascending=False))
        #print(features)
        # Clean and clip features
        for col in data.columns:
            if col != 'target':
                data[col] = data[col].ffill().bfill().fillna(0)
                if col in base_features:
                    lower = data[col].quantile(0.01)
                    upper = data[col].quantile(0.99)
                    data[col] = data[col].clip(lower, upper)

        if len(data) < self.config.seq_len:
            raise ValueError(f"Not enough clean data. Need {self.config.seq_len}, got {len(data)}")

        # Create sequence (last seq_len days)
        feat_cols = [col for col in data.columns if col != 'target']
        x = data[feat_cols].values
        y = data['target'].values

        # Take the last sequence
        X_seq = x[-self.config.seq_len:].reshape(1, self.config.seq_len, -1)
        y_seq = y[-1:].reshape(1, 1)
        seq_date = data.index[-1]

        return X_seq, y_seq, seq_date

    def make_prediction(self, X_seq: np.ndarray, y_seq: np.ndarray):
        """Make prediction using the model"""
        # Compute normalization
        X_mean = np.mean(X_seq, axis=1, keepdims=True)
        X_std = np.std(X_seq, axis=1, keepdims=True) + 1e-8
        y_mean = np.mean(y_seq)
        y_std = np.std(y_seq) + 1e-8

        # Normalize
        X_seq_norm = (X_seq - X_mean) / X_std

        # Apply wavelet decomposition
        X_seq_wavelet = modwt_decompose(X_seq_norm, level=2)

        # Initialize model if needed
        if not self.model_initialized:
            input_dim = X_seq_wavelet.shape[2]
            wavelet_levels = X_seq_wavelet.shape[-1]
            self.model = CleanWaveletTransformer(
                input_dim=input_dim,
                wavelet_levels=wavelet_levels,
                d_model=64,
                nhead=2,
                num_layers=3,
                drop_out=0.5,
                pred_len=self.config.pred_len
            )

            self.model.load_state_dict(self.state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model_initialized = True

        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq_wavelet).to(self.device)
            prediction = self.model(X_tensor)
            prediction = prediction.cpu().numpy()

        # Unscale prediction
        prediction_unscaled = prediction * y_std + y_mean

        return prediction_unscaled

    def load_current_position_simple(self, ticker: str, target_date: str):
        """Load current position - simplified version"""
        try:
            #print(f"[DEBUG] Loading position for {ticker} on {target_date}")

            # Get the latest trade record for this ticker
            latest_trade_query = """
            SELECT * FROM stock_trades 
            WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
            """

            latest_trade = pd.read_sql(latest_trade_query, self.engine, params=(ticker,))

            if latest_trade.empty:
                #print(f"[DEBUG] No trades found for {ticker}")
                return None

            last_trade = latest_trade.iloc[0]
            #print(f"[DEBUG] Latest trade: {last_trade['action']} on {last_trade['date']} at {last_trade['price']}")

            # If last trade was SELL, no current position
            if last_trade['action'] == 'SELL':
                #print(f"[DEBUG] Last action was SELL, no current position")
                return None

            # If last trade was BUY, we have a position
            if last_trade['action'] == 'BUY':
                entry_date = last_trade['date']
                entry_price = last_trade['price']

                # Calculate days held (business days between entry_date and target_date)
                entry_dt = pd.to_datetime(entry_date)
                target_dt = pd.to_datetime(target_date)
                days_held = (target_dt - entry_dt).days

                # Get historical prices to find peak (including today's price)
                historical_query = """
                SELECT "Date", "Adj Close" FROM raw_stock_data 
                WHERE 'Symbol' = %s 
                AND "Date" >= %s 
                AND "Date" <= %s 
                ORDER BY "Date"
                """

                historical_prices = pd.read_sql(
                    historical_query,
                    self.engine,
                    params=(ticker, entry_date.strftime('%Y-%m-%d'), target_date)
                )

                # Calculate peak price since entry
                peak_price = entry_price  # Start with entry price as peak
                if not historical_prices.empty:
                    # Get all prices from entry date onwards
                    all_prices = historical_prices['adj_close'].tolist()
                    peak_price = max(all_prices)
                    #print(f"[DEBUG] Price history since entry: {all_prices}")
                    #print(f"[DEBUG] Peak price found: {peak_price:.2f}")

                position = {
                    'symbol': ticker,
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'peak_price': peak_price,
                    'quantity': last_trade['quantity'],
                    'signal_strength': last_trade.get('signal_strength', 0.0),
                    'days_held': days_held
                }

                #print(f"[DEBUG] Position: entry={entry_price:.2f}, peak={peak_price:.2f}, days={days_held}")
                return position

        except Exception as e:
            #print(f"Error loading position for {ticker}: {e}")
            import traceback
            #traceback.#print_exc()
            return None

    def create_daily_trader(self, current_position=None):
        """Create a trader instance for daily operation"""
        trader = SimpleTradingAlgorithm(
            min_hold_days=2,
            max_hold_days=10,
            strong_signal_threshold=0.04,
            stop_loss=-0.05,  # Not used anymore, replaced by drawdown logic
            portfolio_size=10000
        )

        # Set current position if exists
        if current_position:
            trader.current_position = current_position
            trader.cash = 0  # All cash is invested

        return trader

    def should_sell_position_fixed(self, position, current_price: float):
        """Fixed sell logic - simple and clear"""
        if position is None:
            #print("[DEBUG] No position to sell")
            return False, None

        #print(f"[DEBUG] Checking sell conditions for {position['symbol']}:")
        #print(f"  - Entry date: {position['entry_date']}")
        #print(f"  - Entry price: {position['entry_price']:.2f}")
        #print(f"  - Peak price since entry: {position['peak_price']:.2f}")
        #print(f"  - Current price: {current_price:.2f}")
        #print(f"  - Days held: {position['days_held']}")

        # Update peak price if current price is higher
        current_peak = max(position['peak_price'], current_price)
        if current_price > position['peak_price']:
            #print(f"[DEBUG] New peak price: {current_price:.2f} (was {position['peak_price']:.2f})")
            position['peak_price'] = current_price  # Update for next check

        # Condition 1: Max hold days (10 days)
        if position['days_held'] >= 10:
            #print(f"[DEBUG] ✅ SELL TRIGGER: Max hold days reached ({position['days_held']} >= 10)")
            return True, "max_hold_10_days"

        # Condition 2: 5% drawdown from peak
        drawdown_from_peak = (current_price - current_peak) / current_peak
        drawdown_pct = drawdown_from_peak * 100

        #print(f"  - Current return from peak: {drawdown_pct:.2f}%")

        if drawdown_from_peak <= -0.05:  # 5% drawdown
            #print(f"[DEBUG] ✅ SELL TRIGGER: 5% drawdown from peak ({drawdown_pct:.2f}% <= -5%)")
            return True, "drawdown_5pct_from_peak"

        #print(f"[DEBUG] ❌ HOLD: No sell conditions met")
        return False, None

    def make_trading_decision_fixed(self, trader, ticker: str, date: str, current_price: float, signal_strength: float):
        """Fixed trading decision logic"""
        action = "HOLD"

        # Check if we should sell first
        if trader.current_position is not None:
            should_sell_flag, sell_reason = self.should_sell_position_fixed(trader.current_position, current_price)

            if should_sell_flag:
                # Execute sell
                position = trader.current_position
                quantity = position['quantity']
                proceeds = quantity * current_price
                cost = position['entry_price'] * quantity
                profit_loss = proceeds - cost
                return_pct = profit_loss / cost

                # Create sell record
                sell_record = {
                    "date": date,
                    "action": "SELL",
                    "symbol": ticker,
                    "price": current_price,
                    "quantity": quantity,
                    "cost": None,  # Only for BUY
                    "proceeds": proceeds,
                    "profit_loss": profit_loss,
                    "return_pct": return_pct,
                    "days_held": position['days_held'],
                    "reason": sell_reason,
                    "signal_strength": None
                }

                # Add to trade history and clear position
                trader.trade_history.append(sell_record)
                trader.current_position = None
                trader.cash = proceeds

                action = f"SELL ({sell_reason})"
                #print(f"[DEBUG] 📈 SELL: {ticker} at {current_price:.2f}")
                #print(f"[DEBUG]    Profit: ${profit_loss:.2f} ({return_pct:.2%})")
                #print(f"[DEBUG]    Reason: {sell_reason}")

        # Check if we should buy (only if no position)
        if trader.current_position is None and trader.should_buy(signal_strength):
            # Fix: Use the correct attribute name based on your SimpleTradingAlgorithm class
            # Try these alternatives in order:
            portfolio_value = getattr(trader, 'portfolio_size', None) or \
                              getattr(trader, 'cash', None) or \
                              getattr(trader, 'initial_cash', None) or \
                              10000  # fallback default

            quantity = portfolio_value / current_price
            cost = quantity * current_price

            # Create buy record
            buy_record = {
                "date": date,
                "action": "BUY",
                "symbol": ticker,
                "price": current_price,
                "quantity": quantity,
                "cost": cost,
                "proceeds": None,  # Only for SELL
                "profit_loss": None,
                "return_pct": None,
                "days_held": None,
                "reason": None,
                "signal_strength": signal_strength
            }

            # Update trader state
            trader.current_position = {
                'symbol': ticker,
                'entry_date': date,
                'entry_price': current_price,
                'peak_price': current_price,  # Initialize peak to entry price
                'quantity': quantity,
                'signal_strength': signal_strength,
                'days_held': 0
            }
            trader.cash = 0
            trader.trade_history.append(buy_record)

            action = "BUY"
            #print(f"[DEBUG] 📉 BUY: {ticker} at {current_price:.2f}, qty={quantity:.2f}")
            #print(f"[DEBUG]    Signal strength: {signal_strength:.4f}")

        return action

    def execute_daily_trading_fixed(self, ticker: str, target_date: str):
        """Fixed daily trading execution"""
        try:
            # 1. Fetch recent data
            #print(f"Fetching data for {ticker}...")
            df_raw = self.fetch_recent_data(ticker, target_date)

            if df_raw.empty:
                #print(f"No data found for {ticker}")
                return None

            # 2. Prepare features
            #print(f"Engineering features for {ticker}...")
            df_features = self.prepare_features_for_date(df_raw, ticker, target_date)

            # 3. Create model input
            #print(f"Creating model input for {ticker}...")
            X_seq, y_seq, seq_date = self.create_model_input(df_features)

            # 4. Make prediction
            #print(f"Making prediction for {ticker}...")
            prediction = self.make_prediction(X_seq, y_seq)

            # 5. Create prediction sequence
            pred_array = prediction[0]
            if isinstance(pred_array, np.ndarray):
                pred_list = pred_array.flatten().tolist()
            else:
                pred_list = [pred_array]

            # 6. Load existing position (SIMPLIFIED VERSION)
            current_position = self.load_current_position_simple(ticker, target_date)

            # 7. Initialize trader with loaded position
            trader = self.create_daily_trader(current_position)

            # 8. Get current price
            target_day_data = df_features[df_features['Date'] == pd.to_datetime(target_date)]
            if target_day_data.empty:
                #print(f"No price data for {ticker} on {target_date}")
                return None

            current_price = target_day_data['Adj Close'].iloc[-1]

            # 9. Calculate signal strength
            signal_strength = trader.calculate_signal_strength(pred_list)

            # 10. Make trading decision (FIXED VERSION)
            trading_action = self.make_trading_decision_fixed(trader, ticker, target_date, current_price,
                                                              signal_strength)

            # 11. Save results to database
            self.save_daily_results(ticker, target_date,
                                    {"date": target_date, "symbol": ticker, "y_pred": pred_list},
                                    trading_action, signal_strength, trader)

            print(f"✅ {ticker}: Signal={signal_strength:.4f}, Action={trading_action}")
            return {
                'ticker': ticker,
                'date': target_date,
                'prediction': pred_list,
                'signal_strength': signal_strength,
                'action': trading_action,
                'current_price': current_price
            }

        except Exception as e:
            #print(f"❌ {ticker}: Error - {str(e)}")
            import traceback
            #traceback.#print_exc()
            return None


    def save_daily_results(self, ticker: str, date: str, prediction_data: dict, trading_action: str,
                           signal_strength: float, trader):
        """Save daily results to database"""
        try:
            with self.engine.begin() as conn:
                # 1. Save/update daily signal
                signal_data = {
                    "date": date,
                    "symbol": ticker,
                    "signal_strength": signal_strength,
                }

                # Add prediction values
                for i, val in enumerate(prediction_data["y_pred"]):
                    signal_data[f"y_pred_{i + 1}"] = val

                # Delete existing signal for this ticker/date
                conn.execute(text("DELETE FROM daily_signals WHERE symbol = :symbol AND date = :date"),
                             {"symbol": ticker, "date": date})

                # Insert new signal
                signal_df = pd.DataFrame([signal_data])
                signal_df.to_sql("daily_signals", conn, if_exists="append", index=False)

                # 2. Save trade if action was taken
                if trader.trade_history:
                    trade_record = trader.trade_history[-1]
                    trade_date = trade_record["date"]
                    trade_action = trade_record["action"]
                    trade_symbol = trade_record["symbol"]

                    # Check existing trade
                    existing_trade_query = """
                        SELECT * FROM stock_trades 
                        WHERE symbol = :symbol AND date = :date AND action = :action
                    """
                    existing_trade = pd.read_sql(
                        text(existing_trade_query),
                        conn,
                        params={"symbol": trade_symbol, "date": trade_date, "action": trade_action}
                    )

                    # Decision logic
                    if trade_action == "BUY":
                        if not existing_trade.empty:
                            print(f"[DEBUG] Skipping BUY: Trade already exists for {trade_symbol} on {trade_date}")
                            return  # skip saving BUY
                    elif trade_action == "SELL":
                        if not existing_trade.empty:
                            # Delete existing SELL to replace it
                            conn.execute(
                                text("""
                                    DELETE FROM stock_trades 
                                    WHERE symbol = :symbol AND date = :date AND action = :action
                                """),
                                {"symbol": trade_symbol, "date": trade_date, "action": trade_action}
                            )
                            print(f"[DEBUG] Replacing existing SELL for {trade_symbol} on {trade_date}")

                    # Save the trade
                    trade_df = pd.DataFrame([{
                        "date": trade_date,
                        "action": trade_action,
                        "symbol": trade_symbol,
                        "price": trade_record["price"],
                        "quantity": trade_record["quantity"],
                        "cost": trade_record.get("cost"),
                        "proceeds": trade_record.get("proceeds"),
                        "profit_loss": trade_record.get("profit_loss"),
                        "return_pct": trade_record.get("return_pct"),
                        "days_held": trade_record.get("days_held"),
                        "reason": trade_record.get("reason"),
                        "signal_strength": trade_record.get("signal_strength")
                    }])
                    trade_df.to_sql("stock_trades", conn, if_exists="append", index=False)
                    print(f"[DEBUG] Saved {trade_action} trade for {trade_symbol} on {trade_date}")


        except Exception as e:
            print(f"Error saving results for {ticker}: {e}")


def run_daily_inference(tickers: list, target_date: str = None):
    """Run daily inference for all tickers"""
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    engine = DailyTradingEngine()
    results = []

    #print(f"Running daily inference for {len(tickers)} tickers on {target_date}")
    #print("=" * 60)

    for ticker in tickers:
        result = engine.execute_daily_trading_fixed(ticker, target_date)
        if result:
            results.append(result)

    #print("=" * 60)
    #print(f"Completed: {len(results)}/{len(tickers)} successful")

    return results


# Example usage
if __name__ == "__main__":
    tickers = [
        # Ngân hàng – Tài chính
        'CTG', 'MBB', 'TCB', 'MSB', 'BID', 'EIB', 'LPB', 'OCB', 'NAB', 'VAB', 'SHB', 'VPB', 'ABB', 'STB', 'ACB', 'KLB',

        # Bất động sản – KCN – Hạ tầng
        'DXG', 'KDH', 'HDG', 'NLG', 'IDC', 'KBC', 'DPG',
        'SZC', 'BCM', 'NTC', 'SIP', 'KHG', 'NTL', 'HHS', 'VCG', 'HDC', 'TCH',

        # Bán lẻ – Tiêu dùng
        'MWG', 'DGW', 'PNJ', 'FRT', 'VHC', 'ANV', 'NKG', 'MCH', 'MSN',
        'MSH',

        # Công nghiệp – Hóa chất – VLXD
        'DGC', 'GVR', 'HSG', 'HPG', 'DPM', 'DCM', 'BFC', 'CSV', 'GEX', 'REE', 'GEG', 'NT2', 'BMP',

        # Dầu khí
        'PVS', 'GEE', 'PLC',

        # Công nghệ – Viễn thông
        'CTR', 'ELC', 'VGI', 'VTP', 'TLG', 'FOX',

        # Nông nghiệp – Thực phẩm
        'PAN', 'DBC', 'QNS',

        # Chứng khoán
        'VIX', 'VCI', 'HCM', 'MBS', 'VDS', 'TVS', 'BSI', 'FTS', 'SSI', 'CTS', 'SHS',

        # Khác – Lẻ tiềm năngL
        'HAH', 'SCS', 'PHP', 'TNG', 'TDT', 'AST', 'VSC', 'KOS', 'NAF', 'DGC', 'NTP', 'CMG', 'VGS', 'FCN', 'VOS',

        # Dệt may
        'TCM', 'MSH', 'GIL', 'TNG',

        # Cao su
        'PHR', 'DRC', 'TRC', 'DPR',

        # Tải
        'GMD',

        # VN30
        'HPG', 'VRE', 'VIC', 'VHM',

        # Thêm bừa không biết ngành gì
        'AGG', 'EVG', 'IJC', 'HAG', 'DXS', 'EVF', 'VTO', 'CTD', 'CTI', 'HHV', 'DDV', 'HNG', 'MCH', 'HVN'
    ]
    current_date = '2025-07-18'
    results = run_daily_inference(tickers, current_date)