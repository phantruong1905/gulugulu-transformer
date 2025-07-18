import os
import logging
import pandas as pd
from vnstock import Vnstock


def fetch_stock_data(data_path, stocks, start_date, end_date, output_filename="pooled_stock_data.csv"):
    """
    Fetches historical stock data for multiple stocks, combines them into a single DataFrame,
    sorts them, and saves to a CSV file.
    """
    vnstock = Vnstock()
    data = []

    for symbol in stocks:
        try:
            logging.info(f"Fetching data for {symbol}...")
            stock = vnstock.stock(symbol=symbol, source='VCI')
            df = stock.quote.history(start=start_date, end=end_date, interval='1D')

            df.rename(columns={
                "time": "Date", "close": "Adj Close", "high": "High",
                "low": "Low", "open": "Open", "volume": "Volume"
            }, inplace=True)

            df["Symbol"] = symbol
            df = df[["Date", "Symbol", "Adj Close", "High", "Low", "Open", "Volume"]]
            df["Date"] = pd.to_datetime(df["Date"])

            # --- Key Change 1: Drop NaNs per stock before appending ---
            df.dropna(inplace=True)

            logging.info(f"{symbol} | Shape: {df.shape} | From {df['Date'].min().date()} to {df['Date'].max().date()}")
            data.append(df)

        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
        logging.info(f"Collected {len(data)} valid stock DataFrames")

    if data:
        final_df = pd.concat(data, ignore_index=True)

        # --- Key Change 2: Sort by Symbol and then Date ---
        # This is CRITICAL. It ensures that when we process the data later,
        # operations like .shift() don't leak data from the end of one stock
        # to the beginning of the next.
        final_df.sort_values(by=['Symbol', 'Date'], inplace=True)

        save_path = os.path.join(data_path, output_filename)
        # --- Key Change 3: Save without the pandas index column ---
        final_df.to_csv(save_path, index=False)

        logging.info(f"Final dataset saved to: {save_path}")
        logging.info(f"Final dataset shape: {final_df.shape}")
        logging.info("Final dataset head:\n" + str(final_df.head()))
    else:
        logging.error("No data was collected!")

