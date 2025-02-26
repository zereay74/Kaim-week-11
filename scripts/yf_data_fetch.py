import yfinance as yf
import pandas as pd
import logging
import os
import sys

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'logs.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    def __init__(self, tickers, start_date="2015-01-01", end_date="2025-01-31"):
        """
        Initialize the StockDataFetcher.

        :param tickers: List of stock/ETF tickers to fetch data for.
        :param start_date: Start date for historical data.
        :param end_date: End date for historical data.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        """
        Fetch historical stock data from Yahoo Finance.
        
        :return: Dictionary containing DataFrames for each ticker.
        """
        data = {}
        for ticker in self.tickers:
            try:
                logging.info(f"Fetching data for {ticker} from {self.start_date} to {self.end_date}...")
                stock = yf.download(ticker, start=self.start_date, end=self.end_date)
                if stock.empty:
                    logging.warning(f"No data found for {ticker}")
                else:
                    data[ticker] = stock
            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {e}")
        return data

    def save_to_csv(self, data, folder_path="../week 11 data/"):
        """
        Save fetched data to CSV files.

        :param data: Dictionary containing DataFrames for each ticker.
        :param folder_path: Directory where CSV files will be saved.
        """
        for ticker, df in data.items():
            file_path = f"{folder_path}{ticker}.csv"
            df.to_csv(file_path)
            logging.info(f"Data for {ticker} saved to {file_path}")
'''
# Usage Example
if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    fetcher = StockDataFetcher(tickers)
    stock_data = fetcher.fetch_data()

    # Display first few rows of Tesla data
    print(stock_data["TSLA"].head())

    # Save data to CSV files
    fetcher.save_to_csv(stock_data)
'''