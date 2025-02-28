import yfinance as yf
import pandas as pd
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YFinanceDataCollector:
    def __init__(self, tickers: List[str], start_date: str, end_date: str, columns: List[str] = None):
        """
        Initialize the data collector.
        
        :param tickers: List of stock symbols to fetch data for.
        :param start_date: Start date for historical data (YYYY-MM-DD format).
        :param end_date: End date for historical data (YYYY-MM-DD format).
        :param columns: List of columns to retrieve (default: Open, High, Low, Close, Volume).
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.columns = columns if columns else ["Date", "Open", "High", "Low", "Close", "Volume"]
        logging.info(f"Initialized YFinanceDataCollector with tickers: {self.tickers}, date range: {self.start_date} to {self.end_date}")
    
    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical data from YFinance for a single ticker and date range.
        
        :param ticker: Stock symbol to fetch data for.
        :return: Pandas DataFrame with historical stock data.
        """
        logging.info(f"Fetching data for {ticker}")
        stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
        stock_data = stock_data.reset_index()  # Ensure Date is a column
        available_columns = [col for col in self.columns if col in stock_data.columns]  # Filter existing columns
        stock_data = stock_data[available_columns]  # Select required columns
        logging.info(f"Data fetching complete for {ticker}.")
        return stock_data
    
    def save_to_csv(self, output_dir: str):
        """
        Save each ticker's fetched data to separate CSV files.
        
        :param output_dir: Directory path to save CSV files.
        """
        for ticker in self.tickers:
            file_path = f"{output_dir}/{ticker}.csv"
            logging.info(f"Saving data for {ticker} to {file_path}")
            df = self.fetch_data(ticker)
            df.to_csv(file_path, index=False)
            logging.info(f"Data for {ticker} successfully saved to {file_path}")
''' 
# Usage Example
if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2025-01-31"
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    collector = YFinanceDataCollector(tickers, start_date, end_date, columns)
    output_directory = "./historical_data"  # Specify your desired output directory
    collector.save_to_csv(output_directory)
    print(f"Data saved in separate CSV files under {output_directory}")
'''