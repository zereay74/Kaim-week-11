import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure logging
logging.basicConfig(
    filename="stock_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-notebook")


class StockDataAnalyzer:
    def __init__(self, file_paths):
        """
        Initialize the analyzer with file paths for TSLA, BND, and SPY data.

        :param file_paths: Dictionary with ticker symbols as keys and CSV file paths as values.
        """
        self.file_paths = file_paths
        self.data = {}

    def load_data(self):
        """Load CSV data and check basic statistics before cleaning."""
        for ticker, path in self.file_paths.items():
            logger.info(f"Loading data for {ticker} from {path}...")
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

            # Store the loaded data
            self.data[ticker] = df

    def format_dataframe(self, df):
        """
        Format DataFrame for better table display.
        """
        return df.style.set_properties(**{'text-align': 'left'}) \
            .set_table_styles([{
                'selector': 'th',
                'props': [('font-size', '12px'), ('text-align', 'left')]
            }]).bar(subset=['mean', 'std'], color='#FFA07A')

    def display_basic_statistics(self):
        """Display formatted summary statistics for each dataset."""
        for ticker, df in self.data.items():
            print(f"\nBasic Statistics for {ticker}:\n")
            stats_df = df.describe().transpose().reset_index()
            stats_df.rename(columns={'index': 'Column'}, inplace=True)  # Rename index column for clarity
            display(self.format_dataframe(stats_df))

    def check_missing_values(self):
        """
        Check for missing values in each column, including the Date column.
        :return: DataFrame with Ticker, Column, Missing Values, Missing Percentage, and Data Type.
        """
        missing_data = []
        for ticker, df in self.data.items():
            df_temp = df.reset_index()  # Reset index to include 'Date' as a column
            missing_info = df_temp.isnull().sum()
            missing_percentage = (missing_info / len(df_temp)) * 100

            missing_df = pd.DataFrame({
                'Ticker': ticker,
                'Column': df_temp.columns,
                'Missing Values': missing_info.values,
                'Missing Percentage': missing_percentage.values,
                'Data Type': df_temp.dtypes.values
            })

            missing_data.append(missing_df)

        final_missing_df = pd.concat(missing_data, ignore_index=True)
        print("\nMissing Values Summary:\n")
        display(final_missing_df.style.set_properties(**{'text-align': 'left'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('font-size', '12px'), ('text-align', 'left')]
                }]).bar(subset=['Missing Percentage'], color='#FF6347'))

    def detect_and_remove_outliers(self):
        """Detect and remove outliers in daily returns using the IQR method."""
        logger.info("Detecting and removing outliers in daily returns.")
        for ticker, df in self.data.items():
            if "Daily_Return" not in df.columns:
                logger.warning(f"Skipping {ticker}: 'Daily_Return' not found.")
                continue

            q1 = df["Daily_Return"].quantile(0.25)
            q3 = df["Daily_Return"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df["Daily_Return"] < lower_bound) | (df["Daily_Return"] > upper_bound)]
            num_outliers = len(outliers)

            # Remove outliers
            df_cleaned = df[(df["Daily_Return"] >= lower_bound) & (df["Daily_Return"] <= upper_bound)]

            # Log the changes
            logger.info(f"Removed {num_outliers} outliers from {ticker}. New shape: {df_cleaned.shape}")

            # Replace the original DataFrame with the cleaned version
            self.data[ticker] = df_cleaned

    def detect_outliers(self):
        """Detect outliers in daily returns using the IQR method."""
        logger.info("Detecting outliers in daily returns.")
        outliers_summary = []

        for ticker, df in self.data.items():
            df["Daily_Return"] = df["Close"].pct_change()
            q1 = df["Daily_Return"].quantile(0.25)
            q3 = df["Daily_Return"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df["Daily_Return"] < lower_bound) | (df["Daily_Return"] > upper_bound)]
            outliers_summary.append({
                "Ticker": ticker,
                "Outlier Count": len(outliers),
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound
            })

        outliers_df = pd.DataFrame(outliers_summary)
        print("\nOutliers Summary:\n")
        display(outliers_df.style.set_properties(**{'text-align': 'left'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('font-size', '12px'), ('text-align', 'left')]
                }]).bar(subset=['Outlier Count'], color='#4682B4'))

    def analyze_volatility(self, window=30):
        """Analyze rolling means and standard deviations for volatility."""
        logger.info(f"Analyzing volatility with a rolling window of {window} days.")
        volatility_summary = []

        for ticker, df in self.data.items():
            df["Rolling_Std"] = df["Close"].rolling(window=window).std()
            volatility_summary.append({
                "Ticker": ticker,
                "Mean Rolling Std": df["Rolling_Std"].mean(),
                "Max Rolling Std": df["Rolling_Std"].max(),
                "Min Rolling Std": df["Rolling_Std"].min()
            })

        vol_df = pd.DataFrame(volatility_summary)
        print("\nVolatility Summary:\n")
        display(vol_df.style.set_properties(**{'text-align': 'left'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('font-size', '12px'), ('text-align', 'left')]
                }]).bar(subset=['Mean Rolling Std'], color='#32CD32'))

    def plot_closing_price(self):
        """Visualize closing prices over time for trend identification."""
        logger.info("Plotting closing prices for all assets.")
        plt.figure(figsize=(12, 6))
        for ticker, df in self.data.items():
            plt.plot(df.index, df["Close"], label=ticker)
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title("Stock Closing Prices Over Time")
        plt.legend()
        plt.show()

    def plot_daily_returns(self):
        """Plot daily percentage change to observe volatility."""
        logger.info("Plotting daily returns.")
        plt.figure(figsize=(12, 6))
        for ticker, df in self.data.items():
            plt.plot(df.index, df["Daily_Return"], label=ticker)
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.title("Daily Returns Over Time")
        plt.legend()
        plt.show()

''' 
# Usage Example
if __name__ == "__main__":
    file_paths = {
        "TSLA": "data/TSLA.csv",
        "BND": "data/BND.csv",
        "SPY": "data/SPY.csv"
    }

    analyzer = StockDataAnalyzer(file_paths)
    analyzer.load_data()              # Step 1: Load Data
    analyzer.display_basic_statistics()  # Step 2: Display Basic Statistics
    analyzer.check_missing_values()   # Step 3: Check Missing Values
    analyzer.clean_data()             # Step 4: Clean Data
    analyzer.detect_and_remove_outliers()        # Step 5: Detect Outliers
    analyzer.analyze_volatility()     # Step 6: Volatility Analysis
    analyzer.plot_closing_price()     # Step 7: Plot Closing Prices
    analyzer.plot_daily_returns()     # Step 8: Plot Daily Returns
'''