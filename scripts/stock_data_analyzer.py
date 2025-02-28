import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display

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

    def detect_outliers(self):
        """Detect and remove outliers in daily returns using the IQR method."""
        logger.info("Detecting and removing outliers in daily returns.")
        for ticker, df in self.data.items():
            if "Daily_Return" not in df.columns:
                logger.warning(f"Skipping {ticker}: 'Daily_Return' column not found.")
                continue

            q1 = df["Daily_Return"].quantile(0.25)
            q3 = df["Daily_Return"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df["Daily_Return"] < lower_bound) | (df["Daily_Return"] > upper_bound)]
            logger.info(f"Outliers detected for {ticker}: {len(outliers)} events.")

            # Print the number of rows before and after removal
            print(f"{ticker}: Before Outlier Removal: {len(df)} rows")
            
            # Remove outliers
            df = df[(df["Daily_Return"] >= lower_bound) & (df["Daily_Return"] <= upper_bound)]
            self.data[ticker] = df  # Update with cleaned data
            
            print(f"{ticker}: After Outlier Removal: {len(df)} rows")


    def calculate_daily_returns(self):
        """Calculate daily percentage change for volatility analysis."""
        logger.info("Calculating daily percentage returns.")
        for ticker, df in self.data.items():
            if "Close" not in df.columns:
                logger.warning(f"Skipping {ticker}: 'Close' column not found.")
                continue

            df["Daily_Return"] = df["Close"].pct_change()

            # Ensure there are no NaNs (first row will be NaN)
            df["Daily_Return"].fillna(0, inplace=True)

            logger.info(f"Daily returns statistics for {ticker}:\n{df['Daily_Return'].describe()}")

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
            if "Daily_Return" not in df.columns:
                logger.warning(f"Skipping {ticker}: 'Daily_Return' column not found.")
                continue

            plt.plot(df.index, df["Daily_Return"], label=ticker)

        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.title("Daily Returns Over Time")
        plt.legend()
        plt.show()

    def decompose_seasonality(self, period=252):
        """Decompose the time series into trend, seasonality, and residuals."""
        logger.info("Performing seasonal decomposition for each stock.")
        for ticker, df in self.data.items():
            if "Close" not in df.columns:
                logger.warning(f"Skipping {ticker}: 'Close' column not found.")
                continue

            try:
                result = seasonal_decompose(df["Close"], model="additive", period=period)  # ~1 year of trading days
                fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

                result.observed.plot(ax=axes[0], title=f"{ticker} - Observed (Original Data)")
                result.trend.plot(ax=axes[1], title="Trend")
                result.seasonal.plot(ax=axes[2], title="Seasonality")
                result.resid.plot(ax=axes[3], title="Residuals")

                plt.suptitle(f"Seasonal Decomposition of {ticker}")
                plt.tight_layout()
                plt.show()

            except Exception as e:
                logger.error(f"Error decomposing {ticker}: {e}")

    def calculate_var_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Value at Risk (VaR) and Sharpe Ratio."""
        logger.info("Calculating VaR and Sharpe Ratio.")
        for ticker, df in self.data.items():
            if "Daily_Return" not in df.columns:
                logger.warning(f"Skipping {ticker}: 'Daily_Return' column not found.")
                continue

            daily_returns = df["Daily_Return"].dropna()
            
            if len(daily_returns) < 2:
                logger.warning(f"Not enough data for {ticker} to calculate VaR and Sharpe Ratio.")
                continue

            # Value at Risk (VaR) at 95% confidence
            var_95 = np.percentile(daily_returns, 5)  # 5th percentile
            logger.info(f"VaR (95%) for {ticker}: {var_95:.5f}")

            # Sharpe Ratio calculation
            excess_return = daily_returns.mean() - risk_free_rate / 252
            sharpe_ratio = excess_return / daily_returns.std()
            logger.info(f"Sharpe Ratio for {ticker}: {sharpe_ratio:.5f}")

            # Display the results in a DataFrame
            risk_metrics = pd.DataFrame({
                "Metric": ["VaR (95%)", "Sharpe Ratio"],
                "Value": [var_95, sharpe_ratio]
            })

            print(f"\nRisk Metrics for {ticker}:\n", risk_metrics.to_string(index=False))


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
    analyzer.calculate_daily_returns()   # Step 2.2: Compute Daily Returns
    analyzer.detect_outliers()           # Step 2.4: Outlier Detection
    analyzer.analyze_volatility()     # Step 6: Volatility Analysis
    analyzer.plot_closing_price()     # Step 7: Plot Closing Prices
    analyzer.plot_daily_returns()     # Step 8: Plot Daily Returns
    analyzer.decompose_seasonality()      # Step 3: Decomposing Trends & Seasonality
    analyzer.calculate_var_sharpe_ratio() # Step 5: Risk Assessment (VaR & Sharpe Ratio)

'''