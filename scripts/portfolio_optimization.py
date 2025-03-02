import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

class PortfolioOptimization:
    def __init__(self, tesla, bnd, spy):
        """Initialize with cleaned dataframes."""
        self.df = self.combine_data(tesla, bnd, spy)
        self.returns = self.compute_annual_returns()
        self.weights = np.array([1/3, 1/3, 1/3])  # Default equal weights

    def combine_data(self, tesla, bnd, spy):
        """Step 1: Merge closing prices into one DataFrame."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(tesla['Date']),
            'TSLA': tesla['Close'],
            'BND': bnd['Close'],
            'SPY': spy['Close']
        }).set_index('Date')
        return df

    def compute_annual_returns(self):
        """Step 2: Compute annualized returns."""
        daily_returns = self.df.pct_change().dropna()
        return daily_returns.mean() * 252

    def compute_cov_matrix(self):
        """Step 3: Compute the covariance matrix."""
        return self.df.pct_change().dropna().cov() * 252

    def portfolio_performance(self, weights):
        """Compute portfolio return and volatility."""
        port_return = np.dot(weights, self.returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.compute_cov_matrix(), weights)))
        return port_return, port_volatility

    def negative_sharpe_ratio(self, weights):
        """Step 5: Define the objective function to maximize the Sharpe Ratio."""
        port_return, port_volatility = self.portfolio_performance(weights)
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility
        return -sharpe_ratio

    def optimize_portfolio(self):
        """Step 7: Find the optimal weights to maximize the Sharpe Ratio."""
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1
        bounds = tuple((0, 1) for _ in range(3))  # Weights between 0 and 1
        result = sco.minimize(self.negative_sharpe_ratio, self.weights, bounds=bounds, constraints=constraints)
        self.weights = result.x
        return result.x

    def calculate_risk_metrics(self):
        """Step 6: Analyze portfolio risk & return."""
        avg_return = np.dot(self.weights, self.returns)
        volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.compute_cov_matrix(), self.weights)))
        risk_free_rate = 0.02
        sharpe_ratio = (avg_return - risk_free_rate) / volatility
        VaR_95 = np.percentile(self.df.pct_change().dropna().dot(self.weights), 5) * 100
        return {
            'Avg Return': avg_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'VaR (95%)': VaR_95
        }
    
    def visualize_performance(self):
        """Step 8: Plot cumulative returns with better visibility."""
        
        # Ensure Date is index
        if 'Date' in self.df.columns:
            self.df.set_index('Date', inplace=True)

        # Compute cumulative returns
        cumulative_returns = (1 + self.df.pct_change()).cumprod()

        # Handle NaN values
        cumulative_returns.dropna(inplace=True)

        # Normalize all assets to start from 1
        cumulative_returns /= cumulative_returns.iloc[0]

        # 🔥 Force Tesla to be visible by adjusting linewidth
        plt.figure(figsize=(10, 6))
        for col in cumulative_returns.columns:
            if col == 'TSLA':
                plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=3, linestyle='dashed')  # Thicker, dashed
            else:
                plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=1)  # Normal thickness

        plt.title("Portfolio Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.show()


    # def visualize_performance(self):
    #     """Step 8: Plot cumulative returns."""
    #     cumulative_returns = (1 + self.df.pct_change()).cumprod()
    #     cumulative_returns.plot(figsize=(10, 6), title='Portfolio Performance')
    #     plt.show()

    def summarize_results(self):
        """Step 9: Summarize portfolio metrics and adjustments."""
        optimal_weights = self.optimize_portfolio()
        risk_metrics = self.calculate_risk_metrics()
        print("Optimized Weights:", optimal_weights)
        print("Risk Metrics:", risk_metrics)
        self.visualize_performance()
''' 
# Usage Example
tesla = pd.read_csv("tesla_predictions.csv")
bnd = pd.read_csv("bnd_predictions.csv")
spy = pd.read_csv("spy_predictions.csv")

portfolio = PortfolioOptimization(tesla, bnd, spy)
portfolio.summarize_results()
'''