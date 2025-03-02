# Time Series Forecasting for Portfolio Management Optimization

## 📌 Overview
This project focuses on **Time Series Forecasting** for optimizing portfolio management. It involves fetching financial data, performing time series forecasting using various models, and optimizing portfolio allocations based on forecasted returns.

## 📂 Folder Structure
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── logs                  # Logs for monitoring outputs
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & insights
│   ├── Task_1_Preprocess_and_Explore_the_Data.ipynb  # Fetch, load, and preprocess data from yFinance
│   ├── Task_2_and_Task_3_Time_Series_Forecasting.ipynb  # Time series forecasting using ARIMA, SARIMA, LSTM
│   ├── Task_4_Optimize_Portfolio_Based_on_Forecast.ipynb  # Portfolio optimization
├── scripts               # Python scripts for automation
│   ├── yf_data_fetch.py  # Fetch stock data from yFinance
│   ├── stock_data_analyzer.py  # Preprocess and analyze stock data
│   ├── csv_loader.py  # Load and validate dataframes
│   ├── stock_price_forecasting.py  # Train and forecast stock prices
│   ├── portfolio_optimization.py  # Optimize portfolio allocation
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

## 🚀 Features
- **Data Collection**: Fetch historical stock and ETF data using `yFinance`.
- **Preprocessing**: Clean, normalize, and visualize data.
- **Time Series Forecasting**: Implement ARIMA, SARIMA, and LSTM models.
- **Portfolio Optimization**: Allocate assets based on forecasted returns.
- **Automated Workflows**: Utilize GitHub Actions for CI/CD.
- **Logging & Monitoring**: Track outputs for debugging and validation.

## 🛠️ Installation & Usage
### **1. Clone the Repository**
```sh
git clone https://github.com/zereay74/Kaim-week-11.git
```
### **2. Install Dependencies**
```sh
pip install -r requirements.txt
```
### **3. Run Jupyter Notebooks**
```sh
jupyter notebook
```
Open the notebooks inside the `notebooks/` folder to execute different tasks.

### **4. Automate Data Fetching**
```sh
python scripts/yf_data_fetch.py
```

### **5. Run Portfolio Optimization**
```sh
python scripts/portfolio_optimization.py
```

## 📈 Results & Insights
- **Optimized Portfolio Weights**: Determines the best asset allocation to maximize returns while minimizing risk.
- **Risk Metrics Analysis**: Includes **Sharpe Ratio, Volatility, VaR (95%)**.
- **Forecasting Accuracy**: Evaluates models based on RMSE, MAE, and MAPE.

## 🤝 Contributing
Contributions are welcome! Feel free to submit pull requests or report issues.



