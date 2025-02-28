# **Time Series Forecasting for Portfolio Management Optimization**

## **Overview**
This project focuses on **time series forecasting** for **portfolio management optimization** by analyzing historical financial data of **Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY)**. The goal is to extract insights, assess risk, and build predictive models to optimize investment strategies.

### **Key Features:**
- **Data Collection**: Fetch historical financial data from Yahoo Finance using `yfinance`.
- **Data Preprocessing**: Handle missing values, normalize data, and ensure proper formatting.
- **Exploratory Data Analysis (EDA)**: Visualize trends, volatility, and detect anomalies.
- **Risk Assessment**: Compute **Value at Risk (VaR)** and **Sharpe Ratio** to measure risk-adjusted returns.
- **Predictive Modeling (Upcoming)**: Implement ARIMA, GARCH, and machine learning models.

---

## **Project Structure**
```
├── .github/workflows     # GitHub Actions workflows for CI/CD
├── .vscode               # VS Code settings and extensions
├── logs                  # Logs for monitoring outputs
│   ├── logs.log
├── notebooks             # Jupyter notebooks for data processing & insights
│   ├── Task_1_Preprocess_and_Explore_the_Data.ipynb # Fetch, load, and preprocess data
├── scripts               # Python scripts for automation
│   ├── yf_data_fetch.py   # Fetch historical stock data from yfinance
│   ├── stock_data_analyzer.py # Preprocess and analyze stock data
├── tests                 # Unit tests for data validation & pipeline integrity
├── .gitignore            # Ignore unnecessary files
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

---

## **Installation & Setup**

### **1. Clone the Repository**
```sh
 git clone https://github.com/zereay74/Kaim-week-11.git
cd Kaim-week-11
```

### **2. Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## **Usage**

### **1. Fetch Historical Stock Data**
Run the script to download data for TSLA, BND, and SPY:
```sh
python scripts/yf_data_fetch.py
```

### **2. Preprocess & Explore Data**
Analyze missing values, outliers, and visualize trends:
```sh
python scripts/stock_data_analyzer.py
```

### **3. Run Jupyter Notebook for Analysis**
```sh
jupyter notebook notebooks/Task_1_Preprocess_and_Explore_the_Data.ipynb
```

---

## **Results & Insights**

### **1. Stock Price Trends Over Time**
- TSLA shows high volatility and rapid price surges.
- BND remains stable, with minimal fluctuations.
- SPY exhibits steady growth with periodic corrections.

### **2. Volatility Analysis**
- TSLA’s volatility is significantly higher than SPY and BND.
- Rolling standard deviations help visualize market swings.

### **3. Risk Assessment (VaR & Sharpe Ratio)**
- TSLA has the highest risk exposure but offers higher potential returns.
- BND is a low-risk asset suitable for stability.
- SPY balances risk and return effectively.

---

## **Next Steps**
- Implement **Time Series Forecasting** using ARIMA, GARCH, and ML models.
- Develop a **Portfolio Optimization Strategy** based on historical trends.
- Automate the entire pipeline using CI/CD workflows.

---

