import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockPriceForecaster:
    def __init__(self, file_path):
        self.df = self.load_data(file_path)
        self.train, self.test = self.train_test_split()
        self.scaler = MinMaxScaler()
        self.models = {}
        self.forecasts = {}
    
    def load_data(self, file_path):
        """Load dataset."""
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        logging.info("Data loaded successfully.")
        return df

    def train_test_split(self, split_ratio=0.8):
        """Split data into train and test sets."""
        split_point = int(len(self.df) * split_ratio)
        train, test = self.df[:split_point], self.df[split_point:]
        logging.info(f"Data split into training ({len(train)}) and testing ({len(test)}).")
        return train, test

    def train_arima(self):
        """Train ARIMA model with optimal parameters."""
        auto_model = auto_arima(self.train['Close'], seasonal=False, trace=True)
        order = auto_model.order
        logging.info(f"Optimal ARIMA order: {order}")
        model = ARIMA(self.train['Close'], order=order).fit()
        self.models['ARIMA'] = model
        self.forecasts['ARIMA'] = model.forecast(steps=12)
    
    def train_sarima(self):
        """Train SARIMA model with optimal parameters."""
        auto_model = auto_arima(self.train['Close'], seasonal=True, m=12, trace=True)
        order, seasonal_order = auto_model.order, auto_model.seasonal_order
        logging.info(f"Optimal SARIMA orders: {order}, Seasonal: {seasonal_order}")
        model = SARIMAX(self.train['Close'], order=order, seasonal_order=seasonal_order).fit()
        self.models['SARIMA'] = model
        self.forecasts['SARIMA'] = model.forecast(steps=12)

    def train_lstm(self):
        """Train LSTM model with additional layers and fine-tuning."""
        train_scaled = self.scaler.fit_transform(self.train[['Close']])

        X_train, y_train = [], []
        for i in range(20, len(train_scaled)):
            X_train.append(train_scaled[i-20:i, 0])
            y_train.append(train_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(20, 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        # Generate 12-month forecast iteratively
        last_20_values = train_scaled[-20:].flatten().tolist()
        future_predictions = []
        
        for _ in range(12):
            input_seq = np.array(last_20_values[-20:]).reshape((1, 20, 1))
            pred = model.predict(input_seq)[0, 0]
            future_predictions.append(pred)
            last_20_values.append(pred)  # Append to extend the sequence

        # Inverse transform predictions
        self.models['LSTM'] = model
        self.forecasts['LSTM'] = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    def evaluate_arima(self):
        self.evaluate_model('ARIMA')
    
    def evaluate_sarima(self):
        self.evaluate_model('SARIMA')
    
    def evaluate_lstm(self):
        self.evaluate_model('LSTM')
    
    def evaluate_model(self, name):
        """Evaluate a model using MAE, RMSE, and MAPE."""
        forecast = self.forecasts[name]
        mae = mean_absolute_error(self.test['Close'][:12], forecast)
        rmse = sqrt(mean_squared_error(self.test['Close'][:12], forecast))
        mape = np.mean(np.abs((self.test['Close'][:12] - forecast) / self.test['Close'][:12])) * 100
        logging.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    def plot_forecasts(self):
        """Plot past and 12-month future forecast."""
        plt.figure(figsize=(12,6))
        plt.plot(self.train.index, self.train['Close'], label='Train')
        plt.plot(self.test.index, self.test['Close'], label='Actual')
        future_dates = pd.date_range(self.test.index[-1], periods=13, freq='M')[1:]
        for name, forecast in self.forecasts.items():
            plt.plot(future_dates, forecast, label=f'{name} Forecast')
        plt.legend()
        plt.show()
    # Store LSTM predictions in a DataFrame
    def get_lstm_forecast_df(self, name="LSTM_Predictions.csv"):
        """Store LSTM predictions in a DataFrame with Date and Close columns."""
        future_dates = pd.date_range(self.test.index[-1], periods=13, freq='M')[1:]
        lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'Close': self.forecasts['LSTM']})
        lstm_forecast_df.to_csv(name, index=False)
        logging.info(f"LSTM forecast saved as {name}")
        return lstm_forecast_df

''' 
    def run_all(self):
        """Run all models, evaluate and plot results."""
        self.train_arima()
        self.train_sarima()
        self.train_lstm()
        self.evaluate_arima()
        self.evaluate_sarima()
        self.evaluate_lstm()
        self.plot_forecasts()

if __name__ == "__main__":
    forecaster = StockPriceForecaster("tesla_cleaned.csv")
    forecaster.run_all()
'''