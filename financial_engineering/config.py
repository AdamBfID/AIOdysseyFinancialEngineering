"""
Configuration file for Financial Engineering Project
All parameters and constants
"""

import os
from datetime import datetime, timedelta

# ========== PROJECT PATHS ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========== DATA CONFIGURATION ==========
# Stock symbols to analyze (TOP 10 FREE STOCKS)
STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'META', 'NVDA', 'JPM', 'JNJ', 'V'
]

# Time period for data collection
START_DATE = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')  # 4 years
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Data frequency
DATA_FREQUENCY = 'daily'  # daily, weekly, monthly

# ========== PREPROCESSING CONFIGURATION ==========
TEST_SIZE = 0.2                    # 80% train, 20% test
VALIDATION_SIZE = 0.1              # 10% validation from training set
LOOKBACK_WINDOW = 60               # Use 60 days to predict next day
NORMALIZATION_METHOD = 'minmax'    # minmax or zscore

# ========== LSTM MODEL CONFIGURATION ==========
LSTM_CONFIG = {
    'input_shape': (60, 5),         # (lookback_window, features)
    'hidden_units': [128, 64, 32],  # 3 LSTM layers
    'dropout_rate': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mse',
    'early_stopping_patience': 10,
    'validation_split': 0.1,
}

# ========== PORTFOLIO OPTIMIZATION ==========
PORTFOLIO_CONFIG = {
    'min_weight': 0.05,             # Minimum 5% per stock
    'max_weight': 0.25,             # Maximum 25% per stock
    'target_return': 0.12,          # Target 12% annual return
    'risk_free_rate': 0.05,         # 5% risk-free rate
    'rebalance_frequency': 'monthly',
}

# ========== ESG INTEGRATION ==========
ESG_CONFIG = {
    'esg_weight': 0.3,              # 30% weight for ESG factors
    'min_esg_score': 50,            # Minimum acceptable ESG score
    'exclude_industries': ['fossil_fuels', 'tobacco', 'weapons'],
}

# ========== BACKTESTING CONFIGURATION ==========
BACKTEST_CONFIG = {
    'initial_capital': 100000,      # Start with $100,000
    'transaction_cost': 0.001,      # 0.1% transaction cost
    'slippage': 0.0005,             # 0.05% slippage
    'rebalance_frequency': 'monthly',
    'benchmark': '^GSPC',           # S&P 500 benchmark
}

# ========== FEATURES FOR LSTM ==========
TECHNICAL_INDICATORS = [
    'close',
    'sma_20',                       # 20-day Simple Moving Average
    'rsi_14',                       # 14-day Relative Strength Index
    'macd',                         # MACD indicator
    'bbands',                       # Bollinger Bands
    'atr_14',                       # 14-day Average True Range
]

# ========== MODEL EVALUATION METRICS ==========
METRICS = {
    'regression': ['rmse', 'mae', 'mape', 'r2_score'],
    'portfolio': ['sharpe_ratio', 'max_drawdown', 'annual_return', 'win_rate'],
}

# ========== VISUALIZATION ==========
PLOT_CONFIG = {
    'figure_size': (15, 10),
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
    'dpi': 100,
}