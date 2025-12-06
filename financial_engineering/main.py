"""
Main script to run full Financial Engineering Pipeline:
1. Data loading & preprocessing
2. LSTM prediction
3. Portfolio optimization
4. Backtesting
5. Visualization
"""

import os
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.lstm_model import LSTMStockPredictor
from src.portfolio_optimizer import PortfolioOptimizer
from src.backtester import Backtester
from src.visualization import Visualizer
import config

def main():
    # -------------------------------
    # 1. LOAD STOCK AND ESG DATA
    # -------------------------------
    loader = DataLoader()
    stock_data = loader.download_stock_data()
    esg_csv_path = os.path.join(config.DATA_RAW, "kaggle_esg.csv")
    processed_data = loader.preprocess_data(stock_data, esg_csv_path=esg_csv_path)
    
    # -------------------------------
    # 2. PREPARE LSTM DATA (FIRST STOCK FOR EXAMPLE)
    # -------------------------------
    first_stock = config.STOCKS[0]
    X_scaled = processed_data[first_stock]['scaled']
    X, y = loader.create_sequences(X_scaled, lookback=config.LOOKBACK_WINDOW)

    # Split train/test (80/20)
    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # -------------------------------
    # 3. TRAIN LSTM MODEL
    # -------------------------------
    predictor = LSTMStockPredictor()
    predictor.model = predictor.build_model(input_shape=X_train[0].shape)
    predictor.train(X_train, y_train, X_test, y_test)
    lstm_results = predictor.evaluate(X_test, y_test)
    
    # Save predictions for visualization
    predictions = pd.Series(lstm_results['predictions'], index=range(len(y_test)))
    actuals = pd.Series(y_test, index=range(len(y_test)))

    # -------------------------------
    # 4. CALCULATE DAILY RETURNS FOR PORTFOLIO OPTIMIZATION
    # -------------------------------
    price_data = pd.DataFrame({
        stock: stock_data[stock]['Close'] if 'Close' in stock_data[stock].columns else stock_data[stock]['close']
        for stock in config.STOCKS
    })
    daily_returns = price_data.pct_change().dropna()
    cov_matrix = daily_returns.cov()
    
    # Load ESG scores
    esg_df = pd.read_csv(os.path.join(config.DATA_RAW, "esg_ratings.csv"))
    esg_scores = esg_df.set_index('ticker').loc[config.STOCKS, 'ESG'].values

    # -------------------------------
    # 5. OPTIMIZE PORTFOLIO
    # -------------------------------
    optimizer = PortfolioOptimizer(daily_returns, cov_matrix, esg_scores)
    optimal_weights = optimizer.optimize_portfolio()
    portfolio_stats = optimizer.get_portfolio_stats()
    
    print("\n📈 Portfolio Statistics:")
    for k, v in portfolio_stats.items():
        print(f"{k}: {v}")

    # -------------------------------
    # 6. RUN BACKTEST
    # -------------------------------
    backtester = Backtester(
        initial_capital=config.BACKTEST_CONFIG['initial_capital'],
        portfolio_weights=optimal_weights,
        price_data=price_data
    )
    backtest_results = backtester.run_backtest()
    backtester.save_results(os.path.join(config.RESULTS_DIR, "backtest_results.csv"))
    
    # -------------------------------
    # 7. VISUALIZATION
    # -------------------------------
    viz = Visualizer()
    
    # Stock prices
    fig_prices = viz.plot_stock_prices(price_data)
    fig_prices.show()

    # Returns distribution
    fig_returns = viz.plot_returns_distribution(daily_returns)
    fig_returns.show()

    # Correlation heatmap
    corr_matrix = daily_returns.corr()
    fig_corr = viz.plot_correlation_heatmap(corr_matrix)
    fig_corr.show()

    # Portfolio performance vs benchmark
    # For benchmark, download S&P500
    import yfinance as yf
    benchmark_df = yf.download(config.BACKTEST_CONFIG['benchmark'], start=config.START_DATE, end=config.END_DATE, progress=False)
    benchmark_value = benchmark_df['Close'][:len(backtest_results['portfolio_value'])]
    fig_portfolio = viz.plot_portfolio_performance(backtest_results['portfolio_value'], benchmark_value)
    fig_portfolio.show()

    # Drawdown
    fig_dd = viz.plot_drawdown(backtest_results['portfolio_value'])
    fig_dd.show()

    # Predictions vs actual
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(actuals, label='Actual', linewidth=2)
    plt.plot(predictions, label='Predicted', linewidth=2)
    plt.title(f"LSTM Predictions vs Actual for {first_stock}")
    plt.xlabel('Time')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
