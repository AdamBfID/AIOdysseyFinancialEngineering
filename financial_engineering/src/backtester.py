"""
Backtesting Framework for Portfolio Strategy
"""

import numpy as np
import pandas as pd
from src.data_loader import DataLoader
import config

class Backtester:
    def __init__(self, initial_capital, portfolio_weights, price_data):
        """
        Initialize backtester
        """
        self.initial_capital = initial_capital
        self.portfolio_weights = portfolio_weights
        self.price_data = price_data
        self.results = None
    
    def run_backtest(self):
        """
        Run backtest and calculate performance metrics
        """
        print("\n🔄 Running Backtest...")
        
        # Calculate daily returns
        daily_returns = self.price_data.pct_change().dropna()
        
        # Portfolio returns
        portfolio_returns = (daily_returns * self.portfolio_weights).sum(axis=1)
        
        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = self.initial_capital * cumulative_returns
        
        # Calculate metrics
        total_return = (portfolio_value.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - config.PORTFOLIO_CONFIG['risk_free_rate']) / annual_volatility
        
        # Maximum Drawdown
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        self.results = {
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
        }
        
        print("\n📊 Backtest Results:")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Annual Return: {annual_return*100:.2f}%")
        print(f"Annual Volatility: {annual_volatility*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        
        return self.results
    
    def save_results(self, filepath):
        """
        Save backtest results
        """
        results_df = pd.DataFrame({
            'Portfolio Value': self.results['portfolio_value'],
            'Daily Returns': self.results['portfolio_returns']
        })
        results_df.to_csv(filepath)
        print(f"✅ Results saved to {filepath}")


def main():
    # Load stock data
    loader = DataLoader()
    stock_data = loader.download_stock_data()
    
    # Extract 'close' prices robustly
    price_data_dict = {}
    for stock in stock_data.keys():
        df = stock_data[stock]
        if 'close' in df.columns:
            price_data_dict[stock] = df['close']
        elif 'Close' in df.columns:
            price_data_dict[stock] = df['Close']
        else:
            raise KeyError(f"No close column found for {stock}")
    
    price_data = pd.DataFrame(price_data_dict)
    
    # Example: use equal weights or your optimized weights from portfolio optimizer
    n_stocks = len(price_data.columns)
    portfolio_weights = np.array([1/n_stocks] * n_stocks)  # replace with optimized weights if desired
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=100000,
        portfolio_weights=portfolio_weights,
        price_data=price_data
    )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Save results
    backtester.save_results("backtest_results.csv")


if __name__ == "__main__":
    main()
