"""
Portfolio Optimization using Modern Portfolio Theory + ESG
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config
from src.data_loader import DataLoader

class PortfolioOptimizer:
    def __init__(self, returns_data, cov_matrix, esg_scores):
        """
        Initialize optimizer with historical returns and covariance
        """
        self.returns_data = returns_data
        self.cov_matrix = cov_matrix
        self.esg_scores = esg_scores
        self.optimal_weights = None
    
    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        """
        portfolio_return = np.sum(self.returns_data.mean() * weights) * 252  # Annualized
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))  # Annualized
        sharpe_ratio = (portfolio_return - config.PORTFOLIO_CONFIG['risk_free_rate']) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def objective_function(self, weights):
        """
        Minimize negative Sharpe ratio (maximize Sharpe ratio)
        """
        _, _, sharpe = self.calculate_portfolio_metrics(weights)
        return -sharpe
    
    def esg_constraint(self, weights):
        """
        ESG weighted portfolio constraint
        Weighted average ESG score must be above minimum
        """
        portfolio_esg = np.dot(weights, self.esg_scores)
        return portfolio_esg - config.ESG_CONFIG['min_esg_score']
    
    def optimize_portfolio(self):
        """
        Optimize portfolio weights with constraints
        """
        n_stocks = len(self.returns_data.columns)
        
        # Initial guess: equal weight
        initial_weights = np.array([1/n_stocks] * n_stocks)
        
        # Constraints: sum to 1 & ESG minimum
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': self.esg_constraint}
        ]
        
        # Bounds: min/max weight per stock
        bounds = tuple([
            (config.PORTFOLIO_CONFIG['min_weight'], config.PORTFOLIO_CONFIG['max_weight'])
            for _ in range(n_stocks)
        ])
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print("⚠️ Optimization did not converge:", result.message)
        
        self.optimal_weights = result.x
        
        print("\n📊 Optimal Portfolio Allocation:")
        for stock, weight in zip(self.returns_data.columns, self.optimal_weights):
            print(f"{stock}: {weight*100:.2f}%")
        
        return self.optimal_weights
    
    def get_portfolio_stats(self):
        """
        Get final portfolio statistics
        """
        ret, vol, sharpe = self.calculate_portfolio_metrics(self.optimal_weights)
        stats = {
            'Expected Annual Return': f"{ret*100:.2f}%",
            'Annual Volatility': f"{vol*100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.4f}",
            'Max Weight': f"{np.max(self.optimal_weights)*100:.2f}%",
            'Min Weight': f"{np.min(self.optimal_weights)*100:.2f}%",
        }
        return stats


# -----------------------------
# USAGE
# -----------------------------
if __name__ == "__main__":
    # Load real stock data
    loader = DataLoader()
    stock_data = loader.download_stock_data()
    
    if not stock_data:
        raise ValueError("No stock data downloaded. Cannot optimize portfolio.")
    
    # Calculate daily returns
    returns_data = pd.DataFrame({stock: df['Close'].pct_change().dropna()
                                 for stock, df in stock_data.items()})
    
    # Covariance matrix
    cov_matrix = returns_data.cov()
    
    # Load ESG ratings
    esg_df = pd.read_csv('data/raw/esg_ratings.csv', index_col=0)
    esg_scores = esg_df.loc[config.STOCKS, 'ESG'].values  # align to STOCKS
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_data, cov_matrix, esg_scores)
    
    # Optimize portfolio
    weights = optimizer.optimize_portfolio()
    
    # Portfolio stats
    stats = optimizer.get_portfolio_stats()
    print("\n📈 Portfolio Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
