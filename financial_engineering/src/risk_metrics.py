"""
Risk Metrics Calculation
"""

import numpy as np
import pandas as pd

class RiskMetrics:
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
    
    @staticmethod
    def max_drawdown(returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(returns, risk_free_rate=0.05):
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        max_dd = abs(RiskMetrics.max_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0
    
    @staticmethod
    def sortino_ratio(returns, target_return=0, periods=252):
        """Calculate Sortino ratio"""
        excess_return = returns.mean() * periods - target_return
        downside_returns = returns[returns < target_return]
        downside_std = downside_returns.std() * np.sqrt(periods)
        return excess_return / downside_std if downside_std != 0 else 0
    
    @staticmethod
    def var_95(returns):
        """Value at Risk (95% confidence)"""
        return np.percentile(returns, 5)
    
    @staticmethod
    def cvar_95(returns):
        """Conditional Value at Risk (95% confidence)"""
        var_95 = np.percentile(returns, 5)
        return returns[returns <= var_95].mean()