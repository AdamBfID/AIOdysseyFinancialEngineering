"""
Visualization and Dashboard Functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import config

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_stock_prices(self, price_data):
        """Plot historical stock prices"""
        fig, ax = plt.subplots(figsize=config.PLOT_CONFIG['figure_size'])
        
        for column in price_data.columns:
            normalized = price_data[column] / price_data[column].iloc[0]
            ax.plot(normalized, label=column, linewidth=2)
        
        ax.set_title('Normalized Stock Prices', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_returns_distribution(self, returns_data):
        """Plot returns distribution"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, stock in enumerate(returns_data.columns):
            axes[idx].hist(returns_data[stock], bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{stock} Returns Distribution')
            axes[idx].set_xlabel('Daily Returns')
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix):
        """Plot correlation matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Stock Correlation Matrix', fontsize=14, fontweight='bold')
        return fig
    
    def plot_portfolio_performance(self, portfolio_value, benchmark):
        """Plot portfolio vs benchmark"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=portfolio_value,
            name='Portfolio',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=benchmark,
            name='Benchmark',
            mode='lines',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio Performance vs Benchmark',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_drawdown(self, portfolio_value):
        """Plot drawdown over time"""
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=drawdown,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown Over Time',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        return fig