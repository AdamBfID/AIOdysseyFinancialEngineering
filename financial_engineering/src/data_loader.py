"""
Data collection and preprocessing module
Downloads FREE market data from Yahoo Finance API
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import config
import time

class DataLoader:
    def __init__(self):
        self.stocks = config.STOCKS
        self.start_date = config.START_DATE
        self.end_date = config.END_DATE
        self.scaler = MinMaxScaler() if config.NORMALIZATION_METHOD == "minmax" else StandardScaler()

        # Ensure folders exist
        os.makedirs(config.DATA_RAW, exist_ok=True)
        os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    # --------------------------------------------------
    # 1. DOWNLOAD PRICE DATA
    # --------------------------------------------------
    def download_stock_data(self, retries=3, pause=2):
        print(f"📊 Downloading stock data for {len(self.stocks)} stocks...")
        data = {}

        for stock in tqdm(self.stocks, desc="Downloading"):
            for attempt in range(retries):
                try:
                    df = yf.download(
                        stock,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=True
                    )

                    if df.empty:
                        print(f"⚠️ Warning: {stock} returned no data")
                        break

                    # Flatten columns if multi-index
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    data[stock] = df
                    print(f"✅ {stock}: {len(df)} trading days")
                    break

                except Exception as e:
                    print(f"❌ {stock} attempt {attempt+1} failed: {e}")
                    time.sleep(pause)
                    continue
            else:
                print(f"❌ Failed to download {stock} after {retries} attempts")

        if not data:
            print("❌ No data downloaded. Check internet or Yahoo Finance access.")
            return {}

        combined_data = pd.concat(data, axis=1)
        combined_data.to_csv(f"{config.DATA_RAW}/stock_prices.csv")
        print(f"\n✅ Raw data saved to {config.DATA_RAW}/stock_prices.csv")
        return data

    # --------------------------------------------------
    # 2. LOAD ESG DATA FROM KAGGLE CSV
    # --------------------------------------------------
    def load_kaggle_esg(self, csv_path):
        esg_df = pd.read_csv(csv_path)
        print("ESG CSV columns:", esg_df.columns.tolist())

        if 'ticker' in esg_df.columns and 'total_score' in esg_df.columns:
            # Normalize tickers to uppercase
            esg_df['ticker'] = esg_df['ticker'].str.upper()
            esg_mapping = dict(zip(esg_df['ticker'], esg_df['total_score']))

            # Map only stocks in our list
            esg_mapping = {k: esg_mapping.get(k, 50) for k in self.stocks}

            esg_df_out = pd.DataFrame(list(esg_mapping.items()), columns=['ticker', 'ESG'])
            esg_df_out.to_csv(f"{config.DATA_RAW}/esg_ratings.csv", index=False)
            print(f"✅ ESG data saved to {config.DATA_RAW}/esg_ratings.csv")
            return esg_mapping
        else:
            raise ValueError("CSV missing required columns: 'ticker' or 'total_score'")


    # --------------------------------------------------
    # 3. TECHNICAL INDICATORS
    # --------------------------------------------------
    def add_technical_indicators(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ["Close", "High", "Low"]:
            if col not in df.columns:
                raise ValueError(f"Column {col} missing in dataframe")

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        df["sma_20"] = ta.trend.sma_indicator(close, window=20)
        df["rsi_14"] = ta.momentum.rsi(close, window=14)
        df["macd"] = ta.trend.macd_diff(close)

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bbands_high"] = bb.bollinger_hband()
        df["bbands_mid"] = bb.bollinger_mavg()
        df["bbands_low"] = bb.bollinger_lband()

        df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

        return df

    # --------------------------------------------------
    # 4. PREPROCESS FOR LSTM
    # --------------------------------------------------
    def preprocess_data(self, data, esg_csv_path=None):
        print("\n🔧 Preprocessing data...")
        processed_data = {}

        esg_mapping = {}
        if esg_csv_path:
            esg_mapping = self.load_kaggle_esg(esg_csv_path)

        for stock, df in tqdm(data.items(), desc="Processing"):
            df = df.copy()
            df = self.add_technical_indicators(df)
            df = df.dropna()

            features = ["Close", "sma_20", "rsi_14", "macd", "atr_14"]

            # Add ESG as constant column
            esg_value = esg_mapping.get(stock, 50)
            df["ESG"] = esg_value
            features.append("ESG")

            X = df[features].values
            X_scaled = self.scaler.fit_transform(X)

            processed_data[stock] = {
                "original": df,
                "scaled": X_scaled,
                "features": features,
                "scaler": self.scaler
            }

        combined = pd.concat(
            {
                stock: pd.DataFrame(data["scaled"], columns=data["features"])
                for stock, data in processed_data.items()
            },
            axis=1
        )

        combined.to_csv(f"{config.DATA_PROCESSED}/normalized_prices.csv")
        print(f"✅ Processed data saved to {config.DATA_PROCESSED}/normalized_prices.csv")

        return processed_data

    # --------------------------------------------------
    # 5. LSTM SEQUENCES
    # --------------------------------------------------
    def create_sequences(self, data, lookback=60):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback, 0])
        return np.array(X), np.array(y)


# --------------------------------------------------
# USAGE
# --------------------------------------------------
if __name__ == "__main__":
    loader = DataLoader()
    stock_data = loader.download_stock_data()

    if stock_data:
        kaggle_esg_csv = "data/raw/kaggle_esg.csv"
        processed_data = loader.preprocess_data(stock_data, esg_csv_path=kaggle_esg_csv)
        print("\n✅ Data loading and preprocessing complete!")
    else:
        print("❌ Aborting: No stock data downloaded.")
