"""
LSTM Neural Network for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import config

class LSTMStockPredictor:
    def __init__(self):
        self.model = None
        self.history = None
        self.scaler = None
    

    def build_model(self, input_shape):
        """
        Build LSTM architecture
        3-layer LSTM with dropout for regularization
        Uses keras.Input and will fall back to CPU if GPU JIT compilation fails.
        """
        def _make_model():
            inputs = keras.Input(shape=input_shape)
            x = layers.LSTM(
                units=config.LSTM_CONFIG['hidden_units'][0],
                return_sequences=True,
                activation='tanh'
            )(inputs)
            x = layers.Dropout(config.LSTM_CONFIG['dropout_rate'])(x)

            x = layers.LSTM(
                units=config.LSTM_CONFIG['hidden_units'][1],
                return_sequences=True,
                activation='tanh'
            )(x)
            x = layers.Dropout(config.LSTM_CONFIG['dropout_rate'])(x)

            x = layers.LSTM(
                units=config.LSTM_CONFIG['hidden_units'][2],
                activation='tanh'
            )(x)
            x = layers.Dropout(config.LSTM_CONFIG['dropout_rate'])(x)

            outputs = layers.Dense(1)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            optimizer = keras.optimizers.Adam(learning_rate=config.LSTM_CONFIG['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=config.LSTM_CONFIG['loss'],
                metrics=['mae', 'mse']
            )
            return model

        try:
            model = _make_model()
        except tf.errors.UnknownError:
            print("⚠️ GPU JIT failed — rebuilding model on CPU (this will be slower).")
            with tf.device('/CPU:0'):
                model = _make_model()

        print("\n🧠 LSTM Model Architecture:")
        model.summary()
        return model

    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train LSTM model with early stopping
        """
        print("\n📈 Training LSTM Model...")
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.LSTM_CONFIG['early_stopping_patience'],
            restore_best_weights=True
        )
        
        try:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.LSTM_CONFIG['epochs'],
                batch_size=config.LSTM_CONFIG['batch_size'],
                callbacks=[early_stop],
                verbose=1
            )
        except tf.errors.UnknownError as e:
            print("⚠️ GPU JIT failed during training — rebuilding model on CPU and retrying (slower).")
            # Rebuild model explicitly on CPU and retry
            input_shape = self.model.input_shape[1:]
            with tf.device('/CPU:0'):
                self.model = self.build_model(input_shape)
                self.history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=config.LSTM_CONFIG['epochs'],
                    batch_size=config.LSTM_CONFIG['batch_size'],
                    callbacks=[early_stop],
                    verbose=1
                )
        
        print("✅ Training complete!")
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        predictions = predictions.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        print("\n📊 Model Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²:   {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': predictions
        }
    
    def save_model(self, filepath):
        """
        Save trained model
        """
        self.model.save(filepath)
        print(f"\n💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load pre-trained model
        """
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")

# Usage example
if __name__ == "__main__":
    from src.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    stock_data = loader.download_stock_data()
    processed_data = loader.preprocess_data(stock_data)
    
    # Prepare sequences for first stock
    first_stock = config.STOCKS[0]
    X_scaled = processed_data[first_stock]['scaled']
    X, y = loader.create_sequences(X_scaled, config.LOOKBACK_WINDOW)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    predictor = LSTMStockPredictor()
    predictor.model = predictor.build_model(X_train[0].shape)
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save_model(f"{config.MODELS_DIR}/lstm_model_final.h5")