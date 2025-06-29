# LSTM Stock Price Prediction with Real Sentiment Integration
# Integrated with sentiment_analyzer.py and index.js for future price prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import json
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SENTIMENT INTEGRATION MODULE
# =============================================================================

def convert_sentiment_to_numeric(sentiment_string, confidence_float):
    """
    Convert sentiment string and confidence to numeric values for the model
    
    Args:
        sentiment_string: "positive", "negative", or "neutral"
        confidence_float: confidence value (0.0 to 1.0)
    
    Returns:
        dict: Numeric representation of sentiment
    """
    sentiment_mapping = {
        'positive': 1.0,
        'negative': -1.0,
        'neutral': 0.0
    }
    
    # Get base sentiment score
    base_score = sentiment_mapping.get(sentiment_string.lower(), 0.0)
    
    # Apply confidence weighting
    weighted_score = base_score * confidence_float
    
    return {
        'sentiment_score': weighted_score,
        'sentiment_magnitude': confidence_float,
        'sentiment_category': sentiment_string.lower(),
        'raw_confidence': confidence_float
    }

def get_sentiment_from_analyzer(stock_name, ticker):
    """
    Interface with your sentiment_analyzer.py
    
    Args:
        stock_name: Company name (e.g., "Apple Inc.")
        ticker: Stock ticker (e.g., "AAPL")
    
    Returns:
        dict: Processed sentiment data
    """
    try:
        # Import your sentiment analyzer
        # Adjust the import path based on your file structure
        import sentiment_analyzer
        
        # Call your sentiment analyzer
        # Assuming it has a function that takes stock info and returns sentiment
        sentiment_result = sentiment_analyzer.analyze_stock_sentiment(stock_name, ticker)
        
        # Extract sentiment and confidence
        # Adjust these based on your sentiment_analyzer.py output format
        sentiment_string = sentiment_result.get('sentiment', 'neutral')
        confidence_float = sentiment_result.get('confidence', 0.5)
        
        # Convert to numeric format
        processed_sentiment = convert_sentiment_to_numeric(sentiment_string, confidence_float)
        
        return processed_sentiment
        
    except Exception as e:
        print(f"Error getting sentiment: {e}")
        # Return neutral sentiment as fallback
        return convert_sentiment_to_numeric('neutral', 0.5)

# =============================================================================
# STOCK DATA PROCESSING
# =============================================================================

def get_stock_data_with_sentiment(ticker, stock_name, period="1y"):
    """
    Get stock data and integrate with real-time sentiment
    
    Args:
        ticker: Stock ticker symbol
        stock_name: Full company name
        period: Data period for historical data
    
    Returns:
        DataFrame: Combined stock and sentiment data
    """
    print(f"Fetching data for {stock_name} ({ticker})...")
    
    # Get historical stock data
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period)
    
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Calculate technical indicators
    stock_data = calculate_technical_indicators(stock_data)
    
    # Get current sentiment from your analyzer
    current_sentiment = get_sentiment_from_analyzer(stock_name, ticker)
    
    # Create sentiment data for each date (using current sentiment for simplicity)
    # In production, you might want historical sentiment data
    sentiment_data = []
    for date in stock_data.index:
        sentiment_data.append({
            'Date': date,
            'sentiment_score': current_sentiment['sentiment_score'],
            'sentiment_magnitude': current_sentiment['sentiment_magnitude'],
            'sentiment_category': current_sentiment['sentiment_category'],
            'confidence': current_sentiment['raw_confidence']
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Reset index for merging
    stock_data.reset_index(inplace=True)
    
    # Merge stock and sentiment data
    merged_data = pd.merge(stock_data, sentiment_df, on='Date', how='left')
    merged_data.fillna(method='ffill', inplace=True)
    
    print(f"Data prepared with sentiment: {current_sentiment['sentiment_category']} (confidence: {current_sentiment['raw_confidence']:.2f})")
    
    return merged_data, current_sentiment

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for the stock data
    """
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Price change percentage
    df['Price_Change_Pct'] = df['Close'].pct_change()
    
    # Volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    
    return df

# =============================================================================
# DATA PREPROCESSING FOR FUTURE PREDICTION
# =============================================================================

def prepare_data_for_prediction(merged_data, time_steps=30):
    """
    Prepare data specifically for future prediction (not past fitting)
    """
    # Select features
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                     'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Price_Change_Pct']
    sentiment_features = ['sentiment_score', 'sentiment_magnitude', 'confidence']
    
    # Remove NaN values
    merged_data = merged_data.dropna()
    
    if len(merged_data) < time_steps:
        raise ValueError(f"Not enough data. Need at least {time_steps} days, got {len(merged_data)}")
    
    # Extract feature data
    price_data = merged_data[price_features].values
    sentiment_data = merged_data[sentiment_features].values
    target_data = merged_data['Close'].values
    
    # Initialize scalers
    price_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Fit scalers on all data
    price_scaled = price_scaler.fit_transform(price_data)
    sentiment_scaled = sentiment_scaler.fit_transform(sentiment_data)
    target_scaled = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
    
    # Prepare sequences for training (use 80% for training)
    train_size = int(len(price_scaled) * 0.8)
    
    X_price_train, y_train = create_sequences(price_scaled[:train_size], target_scaled[:train_size], time_steps)
    X_sentiment_train, _ = create_sequences(sentiment_scaled[:train_size], target_scaled[:train_size], time_steps)
    
    # Prepare the most recent sequence for future prediction
    X_price_latest = price_scaled[-time_steps:].reshape(1, time_steps, len(price_features))
    X_sentiment_latest = sentiment_scaled[-time_steps:].reshape(1, time_steps, len(sentiment_features))
    
    return {
        'X_price_train': X_price_train,
        'X_sentiment_train': X_sentiment_train,
        'y_train': y_train,
        'X_price_latest': X_price_latest,
        'X_sentiment_latest': X_sentiment_latest,
        'latest_price': merged_data['Close'].iloc[-1],
        'scalers': {
            'price_scaler': price_scaler,
            'sentiment_scaler': sentiment_scaler,
            'target_scaler': target_scaler
        },
        'feature_names': {
            'price_features': price_features,
            'sentiment_features': sentiment_features
        }
    }

def create_sequences(X, y, time_steps):
    """
    Create sequences for LSTM training
    """
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X[i-time_steps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# =============================================================================
# LSTM MODEL FOR FUTURE PREDICTION
# =============================================================================

def create_future_prediction_model(price_shape, sentiment_shape):
    """
    Create LSTM model optimized for future prediction
    """
    # Price input branch
    price_input = Input(shape=price_shape, name='price_input')
    price_lstm1 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(price_input)
    price_lstm2 = LSTM(32, return_sequences=False, dropout=0.2)(price_lstm1)
    
    # Sentiment input branch
    sentiment_input = Input(shape=sentiment_shape, name='sentiment_input')
    sentiment_lstm = LSTM(16, return_sequences=False, dropout=0.2)(sentiment_input)
    
    # Combine branches
    combined = concatenate([price_lstm2, sentiment_lstm])
    
    # Dense layers with regularization
    dense1 = Dense(64, activation='relu')(combined)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(16, activation='relu')(dropout2)
    
    # Output layer
    output = Dense(1, activation='linear')(dense3)
    
    # Create model
    model = Model(inputs=[price_input, sentiment_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',  # More robust to outliers
        metrics=['mae', 'mse']
    )
    
    return model

def train_prediction_model(model, data, epochs=100, batch_size=32):
    """
    Train the model for future prediction
    """
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='loss', patience=15, restore_best_weights=True, min_delta=0.0001
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=7, min_lr=0.00001
    )
    
    # Train the model
    history = model.fit(
        [data['X_price_train'], data['X_sentiment_train']], 
        data['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        validation_split=0.2
    )
    
    return history

# =============================================================================
# FUTURE PRICE PREDICTION
# =============================================================================

def predict_future_prices(model, data, current_sentiment, days_ahead=5):
    """
    Predict future stock prices using the trained model
    
    Args:
        model: Trained LSTM model
        data: Preprocessed data dictionary
        current_sentiment: Current sentiment data
        days_ahead: Number of days to predict
    
    Returns:
        dict: Future predictions with dates and prices
    """
    print(f"Predicting prices for next {days_ahead} days...")
    
    # Get the latest sequences
    latest_price_seq = data['X_price_latest'].copy()
    latest_sentiment_seq = data['X_sentiment_latest'].copy()
    
    predictions = []
    prediction_dates = []
    
    # Generate future dates
    last_date = datetime.now().date()
    
    for day in range(1, days_ahead + 1):
        # Predict next price
        pred_scaled = model.predict([latest_price_seq, latest_sentiment_seq], verbose=0)
        
        # Inverse transform prediction
        pred_price = data['scalers']['target_scaler'].inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_price)
        
        # Calculate next date (skip weekends for stock market)
        next_date = last_date + timedelta(days=day)
        while next_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            next_date += timedelta(days=1)
        prediction_dates.append(next_date)
        
        # Update sequences for next prediction
        # For price sequence: add predicted price as new 'Close' value
        new_price_row = latest_price_seq[0, -1, :].copy()
        new_price_row[3] = pred_scaled[0, 0]  # Update 'Close' price (index 3)
        
        # Shift sequence and add new row
        latest_price_seq = np.roll(latest_price_seq, -1, axis=1)
        latest_price_seq[0, -1, :] = new_price_row
        
        # For sentiment sequence: keep current sentiment (or update if you have new sentiment)
        # Here we keep the same sentiment for simplicity
        latest_sentiment_seq = np.roll(latest_sentiment_seq, -1, axis=1)
        latest_sentiment_seq[0, -1, :] = latest_sentiment_seq[0, -2, :]
    
    # Calculate prediction confidence and trends
    price_changes = np.diff(predictions)
    trend = "Bullish" if np.mean(price_changes) > 0 else "Bearish"
    
    return {
        'predictions': predictions,
        'dates': prediction_dates,
        'current_price': data['latest_price'],
        'trend': trend,
        'sentiment_impact': current_sentiment['sentiment_category'],
        'confidence': current_sentiment['raw_confidence'],
        'price_change_forecast': price_changes.tolist()
    }

# =============================================================================
# MAIN INTEGRATION FUNCTION (CALLED FROM index.js)
# =============================================================================

def predict_stock_prices(ticker, stock_name, days_ahead=5):
    """
    Main function to be called from your index.js
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        stock_name: Company name (e.g., "Apple Inc.")
        days_ahead: Number of days to predict
    
    Returns:
        dict: Complete prediction results
    """
    try:
        print(f"Starting prediction for {stock_name} ({ticker})")
        
        # 1. Get stock data with sentiment
        merged_data, current_sentiment = get_stock_data_with_sentiment(ticker, stock_name)
        
        # 2. Prepare data for prediction
        processed_data = prepare_data_for_prediction(merged_data, time_steps=30)
        
        # 3. Create and train model
        price_shape = (processed_data['X_price_train'].shape[1], processed_data['X_price_train'].shape[2])
        sentiment_shape = (processed_data['X_sentiment_train'].shape[1], processed_data['X_sentiment_train'].shape[2])
        
        print(f"Training model with shapes - Price: {price_shape}, Sentiment: {sentiment_shape}")
        
        model = create_future_prediction_model(price_shape, sentiment_shape)
        history = train_prediction_model(model, processed_data, epochs=50)
        
        # 4. Make future predictions
        future_results = predict_future_prices(model, processed_data, current_sentiment, days_ahead)
        
        # 5. Format results for your web application
        results = {
            'success': True,
            'ticker': ticker,
            'company_name': stock_name,
            'current_price': float(future_results['current_price']),
            'predictions': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': float(price),
                    'day_ahead': i + 1
                }
                for i, (date, price) in enumerate(zip(future_results['dates'], future_results['predictions']))
            ],
            'sentiment_analysis': {
                'sentiment': current_sentiment['sentiment_category'],
                'confidence': float(current_sentiment['raw_confidence']),
                'score': float(current_sentiment['sentiment_score'])
            },
            'market_forecast': {
                'trend': future_results['trend'],
                'expected_change_pct': float((future_results['predictions'][-1] - future_results['current_price']) / future_results['current_price'] * 100)
            },
            'model_performance': {
                'final_loss': float(history.history['loss'][-1]),
                'final_mae': float(history.history['mae'][-1])
            }
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'ticker': ticker,
            'company_name': stock_name
        }

# =============================================================================
# COMMAND LINE INTERFACE (FOR TESTING)
# =============================================================================

def main():
    """
    Command line interface for testing
    """
    if len(sys.argv) < 3:
        print("Usage: python lstm_predictor.py <TICKER> <COMPANY_NAME> [DAYS_AHEAD]")
        print("Example: python lstm_predictor.py AAPL 'Apple Inc.' 5")
        return
    
    ticker = sys.argv[1]
    company_name = sys.argv[2]
    days_ahead = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    # Run prediction
    results = predict_stock_prices(ticker, company_name, days_ahead)
    
    # Print results
    print("\n" + "="*50)
    print("STOCK PRICE PREDICTION RESULTS")
    print("="*50)
    
    if results['success']:
        print(f"Company: {results['company_name']} ({results['ticker']})")
        print(f"Current Price: ${results['current_price']:.2f}")
        print(f"Sentiment: {results['sentiment_analysis']['sentiment']} (confidence: {results['sentiment_analysis']['confidence']:.2f})")
        print(f"Market Forecast: {results['market_forecast']['trend']}")
        print(f"Expected Change: {results['market_forecast']['expected_change_pct']:.2f}%")
        
        print("\nFuture Price Predictions:")
        for pred in results['predictions']:
            print(f"  {pred['date']}: ${pred['predicted_price']:.2f}")
            
    else:
        print(f"Error: {results['error']}")

# =============================================================================
# EXPORT FUNCTION FOR index.js INTEGRATION
# =============================================================================

def export_prediction_function():
    """
    Export the main function for Node.js integration
    This can be called from your index.js using child_process
    """
    if __name__ == "__main__":
        # Check if called with JSON input (from Node.js)
        if len(sys.argv) == 2 and sys.argv[1].startswith('{'):
            try:
                # Parse JSON input from Node.js
                input_data = json.loads(sys.argv[1])
                ticker = input_data['ticker']
                company_name = input_data['company_name']
                days_ahead = input_data.get('days_ahead', 5)
                
                # Run prediction
                results = predict_stock_prices(ticker, company_name, days_ahead)
                
                # Output JSON for Node.js to parse
                print(json.dumps(results))
                
            except Exception as e:
                error_result = {
                    'success': False,
                    'error': str(e)
                }
                print(json.dumps(error_result))
        else:
            # Run command line interface
            main()

# Run the export function
export_prediction_function()