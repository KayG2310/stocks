import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import sys
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Load model ===
model = joblib.load("/Users/kamakshigupta/Desktop/stocks/backend/python/xgb_stock_predictor.pkl") 

tickers_order = [
    'AXISBANK', 'BAJFINANCE', 'BHARTIARTL', 'HCLTECH', 'HDFCBANK',  'HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC',
    'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC',  'RELIANCE', 'SBIN', 'SUNPHARMA', 
    'TATAMOTORS', 'TCS', 'TITAN'
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_next_day.py <STOCK_TICKER>")
        sys.exit(1)

    input_ticker = sys.argv[1].upper()

    if input_ticker not in tickers_order:
        print(f"❌ '{input_ticker}' not found in training tickers.")
        print("Valid tickers:", ", ".join(tickers_order))
        sys.exit(1)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=70)  # covers ~44 trading days

    yf_ticker = input_ticker + ".NS"
    df = yf.download(yf_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()

    if len(df) < 44:
        print(f"❌ Not enough data for {input_ticker}. Found only {len(df)} days.")
        sys.exit(1)

    one_hot = [1 if t == input_ticker else 0 for t in tickers_order]
    ohlc_flat = df.tail(44).to_numpy().flatten().tolist()
    X_input = np.array(one_hot + ohlc_flat).reshape(1, -1)

    # === Predict ===
    predicted = model.predict(X_input)[0]
    print(json.dumps(predicted.tolist()))

    # === Output ===
    # print(f"\n📈 Predicted OHLC for {input_ticker} (Next Day):")
    # print(f"Open : {predicted[0]:.2f}")
    # print(f"High : {predicted[1]:.2f}")
    # print(f"Low  : {predicted[2]:.2f}")
    # print(f"Close: {predicted[3]:.2f}")

