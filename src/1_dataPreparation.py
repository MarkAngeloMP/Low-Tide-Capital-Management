import yfinance as yf
import pandas as pd
import numpy as np
from Config import *

def get_tickers():
    # Download data
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    for ticker in tickers:
        data = yf.download(ticker, start=date_start, end=date_end).rename(columns={'Adj Close': 'Adj_Close'})
        data.to_csv(os.path.join(os.path.join(DATA_DIR, 'stock'), f'{ticker}.csv'))

def get_index():
    tickers = ['SPY', 'QQQ', 'DJI']
    
    for ticker in tickers:
        data = yf.download(ticker, start=date_start, end=date_end).reset_index().rename(columns={'Adj Close': 'Adj_Close'})
        data.to_csv(os.path.join(os.path.join(DATA_DIR, 'index'), f'{ticker}.csv'))    


if __name__ == "__main__":
    print("Downloading data...")
    get_index()
    get_tickers()
    print("Data downloaded!")
