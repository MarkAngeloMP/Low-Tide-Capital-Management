import yfinance as yf
import pandas as pd
import numpy as np
from Config import *

### params
stock_name = 'AAPL'



if __name__ == '__main__':
    stock_path = os.path.join(DATA_DIR, 'stock')
    file_path  = os.path.join(stock_path, stock_name + '.csv')
    df = pd.read_csv(filepath_or_buffer= file_path, parse_dates=['Date'])
    print(df)