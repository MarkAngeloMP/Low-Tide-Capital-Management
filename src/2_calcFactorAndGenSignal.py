import yfinance as yf
import pandas as pd
import numpy as np
from Config import *
import sys
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
### params
if len(sys.argv) == 3: # for optimization
    long_term = int(sys.argv[1])
    short_term = int(sys.argv[2])
else:
    long_term = 252
    short_term = 30

def calc_necessary_data(df):
    """calc necessary data like pct_change 
    
    """
    df['pct_change'] = df['Close'].pct_change() # daily return
    df['pct_change'].fillna(value=0, inplace=True)
    return df


def calc_factor(df, long_term=252, short_term=30):
    """
    calculate long term and short term 
    SMA and try to add more in the future
    EMA, Bolling Band, RSI, CCI, etc.
    Args:
        df (pd.DataFrame): raw data 
    """

    df[f'sma{long_term}'] = df['Close'].ewm(span=long_term, adjust=False).mean()
    df[f'sma{short_term}'] = df['Close'].ewm(span=short_term, adjust=False).mean()
    return df

def add_more_factor_external(df, long_term=252, short_term=30):  
    """
    pls make sure that the external data have 
    the same length as the original data;
    using merge and check with the config file
    """ 
    return df

def calc_signal(df, long_term=252, short_term=30):
    """calc_signal
    simple strategy with golden cross and dead cross
    all in and all out if signal is triggered
    
    Args:
        df (pd.Dataframe): df with raw data and factors
    """

    condition1 = df[f'sma{short_term}'] > df[f'sma{long_term}']
    condition2 = df[f'sma{short_term}'].shift(1) <= df[f'sma{long_term}'].shift(1)
    df.loc[condition1 & condition2, 'signal'] = 1
    
    condition1 = df[f'sma{short_term}'] < df[f'sma{long_term}']
    condition2 = df[f'sma{short_term}'].shift(1) >= df[f'sma{long_term}'].shift(1)
    df.loc[condition1 & condition2, 'signal'] = 0
    
    return df

if __name__ == '__main__':
    stock_path = os.path.join(DATA_DIR, 'stock')
    file_path  = os.path.join(stock_path, stock_name + '.csv')
    df = pd.read_csv(filepath_or_buffer= file_path, parse_dates=['Date'])
    
    df.sort_values(by='Date', inplace=True)
    df.drop_duplicates(subset='Date', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df['Date'] >= date_start]
    
    # calc necessary data
    df = calc_necessary_data(df)
    
    # calculate factors 
    df = calc_factor(df, long_term=long_term, short_term=short_term)
    
    # if you want to include any other factors that
    # from other files you can add it here
    df = add_more_factor_external(df)
    
    # calc_signal
    df = calc_signal(df, long_term=long_term, short_term=short_term)
    
    # save the file
    file_path = os.path.join(DATA_DIR, 'signal_data', stock_name + '.csv')
    df.to_csv(file_path, index=False)