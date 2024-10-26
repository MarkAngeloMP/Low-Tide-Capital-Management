import yfinance as yf
import pandas as pd
import numpy as np
from Config import *
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

### params
long_term = 20
short_term = 5

def calc_necessary_data(df):
    """calc necessary data like pct_change 
    
    """
    df['pct_change'] = df['Adj_Close'].pct_change() # daily return
    df['pct_change'].fillna(value=0, inplace=True)
    return df


def calc_factor(df):
    """
    calculate long term and short term 
    SMA and try to add more in the future
    EMA, Bolling Band, RSI, CCI, etc.
    Args:
        df (pd.DataFrame): raw data 
    """
    df[f'ma{long_term}'] = df['Adj_Close'].rolling(window=long_term, min_periods=1).mean()
    df[f'ma{short_term}'] = df['Adj_Close'].rolling(window=short_term, min_periods=1).mean()
    
    return df

def add_more_factor_external(df):  
    """
    pls make sure that the external data have 
    the same length as the original data;
    using merge and check with the config file
    """ 
    return df

def calc_signal(df):
    """calc_signal
    simple strategy with golden cross and dead cross
    all in and all out if signal is triggered
    
    Args:
        df (pd.Dataframe): df with raw data and factors
    """

    condition1 = df['ma5'] > df['ma20']
    condition2 = df['ma5'].shift(1) <= df['ma20'].shift(1)
    df.loc[condition1 & condition2, 'signal'] = 1
    
    condition1 = df['ma5'] < df['ma20']
    condition2 = df['ma5'].shift(1) >= df['ma20'].shift(1)
    df.loc[condition1 & condition2, 'signal'] = 0
    
    return df

if __name__ == '__main__':
    stock_path = os.path.join(DATA_DIR, 'stock')
    file_path  = os.path.join(stock_path, stock_name + '.csv')
    df = pd.read_csv(filepath_or_buffer= file_path, parse_dates=['Date'])
    
    df.sort_values(by='Date', inplace=True)
    df.drop_duplicates(subset='Date', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # calc necessary data
    df = calc_necessary_data(df)
    
    # calculate factors 
    df = calc_factor(df)
    
    # if you want to include any other factors that
    # from other files you can add it here
    df = add_more_factor_external(df)
    
    # calc_signal
    df = calc_signal(df)
    
    # save the file
    file_path = os.path.join(DATA_DIR, 'signal_data', stock_name + '.csv')
    df.to_csv(file_path, index=False)