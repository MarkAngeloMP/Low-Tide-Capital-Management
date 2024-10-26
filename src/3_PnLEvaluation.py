import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Config import *
from metrics import *
pd.set_option('expand_frame_repr', False)


def generate_position(df):
    """
    generate position based on signal
    Args:
        df (pd.DataFrame): raw data with signal
    """
    df['signal'].fillna(method='ffill', inplace=True)
    df['signal'].fillna(value=0, inplace=True)
    
    # you can trade on the next day after signal is triggered
    df['position'] = df['signal'].shift()
    df['position'].fillna(value=0, inplace=True)

    return df


def calc_pnl(df):
    '''
    calculate pnl based on position
    plus transaction cost and slippage
    Args:
        df (pd.DataFrame): raw data with position
    '''
    
    # find open positoin time
    condition1 = df['position'] != 0
    condition2 = df['position'] != df['position'].shift(1)
    open_pos_condition = condition1 & condition2
    
    # find close position time
    condition1 = df['position'] != 0
    condition2 = df['position'] != df['position'].shift(-1)
    close_pos_condition = condition1 & condition2   
    
    # time that we don't have positoins is NaT
    df.loc[open_pos_condition, 'start_time'] = df['Date']
    df['start_time'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT
    
    
    
    
    return df

def main():
    file_path = os.path.join(DATA_DIR, 'signal_data', stock_name + '.csv')
    df = pd.read_csv(filepath_or_buffer=file_path, parse_dates=['Date'])
    
    df = generate_position(df)
    
    df = calc_pnl(df)

if __name__ == '__main__':
    main()