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
    df.loc[df['position'] == 0, 'start_time'] = pd.NaT
    
    df.loc[open_pos_condition, 'equity_num'] = np.floor(initial_cash * (1-commission_fee) / (df['Adj_Close'] + slippage))
    df['cash'] = initial_cash - df['equity_num'] * (df['Adj_Close'] + slippage) * (1+commission_fee)
    df['equity_value'] = df['equity_num'] * df['Adj_Close']
    
    df['cash'].fillna(method='ffill', inplace=True)
    df.loc[df['position'] == 0, ['cash']] = None

    group_num = len(df.groupby('start_time'))
    if group_num > 1:
        t = df.groupby('start_time').apply(lambda x:x['Adj_Close'] / x.iloc[0]['Adj_Close'] * x.iloc[0]['equity_value'])  
        t = t.reset_index(level = [0])
        df['equity_value'] = t['Adj_Close']
    elif group_num == 1:
        df['equity_value'] = df.groupby('start_time')[['Adj_Close', 'equity_value']].apply(lambda x:x['Adj_Close'] / x.iloc[0]['Adj_Close'] * x.iloc[0]['equity_value'])
        df['equity_value'] = t.T.iloc[:, 0]
    
    df.loc[close_pos_condition, 'equity_num'] = df['equity_value'] / df['Adj_Close']
    df.loc[close_pos_condition, 'cash'] += df.loc[close_pos_condition, 'equity_num'] * (df['Adj_Close'] - slippage) * (1-commission_fee)
    df.loc[close_pos_condition, 'equity_value'] = 0
    
    df['net_value'] = df['cash'] + df['equity_value']   
    
    # calc pnl
    df['equity_change'] = df['net_value'].pct_change(fill_method=None)
    df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'net_value'] / initial_cash - 1  
    df['equity_change'].fillna(value=0, inplace=True)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()

    return df

def main():
    file_path = os.path.join(DATA_DIR, 'signal_data', stock_name + '.csv')
    df = pd.read_csv(filepath_or_buffer=file_path, parse_dates=['Date'])
    
    df = generate_position(df)
    
    df = calc_pnl(df)
    df.to_csv(os.path.join(DATA_DIR, 'pnl_data', stock_name + '.csv'), index=False)
if __name__ == '__main__':
    main()