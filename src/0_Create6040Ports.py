from Config import *
import pandas as pd
import numpy as np
import quantstats as qs
qs.extend_pandas()

BONDS_TICKER = 'LUATTRUU'
STOCK_TICKER = 'SPTX'

if __name__ == '__main__':
    bond_df = pd.read_csv(os.path.join(os.path.join(DATA_DIR, 'stock'), f'{BONDS_TICKER}.csv'), parse_dates=['Date'])
    stock_df = pd.read_csv(os.path.join(os.path.join(DATA_DIR, 'stock'), f'{STOCK_TICKER}.csv'), parse_dates=['Date'])
    
    bond_df.sort_values('Date', inplace=True)
    stock_df.sort_values('Date', inplace=True)
    
    df = pd.merge(bond_df, stock_df, on='Date', how='inner', suffixes=('_bond', '_stock'))
    df = df[['Date', 'Close_bond', 'Close_stock']]
    df.columns = ['Date', 'Bond', 'Stock']
    for col in ['Bond', 'Stock']:
        df[col+'_pctchange'] = df[col].pct_change()
        df[col+'_pctchange'] = df[col+'_pctchange'].fillna(0)
        df[col+'_pnl'] = (1+df[col+'_pctchange']).cumprod()
    
    # classic 60/40 portfolio, No transaction cost
    df['ports_pctchange'] = 0.6*df['Bond_pctchange'] + 0.4*df['Stock_pctchange']
    df['ports_pnl'] = (1+df['ports_pctchange']).cumprod() 
    
    df.set_index('Date', inplace=True)
    qs.plots.snapshot(df['ports_pnl'], title='60/40Ports Pnl', show=True, savefig='60_40Ports.png')



