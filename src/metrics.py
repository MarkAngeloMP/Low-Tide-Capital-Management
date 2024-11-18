"""
    this file includes some metrics needed for evaluation
    metrics include:
    · Compound annualized growth
    · Annualized volatility
    · Annualized downside deviation
    · Max drawdown each year
    · Annualized Sharpe Ratio
    · Annualized Sortino Ratio
    · Annualized Omega Ratio
    · Annualized Calmar Ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_rolling_corr(benchmark, portfolios, window_size, plot=True):
    '''
    input:
    benchmark: pd.Series or single col pd.Dataframe; correlation of portfolios will be with respect to this benchmark
    portflios: pd.Dataframe; each column is % ret timeseries
    window_size: int; rolling window size
    plot: bool (default: True); Whether to include a plot
    '''
    
    # Check that benchmark is a series, if single col df, cast to series
    if isinstance(benchmark, pd.Series):
        pass
    elif isinstance(benchmark, pd.DataFrame) and len(benchmark.columns) == 1:
        benchmark = benchmark.iloc[:, 0]
    else:
        raise TypeError('Expected a pd.Series or a single-column pd.Dataframe for the benchmark')

    # Check that portfolios is a dataframe
    if not isinstance(portfolios, pd.DataFrame):
        raise TypeError('Expected a pd.DataFrame for the portfolios')
    
    # Check that y and x are the same len    
    if len(benchmark.index) != len(portfolios.index):
        raise ValueError(f'Benchmark and Porfolio data do not have the same length; ({len(benchmark.index)} and {len(portfolios.index)})')
    
    rolling_corr = portfolios.rolling(window=window_size).corr(benchmark).dropna()
    
    if plot:
        plt.figure(figsize=(14, 7))
        for column in portfolios.columns:
            plt.plot(rolling_corr[column], label=f'{column}')
        plt.title(f'Rolling Correlation with {benchmark.name}')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.axhline(0, color='black', lw=0.5, ls='--')  # Add a horizontal line at y=0
        plt.legend()
        plt.show()
    
    return rolling_corr
    
if __name__ == '__main__':
    
    FILEPATH = '/Users/gelo/REPOS/Low-Tide-Capital-Management/data/'
    strategy_data =  pd.read_csv(FILEPATH + 'pnl_data/XAUBNG.csv', parse_dates=['Date'], index_col=0)[['equity_change']].rename(columns={'pct_change':'Gold Strategy'})
    base_data =  pd.read_csv(FILEPATH + 'pnl_data/60_40Ports.csv', parse_dates=['Date'], index_col=0)[['ports_pctchange']].rename(columns={'ports_pctchange':'6040 Portfolio'})
    spx_data =  pd.read_csv(FILEPATH + 'stock/SPX.csv', parse_dates=['Date'], index_col=0).pct_change().rename(columns={'Close':'SPX'}).fillna(0)
    gold_data =  pd.read_csv(FILEPATH + 'stock/XAUBNG.csv', parse_dates=['Date'], index_col=0).pct_change().fillna(0).rename(columns={'Close':'XAU'})
    
    # combine data to test
    start = '1975'
    benchmark = base_data.loc[start:]
    portfolios = pd.DataFrame()
    for rets in [strategy_data, gold_data, spx_data]:
        portfolios = pd.concat([portfolios, rets.loc[start:]], axis=1)
    assert len(portfolios) == len(benchmark)
    
    rolling_corr = calc_rolling_corr(benchmark, portfolios, window_size=60)