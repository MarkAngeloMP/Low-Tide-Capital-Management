import pandas as pd 
import quantstats as qs
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
import warnings
from metrics import calc_rolling_corr
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

######################
#### Input Params ####
######################
alternative_weight = 0.2 # weight of alternative strategy
leveraged = True # will add the alternative weight to total if leveraged, else, total weight will sum to 1

# params for data-driven event windows
dd_threshold = -0.1 # minimum dd needed to count the event, include negative for loss
min_days = 14 # min number of days to consider event

# FILEPATH for pnl for baseline portfolio and alternative strategy
base_path = '../data/pnl_data/60_40Ports.csv'
alternative_path = '../data/pnl_data/XAUBNG.csv'
results_path = '../reports/event_study/'

# use full_window
full_window = False # True if we use the full geopolitcal window as specified, otherwise, will use a below 'during' specified date window from the 'start' date
rolling_window = 63
# days to look at pre / during / post geopolitical window
pre = 30
during = 20
post = 30

# Specify Geo-political windows
windows_events_dict = [{'event': 'Apartheid Sanctions', 'start':'1986-09-01', 'end':'1988-09-01'},
                  {'event': 'Invasion of Kuwait', 'start':'1990-08-01', 'end':'1991-03-01'},
                  {'event': '9-11 Attacks', 'start':'2001-09-01', 'end':'2003-09-01'},
                  {'event': 'Iraq War', 'start':'2003-01-01', 'end':'2003-05-01'},
                  {'event': 'Lebanon War', 'start':'2006-07-01', 'end':'2006-10-01'},
                  {'event': 'Lehman Bankruptcy', 'start':'2007-12-01', 'end':'2009-07-01'},
                  {'event': 'Crimea Annexation', 'start':'2014-02-01', 'end':'2014-05-01'},
                  {'event': 'Ukraine Invasion', 'start':'2022-02-01', 'end':'2024-10-01'}]


######################

def get_data(base_path, alternative_path):
    '''
    Reads the csv and pre-processes the data
    '''
    # read csv
    alternative_pnl =  pd.read_csv(alternative_path, index_col=0, parse_dates=['Date'])
    base_pnl =  pd.read_csv(base_path, index_col=0, parse_dates=['Date'])

    # combine returns to 1 df
    pnl = pd.concat([base_pnl['ports_pctchange'].rename('6040 ret').to_frame(), alternative_pnl['equity_change'].rename('Strategy ret').to_frame()], axis=1).dropna()
    # create combined portfolio
    base_weight = 1 if leveraged else 1-alternative_weight
    pnl[f'Combined ({base_weight*100}% / {alternative_weight*100}%)'] = pnl['6040 ret'] * base_weight + pnl['Strategy ret'] * alternative_weight

    # Gets the DD for each period
    pnl['6040 Equity'] = (1 + pnl['6040 ret']).cumprod()
    pnl['Rolling Max'] = pnl['6040 Equity'].cummax()
    pnl['Drawdown'] = (pnl['6040 Equity'] - pnl['Rolling Max'])/pnl['Rolling Max']
    
    return pnl

def get_data_driven_windows(pnl):
    '''
    This functions gets the windows which fall under the threshold and are at least as long as the min_days
    '''
    pnl['below_threshold'] = pnl['Drawdown'] <= dd_threshold
    pnl['group'] = (pnl['below_threshold']!= pnl['below_threshold'].shift(1)).cumsum()
    windows_dd = pnl[pnl['below_threshold']].groupby('group').apply(
        lambda x: pd.Series({'start': x.index[0], 'end': x.index[-1], 'duration':(x.index[-1] - x.index[0]).days + 1})
    ).reset_index(drop=True)
    windows_dd = windows_dd[windows_dd['duration'] >= min_days]
    windows_dd['event'] = windows_dd["start"].dt.strftime('%Y-%m') + " to " + windows_dd["end"].dt.strftime('%Y-%m')
    
    return windows_dd

def get_geopolitical_windows(pnl, windows_events=windows_events_dict):
    '''
    This function preprocesses the geopolitical windows
    '''
    windows_events = pd.DataFrame(windows_events)
    windows_events['start'] = pd.to_datetime(windows_events['start'])
    windows_events['end'] = pd.to_datetime(windows_events['end'])

    # adjust dates to align with original data index
    windows_events['start'] = pnl.index[pnl.index.get_indexer(windows_events['start'], method='nearest')]
    windows_events['end'] = pnl.index[pnl.index.get_indexer(windows_events['end'], method='nearest')]
    windows_events['duration'] = (windows_events['end']-windows_events['start']).dt.days
    
    return windows_events

def get_metrics(data, days=None, plot=False, name = None):
    '''
    This function gets some common metrics for evaluation
    '''
    if not days:
        days = len(data)
    res = pd.DataFrame(index = data.columns)
    res['Annualized Mean'] = data.mean() * (252)
    res['Annualized Vol'] = data.std() * np.sqrt(252)
    res['Annualized Downside Vol'] = data[data<0].std() * np.sqrt(252)
    res['Annualized Sharpe'] = res['Annualized Mean']/res['Annualized Vol']
    res['Annualized Sortino'] = res['Annualized Mean']/res['Annualized Downside Vol']
    IS_corr = data.corr()
    IS_corr = data.corr().rename(columns = lambda c: f'{c} corr')
    if name:
        res.index.name = name
    if plot:
        figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(IS_corr, annot=True, cmap='coolwarm', ax=ax)
        plt.title(name if name else 'IS Correlation')
    res = pd.concat([res, IS_corr], axis=1)
    return res

def run_event_study(pnl, windows):
    res = get_metrics(pnl, plot=False, name='Full-Sample')
    all_res = {}
    all_res['Full-sample'] = res
    all_res['Full-sample']['event'] = 'Full-sample'
    
    compiled_df = pd.DataFrame()
    for idx,row in windows.iterrows():
        data = pnl.loc[row['start']:row['end']]
        all_res[row['event']] = get_metrics(data, name = row['event'], plot=False)
        all_res[row['event']]['event'] = row['event']
        compiled_df = pd.concat([compiled_df, all_res[row['event']]], axis=0)
    compiled_df = compiled_df.reset_index().rename(columns={'index': 'Portfolio'})
    compiled_df = compiled_df.set_index(['event', 'Portfolio'])
        
    return compiled_df

def get_corr(results_df):
    events = results_df.index.get_level_values(0).unique().tolist()
    correlations = pd.DataFrame(index = events)
    correlations['Alternative Strategy correlation w/ Base'] = None
    for event in events:
        correlations.loc[event, 'Alternative Strategy correlation w/ Base'] = results_df.loc[event].loc['6040 ret', 'Strategy ret corr']
    correlations = correlations.applymap(lambda x: np.nan if pd.isna(x) else f'{x*100:.2f}%')
    correlations.rename_axis('event', axis=0, inplace=True)
    windows.set_index('event', inplace=True)
    correlations = pd.concat([correlations, windows], axis=1)
    return correlations

def save_rolling_corr(correlations):
    highlights = list(zip(correlations['start'], correlations['end']))
    focused_highlights = list(zip(correlations['start'], correlations['start'] + pd.offsets.Day(during)))
    event_periods = list(zip(correlations['start'] - pd.offsets.Day(pre), correlations['start'] + pd.offsets.Day(during + post)))
    calc_rolling_corr(pnl.iloc[:,0], pnl.iloc[:,1:], window_size=252, highlight_regions=highlights, plot_path=results_path, plot_name='Full-sample')
    
    for idx, event in enumerate(correlations.index):
        start, end = event_periods[idx][0], event_periods[idx][1]
        calc_rolling_corr(pnl.iloc[:,0], pnl.iloc[:,1:], window_size=rolling_window, highlight_regions=[focused_highlights[idx]], plot_path=results_path, plot_name=event, xlims = [start, end])
    

if __name__ == '__main__':
    
    # load data
    pnl = get_data(base_path, alternative_path)
    
    # get windows
    windows_dd = get_data_driven_windows(pnl)
    windows_events = get_geopolitical_windows(pnl, windows_events=windows_events_dict)
    # removes preprocessing cols
    pnl = pnl.iloc[:,0:3]
    windows = pd.concat([windows_events, windows_dd], axis=0).reset_index(drop=True)
    # results
    results_df = run_event_study(pnl, windows)
    full_correlations = get_corr(results_df)
    
    ### save results in excel
    # split data driven and geopolitical
    compiled_df_dd = results_df.loc[list(set([c for c,h in results_df.index if " to " in c]))]
    compiled_df_geopolitical = results_df.loc[list(set([c for c,h in results_df.index if " to " not in c]))]
    with pd.ExcelWriter( results_path + 'event_study_results.xlsx', engine='openpyxl') as writer:
        compiled_df_dd.to_excel(writer, sheet_name='Data Driven')
        compiled_df_geopolitical.to_excel(writer, sheet_name='Geopolitical')
        full_correlations.to_excel(writer, sheet_name='Full-Window Corrs')
    save_rolling_corr(full_correlations)
    