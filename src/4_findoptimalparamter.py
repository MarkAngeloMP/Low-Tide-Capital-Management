import os
import sys
import pandas as pd
from Config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
long_terms = [i for i in range(10, 256, 5)]
short_terms = [i for i in range(5, 97, 5)]

# find all combinations of long_term and short_term
# drop if short_term >= long_term
combinations = [(long_term, short_term) for long_term in long_terms for short_term in short_terms if short_term < long_term]
print(f'combinations: {len(combinations)}')
# use the combinations to find the optimal parameter
# 1. run 2_calcFactorAndGenSignal.py
# 2. run 3_PnLEvaluation.py

# for long_term, short_term in tqdm(combinations):
#     # print(f'processing : long_term: {long_term}, short_term: {short_term}')
#     os.system(f'python src/2_calcFactorAndGenSignal.py {long_term} {short_term}')
#     os.system(f'python src/3_PnLEvaluation.py {long_term} {short_term}')



df = pd.read_csv(os.path.join(DATA_DIR, 'summary', 'ma_strategy.csv'))

df['long_term'] = df['Unnamed: 0'].apply(lambda x: int(x.split(',')[0].strip('()"')))
df['short_term'] = df['Unnamed: 0'].apply(lambda x: int(x.split(',')[1].strip('()"')))
df.drop(columns=['Unnamed: 0'], inplace=True)

long_terms = sorted(df['long_term'].unique())
short_terms = sorted(df['short_term'].unique())
Long_Term, Short_Term = np.meshgrid(long_terms, short_terms)

Sharpe_Ratio = df.pivot_table(index='short_term', columns='long_term', values='sharp').values
max_drawdown = df.pivot_table(index='short_term', columns='long_term', values='max_drawdown').values
fig = go.Figure(data=[go.Surface(z=Sharpe_Ratio, x=long_terms, y=short_terms, surfacecolor=max_drawdown, colorscale='Viridis')])

# 添加标题和轴标签
fig.update_layout(
    title='Sharpe Ratio 3D Heatmap with Max Drawdown as Color',
    scene=dict(
        xaxis_title='Long Term',
        yaxis_title='Short Term',
        zaxis_title='Sharpe Ratio'
    ),
    coloraxis_colorbar=dict(
        title="max_drawdown"
    )
)

fig.write_html('sharpe_ratio_3d_heatmap.html')

fig.show()