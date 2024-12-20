# CTA Backtesting Framework for Low Tide Captial
**author: LTC project lab Group**
**date: 2024-10-24**

## Roadmap
1) Buildout a 60/40 portfolio
2) Compute traditionals ratios
3) Develop / Backtest the optimal method to trigger a trend
4) Implement the trend model into the 60/40 portfolio with an additional 10% allocation when triggered.
5) Develop a dynamic method to optimize the allocation percentage
6) Develop a new metric Portfolio C’s “resiliency” relative to Portfolio A.
7) Develop a metric to decide whether Port D should be used instead of Port B in Port C. 
8) try other alternative assets repalcing XAU with HFR, EHFI and SGCTA
9) try leveraged portfolio

## requirements

see attached requirements.txt

## usage

```bash
pip install -r requirements.txt

python 1_dataPreparation.py

python 2_calcFactorAndGenSignal.py

python 3_PnLEvaluation.py
```

**you can use the same environment to run the jupyter notebooks under the main folder**

## file structure

- data: store the data used in the project
- src: store the source code of the project
- risk-metric-models: store the risk metric models that we explored in the academic paper
- some jupyter notebooks under main folder: summary description about the strategy
  
## develop Calender

- 2024-10-24: start the project
- 2024-10-26: finish the basic structure of the project; can generate trend following strategy's PnL; can see the graph using excel
- 2024-10-28: thanks to Marshal, we get the raw data from bloomberg including bonds, SPY and XAU. We will build a classic
40/60 portfolio based on bonds and SPY. And add some metrics. Using quantstats to draw graphs.
- 2024-10-30：finish basic 60/40 portfolio and its PnL; build metrics to evaluate; use 6040 Portfolio as benchmark; Plot graphs for visiualization
- 2024-11-03: change Simple Moving Average to Exponential Moving Average; add more metrics to evaluate the strategy; add more graphs for visualization; grid search and 3D heatmap to find the best parameters
