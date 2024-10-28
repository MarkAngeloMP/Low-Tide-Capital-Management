# single cta backtesting framework for Low tide captial
**author: benchen liu**

**date: 2024-10-24(celebrating the programmer's day Lol)**

**email: benchenl@uchicago.edu**

## introduction
- Single cta backtesting framework for Low tide captial, mainly using simple moving average to generate signals.

- Optimizing the parameters of the moving average to maximize the return of the strategy.

- Aiming to create a new trading strategy that can be added to classic 60/40 paramter

- Add some metrics to evaluate the performance of the strategy

## TODOS
- require basic 60/40 portfolio and its PnL
- build metrics to evaluate
- add benchmark to compare the performance of the strategy
- plot a graph to show the performance of the strategy
- optimize the parameters of the moving average

## develop Calender
- 2024-10-24: start the project
- 2024-10-26: finish the basic structure of the project; can generate trend following strategy's PnL; can see the graph using excel
- 2024-10-28: thanks to Marshal, we get the raw data from bloomberg including bonds, SPY and XAU. We will build a classic 
40/60 portfolio based on bonds and SPY. And add some metrics.

## requirements
see attached requirements.txt

## usage
```bash
pip install -r requirements.txt

python 1_dataPreparation.py

python 2_calcFactorAndGenSignal.py

python 3_PnLEvaluation.py
```

## file structure
- data: store the data used in the project
- src: store the source code of the project
