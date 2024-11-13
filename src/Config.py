import os
import sys
from glob import glob
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ignore all warnings
warnings.filterwarnings("ignore")

stock_name = 'XAUBNG'
initial_cash = 100000000
commission_fee = 0
slippage = 0.01
strategy_name = 'ma_strategy'

date_start = "1975-01-01"
# date_end = "2020-12-31"
date_end = None


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
ROOT_DIR = os.path.join(SRC_DIR, '..')
