import os
import sys
from glob import glob
import warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ignore all warnings
warnings.filterwarnings("ignore")

stock_name = 'AAPL'
initial_cash = 1000000  
commission_fee = 1.2 / 1000
slippage = 0.01

date_start = "2010-01-01"
date_end = "2020-12-31"


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
ROOT_DIR = os.path.join(SRC_DIR, '..')
