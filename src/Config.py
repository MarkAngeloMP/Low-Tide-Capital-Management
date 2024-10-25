import os
import sys
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
ROOT_DIR = os.path.join(SRC_DIR, '..')

commission_fee = 1.2 / 1000

date_start = "2010-01-01"
date_end = "2020-12-31"