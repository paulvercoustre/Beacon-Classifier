import os
import numpy as np
import pandas as pd

import data_manager as dm
import utils

from sklearn.model_selection import train_test_split

FLAGS = None
ROOT = os.path.join(os.path.dirname(__file__), '../')

seed = 0

csv_file_ = os.path.join(ROOT, 'data', 'no_filter.csv')
pickle_file = os.path.join(ROOT, 'cache', 'telemetry.npy')

