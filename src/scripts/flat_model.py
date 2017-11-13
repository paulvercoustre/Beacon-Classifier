import os

FLAGS = None
ROOT = os.path.join(os.path.dirname(__file__), '../')

seed = 0

csv_file_ = os.path.join(ROOT, 'data', 'no_filter.csv')
pickle_file = os.path.join(ROOT, 'cache', 'telemetry.npy')

