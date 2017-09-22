
# loosely based on/following the outline of
# https://jessesw.com/XG-Boost/

import numpy as np
import pandas as pd
import xgboost as xgb
# todo deal with deprecation here
from sklearn.grid_search import GridSearchCV
import seaborn as sns
import logging
import time

from sklearn.metrics import accuracy_score

start_time = time.time()
# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

sns.set(font_scale = 1.5)



logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
