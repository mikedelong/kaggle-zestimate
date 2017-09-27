import logging
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import stateplane
from sklearn.preprocessing import MinMaxScaler

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

properties_file = '../input/properties_2016.csv'
training_file = '../input/train_2016_v2.csv'

properties = pd.read_csv(properties_file, dtype={
    'fireplaceflag': np.bool, 'hashottuborspa': np.bool,
    'propertycountylandusecode': np.str,
    'propertyzoningdesc': np.str}, converters={
    'taxdelinquencyflag': lambda x: np.bool(True) if x == 'Y' else np.bool(False)})  # avoid mixed type warning

properties['taxdelinquencyyear'] = properties['taxdelinquencyyear'].apply(
    lambda x: (2000 + x if x < 20 else 1900 + x) if pd.notnull(x) else x)

train_df = pd.read_csv(training_file)
train = train_df.merge(properties, how='left', on='parcelid')


logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
