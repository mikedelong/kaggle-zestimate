
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
from sklearn.preprocessing import LabelEncoder

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

properties_file = '../input/properties_2016.csv'
training_file = '../input/train_2016_v2.csv'

properties = pd.read_csv(properties_file, dtype={
    'fireplaceflag': np.bool, 'hashottuborspa': np.bool,
    'propertycountylandusecode': np.str,
    'propertyzoningdesc': np.str}, converters={
    'taxdelinquencyflag': lambda x: np.bool(True) if x == 'Y' else np.bool(False)})  # avoid mixed type warning
logger.debug('properties read from %s complete' % properties_file)

# let's patch up the data before we make our training set
bool_columns = ['hashottuborspa', 'fireplaceflag']
for column_name in bool_columns:
    properties[column_name] = properties[column_name].apply(lambda x: False if pd.isnull(x) else True)

# transform these from labels to integers
for column_name in ['propertycountylandusecode', 'propertyzoningdesc']:
    label_encoder = LabelEncoder()
    label_encoder.fit(list(properties[column_name].values))
    properties[column_name] = label_encoder.transform(list(properties[column_name].values))

# transform from labels to integers and fill in NAs
for column_name in ['fips', 'regionidzip']:
    properties[column_name] = properties[column_name].fillna('ZZZ')
    label_encoder = LabelEncoder()
    label_encoder.fit(list(properties[column_name].values))
    properties[column_name] = label_encoder.transform(list(properties[column_name].values))

train_data = pd.read_csv(training_file)
logger.debug('training data read from %s complete' % training_file)
train = train_data.merge(properties, how='left', on='parcelid')


logger.debug(train.info())



logger.debug('finished')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
