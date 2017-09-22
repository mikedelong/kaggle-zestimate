
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

# transform tax delinquency year
properties['taxdelinquencyyear'] = properties['taxdelinquencyyear'].apply(
    lambda x: (17 - x if x < 20 else 117 - x) if pd.notnull(x) else x)

# min-max scaling with imputation via mean
min_max_scaler = MinMaxScaler(copy=True)
scaled_columns = list()
location_columns = ['latitude', 'longitude']
for column_name in location_columns:
    logger.debug('column %s has %d null values' % (column_name, properties[column_name].isnull().sum()))
    mean_value = properties[column_name].mean()
    logger.debug('column %s has mean value %.2f' % (column_name, mean_value))
    properties[column_name].fillna(inplace=True, value=mean_value)
    scaled_columns.append(column_name)
properties[scaled_columns] = min_max_scaler.fit_transform(properties[scaled_columns])

# take the log of select columns
log_columns = ['landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'taxamount', 'taxvaluedollarcnt',
               'calculatedfinishedsquarefeet']
for column_name in log_columns:
    properties[column_name] = properties[column_name].apply(lambda x: np.log(x) if pd.notnull(x) else x)


train_data = pd.read_csv(training_file)
logger.debug('training data read from %s complete' % training_file)

# drop out outliers
lower_limit = -0.4
upper_limit = 0.5
train_data = train_data[(train_data.logerror < upper_limit) & (train_data.logerror > lower_limit)]


train = train_data.merge(properties, how='left', on='parcelid')


logger.debug(train.info())




logger.debug('finished')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
