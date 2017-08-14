import logging
import operator
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import lightgbm

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

# load in the data from CSV files
properties_file = '../input/properties_2016.csv'
training_file = '../input/train_2016_v2.csv'
logger.debug('loading properties data from %s' % properties_file)
properties = pd.read_csv(properties_file, dtype={
    'fireplaceflag': np.bool, 'hashottuborspa': np.bool,
    'propertycountylandusecode': np.str,
    'propertyzoningdesc': np.str}, converters={
    'taxdelinquencyflag': lambda x: np.bool(True) if x == 'Y' else np.bool(False)})  # avoid mixed type warning
logger.debug('loading training data from %s' % training_file)
train = pd.read_csv(training_file)
logger.debug('data load complete.')

# encode labels as integers as needed
for c in properties.columns:
    properties[c] = properties[c].fillna(1)
    if properties[c].dtype == 'object':
        label_encoder = LabelEncoder()
        label_encoder.fit(list(properties[c].values))
        properties[c] = label_encoder.transform(list(properties[c].values))

logger.debug('merging training data and properties on parcel ID')
train_df = train.merge(properties, how='left', on='parcelid')
logger.debug('dropping columns parcel ID, log error, and transaction date to get training data')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
logger.debug('dropping parcel ID from properties to get test data.')

# additional_columns_to_drop = ['typeconstructiontypeid', 'regionidcounty', 'architecturalstyletypeid',
#                               'threequarterbathnbr']
additional_columns_to_drop = []
test_columns_to_drop = ['parcelid'] + additional_columns_to_drop
x_test = properties.drop(test_columns_to_drop, axis=1)
# shape
logger.debug('train shape: %s, test shape: %s' % ((x_train.shape,), (x_test.shape,)))

# drop out outliers
outlier_limit = 0.36

train_df = train_df[abs(train_df.logerror) < outlier_limit]
logger.debug('After removing outliers train shape: {}; test shape unchanged.'.format(x_train.shape, ))
# todo figure out how to do this only once
train_columns_to_drop = ['parcelid', 'logerror', 'transactiondate'] + additional_columns_to_drop
x_train = train_df.drop(train_columns_to_drop, axis=1)
y_train = train_df['logerror'].values.astype(np.float32)
y_mean = np.mean(y_train)
logger.debug('y_train shape: %s' % (y_train.shape,))

dtrain = lightgbm.Dataset(x_train, label=y_train)
dtest = lightgbm.Dataset(x_test)

random_seed = 1
# xgboost parameters
lightgbm_parameters = {
    'bagging_fraction': 0.85,
    'bagging_freq': 20, # was 40,
    'bagging_seed': random_seed, # was 3
    'boosting_type': 'gbdt',
    'feature_fraction_seed': random_seed, # was 2
    'learning_rate': 0.002,
    'max_bin': 9, # was 10,
    'metric': 'mae', # was 'l1', 'l2', 'mae'
    'min_data': 500,
    'min_hessian': 0.05,
    'num_leaves': 60, # was 512,
    'objective': 'regression',
    'sub_feature': 0.3, # was 0.3,
    'verbose': 1
}


logger.debug('lightgbm parameters: %s' % lightgbm_parameters)
lightgbm_rounds = 200  # was 1000
watchlist = [(dtrain, 'train')]
watchlist = [dtest]

# todo figure out how to do early stopping
model = lightgbm.train(lightgbm_parameters, dtrain, lightgbm_rounds, watchlist)
logger.debug('model trained.')
t0 = model.best_iteration
logger.debug('model best iterations: %s' % str(model.best_iteration))
# predict
predictions = model.predict(x_test, num_iteration=max(model.best_iteration, 10))
y_predictions = np.array([str(round(each, 10)) for each in predictions])

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_predictions, '201611': y_predictions, '201612': y_predictions,
                       '201710': y_predictions, '201711': y_predictions, '201712': y_predictions})
# get the column headers
output_columns = output.columns.tolist()

# rearrange the columns to put the last one first
output = output[output_columns[-1:] + output_columns[:-1]]
logger.debug('our submission file has %d rows (should be 18232?)' % len(output))

use_gzip_compression = True
submission_prefix = 'zestimate'
output_filename = '{}{}.csv'.format(submission_prefix, datetime.now().strftime('%Y%m%d_%H%M%S'))
if use_gzip_compression:
    output_filename += '.gz'
    logger.debug('writing submission to %s' % output_filename)
    output.to_csv(output_filename, index=False, float_format='%.4f', compression='gzip')
else:
    logger.debug('writing submission to %s' % output_filename)
    output.to_csv(output_filename, index=False, float_format='%.4f')


zipped = zip(model.feature_name(), model.feature_importance())
importance = sorted(zipped, key=lambda x: x[1])
logger.debug('features by importance (ascending): %s' % importance)

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
