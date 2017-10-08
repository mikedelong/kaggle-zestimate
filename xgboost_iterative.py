import logging
import operator
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os

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

result_file = './iterative_result.csv'

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

do_data_cleanup = True
if do_data_cleanup:
    properties['hasbasement'] = properties["basementsqft"].apply(lambda x: 0 if np.isnan(x) else 1).astype(float)
    properties['fireplacecnt'] = properties['fireplacecnt'].apply(lambda x: 0 if np.isnan(x) else x).astype(float)

    # drop a duplicate column that Zillow doesn't use
    properties = properties.drop(['bathroomcnt'], axis=1)

# drop out outliers
outlier_limit = 0.36
lower_limit = -0.36
upper_limit = 0.375

properties_copy = properties.copy(deep=True)
t0 = train.merge(properties_copy, how='left', on='parcelid')
if False:
    t0 = t0[abs(t0.logerror) < outlier_limit]
else:
    t0 = t0[(t0.logerror < upper_limit) & (t0.logerror > lower_limit)]

t0['transactiondate'] = pd.to_datetime(t0['transactiondate'])

t0['month'] = t0['transactiondate'].dt.month
t1 = t0[t0['month'] < 10]
t3 = t0.shape
t4 = t1.shape
t6 = t1.logerror.values
t7 = t6.shape

t8 = t1.drop(['logerror', 'transactiondate'], axis=1)
t2 = t0[t0['month'] >= 10]
t5 = t2.shape
t12 = t2.logerror.values
t11 = t2.drop(['logerror', 'transactiondate'], axis=1)
for c in t8.dtypes[t8.dtypes == object].index.values:
    t8[c] = (t8[c] is True)

# t8 = t8.values.astype(np.float32, copy=False)
t9 = xgb.DMatrix(t8, t6)
t10 = xgb.DMatrix(t11)

logger.debug('merging training data and properties on parcel ID')
train_df = train.merge(properties, how='left', on='parcelid')
logger.debug('dropping columns parcel ID, log error, and transaction date to get training data')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
logger.debug('dropping parcel ID from properties to get test data.')

if False:
    additional_columns_to_drop = ['typeconstructiontypeid', 'regionidcounty', 'architecturalstyletypeid',
                                  'threequarterbathnbr']
else:
    additional_columns_to_drop = []
test_columns_to_drop = ['parcelid'] + additional_columns_to_drop
x_test = properties.drop(test_columns_to_drop, axis=1)
# shape
logger.debug('train shape: %s, test shape: %s' % ((x_train.shape,), (x_test.shape,)))

if False:
    train_df = train_df[abs(train_df.logerror) < outlier_limit]
else:
    train_df = train_df[(train_df.logerror > lower_limit) & (train_df.logerror < upper_limit)]

logger.debug('After removing outliers train shape: {}; test shape unchanged.'.format(x_train.shape, ))
# todo figure out how to do this only once
train_columns_to_drop = ['parcelid', 'logerror', 'transactiondate'] + additional_columns_to_drop
x_train = train_df.drop(train_columns_to_drop, axis=1)
y_train = train_df['logerror'].values.astype(np.float32)
y_mean = np.mean(y_train)
logger.debug('y_train shape: %s' % (y_train.shape,))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

random_seed = 1
# xgboost parameters
xgboost_parameters = {
    'alpha': 0.0,
    'base_score': y_mean,
    'eta': 0.025,  # todo try a range of values from 0 to 0.1 (?) default = 0.03 # was 0.003
    'eval_metric': 'mae',
    'gamma': 0.0,  # default is 0
    'lambda': 1.0,  # default is 1.0
    'max_depth': 7,  # todo try a range of values from 3 to 7 (?) default = 6
    'objective': 'reg:linear',
    'seed': random_seed,
    'silent': 1,
    'subsample': 0.80
}
best_xgboost_parameters = xgboost_parameters.copy()
best_error = sys.maxint

xgb_boost_rounds = 1200  # was 1000
# cross-validation
cross_validation_nfold = 5
for eta in [0.0252, 0.0255, 0.257]:
    for subsample in [0.7, 0.725, 0.75, 0.775, 0.785, 0.8]:
        for max_depth in [6, 7, 8, 9]:
            xgboost_parameters['max_depth'] = max_depth
            xgboost_parameters['eta'] = eta
            xgboost_parameters['subsample'] = subsample
            logger.debug('xgboost parameters: %s' % xgboost_parameters)

            cv_result_small = xgb.cv(xgboost_parameters, t9,
                                     early_stopping_rounds=30,
                                     nfold=cross_validation_nfold,
                                     num_boost_round=xgb_boost_rounds,
                                     seed=random_seed,
                                     show_stdv=False,
                                     verbose_eval=25)
            actual_small_boost_rounds = len(cv_result_small)
            logger.debug('for small boost we actually used %d rounds' % actual_small_boost_rounds)
            model_small = xgb.train(dict(xgboost_parameters, silent=1), dtrain=t9,
                                    num_boost_round=actual_small_boost_rounds)  # ,evals=[(t9, 'train')])
            predictions_small = model_small.predict(t10)

            error_result = mean_absolute_error(t12, predictions_small)
            logger.debug('mean absolute error from small model: %.6f' % error_result)
            logger.debug('iterate data: %.3f %.3f %d %.4f %.3f %d: %d %.7f' %
                         (lower_limit, upper_limit, cross_validation_nfold, eta, subsample, max_depth,
                          actual_small_boost_rounds, error_result))

            if os.path.exists(result_file):
                append_write = 'ab'
            else:
                append_write = 'wb'
            with open(result_file, append_write) as result_fp:
                result_fp.write(
                    '%.3f,%.3f,%d,%.4f,%.3f,%d,%d,%.7f\n' % (lower_limit, upper_limit, cross_validation_nfold, eta,
                                                                subsample, max_depth, actual_small_boost_rounds,
                                                                error_result))
            if error_result < best_error:
                best_error = error_result
                best_xgboost_parameters['eta'] = eta
                best_xgboost_parameters['max_depth'] = max_depth
                best_xgboost_parameters['subsample'] = subsample
                logger.debug('switching eta to %.3f, subsample to %.3f' % (eta, subsample))
logger.debug('after iteration through several parameter candidates the best parameters are %s' %
             best_xgboost_parameters)

cv_result = xgb.cv(best_xgboost_parameters,
                   dtrain,
                   early_stopping_rounds=30,
                   nfold=cross_validation_nfold,
                   num_boost_round=xgb_boost_rounds,
                   seed=random_seed,
                   show_stdv=False,
                   verbose_eval=50)
actual_boost_rounds = len(cv_result)
logger.debug('for boost we actually used %d rounds' % actual_boost_rounds)
if False:
    logger.debug(cv_result)

# train model
if False:
    watchlist = [(dtrain, 'train')]
else:
    watchlist = []

model = xgb.train(dict(xgboost_parameters, silent=1), dtrain=dtrain, num_boost_round=actual_boost_rounds,
                  evals=watchlist)
logger.debug('model trained.')
# predict
predictions = model.predict(dtest)
y_predictions = np.array([str(round(each, 10)) for each in predictions])

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_predictions, '201611': y_predictions, '201612': y_predictions,
                       '201710': y_predictions, '201711': y_predictions, '201712': y_predictions})
# get the column headers
output_columns = output.columns.tolist()

# rearrange the columns to put the last one first
output = output[output_columns[-1:] + output_columns[:-1]]
logger.debug('our submission file has %d rows (should be 18232?)' % len(output))

make_submission = False
use_gzip_compression = True
submission_prefix = 'zestimate'
output_filename = '{}{}.csv'.format(submission_prefix, datetime.now().strftime('%Y%m%d_%H%M%S'))
if use_gzip_compression:
    output_filename += '.gz'
    if make_submission:
        logger.debug('writing submission to %s' % output_filename)
        output.to_csv(output_filename, index=False, float_format='%.4f', compression='gzip')
else:
    if make_submission:
        logger.debug('writing submission to %s' % output_filename)
        output.to_csv(output_filename, index=False, float_format='%.4f')

f_score = model.get_fscore()
importance = sorted(f_score.items(), key=operator.itemgetter(1), reverse=True)
logger.debug('features by importance (ascending): %s' % importance)
logger.debug('of %d features the model considers %d of them significant' % (len(list(x_train)), len(importance)))
insignificant_features = set([item[0] for item in f_score.items()]).symmetric_difference(set(list(x_train)))
logger.debug('here are the insignificant features: %s' % sorted(list(insignificant_features)))
output_pickle_file = './xgboost_reduced.pickle'
with open(output_pickle_file, 'wb') as outfile_fp:
    pickle.dump(importance, outfile_fp)
logger.debug('wrote feature importance to %s' % output_pickle_file)

features = zip(*importance)[0]
scores = zip(*importance)[1]
x_pos = np.arange(len(features))
plt.figure(figsize=(16, 9))
plt.bar(x_pos, scores, align='center')
plt.xticks(x_pos, features, rotation='vertical')
plt.tight_layout()
plt.title('Feature importance')

use_gzip_compression = True
if use_gzip_compression:
    figure_filename = output_filename.replace('.csv.gz', '.png')
else:
    figure_filename = output_filename.replace('.gz', '.png')

make_feature_graph = True
if make_feature_graph:
    plt.savefig(figure_filename)

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
