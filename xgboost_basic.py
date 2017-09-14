import logging
import operator
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
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

do_consolidate_columns = False
if do_consolidate_columns:
    # Columns to be consolidated
    properties['yardbuildingsqft17'] = properties['yardbuildingsqft17'].apply(lambda x: 0 if np.isnan(x) else x).astype(
        float)
    properties['yardbuildingsqft26'] = properties['yardbuildingsqft26'].apply(lambda x: 0 if np.isnan(x) else x).astype(
        float)
    properties['yard_building_square_feet'] = properties['yardbuildingsqft17'].astype(int) + properties[
        'yardbuildingsqft26'].astype(float)
    properties = properties.drop(['yardbuildingsqft17', 'yardbuildingsqft26'], axis=1)

logger.debug(list(properties))

# properties['transactiondate'] = pd.to_datetime(properties['transactiondate'])
# properties['Month'] = properties['transactiondate'].dt.month
# properties['dayofweek'] = properties['transactiondate'].dt.dayofweek

# one-hot for counties instead of FIPS
fips_map = {6037: 'Los Angeles', 6059: 'Orange', 6111: 'Ventura'}
properties['fips'].replace(fips_map, inplace=True)
fips_one_hot = pd.get_dummies(properties['fips'])
fips_one_hot = fips_one_hot.drop([1.0], axis=1)
# properties = properties.drop(['fips', 'regionidcounty'], axis=1)
# properties = properties.join(fips_one_hot)
properties = properties.drop(
    ['fips', 'regionidcounty', 'assessmentyear', 'fireplaceflag', 'hashottuborspa', 'poolcnt', 'pooltypeid10',
     'pooltypeid2', 'pooltypeid7', 'storytypeid', 'typeconstructiontypeid'], axis=1)

logger.debug('merging training data and properties on parcel ID')
train_df = train.merge(properties, how='left', on='parcelid')

# drop duplicate transactions from the training data
if True:
    duplicate_rows = train_df[train_df['parcelid'].duplicated()]['parcelid'].index
    train_df = train_df.drop(duplicate_rows)

logger.debug(list(train_df))
logger.debug('dropping columns parcel ID, log error, and transaction date to get training data')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'taxvaluedollarcnt'], axis=1)
logger.debug(list(x_train))
logger.debug('dropping parcel ID from properties to get test data.')

additional_columns_to_drop = []
test_columns_to_drop = ['parcelid'] + additional_columns_to_drop
x_test = properties.drop(test_columns_to_drop, axis=1)
# shape
logger.debug('train shape: %s, test shape: %s' % ((x_train.shape,), (x_test.shape,)))

# drop out outliers
outlier_limit = 0.36
lower_limit = -0.36
upper_limit = 0.375
lower_limit = -0.4
upper_limit = 0.4
if False:
    train_df = train_df[abs(train_df.logerror) < outlier_limit]
else:
    train_df = train_df[(train_df.logerror < upper_limit) & (train_df.logerror > lower_limit)]

train_df = train_df[abs(train_df.logerror) < outlier_limit]
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
    'booster' : 'gbtree', # 'gbtree',
    'colsample_bytree': 1.0,
    'eta': 0.025,  # todo try a range of values from 0 to 0.1 (?) default = 0.03 # was 0.003
    'eval_metric': 'mae',
    'gamma': 0.0,  # default is 0
    'lambda': 1.05,  # default is 1.0
    'max_depth': 7,  # todo try a range of values from 3 to 7 (?) default = 6
    'objective':  'reg:linear',
    'seed': random_seed,
    'silent': 1,
    'subsample': 0.7
}
logger.debug('xgboost parameters: %s' % xgboost_parameters)
xgb_boost_rounds = 1200  # was 1000
# cross-validation
cross_validation_nfold = 5

cv_result = xgb.cv(xgboost_parameters, dtrain,
                   early_stopping_rounds=25,
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
watchlist = [(dtrain, 'train')]

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

make_submission = True
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

# model.importance()
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
logger.debug('features by importance (descending): %s' % importance)

f_score = model.get_fscore()
importance = sorted(f_score.items(), key=operator.itemgetter(1), reverse=True)
logger.debug('features by importance (ascending): %s' % importance)
logger.debug('of %d features the model considers %d of them significant' % (len(list(x_train)), len(importance)))
insignificant_features = set([item[0] for item in f_score.items()]).symmetric_difference(set(list(x_train)))
logger.debug('here are the insignificant features: %s' % sorted(list(insignificant_features)))

features = zip(*importance)[0]
scores = zip(*importance)[1]
x_pos = np.arange(len(features))
plt.figure(figsize=(16, 9))
plt.bar(x_pos, scores, align='center')
plt.xticks(x_pos, features, rotation='vertical')  # was 'vertical'
plt.tight_layout()
plt.ylabel('Feature importance')

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
