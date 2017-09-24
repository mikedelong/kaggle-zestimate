# loosely based on/following the outline of
# https://jessesw.com/XG-Boost/

import logging
import time

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
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

sns.set(font_scale=1.5)

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
print(train.info())

train_columns_to_drop = ['parcelid', 'logerror', 'transactiondate']
x_train = train.drop(train_columns_to_drop, axis=1)
y_train = train['logerror'].values.astype(np.float32)
y_mean = np.mean(y_train)
logger.debug('y_mean : %.8f' % y_mean)
logger.debug('y_train shape: %s' % (y_train.shape,))

dtrain = xgb.DMatrix(x_train, label=y_train)

test_columns_to_drop = ['parcelid']
x_test = properties.drop(test_columns_to_drop, axis=1)

dtest = xgb.DMatrix(x_test)

cv_params = {'max_depth': [7]}
# , 'min_child_weight': [1,3,5]}
# ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
#              'objective': 'binary:logistic'}
random_seed = 1
# xgboost parameters
xgboost_parameters = {
    # 'alpha': 0.0,
    'base_score': y_mean,
    # 'booster': 'gbtree',
    'colsample_bytree': 1.0,
    # 'eta': 0.025,  # todo try a range of values from 0 to 0.1 (?) default = 0.03 # was 0.003
    # 'eval_metric': 'mae', # defer to training
    'gamma': 0.0,  # default is 0
    # 'lambda': 1.05,  # default is 1.0
    # 'max_depth': 7,  # todo try a range of values from 3 to 7 (?) default = 6
    'n_estimators': 100,
    'objective': 'reg:linear',
    'seed': random_seed,
    'silent': 1,
    'subsample': 0.7
}

cross_validation_folds = 5
optimized_GBM = GridSearchCV(xgb.XGBRegressor(**xgboost_parameters), cv_params, scoring='accuracy',
                             cv=cross_validation_folds, n_jobs=-1)

logger.debug('created optimized model')
optimized_GBM.fit(x_train, y_train)
ogger.debug('fit optimized model')

GridSearchCV(cv=cross_validation_folds, error_score='raise',
             estimator=xgb.XGBRegressor(base_score=y_mean,
                                        # colsample_bylevel=1,
                                        colsample_bytree=1.0,
                                        gamma=0,
                                        # learning_rate=0.1,
                                        # max_delta_step=0,
                                        # max_depth=3,
                                        # min_child_weight=1,
                                        #                            missing=None,
                                        n_estimators=100,
                                        nthread=-1,
                                        objective='reg:linear',
                                        reg_alpha=0.0,
                                        reg_lambda=1.05,
                                        # scale_pos_weight=1,
                                        seed=random_seed,
                                        silent=False,
                                        subsample=0.7),
             # estimator=optimized_GBM,

             fit_params={}, iid=True,
             # n_jobs=-1,
             n_jobs=1,

             param_grid={
                 # 'min_child_weight': [1, 3, 5],
                 'max_depth': [7]},
             # pre_dispatch='2*n_jobs',
             refit=True, scoring='accuracy', verbose=0)

logger.debug('finished')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
