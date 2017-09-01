import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

train_df = pd.read_csv(training_file)
# test_df = pd.read_csv("../input/sample_submission.csv")
# test_df = test_df.rename(columns={'ParcelId': 'parcelid'})

train = train_df.merge(properties, how='left', on='parcelid')
# test = test_df.merge(properties, on='parcelid', how='left')

logger.debug('training data shape: %s' % (train.shape,))

# make a scatterplot of the training data
if False:
    column_names = ['latitude', 'longitude', 'fips']
    logger.debug(train.fips.unique())
    train.plot(kind='scatter', x='latitude', y='longitude', c='fips')

colors = {6037: 'red', 6059: 'blue', 6111: 'green'}
fig, ax = plt.subplots()
ax.scatter(train['latitude'], train['longitude'], c=train['fips'].apply(lambda x: colors[x]))
figure_filename = 'latitude-longitude-fips.png'
plt.savefig(figure_filename)

column_name = 'lotsizesquarefeet'
fig, ax = plt.subplots()
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
train.hist(ax=ax, bins=40, column=column_name)
# ax.set_xscale('log')
ax.set_yscale('log')
figure_filename = column_name + '-log.png'
plt.savefig(figure_filename)

fig, ax = plt.subplots()
limit = train[column_name].max()
properties[properties[column_name] < limit].hist(ax=ax, column=column_name)
ax.set_yscale('log')
figure_filename = '-'.join([column_name, 'properties', 'log']) + '.png'
plt.savefig(figure_filename)

# need to use enough bins to get quarter-bath accuracy
column_name = 'calculatedbathnbr'
fig, ax = plt.subplots()
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
min_bath_count = train[column_name].min()
max_bath_count = train[column_name].max()
train.hist(ax=ax, bins=4 * (max_bath_count - min_bath_count + 1), column=column_name)
ax.set_yscale('log')
figure_filename = column_name + '-log.png'
plt.savefig(figure_filename)
# plt.show()

column_name = 'bedroomcnt'
fig, ax = plt.subplots()
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
quantile = 0.05
logger.debug('at quantile level %f we have %f' % (quantile, train[column_name].quantile(quantile)))
min_bedroom_count = train[column_name].min()
max_bedroom_count = train[column_name].max()
train.hist(ax=ax, bins=(max_bedroom_count - min_bedroom_count + 1), column=column_name)
ax.set_yscale('log')
figure_filename = column_name + '-log.png'
plt.savefig(figure_filename)

column_name = 'calculatedfinishedsquarefeet'
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
quantile = 0.05
logger.debug('at quantile level %f we have %f' % (quantile, train[column_name].quantile(quantile)))
fig, ax = plt.subplots()
min_count = train[column_name].min()
max_count = train[column_name].max()
train.hist(ax=ax, bins=30, column=column_name)
do_log = False
if do_log:
    ax.set_yscale('log')
    figure_filename = column_name + '-log.png'
else:
    figure_filename = column_name + '.png'
plt.savefig(figure_filename)
# plt.show()

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
