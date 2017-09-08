import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stateplane

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

# let's get some unique values:
# for column_name in ['fips', 'regionidcity', 'regionidcounty', 'regionidzip', 'regionidneighborhood', 'storytypeid']:
#     logger.debug('%s : %s' % (column_name, train[column_name].unique()))

for column_name in ['buildingclasstypeid', 'decktypeid', 'hashottuborspa', 'poolcnt', 'pooltypeid10', 'pooltypeid2',
                    'pooltypeid7', 'typeconstructiontypeid', 'assessmentyear', 'taxdelinquencyyear']:
    logger.debug('%s : %s' % (column_name, train[column_name].unique()))
for column_name in list(train):
    logger.debug('%s : %d' % (column_name, len(train[column_name].unique())))

if False:
    logger.debug(
        train['fips'].head(20)
    )

    logger.debug(train[['latitude', 'longitude']].head(20))

    train['t0'] = train.apply(lambda row: stateplane.identify(
        row['longitude'] / 1000000.0,
        row['latitude'] / 1000000.0,
        fmt='fips'), axis=1)
    logger.debug(
        train['t0'].head(20)
)

logger.debug('training data shape: %s' % (train.shape,))

# make a scatterplot of the training data
if False:
    column_names = ['latitude', 'longitude', 'fips']
    logger.debug(train.fips.unique())
    train.plot(kind='scatter', x='latitude', y='longitude', c='fips')

colors = {6037: 'red', 6059: 'blue', 6111: 'green'}

column_name = 'logerror'
# visualize the error
fig, ax = plt.subplots()
ax.scatter(range(train.shape[0]), np.sort(train.logerror.values))
plt.ylabel(column_name)
# plt.show()
figure_filename = 'train-error-scatter.png'
plt.savefig(figure_filename)
logger.debug('wrote file %s' % figure_filename)

# describe the error
logger.debug(train[column_name].describe())

# visualize the percentile of the error on a map
quantile = 0.995
log_error_abs_quantile = train[column_name].abs().quantile(quantile)
logger.debug('%s: at quantile level %f we have %f' % (column_name,quantile, log_error_abs_quantile))
outliers = train.loc[abs(train[column_name]) > log_error_abs_quantile][['latitude', 'longitude', 'fips']]
logger.debug('outliers shape : %s', (outliers.shape,))
fig, ax = plt.subplots()
ax.scatter(outliers['latitude'], outliers['longitude'], c=outliers['fips'].apply(lambda x: colors[x]))
figure_filename = 'outliers-latitude-longitude-fips.png'
plt.savefig(figure_filename)
logger.debug('wrote file %s' % figure_filename)

fig, axes = plt.subplots(ncols=2)
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
bins=40
train.hist(ax=axes[0], bins=bins, column=column_name)
train.hist(ax=axes[1], bins=bins, column=column_name)
axes[1].set_yscale('log')
figure_filename = 'train-error-histogram.png'
plt.savefig(figure_filename)
logger.debug('wrote file %s' % figure_filename)

fig, ax = plt.subplots()
ax.scatter(train['latitude'], train['longitude'], c=train['fips'].apply(lambda x: colors[x]))
figure_filename = 'train-latitude-longitude-fips.png'
plt.savefig(figure_filename)
logger.debug('wrote file %s' % figure_filename)

fig, ax = plt.subplots()
t0 = properties[['latitude', 'longitude', 'fips']].dropna()
logger.debug('if we filter out n/as from the test data we have %s' % (t0.shape,))
ax.scatter(t0['latitude'], t0['longitude'], c=t0['fips'].apply(lambda x: colors[x]))
figure_filename = 'properties-latitude-longitude-fips.png'
plt.savefig(figure_filename)
logger.debug('wrote file %s' % figure_filename)

column_name = 'taxvaluedollarcnt'
fig, axes = plt.subplots(ncols=2)
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
train.hist(ax=axes[0], bins=50, column=column_name)
train.hist(ax=axes[1], bins=50, column=column_name)
axes[1].set_yscale('log')
figure_filename = column_name + '-train.png'
plt.savefig(figure_filename)

fig, axes = plt.subplots(ncols=2)
limit = train[column_name].max()
properties[properties[column_name] < limit].hist(ax=axes[0], bins=50, column=column_name)
properties[properties[column_name] < limit].hist(ax=axes[1], bins=50, column=column_name)
axes[1].set_yscale('log')
figure_filename = '-'.join([column_name, 'properties']) + '.png'
plt.savefig(figure_filename)

# todo combine these into a single plot
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

# todo combine these into a single plot(?)
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
