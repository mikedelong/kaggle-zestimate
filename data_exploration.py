import logging
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

train_df = pd.read_csv(training_file)

train = train_df.merge(properties, how='left', on='parcelid')

# todo add training data and make a PNG
na_counts = {column_name: properties[column_name].isnull().sum() for column_name in list(properties) if
             column_name not in ['parcelid']}
x_pos = np.arange(len(na_counts))
plt.figure(figsize=(16, 9))
# let's sort these values before we graph them
sorted_counts = sorted(na_counts.items(), key=operator.itemgetter(1), reverse=True)
sorted_values = [item[1] for item in sorted_counts]
plt.bar(x_pos, sorted_values, align='center')
sorted_keys = [item[0] for item in sorted_counts]
plt.xticks(x_pos, sorted_keys, rotation='vertical')
plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.ylabel('Column N/A counts')
figure_filename = 'properties-na-counts.png'
plt.savefig(figure_filename)

na_counts = {column_name: train[column_name].isnull().sum() for column_name in list(train) if
             column_name not in ['parcelid']}
x_pos = np.arange(len(na_counts) - 2)
plt.figure(figsize=(16, 9))
# let's use the ordering from the properties
sorted_values = [na_counts[key] for key in sorted_keys]
plt.bar(x_pos, sorted_values, align='center')
plt.xticks(x_pos, sorted_keys, rotation='vertical')
plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.ylabel('Column N/A counts')
figure_filename = 'train-na-counts.png'
plt.savefig(figure_filename)

logger.debug(properties['latitude'].isnull().sum())
logger.debug(properties['longitude'].isnull().sum())
if False:
    min_max_scaler = MinMaxScaler(copy=True)
    properties_latitude_mean = properties['latitude'].mean()
    properties_longitude_mean = properties['longitude'].mean()
    properties['latitude'].fillna(inplace=True, value=properties_latitude_mean)
    properties['longitude'].fillna(inplace=True, value=properties_longitude_mean)
    properties[['latitude', 'longitude']] = min_max_scaler.fit_transform(properties[['latitude', 'longitude']])

# before we go any further let's check how many parcels sold more than once
t0 = train_df['parcelid'].count()
t1 = len(train_df['parcelid'].unique())
logger.debug('we have %d parcels of which %d are unique' % (t0, t1))
t2 = train_df['parcelid'].duplicated(keep=False)
t3 = train_df[t2 == True].sort_values(['parcelid'])
logger.debug('duplicated shape: %s' % (t3.shape,))
t4 = train[train['parcelid'].duplicated()]['parcelid'].index
t5 = t4.shape
t6 = train.drop(t4)
t7 = t6.shape

# let's get the mean and variance of the latitude and longitude
logger.debug('train-longitude mean : %.2f std: %.2f min: %.2f max: %.2f' % (
    train['longitude'].mean(), train['longitude'].std(), train['longitude'].min(), train['longitude'].max()))
logger.debug('train-latitude mean : %.2f std: %.2f min: %.2f max: %.2f' % (
    train['latitude'].mean(), train['latitude'].std(), train['latitude'].min(), train['latitude'].max()))
logger.debug('properties-longitude mean : %.2f std: %.2f min: %.2f max: %.2f' % (
    properties['longitude'].mean(), properties['longitude'].std(), properties['longitude'].min(),
    properties['longitude'].max()))
logger.debug('properties-latitude mean : %.2f std: %.2f min: %.2f max: %.2f' % (
    properties['latitude'].mean(), properties['latitude'].std(), properties['latitude'].min(),
    properties['latitude'].max()))

logger.debug('%s' % train['latitude'].describe())

# let's get some unique values:
# for column_name in ['fips', 'regionidcity', 'regionidcounty', 'regionidzip', 'regionidneighborhood', 'storytypeid']:
#     logger.debug('%s : %s' % (column_name, train[column_name].unique()))

for column_name in ['buildingclasstypeid', 'decktypeid', 'hashottuborspa', 'poolcnt', 'pooltypeid10', 'pooltypeid2',
                    'pooltypeid7', 'typeconstructiontypeid', 'assessmentyear', 'taxdelinquencyyear',
                    'propertycountylandusecode', 'propertyzoningdesc']:
    uniques = train[column_name].unique()
    logger.debug('%s : %d :: %s' % (column_name, len(uniques), uniques))
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
logger.debug('%s: at quantile level %f we have %f' % (column_name, quantile, log_error_abs_quantile))
outliers = train.loc[abs(train[column_name]) > log_error_abs_quantile][['latitude', 'longitude', 'fips']]
logger.debug('outliers shape : %s', (outliers.shape,))
fig, ax = plt.subplots()
ax.scatter(outliers['latitude'], outliers['longitude'], c=outliers['fips'].apply(lambda x: colors[x]))
figure_filename = 'outliers-latitude-longitude-fips.png'
plt.savefig(figure_filename)
logger.debug('wrote file %s' % figure_filename)

fig, axes = plt.subplots(ncols=2)
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
bins = 40
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

# column_name = 'latitude'
# fig, axes = plt.subplots(ncols=2, nrows=2)
# logger.debug('train %s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
# train.hist(ax=axes[0, 0], bins=50, column=column_name)
# train.hist(ax=axes[0, 1], bins=50, column=column_name)
# axes[0, 1].set_yscale('log')
# logger.debug('properties %s min: %d max: %d' % (column_name, properties[column_name].min(), properties[column_name].max()))
# properties.hist(ax=axes[1, 0], bins=50, column=column_name)
# properties.hist(ax=axes[1, 1], bins=50, column=column_name)
# axes[1, 1].set_yscale('log')
# figure_filename = column_name + '-both.png'
# plt.savefig(figure_filename)
#
# column_name = 'longitude'
# fig, axes = plt.subplots(ncols=2, nrows=2)
# logger.debug('train %s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
# train.hist(ax=axes[0, 0], bins=50, column=column_name)
# train.hist(ax=axes[0, 1], bins=50, column=column_name)
# axes[0, 1].set_yscale('log')
# logger.debug('properties %s min: %d max: %d' % (column_name, properties[column_name].min(), properties[column_name].max()))
# properties.hist(ax=axes[1, 0], bins=50, column=column_name)
# properties.hist(ax=axes[1, 1], bins=50, column=column_name)
# axes[1, 1].set_yscale('log')
# figure_filename = column_name + '-both.png'
# plt.savefig(figure_filename)


fig, axes = plt.subplots(ncols=2, nrows=2)
column_name_1 = 'latitude'
logger.debug('train %s min: %d max: %d' % (column_name_1, train[column_name_1].min(), train[column_name_1].max()))
train.hist(ax=axes[0, 0], bins=50, column=column_name_1)
logger.debug(
    'properties %s min: %d max: %d' % (column_name_1, properties[column_name_1].min(), properties[column_name_1].max()))
properties.hist(ax=axes[1, 0], bins=50, column=column_name_1)
column_name_2 = 'longitude'
logger.debug('train %s min: %d max: %d' % (column_name_2, train[column_name_2].min(), train[column_name_2].max()))
train.hist(ax=axes[0, 1], bins=50, column=column_name_2)
logger.debug(
    'properties %s min: %d max: %d' % (column_name_2, properties[column_name_2].min(), properties[column_name_2].max()))
properties.hist(ax=axes[1, 1], bins=50, column=column_name_2)
figure_filename = '-'.join([column_name_1, column_name_2, 'both']) + '.png'
plt.savefig(figure_filename)

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
