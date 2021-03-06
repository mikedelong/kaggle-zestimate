import logging
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
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

properties_file = ''
training_file = ''
year_to_use = 2016
if year_to_use == 2016:
    properties_file = '../input/properties_2016.csv'
    training_file = '../input/train_2016_v2.csv'
elif year_to_use == 2017:
    properties_file = '../input/properties_2017.csv'
    training_file = '../input/train_2017.csv'
else:
    logging.warn('need to pick a year to use; chose %d. Quitting. ' % year_to_use)
    exit()

properties = pd.read_csv(properties_file, dtype={
    'fireplaceflag': np.bool, 'hashottuborspa': np.bool,
    'propertycountylandusecode': np.str,
    'propertyzoningdesc': np.str}, converters={
    'taxdelinquencyflag': lambda x: np.bool(True) if x == 'Y' else np.bool(False)})  # avoid mixed type warning

properties['taxdelinquencyyear'] = properties['taxdelinquencyyear'].apply(
    lambda x: (2000 + x if x < 20 else 1900 + x) if pd.notnull(x) else x)

train_df = pd.read_csv(training_file)
train = train_df.merge(properties, how='left', on='parcelid')

fig, axes = plt.subplots(ncols=2)
column_name = 'taxdelinquencyyear'
train.hist(ax=axes[0], bins=10, column=column_name)
axes[0].set_yscale('log')
axes[0].set_xticks([1970, 1980, 1990, 2000, 2010, 2020])
properties.hist(ax=axes[1], bins=10, column=column_name)
axes[1].set_yscale('log')
axes[1].set_xticks([1970, 1980, 1990, 2000, 2010, 2020])
figure_filename = '-'.join([column_name, 'histogram']) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, axes = plt.subplots(ncols=2)
column_name = 'yearbuilt'
train.hist(ax=axes[0], bins=50, column=column_name)
axes[0].set_yscale('log')
axes[0].set_xlim(1800, 2020)
properties.hist(ax=axes[1], bins=50, column=column_name)
axes[1].set_yscale('log')
axes[1].set_xlim(1800, 2020)
figure_filename = '-'.join([column_name, 'histogram']) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, axes = plt.subplots(ncols=2, nrows=2)
column_name_00 = 'calculatedfinishedsquarefeet'
train.hist(ax=axes[0, 0], bins=50, column=column_name_00)
axes[0, 0].set_yscale('log')
axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
column_name_10 = 'bedroomcnt'
train.hist(ax=axes[1, 0], bins=50, column=column_name_10)
axes[1, 0].set_yscale('log')
column_name_01 = 'lotsizesquarefeet'
train.hist(ax=axes[0, 1], bins=50, column=column_name_01)
axes[0, 1].set_yscale('log')
axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
column_name_11 = 'bathroomcnt'
train.hist(ax=axes[1, 1], bins=50, column=column_name_11)
axes[1, 1].set_yscale('log')
figure_filename = '-'.join(['train', column_name_00, column_name_01, column_name_10, column_name_11]) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, axes = plt.subplots(ncols=2, nrows=2)
properties.hist(ax=axes[0, 0], bins=50, column=column_name_00)
axes[0, 0].set_yscale('log')
axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
properties.hist(ax=axes[1, 0], bins=50, column=column_name_10)
axes[1, 0].set_yscale('log')
properties.hist(ax=axes[0, 1], bins=50, column=column_name_01)
axes[0, 1].set_yscale('log')
axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
properties.hist(ax=axes[1, 1], bins=50, column=column_name_11)
axes[1, 1].set_yscale('log')
figure_filename = '-'.join(['properties', column_name_00, column_name_01, column_name_10, column_name_11]) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, axes = plt.subplots(ncols=2, nrows=2)
column_name_00 = 'taxvaluedollarcnt'
train.hist(ax=axes[0, 0], bins=50, column=column_name_00)
axes[0, 0].set_yscale('log')
axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
column_name_10 = 'taxamount'
train.hist(ax=axes[1, 0], bins=50, column=column_name_10)
axes[1, 0].set_yscale('log')
axes[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
column_name_01 = 'structuretaxvaluedollarcnt'
train.hist(ax=axes[0, 1], bins=50, column=column_name_01)
axes[0, 1].set_yscale('log')
axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
column_name_11 = 'landtaxvaluedollarcnt'
train.hist(ax=axes[1, 1], bins=50, column=column_name_11)
axes[1, 1].set_yscale('log')
figure_filename = '-'.join(['train', column_name_00, column_name_01, column_name_10, column_name_11]) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, axes = plt.subplots(ncols=2, nrows=2)
properties.hist(ax=axes[0, 0], bins=50, column=column_name_00)
axes[0, 0].set_yscale('log')
axes[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
properties.hist(ax=axes[1, 0], bins=50, column=column_name_10)
axes[1, 0].set_yscale('log')
axes[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
properties.hist(ax=axes[0, 1], bins=50, column=column_name_01)
axes[0, 1].set_yscale('log')
axes[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
properties.hist(ax=axes[1, 1], bins=50, column=column_name_11)
axes[1, 1].set_yscale('log')
figure_filename = '-'.join(['properties', column_name_00, column_name_01, column_name_10, column_name_11]) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

# count nulls for some representative fields:
properties_count = len(properties)
not_null_percentages = {column_name: 100 * float(properties[column_name].count()) / float(properties_count) for
                        column_name in list(properties)}
sorted_percentages = sorted(not_null_percentages.items(), key=operator.itemgetter(1))
print ('|column | percent not null|')
print('|---|---|')
for item in sorted_percentages:
    print('|%s | %.2f|' % (item[0], item[1]))

log_columns = sorted(['landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'taxamount', 'taxvaluedollarcnt',
                      'calculatedfinishedsquarefeet'])
for column_name in log_columns:
    t0 = stats.skew(train[column_name].dropna())
    t1 = stats.skew(properties[column_name].dropna())
    t2 = stats.skew(np.log(train[column_name].dropna()))
    t3 = stats.skew(np.log(properties[column_name].dropna()))
    logger.debug('%s : train skew: %.2f log train skew: %.2f properties skew: %.2f log properties skew: %.2f' %
                 (column_name, t0, t2, t1, t3))

big_error = train[abs(train['logerror']) > 0.3]
logger.debug('big error has length %d (%.2f percent of total)' % (
    len(big_error), 100 * (float(len(big_error)) / float(len(train_df)))))
columns_to_drop = ['poolcnt', 'parcelid', 'buildingclasstypeid', 'decktypeid', 'pooltypeid2',
                   'pooltypeid7', 'pooltypeid10', 'storytypeid', 'assessmentyear',
                   'architecturalstyletypeid', 'basementsqft', 'finishedsquarefeet6', 'finishedsquarefeet13',
                   'poolsizesum', 'typeconstructiontypeid', 'yardbuildingsqft26', 'taxdelinquencyflag',
                   'taxdelinquencyyear']
correlation_input_data = big_error.drop(columns_to_drop, axis=1)

color_map  = 'YlGnBu'

# let's get the Pearson correlations for the training data
train_pearson_correlations = correlation_input_data.corr(method='pearson')
correlations_len = len(train_pearson_correlations)
correlations_columns = train_pearson_correlations.columns
plt.figure()
plt.imshow(train_pearson_correlations, cmap=color_map, interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(correlations_len), correlations_columns, rotation='vertical', fontsize=8)
plt.yticks(range(correlations_len), correlations_columns, fontsize=8)
plt.suptitle('Training data Pearson correlations - outliers', fontsize=8, fontweight='bold')
plt.tight_layout()
figure_filename = 'training-data-outliers-pearson-correlations.png'
plt.savefig(figure_filename)
plt.close()

# todo factor out columns to drop as a variable
thin_train = train.drop(['assessmentyear', 'decktypeid', 'parcelid',
                         'poolcnt', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10',
                         'storytypeid'], axis=1)
train_pearson_correlations = thin_train.corr(method='pearson')
correlations_len = len(train_pearson_correlations)
correlations_columns = train_pearson_correlations.columns
plt.figure()
plt.imshow(train_pearson_correlations, cmap=color_map, interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(correlations_len), correlations_columns, rotation='vertical', fontsize=8)
plt.yticks(range(correlations_len), correlations_columns, fontsize=8)
plt.suptitle('Training data Pearson correlations', fontsize=8, fontweight='bold')
plt.tight_layout()
figure_filename = 'training-data-pearson-correlations.png'
plt.savefig(figure_filename)
plt.close()

# todo factor out columns to drop as a variable
thin_properties = properties.drop(['assessmentyear', 'decktypeid', 'poolcnt', 'pooltypeid2', 'pooltypeid7',
                                   'pooltypeid10', 'storytypeid'], axis=1)
properties_pearson_correlations = thin_properties.corr(method='pearson')
correlations_len = len(properties_pearson_correlations)
correlations_columns = properties_pearson_correlations.columns
plt.figure()
plt.imshow(properties_pearson_correlations, cmap=color_map, interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(correlations_len), correlations_columns, rotation='vertical', fontsize=8)
plt.yticks(range(correlations_len), correlations_columns, fontsize=8)
plt.suptitle('Test data Pearson correlations', fontsize=8, fontweight='bold')
plt.tight_layout()
figure_filename = 'property-data-pearson-correlations.png'
plt.savefig(figure_filename)
plt.close()

na_counts = {column_name: properties[column_name].isnull().sum() for column_name in list(properties) if
             column_name not in ['parcelid']}
x_pos = np.arange(len(na_counts))
plt.figure()
# let's sort these values before we graph them
sorted_counts = sorted(na_counts.items(), key=operator.itemgetter(1), reverse=True)
sorted_values = [item[1] for item in sorted_counts]
plt.bar(x_pos, sorted_values, align='center')
sorted_keys = [item[0] for item in sorted_counts]
plt.xlim([-1, len(x_pos)])
plt.xticks(x_pos, sorted_keys, rotation='vertical', fontsize=8)
plt.yscale('log', nonposy='clip')
plt.tight_layout()
# plt.ylabel('Column N/A counts')
plt.suptitle('Properties data null counts by column')
figure_filename = 'properties-na-counts.png'
plt.savefig(figure_filename)
plt.close()

na_counts = {column_name: train[column_name].isnull().sum() for column_name in list(train) if
             column_name not in ['parcelid']}
x_pos = np.arange(len(na_counts) - 2)
plt.figure()
# let's use the ordering from the properties
sorted_values = [na_counts[key] for key in sorted_keys]
plt.bar(x_pos, sorted_values, align='center')
plt.xlim([-1, len(x_pos)])
plt.xticks(x_pos, sorted_keys, rotation='vertical', fontsize=8)
plt.yscale('log', nonposy='clip')
plt.tight_layout()
# plt.ylabel('Column N/A counts')
plt.suptitle('Training data null counts by column')
figure_filename = 'train-na-counts.png'
plt.savefig(figure_filename)
plt.close()

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

# let's get some unique values:
# for column_name in ['fips', 'regionidcity', 'regionidcounty', 'regionidzip', 'regionidneighborhood', 'storytypeid']:
#     logger.debug('%s : %s' % (column_name, train[column_name].unique()))

logger.debug('%s %s' % ('yearbuilt', train['yearbuilt'].describe()))

columns_of_interest = ['buildingclasstypeid', 'decktypeid', 'hashottuborspa', 'poolcnt', 'pooltypeid10', 'pooltypeid2',
                       'pooltypeid7', 'typeconstructiontypeid', 'assessmentyear', 'taxdelinquencyyear',
                       'propertycountylandusecode', 'propertyzoningdesc', 'yearbuilt', 'storytypeid', 'fireplaceflag',
                       'taxdelinquencyflag', 'buildingqualitytypeid', 'fips']
columns_of_interest = sorted(list(set(columns_of_interest)))
for column_name in columns_of_interest:
    train_uniques = train[column_name].unique()
    logger.debug('%s : %d :: %s' % (column_name, len(train_uniques), train_uniques))

for column_name in list(train):
    logger.debug('%s : train unique: %d properties unique: %d' % (column_name, len(train[column_name].unique()),
                                                                  (len(properties[
                                                                           column_name].unique())) if column_name in list(
                                                                      properties) else -1))

if False:
    import stateplane
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
plt.close()
del fig
logger.debug('wrote file %s' % figure_filename)

# describe the error and the tax amount
for column_name in ['logerror', 'taxamount']:
    logger.debug('describing column %s : %s' % (column_name, train[column_name].describe()))

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
plt.close()
del fig
logger.debug('wrote file %s' % figure_filename)

column_name = 'logerror'
fig, axes = plt.subplots(ncols=2)
logger.debug('%s min: %.4f max: %.4f' % (column_name, train[column_name].min(), train[column_name].max()))
bins = 40
train.hist(ax=axes[0], bins=bins, column=column_name)
train.hist(ax=axes[1], bins=bins, column=column_name)
axes[1].set_yscale('log')
figure_filename = 'train-error-histogram.png'
plt.savefig(figure_filename)
plt.close()
del fig
logger.debug('wrote file %s' % figure_filename)

fig, ax = plt.subplots()
t1 = train[['latitude', 'longitude', 'fips']].dropna()
ax.scatter(t1['latitude'], t1['longitude'], c=t1['fips'].apply(lambda x: colors[x]))
figure_filename = 'train-latitude-longitude-fips.png'
plt.savefig(figure_filename)
plt.close()
del fig
logger.debug('wrote file %s' % figure_filename)

fig, ax = plt.subplots()
t0 = properties[['latitude', 'longitude', 'fips']].dropna()
logger.debug('if we filter out n/as from the test data we have %s' % (t0.shape,))
ax.scatter(t0['latitude'], t0['longitude'], c=t0['fips'].apply(lambda x: colors[x]))
figure_filename = 'properties-latitude-longitude-fips.png'
plt.savefig(figure_filename)
plt.close()
del fig
logger.debug('wrote file %s' % figure_filename)

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
plt.close()
del fig

column_name = 'taxvaluedollarcnt'
fig, axes = plt.subplots(ncols=2)
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
train.hist(ax=axes[0], bins=50, column=column_name)
train.hist(ax=axes[1], bins=50, column=column_name)
axes[1].set_yscale('log')
figure_filename = column_name + '-train.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, axes = plt.subplots(ncols=2)
limit = train[column_name].max()
properties[properties[column_name] < limit].hist(ax=axes[0], bins=50, column=column_name)
properties[properties[column_name] < limit].hist(ax=axes[1], bins=50, column=column_name)
axes[1].set_yscale('log')
figure_filename = '-'.join([column_name, 'properties']) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

# todo combine these into a single plot
column_name = 'lotsizesquarefeet'
fig, ax = plt.subplots()
logger.debug('%s min: %d max: %d' % (column_name, train[column_name].min(), train[column_name].max()))
train.hist(ax=ax, bins=40, column=column_name)
ax.set_yscale('log')
figure_filename = column_name + '-log.png'
plt.savefig(figure_filename)
plt.close()
del fig

fig, ax = plt.subplots()
limit = train[column_name].max()
properties[properties[column_name] < limit].hist(ax=ax, column=column_name)
ax.set_yscale('log')
figure_filename = '-'.join([column_name, 'properties', 'log']) + '.png'
plt.savefig(figure_filename)
plt.close()
del fig

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
plt.close()
del fig

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
plt.close()
del fig

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
plt.close()
del fig

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
