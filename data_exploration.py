
import logging
import operator
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


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

column_name = ['lotsizesquarefeet']
fig, ax = plt.subplots()
train.hist(ax=ax, bins=40, column=column_name)
# ax.set_xscale('log')
ax.set_yscale('log')
figure_filename = column_name[0] + '-log.png'
plt.savefig(figure_filename)

# need to use enough bins to get quarter-bath accuracy
column_name = ['calculatedbathnbr']
fig, ax = plt.subplots()
min_bath_count = train[column_name[0]].min()
max_bath_count = train[column_name[0]].max()
train.hist(ax=ax, bins=4*(max_bath_count-min_bath_count+1),column=column_name)
ax.set_yscale('log')
figure_filename = column_name[0] + '-log.png'
plt.savefig(figure_filename)
# plt.show()

# need to use enough bins to get quarter-bath accuracy
column_name = ['bedroomcnt']
fig, ax = plt.subplots()
min_bath_count = train[column_name[0]].min()
max_bath_count = train[column_name[0]].max()
train.hist(ax=ax, bins=(max_bath_count-min_bath_count+1),column=column_name)
ax.set_yscale('log')
# ax.set_ylim(bottom=-100)
# figure_filename = column_name[0] + '-log.png'
# plt.savefig(figure_filename)
plt.show()

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
