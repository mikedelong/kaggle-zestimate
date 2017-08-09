import logging
import time

import numpy
import pandas

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

properties_file = '../input/properties_2016.csv'
logger.debug('Reading data from %s ' % properties_file)

properties = pandas.read_csv(properties_file, dtype={
    'fireplaceflag': numpy.bool, 'hashottuborspa': numpy.bool,
    'propertycountylandusecode': numpy.str,
    'propertyzoningdesc': numpy.str}, converters={
    'taxdelinquencyflag': lambda x: numpy.bool(True) if x == 'Y' else numpy.bool(False)})  # avoid mixed type warning
logger.debug('Properties read done.')


# let's write out some summary statistics about the properties before we go on.
logger.debug('properties data shape: %s' % (properties.shape,))
columns_with_missing_values = properties.columns[properties.isnull().any()].tolist()
logger.debug('we have %d columns that have missing values: %s' %
             (len(columns_with_missing_values), columns_with_missing_values))
columns_with_no_missing_values = properties.columns[~(properties.isnull().any())].tolist()
logger.debug('we have %d columns that have no missing values: %s' %
             (len(columns_with_no_missing_values), columns_with_no_missing_values))

logger.debug('loading training data')
training_file = '../input/train_2016_v2.csv'
train = pandas.read_csv(training_file)
logger.debug('training data load complete.')
logger.debug('training data shape: %s' % (train.shape,))


logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
