import logging
import operator
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
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

training_file = '../input/train_2016_v2.csv'

train_df = pd.read_csv(training_file)

sample_submission_file = '../input/sample_submission.csv'
submission = pd.read_csv(sample_submission_file)

value = train_df['logerror'].mean()
logger.debug('mean log error from training data is %.4f' % value)

columns = ['201610', '201611', '201612', '201710', '201711', '201712']
submission[columns] = value

make_submission = True
use_gzip_compression = True
submission_prefix = 'zestimate'
output_filename = 'baseline.csv'
if use_gzip_compression:
    output_filename += '.gz'
    if make_submission:
        logger.debug('writing submission to %s' % output_filename)
        submission.to_csv(output_filename, index=False, float_format='%.4f', compression='gzip')
else:
    if make_submission:
        logger.debug('writing submission to %s' % output_filename)
        submission.to_csv(output_filename, index=False, float_format='%.4f')

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
