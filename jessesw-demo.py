import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.grid_search import GridSearchCV

train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                       skiprows=1, header=None)  # Make sure to skip a row for the test set

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
              'wage_class']

train_set.columns = col_labels
test_set.columns = col_labels

train_nomissing = train_set.replace(' ?', np.nan).dropna()
test_nomissing = test_set.replace(' ?', np.nan).dropna()

test_nomissing['wage_class'] = test_nomissing.wage_class.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})

combined_set = pd.concat([train_nomissing, test_nomissing], axis=0)  # Stacks them vertically

for feature in combined_set.columns:  # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object':  # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes  # Replace strings with an integer

final_train = combined_set[:train_nomissing.shape[0]]  # Up to the last initial training set row
final_test = combined_set[train_nomissing.shape[0]:]  # Past the last initial training set row

y_train = final_train.pop('wage_class')
y_test = final_test.pop('wage_class')

cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 100, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic'}
print('about to do grid CV')
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                             cv_params,
                             scoring='accuracy', cv=5, n_jobs=-1)
# Optimize for accuracy since that is the metric used in the Adult Data Set notation

print('did grid CV; about to fit')
optimized_GBM.fit(final_train, y_train)

print('fit done; about to gridsearchCV again')

GridSearchCV(cv=5, error_score='raise',
             estimator=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
                                         gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
                                         min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                                         objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                                         scale_pos_weight=1, seed=0, silent=True, subsample=0.8),
             fit_params={}, iid=True, n_jobs=-1,
             param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},
             pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)

print (optimized_GBM.grid_scores_)
