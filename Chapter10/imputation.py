'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 10: Machine Learning Best Practices
Author: Yuxi (Hayden) Liu
'''


import numpy as np
from sklearn.preprocessing import Imputer


data_origin = [[30, 100],
               [20, 50],
               [35, np.nan],
               [25, 80],
               [30, 70],
               [40, 60]]


imp_mean = Imputer(missing_values='NaN', strategy='mean')
imp_mean.fit(data_origin)
data_mean_imp = imp_mean.transform(data_origin)
print(data_mean_imp)


imp_median = Imputer(missing_values='NaN', strategy='median')
imp_median.fit(data_origin)
data_median_imp = imp_median.transform(data_origin)
print(data_median_imp)

# New samples
new = [[20, np.nan],
       [30, np.nan],
       [np.nan, 70],
       [np.nan, np.nan]]
new_mean_imp = imp_mean.transform(new)
print(new_mean_imp)



# Effects of discarding missing values and imputation
from sklearn import datasets
dataset = datasets.load_diabetes()
X_full, y = dataset.data, dataset.target



m, n = X_full.shape
m_missing = int(m * 0.25)
print(m, m_missing)


np.random.seed(42)
missing_samples = np.array([True] * m_missing + [False] * (m - m_missing))
np.random.shuffle(missing_samples)


missing_features = np.random.randint(low=0, high=n, size=m_missing)

X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = np.nan


# Discard samples containing missing values
X_rm_missing = X_missing[~missing_samples, :]
y_rm_missing = y[~missing_samples]

# Estimate R^2 on the data set with missing samples removed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_rm_missing = cross_val_score(regressor, X_rm_missing, y_rm_missing).mean()
print('Score with the data set with missing samples removed: {0:.2f}'.format(score_rm_missing))


# Imputation with mean value
imp_mean = Imputer(missing_values='NaN', strategy='mean')
X_mean_imp = imp_mean.fit_transform(X_missing)
# Estimate R^2 on the data set with missing samples removed
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_mean_imp = cross_val_score(regressor, X_mean_imp, y).mean()
print('Score with the data set with missing values replaced by mean: {0:.2f}'.format(score_mean_imp))


# Estimate R^2 on the full data set
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=500)
score_full = cross_val_score(regressor, X_full, y).mean()
print('Score with the full data set: {0:.2f}'.format(score_full))


