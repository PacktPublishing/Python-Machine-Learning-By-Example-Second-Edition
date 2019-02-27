'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 9: Stock Price Prediction with Regression Algorithms
Author: Yuxi (Hayden) Liu
'''

from sklearn import datasets
diabetes = datasets.load_diabetes()
num_test = 30    # the last 30 samples as testing set
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]
param_grid = {
    "alpha": [1e-07, 1e-06, 1e-05],
    "penalty": [None, "l2"],
    "eta0": [0.001, 0.005, 0.01],
    "n_iter": [300, 1000, 3000]
}

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
regressor = SGDRegressor(loss='squared_loss',
                             learning_rate='constant')
grid_search = GridSearchCV(regressor, param_grid, cv=3)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

regressor_best = grid_search.best_estimator_


predictions = regressor_best.predict(X_test)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(mean_squared_error(y_test, predictions))

print(mean_absolute_error(y_test, predictions))

print(r2_score(y_test, predictions))

