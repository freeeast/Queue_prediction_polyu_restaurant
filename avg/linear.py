import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from joblib import dump
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

os.environ["JOBLIB_TEMP_FOLDER"] = "path_to_temp_folder"

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true[y_true != 0]
    y_pred = y_pred[:len(y_true)]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data = pd.read_csv('Dataset/waiting_time.csv', encoding='utf-8')

window_mapping = {'a': 0, 'd': 1}
data['window_encoded'] = data.iloc[:, 6].map(window_mapping)

X = data.iloc[:, [2,7]]
time_data = data.iloc[:, 1].str.split(':')
minutes = time_data.str[0].astype(int)
seconds = time_data.str[1].astype(int)
y = minutes * 60 + seconds

X_t, X_te, y_t, y_te = train_test_split(X, y, test_size=0.3, random_state=818)

#model=Lasso()
#model=LinearRegression()
#model=DecisionTreeClassifier()
model=SVR(kernel='linear')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
r2l = []
rmsel = []
mael = []
for train_index, test_index in kf.split(X_t):
    X_train, X_test = X_t.iloc[train_index], X_t.iloc[test_index]
    y_train, y_test = y_t.iloc[train_index], y_t.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print("Mean Squared Error:", mse)

    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)
    rmsel.append(rmse)

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    mael.append(mae)

    r2 = r2_score(y_test, y_pred)
    r2l.append(r2)
    print("R-squared:", r2)

print("Average MSE:", sum(mse_scores) / len(mse_scores))
print("Average RMSE:", sum(rmsel) / len(rmsel))
print("Average MAE:", sum(mael) / len(mael))
print("Average R2:", sum(r2l) / len(r2l))

'''param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [-1],
}'''

#param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['rbf', 'linear']
}

'''param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 5],
    'criterion': ['gini', 'entropy']
}'''

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_t, y_t)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_te)
mse = mean_squared_error(y_te, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
mae = mean_absolute_error(y_te, y_pred)
r2 = r2_score(y_te, y_pred)
print("R-squared:", r2)
print("MAE:", mae)
dump(best_model, 'best_model.joblib')
