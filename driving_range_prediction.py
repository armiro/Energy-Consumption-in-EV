import warnings
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor


warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

old_path = "./data files/data.csv"
new_path = "./data files/new_data.csv"


"""remove missing values (comment it after the first run)"""
# ds = pd.read_csv(filepath_or_buffer=old_path)
# ds = ds[pd.notnull(obj=ds['quantity(kWh)'])]
# ds = ds[pd.notnull(obj=ds['avg_speed(km/h)'])]
# ds.to_csv(path_or_buf=new_path)


"""load the data"""
dataset = pd.read_csv(filepath_or_buffer=new_path)
# print(dataset.head(n=5))
# print(dataset.describe())

X = dataset.iloc[:, 5:14].values
y = dataset.iloc[:, 14].values

# if the data has only one feature, reshape it
# X = np.reshape(X, newshape=(-1, 1))
# y = np.reshape(y, newshape=(-1, 1))


"""do the preprocessing tasks on the data"""
# encode categorical features
label_encoder_1 = LabelEncoder()
X[:, 1] = label_encoder_1.fit_transform(y=X[:, 1])
label_encoder_2 = LabelEncoder()
X[:, 5] = label_encoder_2.fit_transform(y=X[:, 5])

# onehot encoding for categorical features with more than 2 categories
onehot_encoder = OneHotEncoder(categorical_features=[5])
X = onehot_encoder.fit_transform(X=X).toarray()

# delete the first column to avoid the dummy variable
X = X[:, 1:]

# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.fit_transform(X=X_test)
# y_train = sc.fit_transform(X=y_train)
# y_test = sc.fit_transform(X=y_test)


"""define the linear regression model"""
# linear_regressor = LinearRegression()


"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# mae_values = cross_val_score(estimator=linear_regressor, X=X, y=y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=2)
# r2_scores = cross_val_score(estimator=linear_regressor, X=X, y=y, cv=cv, scoring='r2', n_jobs=2)
# print("\n ------ Linear Regression ------")
# print("average MAE values (bias) is:", abs(round(number=mae_values.mean(), ndigits=3)))
# print("std deviation of MAE values (variance) is:", round(number=mae_values.std(), ndigits=3))
# best_mae = sorted(mae_values, reverse=False)[-1]
# print("best MAE value is:", abs(round(number=best_mae, ndigits=3)))
#
# print("average r2 scores (bias) is:", round(number=r2_scores.mean() * 100, ndigits=3))
# print("std deviation of r2 scores (variance) is:", round(number=r2_scores.std() * 100, ndigits=3))
# best_r2 = sorted(r2_scores, reverse=False)[-1] * 100
# print("best r2 score is:", round(number=best_r2, ndigits=3))


"""train the linear regression model and print the results on the never-seen-before test data"""
# linear_regressor.fit(X=X_train, y=y_train)
# reg_training_pred = linear_regressor.predict(X=X_train)
# reg_test_pred = linear_regressor.predict(X=X_test)
# # print("linear reg RMSE on training data: %.3f" % sqrt(mean_squared_error(y_true=y_train, y_pred=reg_training_pred)))
# # print("linear reg RMSE on test data: %.3f" % sqrt(mean_squared_error(y_true=y_test, y_pred=reg_test_pred)))
# print("-------------")
# print("linear reg MAE on training data: %.3f" % mean_absolute_error(y_true=y_train, y_pred=reg_training_pred))
# print("linear reg MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=reg_test_pred))
# print("-------------")
# print("linear reg variance score on training data: %.3f" % r2_score(y_true=y_train, y_pred=reg_training_pred))
# print("linear reg variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=reg_test_pred))
# print("-------------------------------")


"""define the multi-layer perceptron model"""
# mlp = MLPRegressor(hidden_layer_sizes=(20, 11, 11, 7, 7, 3, 3,), max_iter=4000, n_iter_no_change=30,
#                    activation='relu', solver='adam', verbose=False, warm_start=False)


"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# mae_values = cross_val_score(estimator=mlp, X=X, y=y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=2)
# r2_scores = cross_val_score(estimator=mlp, X=X, y=y, cv=cv, scoring='r2', n_jobs=2)
# print("\n ------ MLP ------")
# print("average MAE values (bias) is:", abs(round(number=mae_values.mean(), ndigits=3)))
# print("std deviation of MAE values (variance) is:", round(number=mae_values.std(), ndigits=3))
# best_mae = sorted(mae_values, reverse=False)[-1]
# print("best MAE value is:", abs(round(number=best_mae, ndigits=3)))
# print("average r2 scores (bias) is:", round(number=r2_scores.mean() * 100, ndigits=3))
# print("std deviation of r2 scores (variance) is:", round(number=r2_scores.std() * 100, ndigits=3))
# best_r2 = sorted(r2_scores, reverse=False)[-1] * 100
# print("best r2 score is:", round(number=best_r2, ndigits=3))


"""train the MLP model and print the results on the never-seen-before test data"""
# mlp.fit(X=X_train, y=y_train)
# mlp_training_pred = mlp.predict(X=X_train)
# mlp_test_pred = mlp.predict(X=X_test)
# # print("mlp RMSE on training data: %.3f" % sqrt(mean_squared_error(y_true=y_train, y_pred=mlp_training_pred)))
# # print("mlp RMSE on test data: %.3f" % sqrt(mean_squared_error(y_true=y_test, y_pred=mlp_test_pred)))
# print("-------------")
# print("mlp MAE on training data: %.3f" % mean_absolute_error(y_true=y_train, y_pred=mlp_training_pred))
# print("mlp MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=mlp_test_pred))
# print("-------------")
# print("mlp variance score on training data: %.3f" % r2_score(y_true=y_train, y_pred=mlp_training_pred))
# print("mlp variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=mlp_test_pred))
# print("-------------------------------")


"""define the random forest ensemble model"""
# random_forest = RandomForestRegressor(n_estimators=200, criterion="mae", warm_start=False)


"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# mae_values = cross_val_score(estimator=random_forest, X=X, y=y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=2)
# r2_scores = cross_val_score(estimator=random_forest, X=X, y=y, cv=cv, scoring='r2', n_jobs=2)
# print("\n ------ Random Forest ------")
# print("average MAE values (bias) is:", abs(round(number=mae_values.mean(), ndigits=3)))
# print("std deviation of MAE values (variance) is:", round(number=mae_values.std(), ndigits=3))
# best_mae = sorted(mae_values, reverse=False)[-1]
# print("best MAE value is:", abs(round(number=best_mae, ndigits=3)))
# print("average r2 scores (bias) is:", round(number=r2_scores.mean() * 100, ndigits=3))
# print("std deviation of r2 scores (variance) is:", round(number=r2_scores.std() * 100, ndigits=3))
# best_r2 = sorted(r2_scores, reverse=False)[-1] * 100
# print("best r2 score is:", round(number=best_r2, ndigits=3))


"""train the RF model and print the results on the never-seen-before test data"""
# random_forest.fit(X=X_train, y=y_train)
# rf_train_pred = random_forest.predict(X=X_train)
# rf_test_pred = random_forest.predict(X=X_test)
# # print("rf RMSE on training data: %.3f" % sqrt(mean_squared_error(y_true=y_train, y_pred=rf_train_pred)))
# # print("rf RMSE on test data: %.3f" % sqrt(mean_squared_error(y_true=y_test, y_pred=rf_test_pred)))
# print("-------------")
# print("rf MAE on training data: %.3f" % mean_absolute_error(y_true=y_train, y_pred=rf_train_pred))
# print("rf MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=rf_test_pred))
# print("-------------")
# print("rf variance score on training data: %.3f" % r2_score(y_true=y_train, y_pred=rf_train_pred))
# print("rf variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=rf_test_pred))
# print("-------------------------------")


"""define the ada-boost ensemble model"""
# ada_boost = AdaBoostRegressor(n_estimators=50, learning_rate=1.)


"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# mae_values = cross_val_score(estimator=ada_boost, X=X, y=y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=2)
# r2_scores = cross_val_score(estimator=ada_boost, X=X, y=y, cv=cv, scoring='r2', n_jobs=2)
# print("\n ------ Adaboost ------")
# print("average MAE values (bias) is:", abs(round(number=mae_values.mean(), ndigits=3)))
# print("std deviation of MAE values (variance) is:", round(number=mae_values.std(), ndigits=3))
# best_mae = sorted(mae_values, reverse=False)[-1]
# print("best MAE value is:", abs(round(number=best_mae, ndigits=3)))
# print("average r2 scores (bias) is:", round(number=r2_scores.mean() * 100, ndigits=3))
# print("std deviation of r2 scores (variance) is:", round(number=r2_scores.std() * 100, ndigits=3))
# best_r2 = sorted(r2_scores, reverse=False)[-1] * 100
# print("best r2 score is:", round(number=best_r2, ndigits=3))


"""train the ada-boost model and print the results on the never-seen-before test data"""
# ada_boost.fit(X=X_train, y=y_train)
# ab_train_pred = ada_boost.predict(X=X_train)
# ab_test_pred = ada_boost.predict(X=X_test)
# # print("ada-boost RMSE on training data: %.3f" % sqrt(mean_squared_error(y_true=y_train, y_pred=ab_train_pred)))
# # print("ada-boost RMSE on test data: %.3f" % sqrt(mean_squared_error(y_true=y_test, y_pred=ab_test_pred)))
# print("-------------")
# print("ada-boost MAE on training data: %.3f" % mean_absolute_error(y_true=y_train, y_pred=ab_train_pred))
# print("ada-boost MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=ab_test_pred))
# print("-------------")
# print("ada-boost variance score on training data: %.3f" % r2_score(y_true=y_train, y_pred=ab_train_pred))
# print("ada-boost variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=ab_test_pred))
# print("-------------------------------")


# """plot driving range based on the battery quantity"""
# quantity = X[:, 2]
# distance = y
# quantity = np.reshape(quantity, newshape=(-1, 1))
# distance = np.reshape(distance, newshape=(-1, 1))
#
# quantity_linear_reg = LinearRegression()
# quantity_linear_reg.fit(X=quantity, y=distance)
# q_slope = quantity_linear_reg.coef_[0]
# q_intercept = quantity_linear_reg.intercept_
# q_predicted_distances = q_intercept + q_slope * quantity
#
# plt.scatter(x=quantity, y=distance, s=15, c='black', linewidths=0.1)
# plt.plot(quantity, q_predicted_distances, c='red', linewidth=2)
# plt.legend(('fitted line', 'data records'), loc='lower right')
# plt.title(label='Linear Regression Plot')
# plt.xlabel(xlabel='quantity (kWh)'), plt.ylabel(ylabel='driving range (km)')
# plt.show()


# """plot driving range based on the average speed"""
# avg_speed = X[:, 9]
# avg_speed = np.reshape(avg_speed, newshape=(-1, 1))
#
# speed_linear_reg = LinearRegression()
# speed_linear_reg.fit(X=avg_speed, y=distance)
# s_slope = speed_linear_reg.coef_[0]
# s_intercept = speed_linear_reg.intercept_
# s_predicted_distances = s_intercept + s_slope * quantity
#
# plt.scatter(x=avg_speed, y=distance, s=15, c='orange', linewidths=0.1)
# plt.plot(quantity, s_predicted_distances, c='blue', linewidth=2)
# plt.legend(('fitted line', 'data records'), loc='upper left')
# plt.title(label='Linear Regression Plot')
# plt.xlabel(xlabel='average speed (km/h)'), plt.ylabel(ylabel='driving range (km)')
# plt.xlim(-5, 110), plt.ylim(-30, 650)
# plt.show()
