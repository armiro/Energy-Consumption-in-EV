import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_validate
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


def report_cross_val_results(mae_values, r2_scores):
    print("average MAE values (bias) is:", abs(round(number=mae_values.mean(), ndigits=3)))
    print("std deviation of MAE values (variance) is:", round(number=mae_values.std(), ndigits=3))
    best_mae = sorted(mae_values, reverse=False)[-1]
    print("best MAE value is:", abs(round(number=best_mae, ndigits=3)))

    print("average r2 scores (bias) is:", round(number=r2_scores.mean() * 100, ndigits=3))
    print("std deviation of r2 scores (variance) is:", round(number=r2_scores.std() * 100, ndigits=3))
    best_r2 = sorted(r2_scores, reverse=False)[-1] * 100
    print("best r2 score is:", round(number=best_r2, ndigits=3))


def report_results(training_pred, test_pred):
    print("RMSE on training data: %.3f" % np.sqrt(mean_squared_error(y_true=y_train, y_pred=training_pred)))
    print("RMSE on test data: %.3f" % np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_pred)))
    print("MAE on training data: %.3f" % mean_absolute_error(y_true=y_train, y_pred=training_pred))
    print("MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=test_pred))
    print("variance score on training data: %.3f" % r2_score(y_true=y_train, y_pred=training_pred))
    print("variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=test_pred))


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
# reg_results = cross_validate(estimator=linear_regressor, X=X, y=y, cv=cv, scoring=['neg_mean_absolute_error', 'r2'])
#
# reg_mae_values = reg_results['test_neg_mean_absolute_error']
# reg_r2_scores = reg_results['test_r2']
#
# print("\n ------ Linear Regression CrossVal ------")
# report_cross_val_results(mae_values=reg_mae_values, r2_scores=reg_r2_scores)


"""train the linear regression model and print the results on the never-seen-before test data"""
# linear_regressor.fit(X=X_train, y=y_train)
# reg_training_pred = linear_regressor.predict(X=X_train)
# reg_test_pred = linear_regressor.predict(X=X_test)
#
# print("\n ------ Linear Regression TrainTest ------")
# report_results(training_pred=reg_training_pred, test_pred=reg_test_pred)
# print("-------------------------------")


"""define the shallow multi-layer perceptron model"""
# mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, n_iter_no_change=100,
#                    activation='relu', solver='adam', verbose=False, warm_start=False)


"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# mlp_results = cross_validate(estimator=mlp, X=X, y=y, cv=cv, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=2)
#
# mlp_mae_values = mlp_results['test_neg_mean_absolute_error']
# mlp_r2_scores = mlp_results['test_r2']
#
# print("\n ------ MLP CrossVal ------")
# report_cross_val_results(mae_values=mlp_mae_values, r2_scores=mlp_r2_scores)


"""train the MLP model and print the results on the never-seen-before test data"""
# mlp.fit(X=X_train, y=y_train)
# mlp_training_pred = mlp.predict(X=X_train)
# mlp_test_pred = mlp.predict(X=X_test)
#
# print("\n ------ MLP TrainTest ------")
# report_results(training_pred=mlp_training_pred, test_pred=mlp_test_pred)
# print("-------------------------------")


"""define the random forest ensemble model"""
# rf = RandomForestRegressor(n_estimators=200, criterion="mae", warm_start=False)

"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# rf_results = cross_validate(estimator=rf, X=X, y=y, cv=cv, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=2)
#
# rf_mae_values = rf_results['test_neg_mean_absolute_error']
# rf_r2_scores = rf_results['test_r2']
#
# print("\n ------ Random Forest CrossVal ------")
# report_cross_val_results(mae_values=rf_mae_values, r2_scores=rf_r2_scores)


"""train the RF model and print the results on the never-seen-before test data"""
# rf.fit(X=X_train, y=y_train)
# rf_train_pred = rf.predict(X=X_train)
# rf_test_pred = rf.predict(X=X_test)
#
# print("\n ------ Random Forest TrainTest ------")
# report_results(training_pred=rf_train_pred, test_pred=rf_test_pred)
# print("-------------------------------")


"""define the ada-boost ensemble model"""
# ab = AdaBoostRegressor(n_estimators=50, learning_rate=1.)


"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
# cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
# ab_results = cross_validate(estimator=ab, X=X, y=y, cv=cv, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=2)
#
# ab_mae_values = ab_results['test_neg_mean_absolute_error']
# ab_r2_scores = ab_results['test_r2']
#
# print("\n ------ AdaBoost CrossVal ------")
# report_cross_val_results(mae_values=ab_mae_values, r2_scores=ab_r2_scores)


"""train the ada-boost model and print the results on the never-seen-before test data"""
# ab.fit(X=X_train, y=y_train)
# ab_train_pred = ab.predict(X=X_train)
# ab_test_pred = ab.predict(X=X_test)
#
# print("\n ------ AdaBoost TrainTest ------")
# report_results(training_pred=ab_train_pred, test_pred=ab_test_pred)
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


import keras
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


class LossHistory(keras.callbacks.Callback):
    losses = list()

    def on_epoch_end(self, epoch, logs={}):
        if epoch == num_total_epochs - 1:
            self.losses.append(logs.get('loss'))


def build_regressor():
    regressor = keras.models.Sequential()
    regressor.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=10))
    regressor.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=13, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
    # activation func of the output layer must be 'linear' for regression tasks
    regressor.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_absolute_error')
    return regressor


num_total_epochs = 1000
mae_history = LossHistory()

deep_mlp = KerasRegressor(build_fn=build_regressor, batch_size=16, epochs=num_total_epochs, verbose=False)
cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
r2_scores = cross_val_score(estimator=deep_mlp, X=X, y=y, cv=cv, n_jobs=1, scoring='r2',
                            fit_params={'callbacks': [mae_history]})
mae_scores = np.array(map(lambda x: round(x, ndigits=3), mae_history.losses))

print(mae_scores)
print(np.mean(a=mae_scores))
print(np.std(a=mae_scores))
print(r2_scores)
print(np.mean(a=r2_scores))
print(np.std(a=r2_scores))

