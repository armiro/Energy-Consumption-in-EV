import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def do_kfold(k, test_size, classifier):
    cv = ShuffleSplit(n_splits=k, test_size=test_size, random_state=41)
    acc_scores = cross_val_score(estimator=classifier, X=X, y=y, scoring='balanced_accuracy', n_jobs=2, cv=cv)
    print('accuracy scores:', acc_scores)
    print("average accuracy score (bias) is:", abs(round(number=acc_scores.mean() * 100, ndigits=3)))
    print("std deviation of MAE scores (variance) is:", round(number=acc_scores.std() * 100, ndigits=3))
    best_acc = sorted(acc_scores, reverse=False)[-1]
    print("best accuracy score is:", abs(round(number=best_acc * 100, ndigits=3)))


warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

old_path = "./data files/data_2.csv"
new_path = "./data files/new_data_2.csv"


"""remove missing values (comment it after the first run)"""
ds = pd.read_csv(filepath_or_buffer=old_path)
ds = ds[pd.notnull(obj=ds['quantity(kWh)'])]
ds = ds[pd.notnull(obj=ds['avg_speed(km/h)'])]
ds = ds[pd.notnull(obj=ds['consumption(kWh/100km)'])]
ds.to_csv(path_or_buf=new_path)


"""load the data"""
dataset = pd.read_csv(filepath_or_buffer=new_path)
# print(dataset.head(n=5))
# print(dataset.describe())

X = dataset.iloc[:, 5:15].values
y = dataset.iloc[:, 15].values
# consumption_values = dataset.iloc[:, 16].values


"""change ECR deviation values into binary values:
if real ECR is more than manufacture pre-defined ECR -> put 1
if real ECR is less than manufacture pre-defined ECR -> put 0"""
y = (y >= 0)
y = np.array(y, dtype='int')


"""do the preprocessing tasks on the data"""
# encode categorical features
label_encoder_1 = LabelEncoder()
X[:, 2] = label_encoder_1.fit_transform(y=X[:, 2])
label_encoder_2 = LabelEncoder()
X[:, 6] = label_encoder_2.fit_transform(y=X[:, 6])

# onehot encoding for categorical features with more than 2 categories
onehot_encoder = OneHotEncoder(categorical_features=[6])
X = onehot_encoder.fit_transform(X=X).toarray()

# delete the first column to avoid the dummy variable
X = X[:, 1:]

# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.fit_transform(X=X_test)


"""find the best parameters for the SVM classifier using GridSearch (comment it after the first run)"""
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.1, 1, 10, 100], 'degree': [1, 2, 3, 4, 5]}
svm_classifier = svm.SVC()
clf = GridSearchCV(estimator=svm_classifier, param_grid=parameters)
clf.fit(X=X_train, y=y_train)
print(clf.best_params_)


"""define the SVM classification model with best parameters obtained from above"""
svm_classifier = svm.SVC(C=1000, kernel='rbf', gamma=0.12)


"""KFold cross-validation"""
print("\n ------ SVM ------")
do_kfold(k=10, test_size=0.25, classifier=svm_classifier)


"""evaluate on the never-seen-before test data"""
svm_classifier.fit(X=X_train, y=y_train)
svm_training_pred = svm_classifier.predict(X=X_train)
svm_test_pred = svm_classifier.predict(X=X_test)

svm_training_acc = accuracy_score(y_true=y_train, y_pred=svm_training_pred)
svm_test_acc = accuracy_score(y_true=y_test, y_pred=svm_test_pred)

print("SVM accuracy on training-set:", svm_training_acc)
print("SVM accuracy on test-set:", svm_test_acc)

svm_cm = confusion_matrix(y_true=y_test, y_pred=svm_test_pred)
print("SVM confusion matrix on test-set:", svm_cm)


"""define the MLP as the classifier"""
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50,), activation='relu', solver='adam',
                               verbose=False, max_iter=1000, n_iter_no_change=100, warm_start=True)


"""KFold cross-validation"""
print("\n ------ MLP Classifier ------")
do_kfold(k=10, test_size=0.25, classifier=mlp_classifier)


"""evaluate on the never-seen-before test data"""
mlp_classifier.fit(X=X_train, y=y_train)
mlp_training_pred = mlp_classifier.predict(X=X_train)
mlp_test_pred = mlp_classifier.predict(X=X_test)

mlp_training_acc = accuracy_score(y_true=y_train, y_pred=mlp_training_pred)
mlp_test_acc = accuracy_score(y_true=y_test, y_pred=mlp_test_pred)

print("MLP accuracy on training-set:", mlp_training_acc)
print("MLP accuracy on test-set:", mlp_test_acc)

mlp_cm = confusion_matrix(y_true=y_test, y_pred=mlp_test_pred)
print("MLP confusion matrix on test-set:", mlp_cm)


"""define the RF as the classifier"""
rf_classifier = RandomForestClassifier(n_estimators=50)


"""KFold cross-validation"""
print("\n ------ Random Forest Classifier ------")
do_kfold(k=10, test_size=0.25, classifier=rf_classifier)


"""evaluate on the never-seen-before test data"""
rf_classifier.fit(X=X_train, y=y_train)
rf_training_pred = rf_classifier.predict(X=X_train)
rf_test_pred = rf_classifier.predict(X=X_test)

rf_training_acc = accuracy_score(y_true=y_train, y_pred=rf_training_pred)
rf_test_acc = accuracy_score(y_true=y_test, y_pred=rf_test_pred)

print("RF accuracy on training-set:", rf_training_acc)
print("RF accuracy on test-set:", rf_test_acc)

rf_cm = confusion_matrix(y_true=y_test, y_pred=rf_test_pred)
print("RF confusion matrix on test-set:", rf_cm)
