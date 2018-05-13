from sklearn import multiclass, svm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
import sklearn as sk
import csv

def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    '''
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier,
    multiclass.OneVsOneClassifier and linear_model.LogisticRegression
    to perform multiclass classification.

    Arguments:
        mode: one of 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 1 to 8.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    '''

    if mode == 'crammer':
        clf = LinearSVC(random_state=12345, multi_class='crammer_singer')
    elif mode == 'ovo':
        clf = OneVsOneClassifier(LinearSVC(degree=3, random_state=12345))
    else:
        clf = sk.linear_model.LogisticRegression(max_iter=100)


    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    return (y_pred_train, y_pred_test)


# Read training file
data = pd.read_csv('training.csv')
data = np.array(data)
print('data', data)

# If there is no data, use mean of the whole column alternatively
X_train = data[:,3:80].astype('float')
for j in range(X_train.shape[1]):
    mean = np.nanmean(X_train[:,j], axis=0)
    for i in range(X_train.shape[0]):
        if pd.isnull(X_train[i][j]):
            X_train[i][j] = mean

# Check if all nan are changed into number
a = pd.isnull(X_train)
b = np.sum(a)
print('b',b)

y_train = data[:,-1].astype('int')

# Read testing file
data2 = pd.read_csv('testing.csv')
data2 = np.array(data2)
print('data2', data2)
X_test = data2[:,3:80].astype('float')
for j in range(X_test.shape[1]):
    mean = np.nanmean(X_test[:,j], axis=0)
    for i in range(X_test.shape[0]):
        if pd.isnull(X_test[i][j]):
            X_test[i][j] = mean

# Train the get accuracy
y_pred_train, y_pred_test = sklearn_multiclass_prediction(
            'ovr', X_train, y_train, X_test)
print('y_pred_train', y_pred_train)
train_acc = metrics.accuracy_score(y_train, y_pred_train)
print('train acc', train_acc)

# Output file
y_pred_test = np.reshape(y_pred_test, (y_pred_test.shape[0],1))
y_ID = data2[:,0]
y_ID = np.reshape(y_ID, (y_ID.shape[0],1))
y = np.append(y_ID, y_pred_test, axis=1)
print('y', y)

Title = np.array(['Id','Response'])
Title = np.reshape(Title, (1,2))
y = np.append(Title,y,axis=0)

my_df = pd.DataFrame(y)
my_df.to_csv('out.csv', index=False, header=False)
