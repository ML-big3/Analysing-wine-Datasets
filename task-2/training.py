import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

# Linear Regression Algorithm
def linear_regression(X, y):
    print("LinearRegression")
    regressor = LinearRegression()
    #RMSE metric
    print("RMSE", np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean())/(y_max-y_min))
    #R2 metric
    print("R2  ", cross_val_score(regressor, X, y, cv=10, scoring="r2").mean())

def knn(X, y):
    print("KNN")
    clf = KNeighborsClassifier(10, weights="uniform")
    # Accuracy metric
    print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    # f1 score
    print("f1 ", cross_val_score(clf, X, y, cv=10, scoring="f1").mean())