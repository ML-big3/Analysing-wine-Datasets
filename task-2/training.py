# from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import config
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

def feature_selection(X, y):
    print('feature_selection')
    rf = RandomForestClassifier()
    rf.fit(X, y)
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), config.FEATURE),
             reverse=True))

def KNN(X, y):
    print("KNN")
    clf = KNeighborsClassifier(16, weights="uniform")
    print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())


def SVM(X, y):
    print("SVM")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = SVC()
    # clf.fit(X_train, y_train)
    print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    # print("accuracy", clf.score(X_test, y_test))


def decision_tree(X, y):
    print("Decision Tree")
    clf = DecisionTreeClassifier()
    print("DecisionT  ", cross_val_score(clf, X, y, cv=10).mean())


def logistic_regression(X, y):
    print("LogisticRegression")
    clf = LogisticRegression()
    print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean()
