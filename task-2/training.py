import evaluation
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
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), config.FEATURE), reverse=True))

def KNN(X, y):
    print("KNN")
    clf = KNeighborsClassifier(17, weights="uniform")
    # print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    get_mtrics(clf, X, y, "KNN")


def SVM(X, y):
    print("SVM")
    clf = SVC(probability=True)

    get_mtrics(clf, X, y, "SVM")
    
    # print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())


def decision_tree(X, y):
    print("decision_tree")
    clf = DecisionTreeClassifier()
    get_mtrics(clf, X, y, "decision_tree")
    # print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())


def logistic_regression(X, y):
    print("logistic_regression")
    clf = LogisticRegression()
    get_mtrics(clf, X, y, "logistic_regression")
    # print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())


def get_mtrics(clf, X, y, name):
    metrics = evaluation.EvaluationMetrics(clf, X, y, name)
    metrics.cross_validate_confusion_matrix()
    metrics.cross_validate_precision_score()
    metrics.cross_validate_auc_roc()
    metrics.cross_validate_for_accuracy()
    metrics.cross_validate_f1()
    metrics.cutoff_predict()
    metrics.cross_validate_recall()
    