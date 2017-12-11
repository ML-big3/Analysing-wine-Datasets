# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# Linear Regression Algorithm
# def linear_regression(X, y):
#     print(y)
#     print("LinearRegression")
#     regressor = LinearRegression()
#     # Accuracy metric
#     # print("RMSE", cross_val_score(regressor, X, y, cv=10, scoring="accuracy").mean())
#     # R2 metric
#     print("R2  ", cross_val_score(regressor, X, y, cv=10, scoring="r2").mean())


def KNN(X, y):
    print("KNN")
    clf = KNeighborsClassifier(10, weights="uniform")
    # Accuracy metric
    return("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    # f1 score
    # print("f1 ", cross_val_score(clf, X, y, cv=10, scoring="f1").mean())


def SVM(X, y):
    print("SVM")
    clf = SVC(kernel='linear', random_state=0)
    # Accuracy metric
    return("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
