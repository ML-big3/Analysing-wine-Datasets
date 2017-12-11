# from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


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
    clf = KNeighborsClassifier(6, weights="uniform")
    # Accuracy metric
    print("accuracy ", cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    return(cross_val_score(clf, X, y, cv=10, scoring="accuracy").mean())
    # f1 score
    # print("f1 ", cross_val_score(clf, X, y, cv=10, scoring="f1").mean())


def SVM(X, y):
    print("SVM")
    clf = SVC(kernel='linear', random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = SVC(kernel='linear', random_state=0)
    clf.fit(X_train, y_train)
    print("accuracy", clf.score(X_test, y_test))
    return clf.score(X_test, y_test)


def linear_regression(X, y):
    regressor = LinearRegression()
    print("R2  ", cross_val_score(regressor, X, y, cv=10, scoring="r2").mean())
    return cross_val_score(regressor, X, y, cv=10, scoring="r2").mean()
