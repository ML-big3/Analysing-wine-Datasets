# PREFIX = "/Users/zen/Dropbox/Wine Quality Ratings and Chemicals/"
PREFIX = "Wine Quality Ratings and Chemicals/"

WHITE_WINE = PREFIX + "winequality-white.csv"
RED_WINE = PREFIX + "winequality-red.csv"
COMBINE = PREFIX + "combine.csv"

FEATURE = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"]

# FEATURE = ['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide', 'density', 'chlorides', 'fixed acidity', 'pH', 'free sulfur dioxide']
FEATURE = ['alcohol', 'total sulfur dioxide', 'sulphates', 'volatile acidity', 'density', 'chlorides', 'citric acid', 'pH', 'fixed acidity', 'free sulfur dioxide', 'residual sugar']

def subs(l):
    if l == []:
        return [[]]

    x = subs(l[1:])

    return x + [[l[0]] + y for y in x]

FEATURES = subs(["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"])


FEATURES_WITH_TYPE = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                      "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH",
                      "sulphates", "alcohol", "type"]

TARGET = "quality"
