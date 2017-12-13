# PREFIX = "/Users/zen/Dropbox/Wine Quality Ratings and Chemicals/"
PREFIX = "Wine Quality Ratings and Chemicals/"

WHITE_WINE = PREFIX + "winequality-white.csv"
RED_WINE = PREFIX + "winequality-red.csv"
COMBINE = PREFIX + "combine.csv"

FEATURE = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"]

RED_FEATURES = ['alcohol', 'sulphates', 'total sulfur dioxide', 'volatile acidity', 'density', 'pH', 'chlorides', 'citric acid', 'fixed acidity', 'residual sugar', 'free sulfur dioxide']
WHITE_FEATURES = ['alcohol', 'density', 'volatile acidity', 'free sulfur dioxide', 'total sulfur dioxide', 'residual sugar', 'chlorides', 'pH', 'citric acid', 'sulphates', 'fixed acidity']
TARGET = "quality"
