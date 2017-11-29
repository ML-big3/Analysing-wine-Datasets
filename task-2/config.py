PREFIX = "/Users/zen/Dropbox/Wine Quality Ratings and Chemicals/"
WHITE_WINE = PREFIX + "winequality-white.csv"
RED_WINE = PREFIX + "winequality-red.csv"
FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

FEATURES_WITH_TYPE = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "type"]

TARGET = "quality"