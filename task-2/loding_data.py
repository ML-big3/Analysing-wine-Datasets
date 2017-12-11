import config
import training
import pandas as pd


def train_white_wine_dataset():
    print("Start loading white wine dataset")
    white_wine_dataset = pd.read_csv(config.WHITE_WINE, sep=';')
    start_training(white_wine_dataset)


def train_red_wine_dataset():
    print("Start loading red wine dataset")
    red_wine_dataset = pd.read_csv(config.RED_WINE, sep=';')
    start_training(red_wine_dataset)


def start_training(dataset):
    temp = 0
    f = []
    for i in range(1, len(config.FEATURES)):
        X = dataset[config.FEATURES[i]].values
        y = dataset[config.TARGET].values
        # score = training.KNN(X, y)
        score = training.SVM(X, y)
        if score > temp:
            temp = score
            f = config.FEATURES[i]
    print(temp)
    print(f)


train_red_wine_dataset()
# train_white_wine_dataset()
