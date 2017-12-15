import config
import training
import pandas as pd
import numpy as np


def train_white_wine_dataset():
    print("Start loading white wine dataset")
    white_wine_dataset = pd.read_csv(config.WHITE_WINE, sep=';')
    start_training(white_wine_dataset)


def train_red_wine_dataset():
    print("Start loading red wine dataset")
    red_wine_dataset = pd.read_csv(config.RED_WINE, sep=';')
    start_training(red_wine_dataset)


def data_processing(y):
    k=[]
    for i in y:
        if(i >= 7):
            k.append(1)
        elif(i < 7 and i > 4):
            k.append(0)
        else:
            k.append(-1)
    return k

def feature_selection():
    i = 1
    dataset = pd.read_csv(config.WHITE_WINE, sep=';')
    while(i<=11):
        features = config.WHITE_FEATURES[:i]
        print(i)
        X = dataset[features].values
        y = dataset[config.TARGET].values
        y = data_processing(y)
        training.SVM(X, y)
        training.KNN(X, y)
        training.logistic_regression(X, y)
        training.decision_tree(X, y)
        i+=1


def g_features():
    dataset = pd.read_csv(config.WHITE_WINE, sep=';')
    X = dataset[config.FEATURE].values
    y = dataset[config.TARGET].values
    training.feature_selection(X, y)


def start_training(dataset):
    X = dataset[config.FEATURE].values
    y = dataset[config.TARGET].values
    y = data_processing(y)
    training.SVM(X, y)
    training.logistic_regression(X, y)
    training.KNN(X, y)
    training.decision_tree(X, y)


# g_features()
# feature_selection()
# train_red_wine_dataset()
# train_white_wine_dataset()