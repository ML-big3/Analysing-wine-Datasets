import config
import training
import pandas as pd

def train_white_wine_dataset():
    print("Start loading white wine dataset")
    white_wine_dataset = pd.read_csv(config.WHITE_WINE, sep = ';')
    start_training(white_wine_dataset)
    

def train_red_wine_dataset():
    print("Start loading red wine dataset")
    red_wine_dataset = pd.read_csv(config.RED_WINE, sep = ';')
    start_training(red_wine_dataset)

    
def start_training(dataset):
    X = dataset[config.FEATURES].values
    y = dataset[config.TARGET].values
    training.linear_regression(X, y)
    training.knn(X, y)
    
    
train_red_wine_dataset()
train_white_wine_dataset()