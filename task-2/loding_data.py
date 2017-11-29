import config
import training
import pandas as pd
import zipfile

def train_sum_without_noise_data():
    print("Start loading red wine dataset")
    red_wine_dataset = pd.read_csv(config.SUM_WO_NOISE_DS, sep = ';')
    start_training(red_wine_dataset)
    

def train_sum_without_noise_data():
    print("Start loading white wine dataset")
    white_wine_dataset = pd.read_csv(config.SUM_WO_NOISE_DS, sep = ';')

    
def start_training(dataset):
    X = dataset[config.FEATURES].values
    y = dataset[config.TARGET].values
    training.linear_regression(X, y)
    training.knn(X, y)