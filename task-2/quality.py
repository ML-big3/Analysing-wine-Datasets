import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.cross_validation
import config


red_wine_dataset = pd.read_csv(config.RED_WINE, sep=';')
white_wine_dataset = pd.read_csv(config.WHITE_WINE, sep=';')
red_wine_dataset.head()
X_df = red_wine_dataset.iloc[:,:-1]
X_df.head()
X = X_df.as_matrix()
print(X[:3])
y_df = red_wine_dataset["quality"].values
plt.hist(y_df, range=(1, 10))

plt.xlabel('Ratings of Red wines')
plt.ylabel('Amount')
plt.title('Red wine ratings Distribution')
plt.show()



white_wine_dataset.head()
WX_df = white_wine_dataset.iloc[:,:-1]
WX_df.head()
WX = WX_df.as_matrix()
Wy_df = white_wine_dataset["quality"].values
plt.hist(Wy_df, range=(1, 10))

plt.xlabel('Ratings of White wines')
plt.ylabel('Amount')
plt.title('White Wine ratings Distribution')
plt.show()