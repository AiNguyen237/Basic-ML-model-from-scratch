# This space is for testing out the functions. The dataset used here is the Boston dataset 
import pandas as pd
from helper_functions import *
from main import *

boston = pd.read_csv('https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Boston.csv')

X = boston[['lstat']]
y = boston[['crim']]

X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LinearRegression(X_train, y_train, epoch=100, learning_rate=0.01)
lr.fit()

