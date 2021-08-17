# This space is for testing out the functions. The dataset used here is the Boston dataset 
import pandas as pd
from helper_functions import *
from main import *

auto = pd.read_csv('https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Auto.csv')
# print(auto.info())

X = auto[['displacement']].values
y = auto[['mpg']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)


lr = LinearRegression_scratch(X_train, y_train, epoch=100, learning_rate=5e-12)
lr.fit()

# y_pred = lr.predict(X_test)
# print(R2_statistics(y_test, y_pred))

# from sklearn.linear_model import LinearRegression

# lr_sklearn = LinearRegression()
# lr_sklearn.fit(X_train, y_train)

# y_pred_2 = lr_sklearn.predict(X_test)
# print(R2_statistics(y_test, y_pred_2))