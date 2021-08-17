import numpy as np
import pandas as pd 

def train_test_split(X, y, test_frac = 0.2):
   """
   Splitting the dataset into training and testing set.

   Input 
   ------
   X: array-like of shape (n_samples, n_features)
      Features data
   y: array-like of shape (n_samples, 1)
      Label data
   test_frac: float 
      Fraction of data to split into training and test set. The number will indicate how much of the dataset will be taken to make the new training dataset.
   random_state: int 
      Seed for random number generator
   
   Yields 
   ------
   X_train: array-like of shape (n_samples, n_features)
      Training data set
   X_test: array-like of shape (n_samples, n_features)
      Testing data set
   y_train: array-like of shape (n_samples, 1)
      Training data labels
   y_test: array-like of shape (n_samples, 1)
      Testing data labels
   """
   shuffle_idx = np.random.permutation(X.shape[0])
   X = X[shuffle_idx]
   y = y[shuffle_idx]

   n = int(test_frac*X.shape[0])
   X_train = X[:X.shape[0]-n]
   y_train = y[:X.shape[0]-n]
   X_test = X[X.shape[0]-n:n]
   y_test = y[X.shape[0]-n:n]
   return X_train, X_test, y_train, y_test

def initialize_params(X):
    """
    Initializing parameters w and b randomly

    Input
    ------
    X: array-like of shape (n_samples, n_features)
       Training data where n_samples is the number of samples and n_features is the number of features 

    Yields
    -------
    w: random arrays of shape (n_features, 1)
       The coefficients for the features in the data. The number of w will change depending on the number of n_features
    b: an array 0 of shape (1,1)
       The coefficients for the interception
       
    """
    w = np.random.rand(X.shape[1], 1)
    b = np.array(0)
    return w, b

def make_predictions(X, w, b):
   """
   Making predictions using the formula: y = X*w + b

   Input
   -------
   X: array-like of shape (n_samples, n_features)
       Training data where n_samples is the number of samples and n_features is the number of features 
   w: array-like of shape (n_features, 1)
       The coefficients for the features in the data. The number of w will change depending on the number of n_features
   b: array-like of shape (1,1)
       The coefficients for the interception
   
   Yields
   -------
   y: array-like of shape (n_sample, 1)
      Predictions calculated using the above formula
   """
   y_hat = X@w + b 
   return y_hat

def loss_function(y_hat, y):
   """
   Calculating the mean squared error

   Input
   -------
   y_hat:  array-like of shape (n_samples, 1)
      Predictions calculated from the make_predictions function 
   y: array-like of shape (n_samples, 1)
      Target variables 

   Yields 
   -------
   RSE: float 
      The mean squared error that measures the lack of fit of the model 
   """
   MSE = ((y - y_hat)**2).mean()
   return round(MSE, 5)

def update_weights(X, y, y_hat, w, b, learning_rate):
   """
   Function for updating w and b by using gradient descent 

   Input 
   -------
   X: array-like of shape (n_samples, n_features)
       Training data where n_samples is the number of samples and n_features is the number of features 
   w: array-like of shape (n_features, 1)
       The coefficients for the features in the data. The number of w will change depending on the number of n_features
   b: array-like of shape (1,1)
       The coefficients for the interception
   y_hat:  array-like of shape (n_samples, 1)
      Predictions calculated from the make_predictions function 
   y: array-like of shape (n_samples, 1)
      Target variables 
   learning_rate: float
      The amount that the coefficients are updating during learning. Learning rate is a configurable hyperparameter, often between 0.0 and 1.0.
   
   Yields 
   -------
   w: array-like of shape (n_features, 1)
       Newly updated coefficients for each of the features in the data
   b: array-like of shape (1,1)
       Newly updated coefficients for the interception

   """
   m = y.shape[0]
   # Define the number of samples m
   # Calculate the derivative of w 
   dw = (2/m)*(X.T@(y_hat - y))
   # Calculate the derivative of b 
   db = (2/m)*np.sum(y_hat - y)
   # Update the parameters
   w = w - learning_rate*dw 
   b = b - learning_rate*db

   return w,b

def R2_statistics(y, y_hat):
   """
   Calculate the R-statistics. 

   Input 
   -------
   y_hat:  array-like of shape (n_samples, 1)
      Predictions calculated from the make_predictions function 
   y: array-like of shape (n_samples, 1)
      Target variables 

   Yields 
   -------
   R_2: float 
      The R2-statistic. If the R2 statictis is close to 1 indicates that a large proportion of the variability in the response is explained by the regression.
   """
   R2 = 1 - (np.sum((y-y_hat)**2)/np.sum((y-np.mean(y))**2))
   return R2
