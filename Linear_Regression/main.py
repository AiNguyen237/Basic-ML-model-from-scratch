import numpy as np 
from helper_functions import *

class LinearRegression():
    """
    Ordinary Linear Regression model
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train


    def fit(self, epoch, learning_rate):
        """
        Fit Linear model to the data. 

        Input
        ------
        X: array-like of shape (n_samples, n_features)
            Training data. 
        y: array-like of shape (n_samples, 1)
            Training labels.

        Yields 
        ------
        self: returns an instance of the self 
        """
        # Splitting dataset into training and validation set 
        xtrain, xval, ytrain, yval = train_test_split(self.X_train, self.y_train)
        # Implementing the Gradient Descent 
        # Initialize the parameters 
        w, b = initialize_params(xtrain)

        # Calculating y_hat and updating the weights  
        for i in range(epoch):

            # Train the model using the training data 
            y_hat = make_predictions(xtrain, w, b)
            train_loss = loss_function(y_hat, ytrain)
            w, b = update_weights(xtrain, ytrain, y_hat, w, b, learning_rate)

            # Print out the loss of training and validation set every 10 epochs 
            if i%10:
                # Validate the model predictions using the validation data every 10 epochs 
                y_hat_val = make_predictions(xval, w, b)
                val_loss = loss_function(y_hat_val, yval)
                print(f'Epoch {i} ------- train_loss: {train_loss} --------- val_loss: {val_loss}')

        return self
        
    def predict(self, X):
        """
        Predict the label using the pre-trained weights.

        Input 
        ------
        X: array-like of shape (n_samples, n_features)
            Data used to predict the labels.
        
        Output 
        ------
        y_pred: array-like of shape (n_samples, 1)
            Predicted labels using the Linear Model.
        """
        y_pred = make_predictions(X, self.w, self.b)
        return y_pred

