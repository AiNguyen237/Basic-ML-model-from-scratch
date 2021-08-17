import numpy as np 
from helper_functions import *

class LinearRegression_scratch():
    """
    Ordinary Linear Regression model
    """
    def __init__(self, X_train, y_train, epoch, learning_rate):
        self.X_train = X_train 
        self.y_train = y_train
        self.w, self.b = initialize_params(self.X_train)
        self.epoch = epoch
        self.learning_rate = learning_rate

    def fit(self):
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

        # Calculating y_hat and updating the weights  
        for i in range(self.epoch):
            
            # Train the model using the training data 
            y_hat = make_predictions(xtrain, self.w, self.b)
            train_loss = loss_function(y_hat, ytrain)
            R2_acc_train = R2_statistics(ytrain, y_hat)
            self.w, self.b = update_weights(xtrain, ytrain, y_hat, self.w, self.b, self.learning_rate)

            # Print out the loss of training and validation set every 10 epochs 
            if i%10 == 0:
                # Validate the model predictions using the validation data every 10 epochs 
                y_hat_val = make_predictions(xval, self.w, self.b)
                val_loss = loss_function(y_hat_val, yval)
                R2_acc_val = R2_statistics(yval, y_hat_val)
                print(f'Epoch {i} ------- train_loss: {train_loss} --------- train_acc: {R2_acc_train} --------- val_loss: {val_loss} --------- val_acc: {R2_acc_val}')

        return self
        
    def predict(self, xtest):
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
        y_pred = make_predictions(xtest, self.w, self.b)
        return y_pred

