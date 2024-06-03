# FILENAME: LogisticRegression.py
# DUE DATE: 5/5/2024
# AUTHOR:   Reg Gonzalez
# EMAIL:    rdg170330@utdallas.edu (school) or regmckie@gmail.com (personal)
# COURSE:   CS 6375.001, Spring 2024
# VERSION:  1.0
#
# PROJECT DESCRIPTION:
# The project is an essential part of this class. It will allow you to demonstrate
# your Machine Learning (ML) skills and create something that you are proud of.
# It can also be a valuable addition to your projects portfolio that you can
# demonstrate to prospective employers.
#
# The project has two key components to it:
# - Understanding a recent machine learning technique and associated
# algorithm(s)
# - Apply it to a standard dataset of sufficient complexity. You have to code
# the main part of the algorithm without using any built-in library. You can
# use libraries for pre-processing, loading, analysis of results, etc.
#
# FILE DESCRIPTION:
# This file contains the code for the Logistic Regression algorithm from scratch.
# This will help train the Logistic Regression model and compare it to other
# kinds of models.

import numpy as np


class LogisticRegression:

    '''
    Initialize parameters for the Logistic Regression model

    PARAMETERS:
        - max_iterations: number of iterations that model will train for
        - learning_rate: how quickly or slowly we should approach the minimum

    RETURNS:
        N/A
    '''
    def __init__(self, max_iterations=2000, learning_rate=0.01):
        self.iterations = max_iterations
        self.learning_rate = learning_rate
        self.classes = None
        self.biases = None
        self.weights = None

    '''
    Train the model using Logistic Regression

    PARAMETERS:
        - X_train: part of the dataset used for training (features)
        - y_train: part of the dataset used for training (class label)

    RETURNS:
        N/A
    '''
    def train_model(self, X_train, y_train):
        # Get number of samples, features, and unique classes from the dataset
        no_of_samples, no_of_features = X_train.shape
        self.classes = np.unique(y_train)
        no_of_classes = len(self.classes)
        max_iterations = self.iterations

        # Initialize weighs and biases to 0
        self.biases = np.zeros(no_of_classes)
        self.weights = np.zeros((no_of_classes, no_of_features))

        # Train the model
        for counter, classification_label in enumerate(self.classes):
            bias = 0
            weights = np.zeros(no_of_features)
            true_y_values = np.where(y_train == classification_label, 1, 0)

            for _ in range(max_iterations):
                # Use Sigmoid activation function to help with class predictions
                x_for_sigmoid = np.dot(X_train, weights) + bias
                sigmoid_result = 1 / (1 + np.exp(-x_for_sigmoid))
                class_preds = sigmoid_result

                # Get the gradients of the weights and biases
                gradient_biases = (1 / no_of_samples) * np.sum(class_preds - true_y_values)
                bias -= self.learning_rate * gradient_biases
                gradient_weights = (1 / no_of_samples) * np.dot(X_train.T, (class_preds - true_y_values))
                weights -= self.learning_rate * gradient_weights

            # Update weights and biases
            self.biases[counter] = bias
            self.weights[counter] = weights

    '''
    Predicts the class—here, it's the race for different DnD characters
    
    PARAMETERS: 
        - X_test: part of the dataset used for testing (features)
    
    RETURNS:
        - class_prediction: class prediction of DnD race    
    '''
    def predict_class(self, X_test):
        # Use sigmoid equation to make model predictions
        x_for_sigmoid = np.dot(X_test, self.weights.T) + self.biases
        model_predictions = 1 / (1 + np.exp(-x_for_sigmoid))

        # Class prediction is the maximum probability from each model's prediction
        class_prediction = np.argmax(model_predictions, axis=1)

        return class_prediction
