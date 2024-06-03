# FILENAME: kNN.py
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
# This file contains the code for the K-nearest-neighbors algorithm from scratch.
# This will help train the K-nearest-neighbors algorithm and compare it to other
# kinds of algorithms.

from collections import Counter
import numpy as np


class KNearestNeighbors:

    '''
    Initialize parameters for the K-nearest-neighbors model

    PARAMETERS:
        - k_value: hyperparameter that looks for the 'K' nearest neighbors of a data point

    RETURNS:
        N/A
    '''
    def __init__(self, k_value=5):
        self.X_train = None
        self.y_train = None
        self.k_value = k_value

    '''
    Gets the X and y training sets that will be used for training the model
    
    PARAMETERS:
        - X_train: X training set (features)
        - y_train: y training set (labels)
    
    RETURNS:
        N/A
    '''
    def train_algorithm(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    '''
    Calculates the distance between a test data point and every other data point in 
    the training set, sorts the k nearest data points and gets their indices, gets
    the y value associated with those k nearest data points, and finally computes
    the majority label of those k nearest data points.
    
    PARAMETERS:
        - x_test_value: a single data point from the X_test set
    
    RETURNS:
        - most_common_y_value: the y value (i.e., a 'race' label) that makes up the majority
        of the k nearest neighbors. For example, let's say that we have a data point and k=5. 
        If 3/5 data points are labeled "tiefling," then the most_common_y_value would return
        "tiefling"
    '''
    def get_majority_label(self, x_test_value):
        # Use Euclidean Distance as measurement between data points
        euclidean_distances = []
        for x_train_value in self.X_train.values:
            e_distance = np.sqrt(np.sum((x_test_value - x_train_value) ** 2))
            euclidean_distances.append(e_distance)

        # Sort the nearest k points and get their indices
        indices_of_nearest_k = np.argsort(euclidean_distances)[: self.k_value]

        # Get the y values (i.e., the 'race' labels) that correspond to the
        # k nearest points
        nearest_k_y_values = []
        for index in indices_of_nearest_k:
            nearest_k_y_values.append(self.y_train.iloc[index])

        # Classify the data point as the majority label of k nearest neighbors
        common_y_values = Counter(nearest_k_y_values).most_common()
        most_common_y_value = common_y_values[0][0]

        return most_common_y_value

    '''
    Gets the class predictions based on the most common y value (in above method).
    The class prediction IS the most common label.
    
    PARAMETERS:
        - X_test_values: the values from the X testing set
        
    RETURNS:
        - class_predictions: Class prediction of DnD race
    '''
    def predict_class(self, X_test_values):
        class_predictions = []
        for x_test_value in X_test_values:
            class_predictions.append(self.get_majority_label(x_test_value))

        return class_predictions
