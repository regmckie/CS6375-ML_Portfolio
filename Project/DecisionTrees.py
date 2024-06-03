# FILENAME: DecisionTrees.py
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
# This file contains the code for the Decision Trees algorithm from scratch.
# This will help train the Decision Tree model and compare it to other
# kinds of models.

from collections import Counter
import numpy as np

class TreeNode():

    '''
    Initialize parameters for a node in the decision tree

    PARAMETERS
        - feature_divided: the feature/attribute we divide the tree on in this particular node
        - threshold_divided: the threshold we divide the tree on in this particular node
        - right_subtree: points to the right subtree
        - left_subtree: points to the left subtree
        - node_value: the node's value—this will only come into play when the node is a leaf node

    RETURNS:
        N/A
    '''
    def __init__(self, threshold_divided=None, feature_divided=None,
                 right_subtree=None, left_subtree=None, *, node_value=None):
        self.threshold_divided_with = threshold_divided
        self.feature_divided_with = feature_divided
        self.right_subtree = right_subtree
        self.left_subtree = left_subtree
        self.node_value = node_value

    '''
    Checks to see if the node is a leaf node
    
    PARAMETERS:
        N/A
        
    RETURNS:
        is_leaf_node: true is the node is a leaf node and false otherwise
    '''
    def check_if_leaf_node(self):
        is_leaf_node = False
        if self.node_value is not None:
            is_leaf_node = True
            return is_leaf_node
        else:
            return is_leaf_node


class DecisionTree:
    '''
    Initializes the decision tree
    
    PARAMETERS: 
        - max_tree_depth: maximum depth the tree can go (stopping criteria)
        - no_of_features: subset of the number of features we'll use to split the tree on 
        - min_no_of_samples: minimum number of samples that a node can have; if a node has less
        than this number of samples, then we won't split it any further (stopping criteria)
        
    RETURNS:
        N/A
    '''
    def __init__(self, max_tree_depth=100, no_of_features=None, min_no_of_samples=3):
        self.tree_root = None
        self.max_tree_depth = max_tree_depth
        self.no_of_features = no_of_features
        self.min_no_of_samples = min_no_of_samples

    '''
    Calculates the entropy (can be for a parent node and a child node).
    Entropy is defined as: -sum(log_2(p(X) * p(X))

    PARAMETERS:
        - y_train: y training set (labels)

    RETURNS:
        - entropy: entropy for a particular node
    '''
    def calculate_entropy(self, y_train):
        size_of_training_labels = len(y_train)

        # Create histogram of y labels that tells us how many times a certain value has occurred
        # (e.g., "tiefling" has occurred 6 times, "human" has occurred 8 times, etc.)
        histogram_of_labels = np.bincount(y_train)

        # Get the value of each entry in the histogram as a percentage
        percent_of_label_occurrences = histogram_of_labels / size_of_training_labels

        # Calculate and return entropy
        entropy = 0
        for plo in percent_of_label_occurrences:
            if plo > 0:
                entropy = entropy - (np.log(plo) * plo)

        return entropy

    '''
    Calculates the information gain of a feature
    Information gain is defined as: entropy(parent node) - [weighted average] * entropy(child nodes)

    PARAMETERS:
        - threshold: one of the thresholds in a list of possible thresholds (i.e., unique values
        in a particular feature's column)
        - column_of_feature: column of a specific feature (e.g., "intelligence," "charisma," "speed," etc.)
        - y_train: y training set (labels)

    RETURNS:
        - information_gain: information gain calculated for a specific feature
    '''
    def calculate_info_gain(self, threshold, column_of_feature, y_train):
        # Calculate the parent node's entropy
        entropy_parent_node = self.calculate_entropy(y_train)

        # Get the right and left indices
        right_subtree_indices = np.argwhere(column_of_feature > threshold).flatten()
        left_subtree_indices = np.argwhere(column_of_feature <= threshold).flatten()

        # If the tree doesn't expand to its right and left subtrees, then return 0
        if len(right_subtree_indices) == 0 or len(left_subtree_indices) == 0:
            return 0

        # Calculate the entropy of the child nodes (this is a weighted average)
        size_of_training_labels = len(y_train)
        size_of_right_subtree = len(right_subtree_indices)
        size_of_left_subtree = len(left_subtree_indices)
        entropy_right_subtree = self.calculate_entropy(y_train[right_subtree_indices])
        entropy_left_subtree = self.calculate_entropy(y_train[left_subtree_indices])
        entropy_child_node = (size_of_left_subtree / size_of_training_labels) * entropy_left_subtree + \
                             (size_of_right_subtree / size_of_training_labels) * entropy_right_subtree

        # Calculate information gain
        info_gain = entropy_parent_node - entropy_child_node

        return info_gain

    '''
    Finds the best feature and threshold to split a tree's node on

    PARAMETERS:
        - idxs_of_random_features: indices of randomly selected features; these features will
        be looked at when splitting the node
        - X_train: X training set (features)
        - y_train: y training set (labels)

    RETURNS:
        - threshold_to_split: the best threshold to split the node on
        - feature_to_split: the best feature to split the node on
    '''
    def find_best_feat_thres_to_split(self, idxs_of_random_features, X_train, y_train):
        threshold_to_split = None
        feature_to_split = None
        best_information_gain = 0

        for index_of_feature in idxs_of_random_features:
            # Get column that corresponds to the index of a random feature
            # and get possible thresholds from that column (i.e., just the unique values of that column)
            column_of_feature = X_train[:, index_of_feature]
            possible_thresholds = np.unique(column_of_feature)

            for threshold in possible_thresholds:
                # Calculate information gain
                information_gain = self.calculate_info_gain(threshold, column_of_feature, y_train)

                # We want to maximize information gain, so every time a new value for
                # info gain is greater than the [last] best info gain, replace it
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    threshold_to_split = threshold
                    feature_to_split = index_of_feature

        return threshold_to_split, feature_to_split

    '''
    Calculates the majority label of a leaf node. When we get to a leaf node, the value of that
    node will either be the label that occupies that node (if it's a pure node) or the majority
    of labels that occupy that node (if it's not a pure node). For example, if a node has 3 samples
    in it and 2/3 are "tiefling," then the node's value is "tiefling."

    PARAMETERS:
        - y_train: y training set (labels)

    RETURNS:
        - majority_label: the majority label of that leaf node
    '''
    def get_majority_label(self, y_train):
        count = Counter(y_train)
        majority_label = count.most_common(1)[0][0]

        return majority_label

    '''
    Recursively creates the decision tree
    
    PARAMETERS:
        - X_train: X training set (features)
        - y_train: y training set (labels)
        - tree_depth: depth of the decision tree (initially this is 0, but will grow as the 
        depth increases)
        
    RETURNS:
        - root_node: the root node of the decision tree
    '''
    def create_decision_tree(self, X_train, y_train, tree_depth=0):
        # Get number of samples, number of features, and number of classes from
        # training data
        no_of_classes = len(np.unique(y_train))
        no_of_samples, num_features = X_train.shape

        # Ensures that the number of features will not exceed
        # the number of features we actually have
        if not self.no_of_features:
            self.no_of_features = X_train.shape[1]
        else:
            self.no_of_features = min(self.no_of_features, X_train.shape[1])

        # Check for the stopping criteria.
        # That is, check to see if the node only has 1 class (i.e., it's a pure node)
        # or the number of samples is less than the minimum number of samples 
        # or the depth exceeds the max depth.
        if(no_of_classes == 1 or no_of_samples < self.min_no_of_samples or tree_depth >= self.max_tree_depth):
            leaf_node_value = self.get_majority_label(y_train)
            leaf_node = TreeNode(node_value = leaf_node_value)
            return leaf_node

        # Get a subset of features to build the tree one;
        # these features will be randomly selected
        idxs_of_random_features = np.random.choice(self.no_of_features, num_features, replace=False)

        # For internal nodes in the tree, find the best threshold and feature to split on
        threshold_to_split, feature_to_split = self.find_best_feat_thres_to_split(idxs_of_random_features, X_train,
                                                                                  y_train)

        # Recursively grow the right and left subtrees
        tree_depth = tree_depth + 1
        column_of_feature_to_split = X_train[:, feature_to_split]

        right_subtree_indices = np.argwhere(column_of_feature_to_split > threshold_to_split).flatten()
        left_subtree_indices = np.argwhere(column_of_feature_to_split <= threshold_to_split).flatten()

        right_subtree = self.create_decision_tree(X_train[right_subtree_indices, :], y_train[right_subtree_indices],
                                                  tree_depth)
        left_subtree = self.create_decision_tree(X_train[left_subtree_indices, :], y_train[left_subtree_indices],
                                                 tree_depth)

        # Create the new node and return it
        new_node = TreeNode(threshold_to_split, feature_to_split, right_subtree, left_subtree)
        
        return new_node

    '''
    Traverses through the decision tree recursively
    
    PARAMETERS:
        - node_of_tree: the node that the tree is currently on (at first it'll be the root)
        - x_test_value: a sample from the X_test set (i.e., a single record)
        
    RETURNS:
        - leaf_node_value: if the node is a leaf, return its value
        - otherwise, recursively traverse through the decision tree
    '''
    def go_through_tree(self, node_of_tree, x_test_value):
        # If node is a leaf, then return the value
        if node_of_tree.check_if_leaf_node():
            leaf_node_value = node_of_tree.node_value
            return leaf_node_value

        # If the feature you're dividing the tree on is greater than the threshold,
        # traverse down the right side of the tree. Else, if the feature is less than or
        # equal to the threshold, traverse down the left side of the tree.
        if x_test_value[node_of_tree.feature_divided_with] > node_of_tree.threshold_divided_with:
            return self.go_through_tree(node_of_tree.right_subtree, x_test_value)
        elif x_test_value[node_of_tree.feature_divided_with] <= node_of_tree.threshold_divided_with:
            return self.go_through_tree(node_of_tree.left_subtree, x_test_value)

    '''
    Train the model for Decision Trees—calls another method to start growing the tree

    PARAMETERS:
        - X_train: X training set (features)
        - y_train: y training set (labels)

    RETURNS:
        N/A
    '''
    def train_model(self, X_train, y_train):
        self.tree_root = self.create_decision_tree(X_train, y_train)

    '''
    Predicts the class—here, it's the race for different DnD characters

    PARAMETERS: 
        - X_test: part of the dataset used for testing (features)

    RETURNS:
        - class_preds: list of predictions of DnD race 
    '''
    def predict_class(self, X_test):
        class_preds = []
        for x_test_value in X_test:
            class_preds.append(self.go_through_tree(self.tree_root, x_test_value))
        np.array(class_preds)

        return class_preds
