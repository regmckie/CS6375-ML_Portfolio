# FILENAME: main.py
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
# This file contains the code to train the various ML models using different
# ML algorithms. The algorithms we'll be using are logistic regression, K-nearest neighbors,
# and decision trees. We'll be creating the different models, evaluating them using
# metrics like accuracy, precision, recall, etc., and then determining which model
# is the best for a particular dataset. The dataset we'll be using is called 'dnd_stats.csv,'
# which contains information regarding the popular roleplaying game Dungeons and Dragons.
# The features describe a character's stats (e.g., height, weight, speed, strength, dexterity, etc.)
# and the goal is to use those features to predict a character's race (e.g., dragonborn,
# elf, tiefling, etc.).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from LogisticRegression import LogisticRegression
from kNN import KNearestNeighbors
from DecisionTrees import DecisionTree

# Read in the data and replace values in 'race' column
# to integers—this'll make it easier to classify
dnd_data = pd.read_csv("https://raw.githubusercontent.com/regmckie/CS6375-ML_Portfolio/main/dnd_stats.csv")
dnd_data.race = dnd_data.race.replace("tiefling", 8)
dnd_data.race = dnd_data.race.replace("human", 7)
dnd_data.race = dnd_data.race.replace("halfling", 6)
dnd_data.race = dnd_data.race.replace("half.orc", 5)
dnd_data.race = dnd_data.race.replace("half.elf", 4)
dnd_data.race = dnd_data.race.replace("gnome", 3)
dnd_data.race = dnd_data.race.replace("elf", 2)
dnd_data.race = dnd_data.race.replace("dwarf", 1)
dnd_data.race = dnd_data.race.replace("dragonborn", 0)

# Get the X and y data
X_features = dnd_data.drop(columns=["race"])
y_labels = dnd_data.race

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=1234)

# ------------------------------------------------------------------------------------
# TRAIN AND TEST THE MODELS/ALGORITHMS
# ------------------------------------------------------------------------------------

# Make a logistic regression model, train it, and see the results
logReg = LogisticRegression(learning_rate=0.001)
logReg.train_model(X_train, y_train)
class_preds = logReg.predict_class(X_test)

print("LOGISTIC REGRESSION MODEL:")
accuracy = accuracy_score(y_test, class_preds)
print("Accuracy: ", accuracy)
precision = precision_score(y_test, class_preds, average='macro', zero_division=1)
print("Precision: ", precision)
recall = recall_score(y_test, class_preds, average='macro')
print("Recall: ", recall)
f1 = f1_score(y_test, class_preds, average='macro')
print("F1 Score: ", f1)
cm = confusion_matrix(y_test, class_preds)
print("Confusion Matrix: ")
print(cm)
print("\n")

# Make a K-nearest neighbors model, train it, and see the results
knn_model = KNearestNeighbors(k_value=19)
knn_model.train_algorithm(X_train, y_train)
class_preds = knn_model.predict_class(X_test.values)

print("K-NEAREST NEIGHBORS MODEL:")
accuracy = accuracy_score(y_test, class_preds)
print("Accuracy: ", accuracy)
precision = precision_score(y_test, class_preds, average='macro', zero_division=1)
print("Precision: ", precision)
recall = recall_score(y_test, class_preds, average='macro')
print("Recall: ", recall)
f1 = f1_score(y_test, class_preds, average='macro')
print("F1 Score: ", f1)
cm = confusion_matrix(y_test, class_preds)
print("Confusion Matrix: ")
print(cm)
print("\n")

# Make a decision tree model, train it, and see the results
decisionTree = DecisionTree()
decisionTree.train_model(X_train.values, y_train.values)
class_preds = decisionTree.predict_class(X_test.values)

print("DECISION TREE MODEL:")
accuracy = accuracy_score(y_test, class_preds)
print("Accuracy: ", accuracy)
precision = precision_score(y_test, class_preds, average='macro', zero_division=1)
print("Precision: ", precision)
recall = recall_score(y_test, class_preds, average='macro')
print("Recall: ", recall)
f1 = f1_score(y_test, class_preds, average='macro')
print("F1 Score: ", f1)
cm = confusion_matrix(y_test, class_preds)
print("Confusion Matrix: ")
print(cm)
print("\n")

