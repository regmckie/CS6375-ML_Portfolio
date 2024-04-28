# FILENAME: Assignment1_Part2.py
# DUE DATE: 3/20/2024
# AUTHOR:   Reg Gonzalez
# EMAIL:    rdg170330@utdallas.edu (school) or regmckie@gmail.com (personal)
# COURSE:   CS 6375.001, Spring 2024
# VERSION:  1.0
#
# DESCRIPTION:
# "In this part, you will code a neural network (NN) having at least one hidden
# layers, besides the input and output layers. You are required to pre-process the
# data and then run the processed data through your neural net."

import numpy as np
import pandas as pd

class NeuralNetwork:

    X_training_set, Y_training_set, X_testing_set, Y_testing_set = None, None, None, None
    W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer = None, None, None, None
    neural_net_preds = None
    accuracy = None

    '''
    Performs some preprocessing on the MNIST dataset; creates training
    and testing sets from the dataset.

    PARAMETERS: N/A

    RETURNS:
    Training and testing sets
    '''
    def preprocess_data(self):
        # Read in the data
        dataset = pd.read_csv('MNIST_dataset.csv')

        # Transform the data into a numpy array for further processing
        dataset = np.array(dataset)

        # Split the data into training and testing sets
        rows, columns = dataset.shape
        np.random.shuffle(dataset)

        testing_set = dataset[0:1000].T  # Transpose so that each COLUMN is an example, not each ROW
        X_testing_set = testing_set[1:columns]
        X_testing_set = X_testing_set / 255
        Y_testing_set = testing_set[0]

        training_set = dataset[1000:rows].T # Transpose so that each COLUMN is an example, not each ROW
        X_training_set = training_set[1:columns]
        X_training_set = X_training_set / 255
        Y_training_set = training_set[0]

        return X_training_set, Y_training_set, X_testing_set, Y_testing_set


    '''
    Randomly initializes the parameters of the neural network (i.e., the weights
    and biases).
    
    PARAMETERS:
    no_of_neurons - Represents the number of neurons for the hidden layer to have
    
    RETURNS:
    Initial values for the weights and biases of the network (hlayer means 'hidden layer' 
    and olayer means 'output layer')
    '''
    def initialize_parameters(self, no_of_neurons):
        # Structure of the neural network:
        # 1 input layer: 784 nodes
        # 1 hidden layer: # of nodes depends on parameter
        # 1 output layer: 10 nodes

        # Randomly initialize weights and biases
        W_to_hlayer = np.random.rand(no_of_neurons, 784) - 0.5
        b_to_hlayer = np.random.rand(no_of_neurons, 1) - 0.5
        W_to_olayer = np.random.rand(10, no_of_neurons) - 0.5
        b_to_olayer = np.random.rand(10, 1) - 0.5

        return W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer


    '''
    Performs the ReLu activation function. This is used as the activation function for the hidden layer.
    
    PARAMETERS:
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    
    RETURNS:
    The result of the ReLu activation function (i.e., value of net_hidden if it's > 0 
    or 0 if it's <= 0)
    '''
    def ReLu_activation(self, net_hidden):
        return np.maximum(0, net_hidden)


    '''
    Performs the sigmoid activation function. This is used as the activation function for the hidden layer.
    
    PARAMETERS:
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    
    RETURNS:
    The result of the sigmoid activation function (i.e., 1 / (1 + e^(-x)))
    '''
    def sigmoid_activation(self, net_hidden):
        return 1 / (1 + np.exp(-net_hidden))


    '''
    Performs the tanh activation function. This is used as the activation function for the hidden layer.
    
    PARAMETERS:
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    
    RETURNS:
    The result of the tanh activation function (i.e., (e^x - e^(-x)) / (e^x + e^(-x)))
    '''
    def tanh_activation(self, net_hidden):
        return np.tanh(net_hidden)


    '''
    Performs the derivative of the ReLu activation function. This is used to calculate the derivative
    of net_hidden.
    
    PARAMETERS:
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    
    RETURNS:
    The result of the derivative of ReLu (i.e., 1 if value of net_hidden > 0 and 0 otherwise)
    '''
    def ReLu_derivative(self, net_hidden):
        return net_hidden > 0


    '''
    Performs the derivative of the Sigmoid activation function. This is used to calculate the derivative
    of net_hidden.
    
    PARAMETERS:
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    
    RETURNS:
    The result of the derivative of ReLu (i.e., f(x)(1 - f(x), where f(x) is the sigmoid function)
    '''
    def sigmoid_derivative(self, net_hidden):
        sigmoid_result = self.sigmoid_activation(net_hidden)
        return sigmoid_result * (1 - sigmoid_result)


    '''
    Performs the derivative of the tanh activation function. This is used to calculate the derivative
    of net_hidden.
    
    PARAMETERS:
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    
    RETURNS:
    The result of the derivative of ReLu (i.e., 1 - f(x)^2, where f(x) is the tanh function)
    '''
    def tanh_derivative(self, net_hidden):
        return 1 - np.square(np.tanh(net_hidden))


    '''
    Performs the softmax activation function. This is used as the activation function for the output layer.
    
    PARAMETERS: 
    net_outer - Represents the input to the output layer (i.e., output of hidden layer * Weights + bias)
    
    RETURNS:
    The result of the softmax activation function (i.e., value between 0 and 1)
    '''
    def softmax_activation(self, net_outer):
        numerator = np.exp(net_outer)
        denominator = sum(np.exp(net_outer))
        softmax_result = numerator / denominator

        return softmax_result


    '''
    Performs one-hot encoding. This is done to help calculate the derivative of net_outer.
    We want to calculate that for the purposes of backward propagation.
    
    PARAMETERS: 
    Y_training_set - The dataset we're training with (labels)
    
    RETURNS:
    The one-hot encoded matrix
    '''
    def one_hot_encoding(self, Y_training_set):
        no_of_examples = Y_training_set.size
        no_of_output_classes = Y_training_set.max() + 1

        # Create matrix for one-hot encoding
        Y_one_hot_matrix = np.zeros((no_of_examples, no_of_output_classes))

        # For each row, go to the column specified by the label in Y_training_set
        # and set it to 1—this is one-hot encoding
        Y_one_hot_matrix[np.arange(Y_training_set.size), Y_training_set] = 1

        # Transpose the matrix because right now each ROW is an example,
        # we want each COLUMN to be an example
        Y_one_hot_matrix = Y_one_hot_matrix.T

        return Y_one_hot_matrix


    '''
    Performs the forward propagation for the neural network. Generally speaking, the input to a node is:
    input of previous layer's nodes * Weights + bias. We do this for both the hidden and output layers. 
    After we calculate that, we pass the results to an activaton function. For the hidden layer, the activation
    is either ReLu, sigmoid, or tanh. For the output layer, the activation function is softmax.
    
    PARAMETERS:
    X_set - The dataset we're using (this can either be the training set or testing set)
    W_to_hlayer - Weights vector to the hidden layer
    b_to_hlayer - Biases for the hidden layer
    W_to_olayer - Weights vector to the output layer
    b_to_olayer - Biases for the output layer
    activation_fn_choice - Choice of activation function (1 for ReLu, 2 for sigmoid, or 3 for tanh)
    
    RETURNS:
    The values calculated (i.e., (input of previous layer's nodes * Weights + bias) & the results
    when they're passed into the activation functions)
    '''
    def forward_propagation(self, X_set, W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer, activation_fn_choice):
        net_hidden = W_to_hlayer.dot(X_set) + b_to_hlayer

        if activation_fn_choice == 1:
            o_hidden = self.ReLu_activation(net_hidden)
        elif activation_fn_choice == 2:
            o_hidden = self.sigmoid_activation(net_hidden)
        elif activation_fn_choice == 3:
            o_hidden = self.tanh_activation(net_hidden)

        net_outer = W_to_olayer.dot(o_hidden) + b_to_olayer
        o_outer = self.softmax_activation(net_outer)

        return net_hidden, o_hidden, net_outer, o_outer


    '''
    Performs the backward propagation algorithm for the neural network. Here we calculate the derivatives,
    which will be used the update the weights and biases accordingly. 
        
    PARAMETERS:
    X_training_set - The dataset we're training with (features)
    Y_training_set - The dataset we're training with (labels)
    W_to_olayer - Weights vector to the output layer
    net_hidden - Represents the input to the hidden layer (i.e., input X * Weights + bias)
    o_hidden - Represents the output of the hidden layer (i.e., result of net_hidden after going through the activation function)
    o_outer - Represents the output of the output layer (i.e., result of net_outer after going through the activation function)
    activation_fn_choice - Choice of activation function (1 for ReLu, 2 for sigmoid, or 3 for tanh)
    
    RETURNS:
    The derivatives needed to update the weights and biases
    '''
    def backward_propagation(self, X_training_set, Y_training_set, W_to_olayer, net_hidden, o_hidden, o_outer, activation_fn_choice):
        no_of_examples = Y_training_set.size
        Y_one_hot_matrix = self.one_hot_encoding(Y_training_set)

        # Calculate the derivatives for back propagation.
        # The derivatives represent the change in values that we will need
        # to update the weights and biases later on
        deri_net_outer = o_outer - Y_one_hot_matrix
        deri_W_to_olayer = 1 / no_of_examples * deri_net_outer.dot(o_hidden.T)
        deri_b_to_olayer = 1 / no_of_examples * np.sum(deri_net_outer)

        if activation_fn_choice == 1:
            deri_net_hidden = W_to_olayer.T.dot(deri_net_outer) * self.ReLu_derivative(net_hidden)
        elif activation_fn_choice == 2:
            deri_net_hidden = W_to_olayer.T.dot(deri_net_outer) * self.sigmoid_derivative(net_hidden)
        elif activation_fn_choice == 3:
            deri_net_hidden = W_to_olayer.T.dot(deri_net_outer) * self.tanh_derivative(net_hidden)

        deri_W_to_hlayer = 1 / no_of_examples * deri_net_hidden.dot(X_training_set.T)
        deri_b_to_hlayer = 1 / no_of_examples * np.sum(deri_net_hidden)

        return deri_W_to_hlayer, deri_b_to_hlayer, deri_W_to_olayer, deri_b_to_olayer


    '''
    Updates the weights and biases. Because we're using gradient descent with momentum, we're not just 
    going to use the derivatives (calculated in the backward_propagation() function). We're going to use the velocity.
    
    PARAMETERS:
    learning_rate - Represents the learning rate for the neural network—this is arbitrarily set by the programmer
    W_to_hlayer - Weights vector to the hidden layer
    b_to_hlayer - Biases to the hidden layer
    W_to_olayer - Weights vector to the output layer
    b_to_olater - Biases to the output layer
    vel_W_to_hlayer - Velocity of W_to_hlayer calculated for gradient descent with momentum 
    vel_b_to_hlayer - Velocity of b_to_hlayer calculated for gradient descent with momentum 
    vel_W_to_olayer - Velocity of W_to_olayer calculated for gradient descent with momentum 
    vel_b_to_olayer - Velocity of b_to_olayer calculated for gradient descent with momentum 
    
    RETURNS:
    The updated weights and biases using gradient descent with momentum
    '''
    def update_parameters(self, learning_rate, W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer,
                          vel_W_to_hlayer, vel_b_to_hlayer, vel_W_to_olayer, vel_b_to_olayer):
            W_to_hlayer = W_to_hlayer - learning_rate * vel_W_to_hlayer
            b_to_hlayer = b_to_hlayer - learning_rate * vel_b_to_hlayer
            W_to_olayer = W_to_olayer - learning_rate * vel_W_to_olayer
            b_to_olayer = b_to_olayer - learning_rate * vel_b_to_olayer

            return W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer


    '''
    Performs gradient descent with momentum by combining the forward propagation, backward propagation, 
    and updates to the velocities to finally get the updated weights and biases. 
    
    PARAMETERS:
    beta - Represents a constant to calculate the velocities for gradient descent with momentum—this is arbitrarily set by the programmer
    iterations - The number of iterations we want to train the neural network—this is arbitrarily set by the programmer
    learning_rate - Represents the learning rate for the neural network—this is arbitrarily set by the programmer
    X_training_set - The dataset we're training with (features)
    Y_training_set - The dataset we're training with (labels)
    activation_fn_choice - Choice of activation function (1 for ReLu, 2 for sigmoid, or 3 for tanh)
    
    RETURNS:
    The updated weights and biases using gradient descent with momentum
    '''
    def gd_with_momentum(self, beta, iterations, learning_rate, X_training_set, Y_training_set, activation_fn_choice, no_of_neurons):
        # Get the iniital values for the weights and biases
        W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer = self.initialize_parameters(no_of_neurons)

        # Set the initial values of the velocity to 0 (this is for gradient descent w/ momentum)
        vel_W_to_hlayer, vel_b_to_hlayer, vel_W_to_olayer, vel_b_to_olayer = 0, 0, 0, 0

        # Perform the forward & backward propagation, update the velocities, and finally
        # update the weights and biases
        for counter in range(iterations):
            # Forward & backward propagation
            net_hidden, o_hidden, net_outer, o_outer = self.forward_propagation(X_training_set, W_to_hlayer,
            b_to_hlayer, W_to_olayer, b_to_olayer, activation_fn_choice)
            deri_W_to_hlayer, deri_b_to_hlayer, deri_W_to_olayer, deri_b_to_olayer = self.backward_propagation(X_training_set, Y_training_set,
            W_to_olayer, net_hidden, o_hidden, o_outer, activation_fn_choice)

            # Update the velocities
            vel_W_to_hlayer = beta * vel_W_to_hlayer + (1 - beta) * deri_W_to_hlayer
            vel_b_to_hlayer = beta * vel_b_to_hlayer + (1 - beta) * deri_b_to_hlayer
            vel_W_to_olayer = beta * vel_W_to_olayer + (1 - beta) * deri_W_to_olayer
            vel_b_to_olayer = beta * vel_b_to_olayer + (1 - beta) * deri_b_to_olayer

            # Update the weights and biases
            W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer = self.update_parameters(learning_rate, W_to_hlayer, b_to_hlayer,
            W_to_olayer, b_to_olayer, vel_W_to_hlayer, vel_b_to_hlayer, vel_W_to_olayer, vel_b_to_olayer)

            # Get the accuracy of the neural network now that we've built it
            if counter == iterations - 1:
                if activation_fn_choice == 1:
                    neural_net_predictions_ReLu = self.nn_predictions(o_outer)
                    print("Accuracy of model using training data (ReLu): ", self.evaluate_accuracy(Y_training_set, neural_net_predictions_ReLu))
                elif activation_fn_choice == 2:
                    neural_net_predictions_sigmoid = self.nn_predictions(o_outer)
                    print("Accuracy of model using training data (Sigmoid): ", self.evaluate_accuracy(Y_training_set, neural_net_predictions_sigmoid))
                elif activation_fn_choice == 3:
                    neural_net_predictions_tanh = self.nn_predictions(o_outer)
                    print("Accuracy of model using training data (Tanh): ", self.evaluate_accuracy(Y_training_set, neural_net_predictions_tanh))

        return W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer


    '''
    Gets the results from the output layer and takes those as predictions of the neural network
    
    PARAMETERS:
    X_testing_set - The dataset we're using to test the neural network
    W_to_hlayer - Weights vector to the hidden layer
    b_to_hlayer - Biases to the hidden layer
    W_to_olayer - Weights vector to the output layer
    b_to_olater - Biases to the output layer
    activation_fn_choice - Choice of activation function (1 for ReLu, 2 for sigmoid, or 3 for tanh)
    
    RETURNS:
    The predictions made by the neural network
    '''
    def test_NN(self, X_testing_set, W_to_hlayer, b_to_hlayer, W_to_olayer, b_to_olayer, activation_fn_choice):
        # Get the results of the output layer and get the predictions of the neural network
        _, _, _, o_outer = self.forward_propagation(X_testing_set, W_to_hlayer, b_to_hlayer,
        W_to_olayer, b_to_olayer, activation_fn_choice)
        neural_net_preds = self.nn_predictions(o_outer)
        print(neural_net_preds)
        return neural_net_preds


    '''
    Returns the predictions output by the neural network

    PARAMETERS:
    o_outer - Represents the output of the output layer (i.e., result of net_outer after going through the activation function)

    RETURNS:
    Predictions of the neural network 
    '''
    def nn_predictions(self, o_outer):
        return np.argmax(o_outer, 0)


    '''
    Calculates the accuracy by comparing the neural network's predictions to the actual labels

    PARAMETERS:
    Y_training_set - The dataset we're using (this can either be the training or testing set)
    neural_net_predictions - The predictions of the neural net output by the output layer (essentially, o_outer)

    RETURNS:
    The accuracy of our neural network
    '''
    def evaluate_accuracy(self, Y_set, neural_net_predictions):
        accuracy = np.sum(neural_net_predictions == Y_set) / Y_set.size

        return accuracy


# -----------------------------------------------------------------------------------------------------
# Build a neural network and train/test it!
testNetwork = NeuralNetwork()

# PREPROCESS:
# Get the training and testing sets.
testNetwork.X_training_set, testNetwork.Y_training_set, testNetwork.X_testing_set, testNetwork.Y_testing_set = testNetwork.preprocess_data()

# TRAIN:
# Train the network with gradient descent with momentum.
# 'activation_fn_choice' refers to the choice of the activation function to use for the neural network.
# The activation functions can either be ReLu, Sigmoid, or Tanh.

# TEST:
# Make predictions from the neural network and evaluate those predictions.

# Train and test the model using the ReLu activation function
activation_fn_choice = 1
testNetwork.W_to_hlayer, testNetwork.b_to_hlayer, testNetwork.W_to_olayer, testNetwork.b_to_olayer = testNetwork.gd_with_momentum(0.90, 500, 0.10,
testNetwork.X_training_set, testNetwork.Y_training_set, activation_fn_choice, 13)

testNetwork.neural_net_preds = testNetwork.test_NN(testNetwork.X_testing_set, testNetwork.W_to_hlayer, testNetwork.b_to_hlayer,
testNetwork.W_to_olayer, testNetwork.b_to_olayer, activation_fn_choice)
testNetwork.accuracy = testNetwork.evaluate_accuracy(testNetwork.neural_net_preds, testNetwork.Y_testing_set)
print("Accuracy of model using testing data (ReLu):", testNetwork.accuracy)

# Train and test the model using the Sigmoid activation function
activation_fn_choice = 2
testNetwork.W_to_hlayer, testNetwork.b_to_hlayer, testNetwork.W_to_olayer, testNetwork.b_to_olayer = testNetwork.gd_with_momentum(0.90, 500, 0.10,
testNetwork.X_training_set, testNetwork.Y_training_set, activation_fn_choice, 13)

testNetwork.neural_net_preds = testNetwork.test_NN(testNetwork.X_testing_set, testNetwork.W_to_hlayer, testNetwork.b_to_hlayer,
testNetwork.W_to_olayer, testNetwork.b_to_olayer, activation_fn_choice)
testNetwork.accuracy = testNetwork.evaluate_accuracy(testNetwork.neural_net_preds, testNetwork.Y_testing_set)
print("Accuracy of model using testing data (Sigmoid):", testNetwork.accuracy)

# Train and test the model using the Tanh activation function
activation_fn_choice = 3
testNetwork.W_to_hlayer, testNetwork.b_to_hlayer, testNetwork.W_to_olayer, testNetwork.b_to_olayer = testNetwork.gd_with_momentum(0.90, 500, 0.10,
testNetwork.X_training_set, testNetwork.Y_training_set, activation_fn_choice, 13)

testNetwork.neural_net_preds = testNetwork.test_NN(testNetwork.X_testing_set, testNetwork.W_to_hlayer, testNetwork.b_to_hlayer,
testNetwork.W_to_olayer, testNetwork.b_to_olayer, activation_fn_choice)
testNetwork.accuracy = testNetwork.evaluate_accuracy(testNetwork.neural_net_preds, testNetwork.Y_testing_set)
print("Accuracy of model using testing data (Tanh):", testNetwork.accuracy)

