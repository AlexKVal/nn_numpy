import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

nn_architecture = [
    {"input_dim": 2,  "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

def W_key(idx):
    return 'W' + str(idx)

def b_key(idx):
    return 'b' + str(idx)

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {} # cache

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1 # layer indexes are 1 based
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"] # num of neurons in a layer

        # generate random values [-1..+1] for Weights and Biases
        # to escape the "symmetry problem"
        # use small values (<1) b/c of the Sigmoid output function
        # e.g. for [input_dim: 2, output_dim: 4] layer
        # W1: [[w1, w2], [w1, w2], [w1, w2], [w1, w2]]
        # b1: [[b1], [b1], [b1], [b1]]
        rnd_weights = np.random.randn(layer_output_size, layer_input_size) * 0.1
        rnd_biases = np.random.randn(layer_output_size, 1) * 0.1

        params_values[W_key(layer_idx)] = rnd_weights
        params_values[b_key(layer_idx)] = rnd_biases

    return params_values


# Z - is a dot product of input values and weights in a neuron
# x1, x2 - input values, w1, w2 - weights
# Z = x1 * w1 + x2 * w2

# https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# derivative of the sigmoid function
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

# https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7
def relu(Z):
    return np.maximum(0, Z)

# derivative of ReLU https://stats.stackexchange.com/a/333400
def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ


# Z_curr is affine transformation
# A_prev is affine transformation result from the previous layer (activations)
# W_curr the weights of the current layer
# b_curr the bias of the current layer
# The function additinally returns the Z_curr
# for it later being used for backward propagation
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "relu":
        return relu(Z_curr), Z_curr
    elif activation is "sigmoid":
        return sigmoid(Z_curr), Z_curr
    else:
        raise Exception("Non-supported activation function")


def a_key(idx):
    return 'A' + str(idx)

def z_key(idx):
    return 'Z' + str(idx)

# X - input values
# params_values - working "cache" of Weights and Bias values
def full_forward_propagation(X, params_values, nn_architecture):
    memory = {} # cache for backward propagation
    A_curr = X # X vector is the activation for layer 0

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1 # layer indexes are 1 based

        # previous layer's output values are the next layer's input values
        A_prev = A_curr

        W_curr = params_values[W_key(layer_idx)]
        b_curr = params_values[b_key(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, layer["activation"])

        # save calculated forward results
        memory[a_key(idx)] = A_prev # idx contains previous layer index value
        memory[z_key(layer_idx)] = Z_curr

    return A_curr, memory # return final output A_curr along with cache for backpropagation


# Loss function: binary crossentropy
# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#id11
# m - number of classes (dog, cat, fish) in a classification (in binary it should be 2)
# Y - binary indicator (0 or 1) if class label is the correct classification
# Y_hat - predicted probability
def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1] # the second dimension of the Y_hat matrix
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost) # [[[0], [1], [2]]] => [0, 1, 2]

# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# https://youtu.be/tIeHLnjs5U8 Backpropagation calculus
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1] # input_dim, number of examples

    if activation is "relu":
        backward_activation = relu_backward
    elif activation is "sigmoid":
        backward_activation = sigmoid_backward
    else:
        raise Exception("Non-supported backward activation function")

    # calculate gradient descent with activation function derivative
    dZ_curr = backward_activation(dA_curr, Z_curr)
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m

    return dA_prev, dW_curr, db_curr


def dW_key(idx):
    return 'dW' + str(idx)

def db_key(idx):
    return 'db' + str(idx)

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {} # cost function derivatives calculated w.r.t. params_values
    m = Y.shape[1] # number of examples
    Y = Y.reshape(Y_hat.shape) # ensure the same shape of the prediction vector and labels vector

    # initialisation of the gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activation_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory[a_key(layer_idx_prev)]
        Z_curr = memory[z_key(layer_idx_curr)]
        W_curr = params_values[W_key(layer_idx_curr)]
        b_curr = params_values[b_key(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_curr
        )

        grads_values[dW_key(layer_idx_curr)] = dW_curr
        grads_values[db_key(layer_idx_curr)] = db_curr

    return grads_values

# updating parameter values using gradient descent
def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values[W_key(layer_idx)] -= learning_rate * grads_values[dW_key(layer_idx)]
        params_values[b_key(layer_idx)] -= learning_rate * grads_values[db_key(layer_idx)]

    return params_values


# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


# putting it all together
def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2) # different seed

    # lists storing the history of metrics calculated during the learning process
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        # forward
        Y_hat, memory_cache = full_forward_propagation(X, params_values, nn_architecture)

        # calculate and save metrics
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        # backward - calculate gradient
        grads_values = full_backward_propagation(Y_hat, Y, memory_cache, params_values, nn_architecture)
        # update model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        # feedback during the training process
        if i % 50 == 0:
            print("Iter: {:05} - cost: {:.5f} - accur: {:.5f}".format(i, cost, accuracy))

    return params_values


# ===============================================
# run it

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

# create a dataset
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# Training
params_values = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture, 10000, 0.01)

# Prediction
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, nn_architecture)

# Accuracy achieved on the test set
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))

# print("Final Weights and bias values:")
# print(params_values)
