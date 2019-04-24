import numpy as np

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def weight_key(idx):
    return 'W' + str(idx)

def bias_key(idx):
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

        params_values[weight_key(layer_idx)] = rnd_weights
        params_values[bias_key(layer_idx)] = rnd_biases

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
# A_prev is affine transformation result from the previous layer
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
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1 # layer indexes are 1 based

        # previous layer's output values are the next layer's input values
        A_prev = A_curr

        W_curr = params_values[weight_key(layer_idx)]
        b_curr = params_values[bias_key(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, layer["activation"])

        # save calculated forward results
        memory[a_key(idx)] = A_prev # idx contains previous layer index value
        memory[z_key(layer_idx)] = Z_curr

    return A_curr, memory # return final output A_curr along with cache for backpropagation

nn_params_values = init_layers(nn_architecture)
print("Starting Weights and bias values:")
print(nn_params_values)
print("\n\n")

X = [1, 0] # e.g.
output, memory_back = full_forward_propagation(X, nn_params_values, nn_architecture)
print("Output values:")
print(output)
print("\n\n")
print("memory values:")
print(memory_back)
print("\n\n")
