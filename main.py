import numpy as np

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

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

        weight_key = 'W' + str(layer_idx)
        params_values[weight_key] = rnd_weights
        bias_key = 'b' + str(layer_idx)
        params_values[bias_key] = rnd_biases

    return params_values

print(init_layers(nn_architecture))
