import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0,-1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

class Network:
    def __init__(self, activation_func, *layers):
        self.layers = layers
    def forward(self, inputs):
        self.outputs = inputs
        for layer in self.layers:
            layer.forward(self.outputs)
            self.outputs = layer.outputs

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

network = Network(layer1, layer2)
network.forward(X)
print(network.outputs)