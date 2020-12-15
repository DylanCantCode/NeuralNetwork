import numpy as np
np.random.seed(0)

X = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0,-1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

class Layer_Dense_With_Activation(Layer_Dense):
    def __init__(self, n_inputs, n_neurons, activation_func):
        super(). __init__(n_inputs, n_neurons)
        self.activation_func = activation_func
    def forward(self, inputs):
        super().forward(inputs)
        self.activation_func.forward(self.outputs)
        self.outputs = self.activation_func.outputs

class Network:
    def __init__(self, *layers):
        self.layers = layers
    def forward(self, inputs):
        self.outputs = inputs
        for layer in self.layers:
            layer.forward(self.outputs)
            self.outputs = layer.outputs

func1 = Activation_ReLU()
func2 = Activation_ReLU()
layer1 = Layer_Dense_With_Activation(4, 5, func1)
layer2 = Layer_Dense_With_Activation(5, 2, func2)

network = Network(layer1, layer2)
network.forward(X)
print(network.outputs)