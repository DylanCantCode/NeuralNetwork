import numpy as np
import csv

X = []
Y = []

with open("wheat-seeds.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    Z = []
    for row in csv_reader:
        Z.append(row)

print(Z)
np.random.shuffle(Z)
for row in Z:
    X.append(row[0:-1])
    Y.append([row[-1]])
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)

X_train = X[0:50]
Y_train = Y[0:50]
X_test = X[50:]
Y_test = Y[50:]


class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
    def derivative(self, outputs):
        dev = lambda x : 1 if x > 0 else 0
        return np.array([[dev(x) for x in xs] for xs in outputs])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

class Layer_Dense_With_Activation(Layer_Dense):
    def __init__(self, n_inputs, n_neurons, activation_func):
        super(). __init__(n_inputs, n_neurons)
        self.activation_func = activation_func
    def forward(self, inputs):
        super().forward(inputs)
        self.activation_func.forward(self.outputs)
        self.outputs = self.activation_func.outputs
    def back_with_expecteds(self, expecteds):
        self.errors = (expecteds - self.outputs) * self.activation_func.derivative(self.outputs)
    def back_with_errors(self, errors, weights):
        self.errors = np.dot(errors, weights.T) * self.activation_func.derivative(self.outputs)
    def update_weights(self, lrate):
        update = np.dot(self.inputs.T, self.errors)
        update *= lrate / len(self.errors)
        self.weights += update
    def update_biases(self, lrate):
        update = [sum(xs) / len(xs) * lrate for xs in self.errors.T]
        self.biases += update

class Network:
    def __init__(self, *layers):
        self.layers = layers
    def forward(self, inputs):
        self.outputs = inputs
        for layer in self.layers:
            layer.forward(self.outputs)
            self.outputs = layer.outputs
    def back(self, expecteds):
        last_layer = True
        for layer in reversed(self.layers):
            if last_layer:
                layer.back_with_expecteds(expecteds)
                errors = layer.errors
                weights = layer.weights
                last_layer = False
            else:
                layer.back_with_errors(errors, weights)
                errors = layer.errors
                weights = layer.weights
    def update(self, lrate = 0.1):
        for layer in self.layers:
            layer.update_weights(lrate)
            layer.update_biases(lrate)
    def train(self, X, Y, n_epoch, lrate_ = 0.1, batch_size = -1):
        if batch_size = -1:
            batch_size = len(X)
        for i in range(0, n_epoch):
            data_index = 0
            while data_index <= len(X):
                data_last = min(data_index + batch_size, len(X))
                self.forward(X[data_index:data_last])
                self.back(Y[data_index:data_last])
                self.update(lrate = lrate_)
                data_index = data_last

    def predict(self, X):
        network.forward(X)
        return network.outputs



func1 = Activation_ReLU()
func2 = Activation_ReLU()
layer1 = Layer_Dense_With_Activation(4, 5, func1)
layer2 = Layer_Dense_With_Activation(5, 2, func2)

network = Network(layer1, layer2)
"""
network.forward(X)
print(network.outputs)
network.train(X, Y, 20)
print(network.outputs)
"""
