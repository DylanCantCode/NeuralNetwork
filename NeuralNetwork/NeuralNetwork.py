import numpy as np
np.random.seed(0)

X = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0,-1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])

Y = np.array([[1,0],
              [0,1],
              [1,0]])

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
    def derivative(self, outputs):
        dev = lambda x : 1 if x > 0 else 0
        return np.array([[dev(x) for x in xs] for xs in outputs])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
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




func1 = Activation_ReLU()
func2 = Activation_ReLU()
layer1 = Layer_Dense_With_Activation(4, 5, func1)
layer2 = Layer_Dense_With_Activation(5, 2, func2)

network = Network(layer1, layer2)
for i in range(0,10):
    network.forward(X)
    network.back(Y)
    print("outputs:", network.outputs)
    print("layer0 errors:", network.layers[0].errors)
    print("layer1 errors:", network.layers[1].errors)
    print("layer0 weights:", network.layers[0].weights)
    print("layer1 weights:", network.layers[1].weights)
    print("layer0 biases:", network.layers[0].biases)
    print("layer1 biases:", network.layers[1].biases)
    network.update()
