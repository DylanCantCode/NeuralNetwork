import numpy as np
import csv
import json

X = []
Y = []
#Variables
with open("wheat-seeds.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    Z = []
    for row in csv_reader:
        Z.append(row)

def normalise_row(row):
    ys = [0.0, 0.0, 0.0]
    ys[int(row[-1]) - 1] = 1.0
    xs = row[0:-1]
    xs = [float(xs[i]) / col_max[i] for i in range(0,len(xs))]
    return xs + ys
np.random.shuffle(Z)
Z = np.array(Z)
col_max = [max(map(float,col_i)) for col_i in Z.T]
for row in Z:
    z = normalise_row(row)
    X.append(z[0:-3])
    Y.append(z[-3:])
X = np.array(X)
Y = np.array(Y)

X_train = X[0:150]
Y_train = Y[0:150]
X_test = X[150:]
Y_test = Y[150:]

#Stuff
class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
    def derivative(self, outputs):
        dev = lambda x : 1 if x > 0 else 0
        return np.array([[dev(x) for x in xs] for xs in outputs])
    def getjson(self):
        return "ReLU"

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
    def getjson(self):
        data = {"weights" : self.weights.tolist(),
                "biases" : self.biases.tolist(),
                "n_inputs" : self.n_inputs,
                "n_neurons" : self.n_neurons
        }
        return data

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
    def getjson(self):
        data = super().getjson()
        data["activation_func"] = self.activation_func.getjson()
        return data

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
        if batch_size == -1:
            batch_size = len(X)
        for i in range(0, n_epoch):
            data_index = 0
            while data_index < len(X):
                data_last = min(data_index + batch_size, len(X))
                self.forward(X[data_index:data_last])
                self.back(Y[data_index:data_last])
                self.update(lrate = lrate_)
                data_index = data_last

    def predict(self, X, classification_func):
        self.forward(X)
        return map(classification_func, network.outputs)
    def save(self, file_name):
        data = {"layers" : [],
                "layer_count" : len(self.layers)}
        for layer in self.layers:
            data["layers"].append(layer.getjson())
        with open("./savednets/{}.json".format(file_name), "w") as write_file:
            json.dump(data, write_file)
    def load(self, file_name):
        with open("./savednets/{}.json".format(file_name), "r") as read_file:
            data = json.loads(read_file.read())
        self.layers = []
        for layer in data["layers"]:
            func = Activation_ReLU()
            new_layer = Layer_Dense_With_Activation(layer["n_inputs"], layer["n_neurons"], func)
            new_layer.weights = np.array(layer["weights"])
            new_layer.biases = np.array(layer["biases"])
            self.layers.append(new_layer)



func1 = Activation_ReLU()
func2 = Activation_ReLU()
layer1 = Layer_Dense_With_Activation(7, 8, func1)
layer2 = Layer_Dense_With_Activation(8, 3, func2)
classification_func = lambda x: [[1.0, 0.0, 0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]][np.argmax(x)]

network = Network()
network.load("test")
network.train(X_train, Y_train, 1000, batch_size = 10)
results = network.predict(X_test, classification_func)
print(network.outputs)
success_total = 0
for result, answer in zip(results, Y_test):
    if (answer == result).all():
        success_total += 1
print("successes:{}".format(success_total))
print("total:{}".format(len(X_test)))
network.save("test")
