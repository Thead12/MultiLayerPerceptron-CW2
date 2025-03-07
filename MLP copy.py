import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

def linear_derivative(x):
    return 1

class MultiLayerPerceptron():
    def __init__(self, input_size, hidden_size, output_size, activation_function, activation_derivative):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.losses = []

        # Initialise weights and biases
        self.input_weights = np.random.rand(input_size, hidden_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.random.rand(output_size)

    # MSE loss function 
    def loss(self, y):
        return np.mean(np.square(y - self.output))

    def forward(self, x):
        self.hidden_layer_input = np.dot(x, self.input_weights) + self.hidden_bias
        self.hidden_layer_output = self.activation_function(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        self.output = self.output_layer_input  # Linear activation in output
        return self.output

    def backprop(self, x, y, learning_rate):
        # calculate gradients

        # output layer error
        dL_doutput = self.output - y
        # no delta as activvation function is linear

        # hidden layer error
        doutput_dhidden = dL_doutput.dot(self.output_weights.T)

        delta_hidden = doutput_dhidden * relu_derivative(self.hidden_layer_output)
        
        print("Output error:", np.max(dL_doutput))
        print("Hidden error:", np.max(doutput_dhidden))

        # update 
        self.output_weights -= self.hidden_layer_output.T.dot(dL_doutput) * learning_rate
        self.output_bias -= np.sum(dL_doutput, axis=0) * learning_rate
        self.input_weights -= x.T.dot(delta_hidden) * learning_rate
        self.hidden_bias -= np.sum(delta_hidden, axis=0) * learning_rate

    def train(self, x, y, epochs, learning_rate):
        for _ in range(epochs):
            self.forward(x)
            self.backprop(x, y, learning_rate)
            self.losses.append(self.loss(y))

    def predict(self, x):
        return self.forward(x)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show(block=False)

'''
# Example usage
mlp = MultiLayerPerceptron(2, 2, 1, sigmoid)
x = np.array([[1, 0]])
y = np.array([[1]])

mlp.train(x, y, 1000, 0.1)
print(mlp.predict(x))

mlp.plot_loss()
'''
