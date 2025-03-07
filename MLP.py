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

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

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

        self.input_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)  # He Initialization for ReLU
        self.output_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)  # He Initialization

        self.hidden_bias = np.zeros(hidden_size)
        self.output_bias = np.zeros(output_size)

    def loss(self, y):
        return np.mean(np.square(y - self.output))

    def forward(self, x):
        """Forward propagation"""
        self.hidden_layer_input = np.dot(x, self.input_weights) + self.hidden_bias
        self.hidden_layer_output = self.activation_function(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        self.output = self.output_layer_input  # Linear activation in output
        return self.output

    def backprop(self, x, y, learning_rate):
        """Backward propagation"""

        batch_size = x.shape[0]  # Number of samples

        dL_doutput = (self.output - y) / batch_size  # MSE derivative

        doutput_dhidden = dL_doutput.dot(self.output_weights.T)
        delta_hidden = doutput_dhidden * self.activation_derivative(self.hidden_layer_output)
        
        # Update weights and biases
        self.output_weights -= self.hidden_layer_output.T.dot(dL_doutput) * learning_rate
        self.output_bias -= np.sum(dL_doutput, axis=0) * learning_rate
        self.input_weights -= x.T.dot(delta_hidden) * learning_rate
        self.hidden_bias -= np.sum(delta_hidden, axis=0) * learning_rate

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(x)
            self.backprop(x, y, learning_rate)
            loss_value = self.loss(y)
            self.losses.append(loss_value)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.5f}")

    def predict(self, x):
        return self.forward(x)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show(block=False)

