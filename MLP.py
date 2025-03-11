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
    def __init__(self, num_layers=3, layer_sizes=[1, 10, 1],
                output_activation_function=linear, output_activation_derivative=linear_derivative,
                hidden_activation_function=relu, hidden_activation_derivative=relu_derivative):
        
        if num_layers != len(layer_sizes):
            raise Exception("Number of layers and number of layer sizes do not match")
        
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes

        self.output_activation_function = output_activation_function
        self.output_activation_derivative = output_activation_derivative
        self.hidden_activation_function = hidden_activation_function
        self.hidden_activation_derivative = hidden_activation_derivative

        self.losses = []          

        self.weights = []
        self.bias = [np.zeros(layer_sizes[0])]  

        # initialise hidden layers
        for i in range(1, num_layers-1, 1):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1])) # He initialisation
            self.bias.append(np.zeros(layer_sizes[i]))
        
        # initialise final weights between hidden and output layers
        self.weights.append(np.random.randn(layer_sizes[-2], layer_sizes[-1]) * np.sqrt(1 / layer_sizes[-2])) # Xavier Initialization
        self.bias.append(np.zeros(layer_sizes[-1]))


    def loss(self, y):
        return np.mean(np.square(y - self.prediction))

    def forward(self, x):
        # Forward propagation
        print("Input layer")
        print(f"x shape: {x.shape}, weights[0] shape: {self.weights[0].shape}, bias[0] shape: {self.bias[0].shape}")
        
        input = np.dot(x, self.weights[0]) + self.bias[0]
        print(f"After first layer: input shape: {input.shape}")
        
        output = self.hidden_activation_function(input)
        print(f"After first activation: output shape: {output.shape}")
    
        for i in range(1, self.num_layers - 1):
            print(f"Layer {i}: output: {output.shape}")
            print(f"Layer {i}: weight{i}: {self.weights[i].shape}")


            input = np.dot(output, self.weights[i]) + self.bias[i]
            print(f"Layer {i}: input shape: {input.shape}")
            
            output = self.hidden_activation_function(input)
            print(f"Layer {i}: output shape: {output.shape}")
    
        output_layer_input = np.dot(output, self.weights[-1]) + self.bias[-1]
        print(f"Output layer input shape: {output_layer_input.shape}")
        
        self.prediction = self.output_activation_function(output_layer_input)
        print(f"Prediction shape: {self.prediction.shape}")
    
        return self.prediction

    def backprop(self, x, y, learning_rate):
        """Backward propagation"""

        batch_size = x.shape[0]  # Number of samples

        dL_doutput = (self.output - y) / batch_size  # MSE derivative

        doutput_dhidden = dL_doutput.dot(self.hidden_to_output_weights.T)
        delta_hidden = doutput_dhidden * self.activation_derivative(self.hidden_layer_output)
        
        # Update weights and biases
        self.hidden_to_output_weights -= self.hidden_layer_output.T.dot(dL_doutput) * learning_rate
        self.output_bias -= np.sum(dL_doutput, axis=0) * learning_rate
        self.input_to_hidden_weights -= x.T.dot(delta_hidden) * learning_rate
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

    def print_architecture(self):
        print("Number of weight layers: ", len(self.weights))
        print("Number of bias layers: ", len(self.bias))

        print("Dimensions of weights")
        for weight in self.weights:
            print(weight.shape)

        print("Dimensions of biases")
        for layer in self.bias:
            print(layer.shape)



