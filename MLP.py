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

def sigmoidal_anneal(start, end, max_epochs, current_epoch):
    a = 1 + (np.exp(10-((20*current_epoch)/max_epochs)))
    return end + (start-end)*(1-1/a)

class MultiLayerPerceptron():
    def __init__(self, layer_sizes=[1, 10, 1]):
        
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        self.output_activation_function = linear
        self.output_activation_derivative = linear_derivative
        self.hidden_activation_function = relu
        self.hidden_activation_derivative = relu_derivative

        self.losses = []    
        self.learning_rate = 0.001      
        self.momentum_rate = 0.5
        
        self.print_epochs = 100
        self.bold_driver_check = 1000
        self.bold_driver_thresholds = [0.04, -0.001]
        self.bold_driver_annealing_split = 0.5 # refers to fraction of epochs which use bold driver
        
        self.annealing_function = sigmoidal_anneal

        self.weight_decay_coefficient = 0.01

        self.gradient_clipping = 10

        self.weights = []
        self.bias = []  
        self.momentums = []

        # initialise hidden layers
        for i in range(1, self.num_layers-1, 1):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1])) # He initialisation
            self.bias.append(np.zeros(layer_sizes[i]))
            self.momentums.append(np.zeros([layer_sizes[i-1], layer_sizes[i]]))
        
        # initialise final weights between hidden and output layers
        self.weights.append(np.random.randn(layer_sizes[-2], layer_sizes[-1]) * np.sqrt(1 / layer_sizes[-2])) # Xavier Initialization
        self.bias.append(np.zeros(layer_sizes[-1]))
        self.momentums.append(np.zeros([layer_sizes[-2], layer_sizes[-1]]))


    def loss(self, y):
        return np.mean(np.square(y - self.prediction))
    
    def sum_weights(self):
        total_sum = 0
        for weight_matrix in self.weights:
            total_sum += np.sum(weight_matrix**2)
        return total_sum

    def forward(self, x):
        inputs = [x] # container to store pre activations for backprop calculation
        # input layer
        input = np.dot(x, self.weights[0]) + self.bias[0]
        inputs.append(input)    # store pre activation
        output = self.hidden_activation_function(input)
        # hidden layers
        for i in range(1, self.num_layers - 2):
            input = np.dot(output, self.weights[i]) + self.bias[i]
            inputs.append(input)    # store pre activation
            output = self.hidden_activation_function(input)
        # output layers
        output_layer_input = np.dot(output, self.weights[-1]) + self.bias[-1]
        inputs.append(output_layer_input)        
        self.prediction = self.output_activation_function(output_layer_input)

        self.pre_activations = inputs
        return self.prediction

    def backprop(self, x, y, learning_rate=0.001, momentum_rate=0.9):

        self.previous_weights = [w.copy() for w in self.weights]
        self.previous_biases = [b.copy() for b in self.bias]

        # output layer
        #delta = (dL/dy)*g'(z) = (y_hat - y)g'(z)
        delta = 2*(self.prediction - y)*self.output_activation_derivative(self.pre_activations[-1])
        #print(f"Output layer delta shape", delta.shape)

        # update weights and biases for output layer
        # W = W - learning_rate*a(h-1).T*delta
        activation = self.hidden_activation_function(self.pre_activations[-2])

        delta_w = learning_rate * np.dot(activation.T, delta)
        
        self.weights[-1] -= (delta_w + momentum_rate*self.momentums[-1])

        self.momentums[-1] = delta_w

        # b  = b-learning_rate*delta
        self.bias[-1] -= learning_rate * np.sum(delta)    

        # hidden layers
        for i in range(-2, -self.num_layers, -1):
            delta = (np.dot(delta, self.weights[i+1].T))*self.hidden_activation_derivative(self.pre_activations[i])
            
            #print(f"Layer {self.num_layers + i + 1}: weights shape", self.weights[i].shape)
            #print(f"Layer {self.num_layers + i + 1}: delta shape", delta.shape)

            #print("i:", i)
            #print("i-1:", i-1)
            #for y in range(len(self.pre_activations)):
                #print(f"Activations layer: {y}", self.pre_activations[y].shape)

            activation = self.hidden_activation_function(self.pre_activations[i-1])

            delta_w = learning_rate * np.dot(activation.T, delta)
            #print(f"Layer {self.num_layers + i + 1}: activatioins shape", activation.shape)

            gradient = delta_w + momentum_rate * self.momentums[i]
            np.clip(gradient,-self.gradient_clipping,self.gradient_clipping)
            self.weights[i] -= gradient

            self.momentums[i] = delta_w

            self.bias[i] -= learning_rate * np.sum(delta)

        # input layer
        delta = (np.dot(delta, self.weights[0].T))*self.hidden_activation_derivative(self.pre_activations[1])
        delta_w = learning_rate * np.dot(x.T, delta)
        self.weights[0] -= (delta_w + momentum_rate * self.momentums[0])
        self.momentums[0] = delta_w
        self.bias[0] -= learning_rate * np.sum(delta)

    def undo_backprop(self):
        # restore weights and biases to their previous values
        self.weights = [w.copy() for w in self.previous_weights]
        self.bias = [b.copy() for b in self.previous_biases]

    def train(self, x, y, epochs, learning_rate, momentum_rate=0.9, weight_decay_coefficient=0.01):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.weight_decay_coefficient = weight_decay_coefficient

        for epoch in range(epochs):
            self.forward(x)
            self.backprop(x, y, self.learning_rate, momentum_rate)

            weight_decay = 1/(epochs+1) * self.sum_weights()
            loss_value = self.loss(y) + weight_decay

            self.losses.append(loss_value)

            # adaptive learning rates
            if epoch <= round(epochs*self.bold_driver_annealing_split):
                # bold driver
                if (epoch + 1) % self.bold_driver_check == 0:
                    self.bold_driver(learning_rate)
            else:
                #apply annealing
                start = self.learning_rate
                end = learning_rate*0.1
                num_epochs = epochs*self.bold_driver_annealing_split
                self.learning_rate = self.annealing_function(start, end, num_epochs, epoch)

            # print loss every ___ epochs
            if epoch % self.print_epochs == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.5f}")

        print(min(self.losses))

    def bold_driver(self, learning_rate):
        delta_loss = (self.losses[-2] - self.losses[-1])/2  # Calculate change in loss

        if delta_loss >= self.bold_driver_thresholds[0]:  # loss decrease
            print("Previous and new loss: ", self.losses[-2], self.losses[-1])
            print("Delta_loss:", delta_loss)
            print("Previous learning rate: ", self.learning_rate)
            if self.learning_rate <= learning_rate*50: self.learning_rate *= 1.05 # increase learning rate
            print("Learning rate increased: ", self.learning_rate)
        elif delta_loss <= self.bold_driver_thresholds[1]:  # loss increase
            print("Previous and new loss: ", self.losses[-2], self.losses[-1])
            print("Delta_loss:", delta_loss)
            self.undo_backprop()  
            print("Previous learning rate: ", self.learning_rate)
            if self.learning_rate >= learning_rate/10: self.learning_rate *= 0.7 # decrease learning rate
            print("Learning rate decreased: ", self.learning_rate)



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


