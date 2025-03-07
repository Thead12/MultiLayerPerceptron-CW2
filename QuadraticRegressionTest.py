import numpy as np
import matplotlib.pyplot as plt
from MLP import *

X = np.linspace(-10, 10, 1000).reshape(-1, 1) # Input from -10 to 10
y = X**2 # quadratic function output

mlp = MultiLayerPerceptron(1, 100, 1, relu, relu_derivative)
mlp.train(X, y, 20000, 0.001)

print(mlp.hidden_bias.shape)
# Plot loss
plt.figure()
mlp.plot_loss()

# Plot predictions
y_hat = mlp.predict(X)
plt.figure()
plt.plot(X, y_hat, label='Predicted')
plt.plot(X, y, label='True')
plt.legend()
plt.show(block=False)

