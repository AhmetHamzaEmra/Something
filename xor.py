import numpy as np

from something.train import train
from something.nn import NeuralNet
from something.layers import Linear, Tanh
from something.datasets import get_xor

# Load the dataset
inputs, targets = get_xor()

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)