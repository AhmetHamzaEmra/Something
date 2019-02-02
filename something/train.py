"""
Here's a function that can train a neural net
"""

from something.tensor import Tensor
from something.nn import NeuralNet
from something.loss import Loss, MSE
from something.optim import Optimizer, SGD
from something.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(),
          p_every: int = 50) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        if epoch % p_every == 0:
            print(epoch, epoch_loss)