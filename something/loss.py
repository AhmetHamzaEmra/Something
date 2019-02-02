"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""

import numpy as np

from something.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError



class MSE(Loss):
    """
    MSE is mean squared error, although we're
    just going to do total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class CrossEntrophy(Loss):
    """
    Not Implemented Yet 
    

    Cross Entrophy loss for classification problems
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, acutal: Tensor) -> Tensor:
        raise NotImplementedError