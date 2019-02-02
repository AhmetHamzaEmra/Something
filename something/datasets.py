"""
Example datasets for the testing!

"""

from something.tensor import Tensor
import numpy as np 

def get_xor():

    inputs = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    targets = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])

    return inputs, targets
    