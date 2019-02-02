# Something 

## Machine Learning Module 



![](img/something.jpg)



* Machine Learning and Deep learning algorithms written with python and numpy.
* Perfect for every environment including Raspberry Pi


![](https://github.com/AhmetHamzaEmra/Something/blob/master/img/Loss.jpg)


## Example 

```(python)
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
```

## Stay tuned! Nothing is also coming :D 

### References 

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) 

[Coursera Neural Networks and Deep Learning ](https://www.coursera.org/specializations/deep-learning)



