# Something 

## Machine Learning Module 



![](img/something.jpg)



* Machine Learning and Deep learning algorithms written with python and numpy.

## Examples:

```python
# MNIST two layer example
from NNetwork import network_two_layer
net = network_two_layer()
net.init_two_layer_network(input_size=28*28, hidden_size=100, output_size=10)
net.train(x_train,y_train,verbose=True, num_iters=150)
net.predict(x_test)
```



![](https://github.com/AhmetHamzaEmra/Something/blob/master/img/Loss.jpg)

## Notebooks:

* [KNearestNeighbor](https://github.com/AhmetHamzaEmra/Something/blob/master/Examples/Knn%20example.ipynb)

## Todo:

* Softmax Classifier
* Linear SVM
* Neular Networks 
* Conv Nets


## Stay tuned! Nothing is also coming :D 

### References 

 [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) 

[Coursera Neural Networks and Deep Learning ](https://www.coursera.org/specializations/deep-learning)



