import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  # a kNN classifier with L2 distance

  def __init__(self):
    pass

  def train(self, X, y):

    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1):
    
    dists = self.compute_distances(X)
    
    return self.predict_labels(dists, k=k)



  def compute_distances(self, X):
    # Compute Distace Vectorized version 
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    m = X.shape[0] 
    n = self.X_train.shape[0] 
    x2 = np.sum(X**2, axis=1).reshape((m, 1))
    y2 = np.sum(self.X_train**2, axis=1).reshape((1, n))
    xy = X.dot(self.X_train.T) 
    dists = np.sqrt(x2 + y2 - 2*xy)

    return dists

  def predict_labels(self, dists, k=1):
    
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):

      closest_y = []

      closest_y = self.y_train[np.argsort(dists[i])][:k]

      y_pred[i] = np.argmax(np.bincount(closest_y))


    return y_pred

