from something import model, predict
import numpy as np
import pandas as pd

mnist = pd.read_csv('data/mnist.csv')
x = mnist.drop(['label'], axis=1).values
x = x/255
y = mnist.label.values
y = y.reshape((-1,1))
print(x.shape, y.shape)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y)
print(x.shape, y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y.toarray(), test_size=.3, random_state=32)


d = model(x_train, y_train, x_test, y_test, num_iterations = 50000, learning_rate = 0.001, print_cost = True, print_every=500)

from sklearn.metrics import accuracy_score

preds = predict(d['w'], d['b'], x_train)
y_pred = []
for i in preds:
    y_pred.append(np.argmax(i))
y_pred = np.array(y_pred)
y_true = []
for i in y_train:
    y_true.append(np.argmax(i))
y_true = np.array(y_true)
accuracy_score(y_true, y_pred)
