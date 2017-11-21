import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def transpose_all(x):
    a = x.copy()
    a = a.T
    return x, a

def initialize_with_zeros(dim):
    w = np.random.randn(dim,1)*0.01
    b = 0
    return w, b
def propagate(w, b, X, Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)                                     # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    
    dw =    (1 / m) * np.dot(X, (A - Y).T)
    db =    (1 / m) * np.sum(A - Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False, print_every=100):
    
    
    costs = []
    
    for i in range(num_iterations):
        grads, cost =  propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - np.dot(learning_rate, dw)
        b = b -  learning_rate* db
        if i % 50 == 0:
            costs.append(cost)
        
        if print_cost and i % print_every == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    
    
    X, X_T = transpose_all(X)
    
    m = X_T.shape[1]
    w = w.reshape(X_T.shape[0], w.shape[1])
    
    Y_prediction = sigmoid(np.dot(w.T, X_T)+b)
    
    
    
    return Y_prediction.T


def model(X_train, Y_train, X_test, Y_test, num_iterations = 500, learning_rate = 0.5, print_cost = True,print_every=100):
    
    
    X_train, X_train_T = transpose_all(X_train)
    Y_train, Y_train_T = transpose_all(Y_train)
    X_test, X_test_T = transpose_all(X_test)
    Y_test, Y_test_T = transpose_all(Y_test)
    
    
    
    w, b = initialize_with_zeros(X_train_T.shape[0])

    parameters, grads, costs = optimize(w, b, X_train_T, Y_train_T, num_iterations, learning_rate, print_cost, print_every=print_every)
    
    w = parameters["w"]
    b = parameters["b"]
    
    #Y_prediction_test = predict(w, b, X_test)
    #Y_prediction_train = predict(w, b, X_train)


    #print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train_T)) * 100))
    #print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test_T)) * 100))

    
    d = {"costs": costs,
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

