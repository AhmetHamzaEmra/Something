import numpy as np

class network_two_layer(object):
    def __init__(self):
        self.definition = "two layer neular network"
    def init_two_layer_network(self, input_size,hidden_size, output_size):
        model = {}
        model['W1'] = np.random.randn(input_size,hidden_size)*0.001
        model['b1'] = np.zeros(hidden_size)
        model['W2'] = np.random.randn(hidden_size,output_size)*0.001
        model['b2'] = np.zeros(output_size)
        self.model = model
    def train_step(self, x_train, y_train):
        N, D = x_train.shape
        Z1 = x_train.dot(self.model['W1']) + self.model['b1']
        A1 = np.maximum(0,Z1)
        Z2 =  A1.dot(self.model['W2']) + self.model['b2']
        
        
        loss = None
        exp_scores = np.exp(Z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        
        corect_logprobs = -np.log(probs[range(N), y_train])
        loss = np.sum(corect_logprobs) / N
        
        grads={}
        dscores = probs.copy()
        dscores[range(N), list(y_train)] -= 1
        dscores /= N
        grads['W2'] = A1.T.dot(dscores) 
        grads['b2'] = np.sum(dscores, axis = 0)

        dh = dscores.dot( self.model['W2'].T)
        dh_ReLu = (A1 > 0) * dh
        grads['W1'] = x_train.T.dot(dh_ReLu) 
        grads['b1'] = np.sum(dh_ReLu, axis = 0)
        
        return loss, grads
    
    
    def train(self, x, y, learning_rate=1e-3, num_iters=100, verbose=False):
        loss_history = []
        for step in range(num_iters):
            loss, grads = self.train_step(x,y)
            loss_history.append(loss)
            self.model['W2'] -= learning_rate*(grads['W2'])
            self.model['b2'] -= learning_rate*(grads['b2'])
            self.model['W1'] -= learning_rate*(grads['W1'])
            self.model['b1'] -= learning_rate*(grads['b1'])
            if verbose and step % 10 == 0:
                print( 'iteration %d / %d: loss %f' % (step, num_iters, loss))
        self.loss_history = loss_history
        
    def predict(self, x):
        Z1 = x.dot(self.model['W1']) + self.model['b1']
        A1 = np.maximum(0,Z1)
        Z2 =  A1.dot(self.model['W2']) + self.model['b2']
        y_pred = np.argmax(Z2, axis=1)
        return y_pred
              