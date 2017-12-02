
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


class Fully_Connected(object):
    def __init__(self):
        pass
    
    def init_layers(self, layers, sm=0.001):
        model={}
        for i in range(len(layers)-1):
            model['W'+str(i)] = np.random.randn(layers[i], layers[i+1]) * sm
            model['b'+str(i)] = np.zeros(layers[i+1])
        self.model = model
        
    def init_layers_xavier(self, layers):
        model={}
        for i in range(len(layers)-1):
            model['W'+str(i)] = np.random.randn(layers[i], layers[i+1])*0.1 / np.sqrt(layers[i]/2)
            model['b'+str(i)] = np.zeros(layers[i+1])
        self.model = model
        
    def train_step(self, x_train, y_train):
        N, D = x_train.shape
        scores={}
        for i in range(len(self.model)//2):
            if i ==0:
                scores['Z'+str(i)] = x_train.dot(self.model['W'+str(i)]) + self.model['b'+str(i)]
                scores['A'+str(i)] = np.maximum(0,scores['Z'+str(i)])
            else:
                scores['Z'+str(i)] = scores['A'+str(i-1)].dot(self.model['W'+str(i)]) + self.model['b'+str(i)]
                if i!=len(self.model)/2 -1:
                    scores['A'+str(i)] = np.maximum(0,scores['Z'+str(i)])
        
        loss = 0
        exp_scores = np.exp(scores['Z'+str(len(self.model)//2 -1 )])
        
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        
        corect_logprobs = -np.log(probs[range(N), y_train])
        loss = np.sum(corect_logprobs) / N
        
        grads={}
        dscores = probs.copy()
        dscores[range(N), list(y_train)] -= 1
        dscores /= N
        
        for i in range(len(self.model)//2-1,-1,-1):
            
            if i == len(self.model)//2-1:
                grads['W' + str(i)] = scores['A'+str(i-1)].T.dot(dscores) 
                grads['b' + str(i)] = np.sum(dscores, axis = 0)
                grads['dZ' + str(i)] = dscores.dot( self.model['W'+str(i)].T)
            else:
                dh = grads['dZ' + str(i+1)]
                dh_ReLu = (scores['A'+str(i)] > 0) * dh
                if i !=0:

                    grads['W' + str(i)] = scores['A'+str(i-1)].T.dot(dh_ReLu) 
                    grads['b' + str(i)] = np.sum(dh_ReLu, axis = 0)
                    grads['dZ' + str(i)] = dh_ReLu.dot( self.model['W'+str(i)].T)
                else:
                    grads['W' + str(i)] = x_train.T.dot(dh_ReLu) 
                    grads['b' + str(i)] = np.sum(dh_ReLu, axis = 0)
                
                
        return loss, grads
    
    def train(self, x, y, learning_rate=1e-3, num_iters=100, verbose=False):
        loss_history = []
        for step in range(1,num_iters+1):
            loss, grads = self.train_step(x,y)
            loss_history.append(loss)
            
            for i in self.model:
                self.model[i] -= grads[i]*learning_rate
                
            if verbose and step % 100 == 0:
                print( 'iteration %d / %d: loss %f' % (step, num_iters, loss))
        self.loss_history = loss_history
            
                
    def predict(self, x):
        scores={}
        for i in range(len(self.model)//2):
            if i ==0:
                scores['Z'+str(i)] = x.dot(self.model['W'+str(i)]) + self.model['b'+str(i)]
                scores['A'+str(i)] = np.maximum(0,scores['Z'+str(i)])
            else:
                scores['Z'+str(i)] = scores['A'+str(i-1)].dot(self.model['W'+str(i)]) + self.model['b'+str(i)]
                if i!=len(self.model)/2 -1:
                    scores['A'+str(i)] = np.maximum(0,scores['Z'+str(i)])
        y_pred = np.argmax(scores['Z'+str(len(self.model)//2 -1)], axis=1)
        return y_pred
                


              