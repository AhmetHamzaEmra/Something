
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

        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history =[]

    
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
    
    def train(self, x, y, x_val=np.array([]), y_val=np.array([]), learning_rate=1e-3, num_iters=100, verbose=False):
        
        for step in range(1,num_iters+1):
            loss, grads = self.train_step(x,y)
            self.loss_history.append(loss)
            
            for i in self.model:
                self.model[i] -= grads[i]*learning_rate
                
            if verbose and step % 100 == 0:
                train_acc = self.score(x,y)
                self.train_acc_history.append(train_acc)
                if x_val.shape[0] != 0:
                    val_acc = self.score(x_val,y_val)
                    self.val_acc_history.append(val_acc)
                    print('iteration %d / %d: loss %f training accuracy %f val accuracy %f'% (step, num_iters, loss, train_acc,val_acc))
            
                else:
                    print( 'iteration %d / %d: loss %f training accuracy %f'  % (step, num_iters, loss, train_acc))
                    
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
                

    def score(self,x,y):
        pred = self.predict(x)
        correct = pred == y
        return np.sum(correct)/y.shape[0]

    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(40, 40))
        plt.subplot(5, 5, 1)
        plt.plot(self.loss_history)
        plt.title("Loss")
        plt.subplot(5, 5, 2)
        plt.plot(self.train_acc_history, 'b',label='traing accuracy')
        plt.plot(self.val_acc_history, 'r', label='validation accuracy')
        plt.legend()
        plt.title("Accuracy")
        plt.show()


class ConvNet(object):
    def __init__(self):

        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history =[]


    def zero_pad(self, X, pad):

        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad),(0,0)), mode = 'constant')

        return X_pad
    
    def conv_single_step(self, a_slice_prev, W, b):
        s = a_slice_prev*W
        Z = np.sum(s)
        Z = float(Z+b)
        return Z

    
    def conv_forward(self, A_prev, W, b):

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape

        stride = 1
        pad = 1

        n_H = int(np.floor((n_H_prev - f + (2*pad))/stride)+1)
        n_W = int(np.floor((n_W_prev - f + (2*pad))/stride)+1)

        Z = np.zeros([m, n_H, n_W, n_C])
        A_prev_pad = self.zero_pad(A_prev, pad)

        for i in range(m):                                  # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                      # Select ith training example's padded activation
            for h in range(n_H):                            # loop over vertical axis of the output volume
                for w in range(n_W):                        # loop over horizontal axis of the output volume
                    for c in range(n_C):                    # loop over channels (= #filters) of the output volume
                        vert_start = h * stride
                        vert_end =vert_start+pad
                        horiz_start = w* stride
                        horiz_end = horiz_start+pad
                        a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, W[...,c], b[...,c])

        
        return Z

    
    def conv_backward(self, dZ, A_prev, W, b):
       
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve information from "hparameters"
        stride = 1
        pad = 1

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad )
        
        
        
        for i in range(m):                       # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = w * stride
                        vert_end = vert_start + f
                        horiz_start = h * stride
                        horiz_end = horiz_start+f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end , horiz_start:horiz_end , :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] +=  a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        
        return dA_prev, dW, db
    
    
    def init_layers(self,input_size, output_size,  sm=0.1):
        h, w, c = input_size
        model = {}
        model['W1'] = np.random.randn(3,3,c,4)*sm
        model['b1'] = np.zeros((1,1,1,4))
        model['W2'] = np.random.randn(3,3,4,4)*sm
        model['b2'] = np.zeros((1,1,1,4))
        model['W3'] = np.random.randn(3,3,4,4)*sm
        model['b3'] = np.zeros((1,1,1,4))
        model['W4'] = np.random.randn((h)*(w)*4,256)*sm
        model['b4'] = np.zeros((256))
        model['W5'] = np.random.randn(256,output_size)*sm
        model['b5'] = np.zeros((output_size))
        self.model = model
    
    
    
    def train_step(self, x_train, y_train):
        N,h,w,c = x_train.shape
        scores={}
        scores['Z1'] = self.conv_forward( x_train, self.model['W1'], self.model['b1'])
        scores['A1'] = np.maximum(0,scores['Z1'])
        scores['Z2'] = self.conv_forward(scores['A1'], self.model['W2'], self.model['b2'])
        scores['A2'] = np.maximum(0,scores['Z2'])
        scores['Z3'] = self.conv_forward(scores['A2'], self.model['W3'], self.model['b3'])
        scores['A3'] = np.maximum(0,scores['Z3'])
        scores['A3f'] = scores['A3'].reshape([-1,(h)*(w)*4])
        scores['Z4'] = scores['A3f'].dot(self.model['W4']) + self.model['b4']
        scores['A4'] = np.maximum(0,scores['Z4'])
        scores['Z5'] = scores['A4'].dot(self.model['W5']) + self.model['b5']
        
        loss = 0
        exp_scores = np.exp(scores['Z5'])
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        corect_logprobs = -np.log(probs[range(N), y_train])
        loss = np.sum(corect_logprobs) / N
        
        grads={}
        dscores = probs.copy()
        dscores[range(N), list(y_train)] -= 1
        dscores /= N
        # Grads layer5 
        grads['W5'] = scores['A4'].T.dot(dscores) 
        grads['b5'] = np.sum(dscores, axis = 0)
        grads['dZ5'] = dscores.dot( self.model['W5'].T)
        # Grads layer4
        d4_relu = (scores['A4']>0) * grads['dZ5']
        grads['W4'] = scores['A3'].T.dot(d4_relu) 
        grads['W4'] = grads['W4'].reshape([grads['W4'].shape[0]*grads['W4'].shape[1]*grads['W4'].shape[2],grads['W4'].shape[3]])
        grads['b4'] = np.sum(d4_relu, axis = 0)
        grads['dZ4'] = d4_relu.dot( self.model['W4'].T)
        # Grads layer3
        grads['dZ4'] = grads['dZ4'].reshape([-1,h,w,4])
        d3_relu = (scores['A3']>0) * grads['dZ4']
        grads['dZ3'], grads['W3'], grads['b3'] = self.conv_backward(d3_relu, scores['A2'], self.model['W3'], self.model['b3'])
     
        # Grads layer2 
        d2_relu = (scores['A2']>0) * grads['dZ3']
        grads['dZ2'], grads['W2'], grads['b2'] = self.conv_backward(d2_relu, scores['A1'], self.model['W2'], self.model['b2'])
       
        # Grads layer1 
        d1_relu = (scores['A1']>0) * grads['dZ2']
        grads['dZ1'], grads['W1'], grads['b1'] = self.conv_backward(d1_relu, x_train, self.model['W1'], self.model['b1'])
       
        return loss, grads
    
    def train(self, x, y, x_val=np.array([]), y_val=np.array([]), learning_rate=1e-3, num_iters=100, verbose=True):
        
        for step in range(1,num_iters+1):
            loss, grads = self.train_step(x,y)
            self.loss_history.append(loss)
            
            for i in self.model:
               
                self.model[i] -= grads[i]*learning_rate
                
            if verbose:
                #train_acc = self.score(x,y)
                #self.train_acc_history.append(train_acc)
                if x_val.shape[0] != 0:
                    #val_acc = self.score(x_val,y_val)
                    #self.val_acc_history.append(val_acc)
                    print('\riteration %d / %d: loss %f'% (step, num_iters, loss), end = "")
            
                else:
                    print( '\riteration %d / %d: loss %f'  % (step, num_iters, loss), end = "")
                    
    def predict(self, x):
        
        N,h,w,c = x.shape
        scores={}
        scores['Z1'] = self.conv_forward( x, self.model['W1'], self.model['b1'])
        scores['A1'] = np.maximum(0,scores['Z1'])
        scores['Z2'] = self.conv_forward(scores['A1'], self.model['W2'], self.model['b2'])
        scores['A2'] = np.maximum(0,scores['Z2'])
        scores['Z3'] = self.conv_forward(scores['A2'], self.model['W3'], self.model['b3'])
        scores['A3'] = np.maximum(0,scores['Z3'])
        scores['A3f'] = scores['A3'].reshape([-1,(h)*(w)*4])
        scores['Z4'] = scores['A3f'].dot(self.model['W4']) + self.model['b4']
        scores['A4'] = np.maximum(0,scores['Z4'])
        scores['Z5'] = scores['A4'].dot(self.model['W5']) + self.model['b5']
        y_pred = np.argmax(scores['Z5'], axis=1)
        return y_pred
                

    def score(self,x,y):
        pred = self.predict(x)
        correct = pred == y
        return np.sum(correct)/y.shape[0]

    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(40, 40))
        plt.subplot(5, 5, 1)
        plt.plot(self.loss_history)
        plt.title("Loss")
        plt.subplot(5, 5, 2)
        plt.plot(self.train_acc_history, 'b',label='traing accuracy')
        plt.plot(self.val_acc_history, 'r', label='validation accuracy')
        plt.legend()
        plt.title("Accuracy")
        plt.show()
              
              