import numpy as np 


class LogisticRegression(object):
	# Logistic Regression classifier 
	def __init__(self):
		self.definition = 'Logistic Regression for binary classification problems'
		self.W = None 
		self.b = None


	def train(self, x, y, lr=1e-3, num_iter = 1000):
		
	
		num_train, dim = x.shape
		num_classes = np.max(y) + 1
		#print(x.shape)
		if self.W is None:
		# lazily initialize W
			self.W = 0.001 * np.random.randn(dim)
			self.b = np.zeros(1)

		for it in range(num_iter):

			Z = np.dot(self.W, x.T)
			Z += self.b
			A = 1/ (1+np.exp(-Z)) 
			cost = - (1/m)* np.sum(y*np.log(A) + (1-y)*np.log(1-A))
			dz = A-y
			dw =  np.dot(x.T, dz.T) / num_train
			db =  np.sum(dz)/num_train

			self.W -=dw*lr
			self.b -=db*lr

	def predict_prob(self, x):
		Z = np.dot(self.W, x.T)
		Z += self.b
		A = 1/ (1+np.exp(-Z)) 
		return A
	def predict(self,x):
		A = self.predict_prob(x)
		return (A >.5).astype(int)

