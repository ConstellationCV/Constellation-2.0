from .arithmetic_toolkit import Vectors
from collections import Counter
# import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class knn(object):
	"""class which represents a k-nearest neighbors model, used for predicting distance"""
	# training data format -
	# X = [[0], [1], [2], [3]]
	# y = [0, 0, 1, 1]
	def __init__(self, k):
		"""initializes a new k nearest neighbors model with k=k
		and trains model based on X,y"""
		self.model = KNeighborsClassifier(n_neighbors=k)
		self.train(X,y)

	def train(self, X, y):
		"""trains the model on inputs X and classifications y"""
		self.model.fit(X,y)

	def predict(self, x):
		"""returns prediction for classification of x"""
		return self.model.predict(x)

	def probabilities(self, x):
		"""returns list of probablities of each label for input x"""
		return self.model.predict_proba(x)

class neural_network(object):
	def __init__(self):
		self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

	def train(self, X, y):
		X = self.flatten(X)
		self.clf.fit(X,y)

	def flatten(self, matrix):
		new_X = []
		for row in matrix:
			row_arr = []
			for col in row:
				row_arr.append(col[0])
				row_arr.append(col[1])
			new_X.append(row_arr)
		return new_X

	def predict(self, Xarr):
		return self.clf.predict(Xarr)
