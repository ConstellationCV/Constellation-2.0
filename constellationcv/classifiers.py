from arithmetic_toolkit import Vectors
from collections import Counter
# import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class knn_scratch(object):
	def raw_majority_vote(self, labels):
		votes = Counter(labels)
		winner, _ = votes.most_common(1)[0]
		return winner

	def majority_vote(self, labels):
		"""assumes that labels are ordered from nearest to farthest"""
		vote_counts = Counter(labels)
		winner, winner_count = vote_counts.most_common(1)[0]
		num_winners = len([count
							for count in vote_counts.values()
							if count == winner_count])
		if num_winners == 1: # unique winner, so return it
			return winner
		else:
			return self.majority_vote(labels[:-1]) # try again with the k-1 nearest

	def knn_classify(self, k, labeled_points, new_point):
		"""each labeled point should be a pair (point, label)"""
		v = Vectors()
		by_distance = sorted(labeled_points, key=lambda (point, _): abs(point-new_point))
		k_nearest_labels = [label for _, label in by_distance[:k]]
		return self.majority_vote(k_nearest_labels)

class network_scratch(object):
	def sigmoid(t):
		return 1/(1+math.exp(-t))

	def neuron_output(weights,inputs):
		v = Vectors()
		return sigmoid(v.dot(weights,inputs))

	def feed_forward(neural_network, input_vector):
		"""takes in a neural entwork
		(represented as a list of lists of lists of weights)
		and returns the output from forward-propogating the input"""

		outputs = []

		# process one layer at a time
		for layer in neural_network:
			input_with_bias = input_vector + [1]				# add a bias input
			output = [neuron_output(neuron, input_with_bias)	# compute the output
						for neuron in layer]					# for each neuron
			outputs.append(output)								# and remember it

			# then the input to the next layer is the output of this one
			input_vector = output

		return outputs

	def backpropagate(network, input_vector, targets):
		hidden_outputs, outputs = feed_forward(network, input_vector)

		# the output * (1-ouput) is from the derivate of sigmoid
		output_deltas = [output * (1-output) * (output-target) 
						for output, target, in zip(outputs, targets)]

		# the output * (1-output) is from the derivate of sigmoid
		for i, output_neuron in enumerate(network[-1]):
			# focus on the ith output layer neuron
			for j, hidden_output in enumerate(hidden_outputs + [1]):
				# adjust the jth weight based on both
				# this neuron's delta and its jth input
				output_neuron[j] -= output_deltas[i] * hidden_output

		# back-propagate errors to hidden layer
		v = Vectors()
		hidden_deltas = [hidden_output * (1-hidden_output) *
						v.dot(output_deltas, [n[i] for n in network[-1]])
						for i, hidden_output in enumerate(hidden_outputs)]

		# adjust weights for hidden layer, one neuron at a time
		for i, hidden_neuron in enumerate(network[0]):
			for j, input in enumerate(input_vector + [i]):
				hidden_neuron[j] -= hidden_deltas[i] * input

	def predict(input, network):
		return feed_forward(network, input)[-1]

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
		self.clf = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes-(5,2),random_state=1)
	
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

