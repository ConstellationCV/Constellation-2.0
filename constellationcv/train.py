import numpy as np
from edge_detection import edge_detector
import os
from classifiers import neural_network
import pickle

class trainer(object):
	"""docstring for trainer"""
	def __init__(self, pathToImages):
		if not os.path.isdir("pickled_data"):
			os.makedirs("pickled_data")
		else:
			os.remove("pickled_data/trained_net")
		f=open("pickled_data/trained_net","wb")
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
...                     hidden_layer_sizes=(5, 2), random_state=1)
		self.createAllEdgesList(pathToImages)
		# clf.fit(self.all_image_edges,)
		pickle.dump(self.all_image_edges,f)
		f.close()
		
	def createEdgeList(self, pathToImage):
		e = edge_detector(pathToImage)
		return e.createEdgePointsList(e.findEdgeMatrix())

	def createAllEdgesList(self, pathToImages):
		list = []
		for image in os.listdir(pathToImages):
			if ".png" in image:
				print "training on " + image
				list.append(self.createEdgeList(pathToImages+"/"+image))
		self.all_image_edges = self.flatten(list)

	def flatten(self, matrix):
		new_X = []
		for row in matrix:
			row_arr = []
			for col in row:
				row_arr.append(col[0])
				row_arr.append(col[1])
			new_X.append(row_arr)
		return new_X

	def createYs(self, pathToImages):
		list = []