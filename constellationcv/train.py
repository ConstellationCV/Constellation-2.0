import numpy as np
from edge_detection import edge_detector
import os
from classifiers import neural_network

class trainer(object):
	"""docstring for trainer"""
	def __init__(self, pathToImages):
		self.createAllEdgesList(pathToImages)
		
	def createEdgeList(self, pathToImage):
		e = edge_detector(pathToImage)
		return e.createEdgePointsList(e.findEdgeMatrix())

	def createAllEdgesList(self, pathToImages):
		list = []
		for image in os.listdir(pathToImages):
			list.append(self.createEdgeList(pathToImages+image))
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