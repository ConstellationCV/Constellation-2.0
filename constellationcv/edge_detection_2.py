import cv2
from arithmetic_toolkit import Matrices
from arithmetic_toolkit import Vectors
import numpy as np
import copy
import math
from edges_graph import graph_node,graph
# import time # for testing efficiency, remove in production build

class edge_detector_2(object):
	"""class which contains all functions to extract edges from an image"""
	def __init__(self, imagePath):
		self.imagePath = imagePath
		self.img_rgb = cv2.imread(imagePath)
		self.image_gray = cv2.cvtColor(self.img_rgb,cv2.COLOR_BGR2GRAY)
		
	def findEdgeMatrix(self):
		m = Matrices()
		v = Vectors()
		img = self.img_rgb
		num_rows = len(img)
		num_cols = len(img[0])
		edge_mat = np.around(np.zeros(shape=(num_rows,num_cols)),decimals=0)
		row_count = 0
		col_count = 0
		threshold=10
		reached = False
		for row in img:
			for col in row:
				if row_count>=num_rows-1 or col_count>=num_cols-1 or row_count==0 or col_count==0:
					col_count += 1
					continue
				else:
					if v.distance(img[row_count][col_count],img[row_count-1][col_count])>threshold or v.distance(img[row_count][col_count],img[row_count+1][col_count])>threshold or v.distance(img[row_count][col_count],img[row_count][col_count-1])>threshold or v.distance(img[row_count][col_count],img[row_count][col_count+1])>threshold:
						edge_mat[row_count][col_count]=1
				col_count += 1
			col_count = 0
			row_count+=1
		self.edge_mat. = edge_mat
		return edge_mat

	def outputEdgeImage(self,edge_mat, output_path):
		row_count=0
		col_count=0
		for row in self.image_gray:
			col_count=0
			for col in row:
				self.image_gray[row_count][col_count]=edge_mat[row_count][col_count]
				if edge_mat[row_count][col_count]==1:
					self.image_gray[row_count][col_count]=0
				else:
					self.image_gray[row_count][col_count]=252
				col_count+=1
			row_count+=1
		cv2.imwrite(output_path,self.image_gray)


	def createEdgeGraph(self):
		g=graph()
		row_count=0
		col_count=0
		for row in self.edge_mat:
			col_count=0
			for col in row:
				if col==1:
					tempNList = []
					if g.findNode(row,col)==1:
					for neighbor in self.listOfEdgeNeighbors(row_count,col_count):
						tempNList.append()
				col_count+=1
			row_count+=1

	def listOfEdgeNeighbors(self,r,c)
		list=[]
		mat = self.edge_mat
		if mat[r][c+1]==1: # right
			list.append([r,c+1])
		if mat[r-1][c+1]==1: # right down
			list.append([r-1,c+1])
		if mat[r+1][c+1]==1: # right up
			list.append([r+1,c+1])
		if mat[r+1][c]==1: # up
			list.append([r+1,c])
		if mat[r-1][c]==1: # down
			list.append([r-1,c])
		if mat[r][c-1]==1: # left
			list.append([r,c-1])
		if mat[r-1][c-1]==1: # left down
			list.append([r-1,c-1])
		if mat[r+1][c-1]==1: # left up
			list.append([r+1,c-1])
		return list

	# helper functions

	def roundAll(self, mat):
		for row in mat:
			for col in row:
				col = np.around(col)
		return mat

	def printEquations(self, list_of_fxns):
		for fxn in list_of_fxns:
			eqn = "y="+str(fxn[0])+"x+"+str(fxn[1])+" { "+str(fxn[2][0]) + "<=x<="+str(fxn[3][0])+" }"
			print eqn