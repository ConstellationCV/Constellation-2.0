import cv2
from arithmetic_toolkit import Matrices
from arithmetic_toolkit import Vectors
import numpy as np
class edge_detector(object):
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
		edge_mat = np.zeros(shape=(num_rows,num_cols))
		row_count = 0
		col_count = 0
		for row in img:
			for col in row:
				if not row_count>=num_rows-1 and not col_count>=num_cols-1 and not row_count>=0 and not col_count>=0:
					print num_cols
					if v.distance(img[row_count][col_count],img[row_count-1][col_count])>50 or v.distance(img[row_count][col_count],img[row_count+1][col_count])>50 or v.distance(img[row_count][col_count],img[row_count][col_count-1])>50 or v.distance(img[row_count][col_count],img[row_count][col_count+1])>50:
						edge_mat[row_count][col_count]=1
				col_count += 1
			row_count+=1
		return edge_mat