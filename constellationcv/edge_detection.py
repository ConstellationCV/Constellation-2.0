import cv2
from arithmetic_toolkit import Matrices
from arithmetic_toolkit import Vectors
import numpy as np
# import time # for testing efficiency, remove in production build

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
		edge_mat = np.around(np.zeros(shape=(num_rows,num_cols)),decimals=0)
		row_count = 0
		col_count = 0
		threshold=7
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
		return edge_mat

	def outputEdgePic(self,edge_mat):
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
		cv2.imwrite('edges_output.png',self.image_gray)


	def calculateBestFit(self, list_of_pts):
		transform_factor = list_of_pts[0]
		x_transform = float(transform_factor[0])
		y_transform = float(transform_factor[1])
		for pt in list_of_pts:
			pt[0]=float(pt[0])-x_transform
			pt[1]=float(pt[1])-y_transform
		x_sum = 0
		y_sum = 0
		for pt in list_of_pts:
			x_sum+=pt[0]
			y_sum+=pt[1]
		n = len(list_of_pts)
		x_bar = x_sum/n
		y_bar = y_sum/n
		m_numerator = 0
		for pt in list_of_pts:
			m_numerator+=(pt[0]-x_bar)*(pt[1]-y_bar)
		m_denominator = 0
		for pt in list_of_pts:
			m_denominator+=(pt[0]-x_bar)*(pt[0]-x_bar)
		m=m_numerator/m_denominator
		b=m*x_transform*-1+y_transform
		return m,b

	def findAngleBetween(self,m1,m2):
		theta1=math.degrees(atan(m1))
		theta2=math.degrees(atan(m2))
		return theta2-theta1

	def formAllLines(self,edge_mat):
		self.lineFormationMatrix = np.copy(edge_mat)
		edge_lines_list = []
		row_count=0
		col_count=0
		for row in edge_mat:
			col_count=0
			for col in row:
				if edge_mat[row_count][col_count]==1:
					edge_lines_list.append(self.formLine([row_count,col_count]))
				col_count+=1
			row_count+=1

	def formLine(self, base_point):
		edges_remaining = self.lineFormationMatrix
		# return line

	def cost(self,line1,line2):
		return self.findAngleBetween(line1[0],line2[0])

	def value(self,x):
		return math.sqrt(x)








