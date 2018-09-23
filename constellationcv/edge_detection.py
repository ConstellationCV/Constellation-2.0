import cv2
from .arithmetic_toolkit import Matrices
from .arithmetic_toolkit import Vectors
import numpy as np
import copy
import math
# import time # for testing efficiency, remove in production build

class edge_detector(object):
	"""class which contains all functions to extract edges from an image"""
	def __init__(self, imagePath):
		self.imagePath = imagePath
		self.img = cv2.imread(imagePath,0)
		self.laplacian = cv2.Laplacian(img,cv2.CV_64F)
		# self.img_rgb = cv2.imread(imagePath)
		# self.image_gray = cv2.cvtColor(self.img_rgb,cv2.COLOR_BGR2GRAY)

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

	def createEdgePointsList(self, edge_mat):
		r=0
		c=0
		ptslist = []
		for row in edge_mat:
			c=0
			for col in row:
				if edge_mat[r][c]==1:
					ptslist.append([r,c])
				c+=1
			r+=1
		self.edge_pts_list = self.flipXY(sorted(ptslist))
		return self.edge_pts_list

	def formAllLines(self):
		edge_lines_list = []
		self.lines_edge_pts_list = copy.deepcopy(self.edge_pts_list)
		for pt in self.lines_edge_pts_list:
			tempLine = self.formLine(pt)
			if tempLine[0]==-1:
				continue
			else:
				edge_lines_list.append(tempLine)
		return self.cleanListOfLines(edge_lines_list)

	def formLine(self, base_point):
		if self.lines_edge_pts_list==[]:
			return [-1,-1,-1,-1]
		# method setup
		v=Vectors()
		line_pts = []

		# loop setup
		line_pts.append(base_point)
		self.lines_edge_pts_list = sorted(self.lines_edge_pts_list, key=lambda (point): v.distance(point, base_point))[1:]
		try:
			next_pt = self.lines_edge_pts_list[0]
		except Exception as e:
			return [-1,-1,-1,-1]
		line_pts.append(next_pt)
		self.lines_edge_pts_list = sorted(self.lines_edge_pts_list, key=lambda (point): v.distance(point, next_pt))[1:]
		try:
			next_pt = self.lines_edge_pts_list[0]
		except Exception as e:
			return [-1,-1,-1,-1]
		line_pts.append(next_pt)

		# loop variables setup
		last_point=next_pt
		self.lines_edge_pts_list = sorted(self.lines_edge_pts_list, key=lambda (point): v.distance(point, next_pt))[1:]
		try:
			next_pt = self.lines_edge_pts_list[0]
		except Exception as e:
			return [-1,-1,-1,-1]

		# line variables setup
		m,b = self.calculateBestFit(line_pts)
		potential_pts = copy.deepcopy(line_pts)
		potential_pts.append(next_pt)
		nextm,nextb = self.calculateBestFit(potential_pts)

		# print line_pts


		while self.findAngleBetween(m,nextm)<3 and v.distance(last_point,next_pt)<80 and v.distance(last_point,[next_pt[0],self.evalLinearFunction(m,b,next_pt[0])])<60 and not next_pt==[-1,1] and not self.lines_edge_pts_list==[]:
			last_point=next_pt
			line_pts.append(next_pt)
			self.lines_edge_pts_list = sorted(self.lines_edge_pts_list, key=lambda (point): v.distance(point, next_pt))[1:]
			try:
				next_pt = self.lines_edge_pts_list[0]
			except Exception as e:
				break
			m,b = self.calculateBestFit(line_pts)
			potential_pts = copy.deepcopy(line_pts)
			potential_pts.append(next_pt)
			nextm,nextb = self.calculateBestFit(potential_pts)



	# helper functions, commented out as not needed

	def roundAll(self, mat):
		for row in mat:
			for col in row:
				col = np.around(col)
		return mat

	def removeAddedPtsFromLineFormationMatrix(self,list_of_pts):
		for pt in list_of_pts:
			self.lineFormationMatrix[pt[0]][pt[1]]=0

	def cost(self,line1,line2):
		return self.findAngleBetween(line1[0],line2[0])

	def value(self,x):
		return math.sqrt(x)

	def cleanListOfLines(self, list_of_lines):
		for line in list_of_lines:
			for otherLine in list_of_lines:
				if line==otherLine:
					continue
				if abs(self.evalLinearFunction(line[0],line[1],1)-self.evalLinearFunction(otherLine[0],otherLine[1],1))<10 and abs(self.evalLinearFunction(line[0],line[1],2)-self.evalLinearFunction(otherLine[0],otherLine[1],2))<20 and abs(self.evalLinearFunction(line[0],line[1],3)-self.evalLinearFunction(otherLine[0],otherLine[1],3))<10:
					list_of_lines.remove(otherLine)
		return list_of_lines

	def flipXY(self, list_of_pts):
		for pt in list_of_pts:
			pt.reverse()
		return list_of_pts

	def evalLinearFunction(self,m,b,x):
		return (m*x)+b

	def findAngleBetween(self,m1,m2):
		theta1=math.degrees(math.atan(m1))
		theta2=math.degrees(math.atan(m2))
		return theta2-theta1

	def calculateBestFit(self, list_of_pts):
		self.flipXY(list_of_pts)
		transform_factor = list_of_pts[0]
		x_transform = float(transform_factor[0])
		y_transform = float(transform_factor[1])
		new_list = copy.deepcopy(list_of_pts)
		for pt in new_list:
			pt[0]=float(pt[0])-x_transform
			pt[1]=float(pt[1])-y_transform
		x_sum = 0
		y_sum = 0
		for pt in new_list:
			x_sum+=pt[0]
			y_sum+=pt[1]
		n = len(list_of_pts)
		x_bar = x_sum/n
		y_bar = y_sum/n
		m_numerator = 0
		for pt in new_list:
			m_numerator+=(pt[0]-x_bar)*(pt[1]-y_bar)
		m_denominator = 0
		for pt in new_list:
			m_denominator+=(pt[0]-x_bar)*(pt[0]-x_bar)
		try:
			m=m_numerator/m_denominator
		except Exception as e:
			m=0
		b=m*x_transform*-1+y_transform
		return m,b

	# tool functions

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

	def printFullEquations(self, list_of_fxns):
		for fxn in list_of_fxns:
			eqn = "y="+str(fxn[0])+"x+"+str(fxn[1])+" { "+str(fxn[2][0]) + "<=x<="+str(fxn[3][0])+" }"
			# print eqn

	def printEquations(self, list_of_fxns):
		for fxn in list_of_fxns:
			eqn = "y="+str(fxn[0])+"x+"+str(fxn[1])
			print eqn
