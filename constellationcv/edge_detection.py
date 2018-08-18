import cv2
class edge_detector(object):
	"""class which contains all functions to extract edges from an image"""
	def __init__(self, imagePath):
		self.imagePath = imagePath
		self.img_rgb = cv2.imread(imagePath)
		self.image_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
		
	def findEdgeMatrix(self):
		print(imag_rgb)