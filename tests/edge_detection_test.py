import constellationcv as cv
import numpy as np

e = cv.edge_detector("cube.png")
# print(np.where(e.findEdgeMatrix()==1))
edgeMat = e.findEdgeMatrix()
e.outputEdgeImage(edgeMat,"edges_output.png")
# np.savetxt('outputmat.txt', edgeMat)
#print edgeMat[0][0]
# print(e.calculateBestFit([[1,1],[3,3],[10,6],[13,16]]))