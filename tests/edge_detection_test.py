import constellationcv as cv
import numpy as np

e = cv.edge_detector("image.png")
edgeMat = e.findEdgeMatrix()
# e.createEdgePointsList(edgeMat)
# e.outputEdgeImage(edgeMat,"edges_output.png")
# e.printFullEquations(e.formAllLines())
# np.savetxt('outputmat.txt', edgeMat)
#print edgeMat[0][0]
# print(e.calculateBestFit([[1,1],[3,3],[10,6],[13,16]]))
