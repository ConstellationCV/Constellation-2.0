import constellationcv as cv
import numpy as np

e = cv.edge_detector("image.png")
edgeMat = e.findEdgeMatrix()
# e.createEdgePointsList(edgeMat)
<<<<<<< HEAD
# e.outputEdgeImage(edgeMat,"edges_output.png")
=======
e.outputEdgeImage(edgeMat,"edges_output.png")
>>>>>>> be2887370a407dda19e47682273e6b556a343b84
# e.printFullEquations(e.formAllLines())
# np.savetxt('outputmat.txt', edgeMat)
#print edgeMat[0][0]
# print(e.calculateBestFit([[1,1],[3,3],[10,6],[13,16]]))
