import constellationcv as cv
import numpy as np

e = cv.edge_detector("image.png")
print(np.where(e.findEdgeMatrix()==1))