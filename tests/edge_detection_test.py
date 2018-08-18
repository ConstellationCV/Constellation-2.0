import constellationcv as cv

e = cv.edge_detector("image.png")
print(e.findEdgeMatrix())