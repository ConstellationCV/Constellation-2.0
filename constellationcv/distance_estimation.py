from sympy import *
import cv2

class distance_estimation(object):
    """class for determing distance and position of planes from camera in refernce to (0, 0) of the cameras imaging frame."""

    def __init__ (self, fundementalMatrix):
        self.fundementalMatrix = fundementalMatrix

    #functions for determing coordinates of a given point
    def distanceCalculation (self, xVar, xPrimeVar, bVar, fVar, debug):
        '''given two matched points determines distance of points from camera (deprecated due to new k-nearest neighbors distance estimation system).'''
        if debug == true:
            init_session()
        x, xPrime, B, F = symbols('x x* B F')
        Z = (B*F)/(Abs(x - xPrime))
        return Z.evalf(subs={x:xVar, xPrime:xPrimeVar, B:bVar, F:fVar})

    def coordinateComponentX (self, xVar, fVar, zVar, debug):
        '''uses distance and pixel location to determine x coordinate of a pixel.'''
        if debug == true:
            init_session()
        x, F, Z = symbols('x F Z')
        xComponent = (F + Z)/tan(atan(x/F))
        return xComponent.evalf(subs={x:xVar, F:fVar, Z: zVar})

    def coordinateComponentY (self, yVar, fVar, zVar, debug):
        '''uses distance and pixel location to determine y coordinate of a pixel.'''
        if debug == true:
            init_session()
        y, F, Z = symbols('y F Z')
        xComponent = (F + Z)/tan(atan(y/F))
        return xComponent.evalf(subs={y:yVar, F:fVar, Z: zVar})

    def findCoordinate (self, xPx, yPx, fVar, zVar, bVar, zPos, debug):
        '''uses coordinateComponentX() and coordinateComponentY() as well as a distance measurement to create a 3d point object.'''
        if debug == true:
            init_session()
        z = zPos
        x = self.coordinateComponentX(xPx, fVar, zVar, debug)
        y = self.coordinateComponentY(yPx, fVar, zVar, debug)
        return Point(x, y, z)

    #functions for finding and making faces
    def houghTransform(self, edgePoints, rho, theta, threshold, minLineLength, maxLineGap):
        '''runs opencv HoughLinesP and returns lines as sympy line objects.'''
        cvLines = cv2.HoughLinesP(edgePoints, rho, theta, threshold, minLineLength, maxLineGap)
        sympyLines = []
        for line in cvLines:
            for x1, y1, x2, y2 in line:
                sympyLines.append(Line(Point(x1, y1), Point(x2, y2)))
        return sympyLines

    def isInContainer(self, container, objects, debug):
        '''takes a location (polygon) of a detected face and returns an array of objects (lines or points) in that array.'''
        if debug == true:
            init_session()
        containedPoints = []
        for object in objects:
            if type(object) is sympy.geometry.point.Point2D and if container.encloses_point(object):
                containedPoints.append(object)
            else if type(object) is sympy.geometry.line.Line2D and container.encloses_point(object.p1) and container.encloses_point(object.p2):
                containedPoints.append(object)
        return containedPoints

    def makeFace(self, container, points, debug):
        '''runs isInContainer() and then creates a convex hull (polygon) of those objects.'''
        if debug == true:
            init_session()
         containedPoints = self.isInContainer(container, points, debug)
         return convex_hull(*containedPoints, **dict(polygon=True))

    #functions for handline laser points
