from sympy import *

class distance_estimation(object):
    """class for determing distance of planes from the camera, assumes cameras are parallel and colinear."""
    
    def __init__ (self, fundementalMatrix):
        self.fundementalMatrix = fundementalMatrix

    def distanceCalculation (self, xVar, xPrimeVar, bVar, fVar, debug):
        if debug == true:
            init_session()
        x, xPrime, B, F = symbols('x x* B F')
        Z = (B*F)/(Abs(x - xPrime))
        return Z.evalf(subs={x:xVar, xPrime:xPrimeVar, B:bVar, F:fVar})

    def coordinateComponentX (self, xVar, fVar, zVar, debug):
        if debug == true:
            init_session()
        x, F, Z = symbols('x F Z')
        xComponent = (F + Z)/tan(atan(x/F))
        return xComponent.evalf(subs={x:xVar, F:fVar, Z: zVar})

    def coordinateComponentY (self, yVar, fVar, zVar, debug):
        if debug == true:
            init_session()
        y, F, Z = symbols('y F Z')
        xComponent = (F + Z)/tan(atan(y/F))
        return xComponent.evalf(subs={y:yVar, F:fVar, Z: zVar})

    def findCoordinate (self, xPx, yPx, xPrimePx, yPrimePx, fVar, zVar, bVar, side, debug):
        if debug == true:
            init_session()
        z = self.distanceCalculation(xPx, xPrimePx, bVar, fVar, false)
        if side == 'left':
            x = self.coordinateComponentX(xPx, fVar, zVar, false)
            y = self.coordinateComponentY(yPx, fVar, zVar, false)
            return Point(x, y, z)
        elif side == 'right':
            x = self.coordinateComponentX(xPrimePx, fVar, zVar, false)
            y = self.coordinateComponentX(yPrimePx, fVar, zVar, false)
            return Point(x, y, z)
        else:
            print("Select which imaging frame you want to reference using left or right in the side variable.")