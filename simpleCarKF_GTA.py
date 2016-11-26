import numpy as np
import matplotlib.pyplot as plt
import math
import sympy

'''This class establishes a non-linear kalman filter designed to handle the simple
car model'''
class SimpleCarKF_GTA:
    def __init__(self,X, P, hSym, Q, R):
        self.X = X #state vector
        self.P = P #process covariance matrix
        self.hSym = hSym #symbolic function used to compute the predicted measurement from the predicted state
        self.Q = Q #Process noise
        self.R = R #Observation noise

    def getCurrentState(self):
        return self.X

    def getF(self, controlVector, wheelBase, dt):
        '''Computes F, the state transition matrix, using sympy for symbolic
        jacobian calculation and substitution of current state and control vector.
        An alternate solution would be to compute the jacobian by hand once, and
        use a function that does the necessary value substitution.'''

        x,y,theta,theta_dot,v,acc,L,phi,dtSym = sympy.symbols("x,y,theta,theta_dot,v,acc,L,phi,dtSym") #Declare symbolic variables

        #State transition model
    	fSym = sympy.Matrix([[x + (v * sympy.cos(theta) * dtSym)],
                             [y + (v * sympy.sin(theta) * dtSym)],
                             [theta + (v * sympy.tan(phi) * dtSym)/L],
                             [v],
                             [(v * sympy.tan(phi) * dtSym)/L]])

    	#State vector
    	XSym = sympy.Matrix([x,y,theta,v,theta_dot])

        #Calculate jacobian and substitute current state vector and control vector values
    	F = fSym.jacobian(XSym)
    	F = F.subs(x,self.X.item(0))
        F = F.subs(y,self.X.item(1))
        F = F.subs(theta,self.X.item(2))
        # F = F.subs(v,self.X.item(3))
        F = F.subs(theta_dot,self.X.item(4))
        F = F.subs(v,controlVector[0])
        F = F.subs(phi,controlVector[1])
        F = F.subs(dtSym,dt)
        F = F.subs(L,wheelBase)
        return np.matrix(F)

    def getPredictedX(self, controlVector, wheelBase, dt):
        '''Computes predicted state vector'''
        oldX = self.X.item(0)
        oldY = self.X.item(1)
        oldTheta = self.X.item(2)
        oldV = self.X.item(3)
        oldTheta_dot = self.X.item(4)
        velInput = controlVector[0] #m/s
        steerAngle = controlVector[1] #Radians

        newX = oldX + (oldV * math.cos(oldTheta) * dt)
        newY = oldY + (oldV * math.sin(oldTheta) * dt)
        newV = velInput
        newTheta = oldTheta + ((oldV / wheelBase) * math.tan(steerAngle) * dt)
        if newTheta > 360:
            newTheta = newTheta - 360
        elif newTheta < -360:
            newTheta = newTheta + 360
        newTheta_dot = ((oldV / wheelBase) * math.tan(steerAngle))
        predictedX = np.matrix([[newX], [newY], [newTheta], [newV], [(newTheta_dot)]])
        return predictedX

    def step(self, controlVector, measurementVector, wheelBase, dt):
        """PREDICTION"""
        predictedX = self.getPredictedX(controlVector, wheelBase, dt)
        F = self.getF(controlVector, wheelBase, dt)
        predictedP = (F * self.P * np.transpose(F)) + self.Q

        """OBSERVATION"""
        """Assuming we can measure our stateVector directly, our predicted
        measurement (h(predictedX)) becomes simply predictedX."""
        y = measurementVector - predictedX
        if y[2] < 0:
            y[2] = -(abs(y[2]) % (2*math.pi))
        else:
            y[2] = (abs(y[2]) % (2*math.pi))
        H = np.eye(5)
        S = H * predictedP * np.transpose(H) + self.R

        """UPDATESTEP"""
        K = predictedP * np.transpose(H) * np.linalg.inv(S)
        self.X = predictedX + K * y
        size = (self.P.shape[0])
        self.P = (np.eye(size) - (K * H)) * predictedP
