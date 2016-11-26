import math
import random

"""This class simulates a simple car model as seen here:
http://msl.cs.uiuc.edu/planning/node658.html. x,y,theta are relative position and
orientation. v is linear velocity, and theta_dot is angular velocity. L is the
length of the car - needed for radius of curvature."""

class SimpleCar:
    def __init__(self, L, x=0, y=0, theta=0, v=0, theta_dot=0):
        self.x = x #meters
        self.y = y #meters
        self.theta = theta #radians
        self.v = v #meters/sec
        self.theta_dot = theta_dot #radians/sec
        self.L = L #meters

    def getXLocation(self):
        return self.x

    def getYLocation(self):
        return self.y

    def getTheta(self):
        return self.theta

    def getV(self):
        return self.v

    def getTheta_dot(self):
        return self.theta_dot

    '''In order to create noisy measurements, a gaussian distribution is applied to
    the current state.'''
    def getXLocationWithNoise(self, noiseLevel):
        return random.gauss(self.getXLocation(), noiseLevel)

    def getYLocationWithNoise(self, noiseLevel):
        return random.gauss(self.getYLocation(), noiseLevel)

    def getThetaWithNoise(self, noiseLevel):
        return random.gauss(self.getTheta(), noiseLevel)

    def getVWithNoise(self, noiseLevel):
        return random.gauss(self.getV(), noiseLevel)

    def getTheta_dotWithNoise(self, noiseLevel):
        return random.gauss(math.radians(self.getTheta_dot()), noiseLevel)

    def getStateVector(self):
        return (self.getXLocation(), self.getYLocation(), self.getV(), self.getTheta(), self.getTheta_dot())


    def step(self, a, steerAngle, dt):
        '''Steps forward in simulation given the current state, some control
        vector, and a small dt'''
        oldX = self.getXLocation()
        oldY = self.getYLocation()
        oldV = self.getV()
        oldTheta = self.getTheta()
        oldTheta_dot = self.getTheta_dot()

        self.x = oldX + (oldV * math.cos(oldTheta) * dt) + (.5 * a * math.cos(oldTheta) * dt**2)
        self.y = oldY + (oldV * math.sin(oldTheta) * dt) + (.5 * a * math.sin(oldTheta) * dt**2)
        self.v = oldV + a * dt
        self.theta = oldTheta + ((oldV / self.L) * math.tan(steerAngle) * dt)
        if self.theta > 360:
            self.theta = self.theta - 360
        elif self.theta < -360:
            self.theta = self.theta + 360
        self.theta_dot = (oldV / self.L) * math.tan(steerAngle)
