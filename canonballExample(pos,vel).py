"""This example will track the classic physics problem of a cannon ball shot
from a cannon. The state vector will be [posX, posY, velX, velY]. Noisy measurements
will come from a simulated camera tracking the position of the ball, and from
a more accurate sensor in the ball reading acceleration x & y.
Review of equations:
r = r_0 + v_0t + 0.5at^2
g = [0, -9.81]'
X = [x_0 + cos(angle)*v_0*dt
     y_0 + sin(angle)*v_0*dt - 0.5gdt^2
     cos(angle)*v_0
     sin(angle)*v_0 - 0.5gdt^2]
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import kalmanFilterLinear

class Cannon:
    def __init__(self):
        self.velInit = 100 #m/s
        self.accInit = 0 #m/s/s
        self.posInit = [0,0] #[x,y]
        self.g = [0, -9.81] #m/s/s
        self.angle = math.radians(45) #angle of cannon in degrees
        self.noiseLevel = 30

    def getXLocation(self, t):
        return self.posInit[0] + self.velInit*math.cos(self.angle)*t

    def getYLocation(self, t):
        return self.posInit[1] + self.velInit*math.sin(self.angle)*t + 0.5*self.g[1]*(t**2)

    def getXVelocity(self):
        return self.velInit*math.cos(self.angle)

    def getYVelocity(self, t):
        return self.velInit*math.sin(self.angle) + self.g[1]*t

    def getXLocationWithNoise(self, t):
        return random.gauss(self.getXLocation(t), self.noiseLevel)

    def getYLocationWithNoise(self, t):
        return random.gauss(self.getYLocation(t), self.noiseLevel)

    def simulate(self, steps):
        dt = .2
        trueX = []
        trueY = []
        measuredX = []
        measuredY = []
        trueXVel = []
        trueYVel = []
        for t in np.arange(0, steps+dt, dt):
            trueX.append(self.getXLocation(t))
            trueY.append(self.getYLocation(t))
            measuredX.append(self.getXLocationWithNoise(t))
            measuredY.append(self.getYLocationWithNoise(t))
            trueXVel.append(self.getXVelocity())
            trueYVel.append(self.getYVelocity(t))
        return (trueX, trueY, measuredX, measuredY, trueXVel, trueYVel)

cannon1 = Cannon()
(trueX,trueY, measuredX, measuredY, trueXVel, trueYVel) = cannon1.simulate(15)

steps = np.arange(0, 15+.2, .1)
dt = .1
A = np.matrix([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.matrix([[0], [.5*dt**2], [0], [dt]])
H = np.eye(4)
Xinitial= np.matrix([[0],[500],[100*math.cos(math.radians(45))],[100*math.sin(math.radians(45))]])
P = np.eye(4)
Q = np.zeros(4)
R = np.eye(4)*0.2

kf = kalmanFilterLinear.KalmanFilterLinear(A,B,H,Xinitial,P,Q,R)
trueX = []
trueY = []
measuredX = []
measuredY = []
trueXVel = []
trueYVel = []
kfX = []
kfY = []
kfXVel = []
kfYVel = []

for t in range(len(steps)):
    dt = steps.item(t)
    trueX.append(cannon1.getXLocation(steps.item(t)))
    trueY.append(cannon1.getYLocation(steps.item(t)))
    measuredX.append(cannon1.getXLocationWithNoise(steps.item(t)))
    measuredY.append(cannon1.getYLocationWithNoise(steps.item(t)))
    trueXVel.append(cannon1.getXVelocity())
    trueYVel.append(cannon1.getYVelocity(steps.item(t)))
    kfX.append(kf.getCurrentState().item(0))
    kfY.append(kf.getCurrentState().item(1))
    kfXVel.append(kf.getCurrentState().item(2))
    kfYVel.append(kf.getCurrentState().item(3))
    kf.step([-9.81], np.matrix([[measuredX[t]], [measuredY[t]], [trueXVel[t]], [trueYVel[t]]]))

plt.plot(trueX, trueY, 'b', measuredX, measuredY, 'r+', kfX, kfY, 'g')
plt.show()
plt.plot(steps, trueXVel, 'r',steps, trueYVel, 'b', steps, kfXVel, 'y', steps, kfYVel, 'g')
plt.show()
