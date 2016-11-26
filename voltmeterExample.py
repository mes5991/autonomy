"""1D kalman filter example. State estimation for
a single noisy reading from a voltmeter."""

import numpy as np
import random
import kalmanFilterLinear
import matplotlib.pyplot as plt


class Voltmeter:
    """Voltmeter class. Randomly applies gaussian noise to "true"
    voltage readings."""
    def __init__(self, trueVoltage, noiseLevel):
        self.trueVoltage = trueVoltage
        self.noiseLevel = noiseLevel

    def getVoltage(self):
        return self.trueVoltage

    def getVoltageWithNoise(self):
        return random.gauss(self.getVoltage(), self.noiseLevel)

steps = 200 #Number of steps in simulation

A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.1])
xhat = np.matrix([3]) #Initial state estimate
P = np.matrix([1])

Filter1 = kalmanFilterLinear.KalmanFilterLinear(A,B,H,xhat,P,Q,R)
Voltmeter1 = Voltmeter(1.25,0.25)

measuredVoltage = []
trueVoltage = []
kalman = []

for i in range(steps):
    measured = Voltmeter1.getVoltageWithNoise()
    measuredVoltage.append(measured)
    trueVoltage.append(Voltmeter1.getVoltage())
    kalman.append(Filter1.getCurrentState().item(0))
    Filter1.step(np.matrix([0]), np.matrix([measured]))

# # plt.plot(range(steps), measuredVoltage)
# print(kalman)
# plt.plot(range(steps), kalman)

plt.plot(range(steps), trueVoltage, 'r', range(steps), measuredVoltage, 'g', range(steps), kalman, 'b')
plt.show()
