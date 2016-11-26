import simpleCar
import simpleCarKF;
import numpy as np
import matplotlib.pyplot as plt
import sympy
import math
from readCSV import getList
import XML_Reader
import os

"""Establishes GTA vehicle model and filters inputs taken from GTA simulation
output. Inputs are pre-computed and stored in XML files"""

"""Initial Values"""
xi = -10 #meters
yi = 10 #meters
thetai = 3.14 #radians
vi = 0 #meters/sec
theta_doti = 0 #radians/sec
Xi = np.matrix([[xi], [yi], [thetai], [vi], [theta_doti]]) #State vector ('guess')
Pi = np.zeros((5,5)) #Initial process covariance guess
Q = np.eye(5)*0.000001 #Process noise
R = np.eye(5)*.0001 #Measurement noise
wheelBase = 4.5 #meters. "L" in KF state transition model
noiseLevel = .4 #variance of noise to simulate measurement error

'''Get system input and GTA readings'''
xmlList = os.listdir('gtaOutput/gtaOutput1')
xmlList.sort()
frameListLength = len(xmlList) #number of frames taken in GTA
XTruth = []
YTruth = []
thetaTruth = []
velInput = []
steerAngleInput = []
timeStep = []
for i in range(frameListLength):
    data = XML_Reader.getXMLInfo(xmlList[i]) #[posX, posY, theta, vel, steerAngle, dt]
    XTruth.append(data[0])
    YTruth.append(data[1])
    thetaTruth.append(data[2])
    velInput.append(data[3])
    steerAngleInput.append(math.radians(data[4]))
    timeStep.append(data[5])
controlVector = [velInput, steerAngleInput]

"""Generate EKF object"""
EKF = simpleCarKF_GTA.SimpleCarKF_GTA(Xi, Pi, 0, Q, R)

"""Simulate and Filter"""
#Declare lists to store values
xTrue = []
yTrue = []
thetaTrue = []
vTrue = []
xMeasured = []
yMeasured = []
thetaMeasured = []
vMeasured = []
theta_dotMeasured = []
xKF = []
yKF = []
thetaKF = []
vKF = []
theta_dotKF = []
i = 0 #loop counter
for t in np.arange(0, time, dt):
    #Store true car values
    xTrue.append(car1.getXLocation())
    yTrue.append(car1.getYLocation())
    vTrue.append(car1.getV())
    thetaTrue.append(car1.getTheta())
    theta_dotTrue.append(car1.getTheta_dot())

    #Store sensor readings
    xMeasured.append(car1.getXLocationWithNoise(noiseLevel))
    yMeasured.append(car1.getYLocationWithNoise(noiseLevel))
    vMeasured.append(car1.getVWithNoise(noiseLevel))
    thetaMeasured.append(car1.getThetaWithNoise(noiseLevel))
    theta_dotMeasured.append(car1.getTheta_dotWithNoise(noiseLevel))
    measurementVector = np.matrix([[xMeasured[i]], [yMeasured[i]], [thetaMeasured[i]], [vMeasured[i]], [theta_dotMeasured[i]]])

    #Store EKF state values
    currentStateEKF = EKF.getCurrentState()
    xKF.append(currentStateEKF.item(0))
    yKF.append(currentStateEKF.item(1))
    thetaKF.append(currentStateEKF.item(2))
    vKF.append(currentStateEKF.item(3))
    theta_dotKF.append(currentStateEKF.item(4))

    #Advance simulation by dt
    currentControlVector = [controlVector[0][i], controlVector[1][i]]
    car1.step(currentControlVector[0], currentControlVector[1], dt)
    EKF.step(currentControlVector, measurementVector, wheelBase, dt)
    i = i + 1
    # print i

plt.figure()
plt.plot(xTrue, yTrue, 'b', xMeasured, yMeasured, '+r', xKF, yKF, 'g')
plt.title('y over x, blue = Truth, green = KF, red = measurements')

plt.figure()
plt.plot(np.arange(0, time, dt), xTrue, 'b', np.arange(0, time, dt), xKF, 'g')
plt.title('x over time, blue = Truth, green = KF')

plt.figure()
plt.plot(np.arange(0, time, dt), yTrue, 'b', np.arange(0, time, dt), yKF, 'g')
plt.title('y over time, blue = Truth, green = KF')

plt.figure()
plt.plot(np.arange(0, time, dt), thetaTrue, 'b', np.arange(0, time, dt), thetaKF, 'g')
plt.title('theta over time, blue = Truth, green = KF')
plt.show()
