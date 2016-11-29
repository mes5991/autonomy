import simpleCar
import simpleCarKF;
import numpy as np
import matplotlib.pyplot as plt
import sympy
import math
from readCSV import getList
import XML_Reader
import os

'''Get GTA readings'''
# xmlFolder = 'gtaOutput/gtaOutput1'
xmlFolder = 'gtaOutput/28NOV16/first300'
xmlList = os.listdir(xmlFolder)
xmlList.sort()
frameListLength = len(xmlList) #number of frames taken in GTA
XTruth = [] #meters
YTruth = [] #meters
thetaTruth = [] #degrees
velInput = [] #m/s
steerAngleInput = [] #degrees
timeStep = [] #Time from previous frame to current frame
for i in range(frameListLength):
    data = XML_Reader.getXMLInfo(xmlFolder + '/' + xmlList[i]) #[posX, posY, theta, vel, steerAngle, dt]
    XTruth.append(data[0])
    YTruth.append(data[1])
    thetaTruth.append(data[2])
    velInput.append(data[3])
    steerAngleInput.append((data[4]))
    timeStep.append(data[5])

'''Compute total time elapsed'''
time = [0]
for i in range(1,frameListLength):
    time.append(time[i-1] + timeStep[i])

'''Compute average acceleration to be used as EKF control vector input'''
accInput = [0]
for i in range(1, frameListLength):
    accInput.append((velInput[i] - velInput[i-1])/timeStep[i])
accInput[0] = accInput[1] #Making assumption that the acceleration in the first frame is similar to the second. Not enough information to comput acceleration in the first frame

'''Compute theta_dot to be used as EKF measurement'''
theta_dot = [0] #deg/s
for i in range(1, frameListLength):
    theta_dot.append((thetaTruth[i]-thetaTruth[i-1])/timeStep[i])
theta_dot[0] = theta_dot[1] #Making assumption that the theta_dot in the first frame is similar to the second. Not enough information to comput theta_dot in the first frame

'''Creating simple car EKF object'''
xi = XTruth[0] #Initial x guess. Making it the same as first true reading. Meters
yi = YTruth[0] #Initial y guess. Making it the same as first true reading. Meters
thetai = math.radians(thetaTruth[0]) #Initial theta guess. Making it the same as first true reading. Degrees
vi = velInput[0] #Initial velocity guess. Making it the same as first true reading. meters/sec
theta_doti = math.radians(theta_dot[0]) #Initial theta_dot guess. Making it the same as first true reading. Degrees/sec
Xi = np.matrix([[xi], [yi], [thetai], [vi], [theta_doti]]) #Initial state vector guess
Pi = np.zeros((5,5)) #Initial process covariance guess
Q = np.eye(5)*0.000001 #Process noise
R = np.eye(5)*.0001 #Measurement noise
wheelBase = 4.5 #meters. "L" in KF state transition model.
hSym = 0 #symbolic function used to compute the predicted measurement from the predicted state. not needed if readings are taken directly
EKF = simpleCarKF.SimpleCarKF(Xi, Pi, hSym, Q, R)

'''Filter data'''
xKF = [] #estimate produced by EKF
yKF = [] #estimate produced by EKF
vKF = [] #estimate produced by EKF
thetaKF = [] #estimate produced by EKF
theta_dotKF = [] #estimate produced by EKF

for t in range(frameListLength):
    #Store EKF values
    currentStateEKF = EKF.getCurrentState()
    xKF.append(currentStateEKF.item(0))
    yKF.append(currentStateEKF.item(1))
    thetaKF.append(math.degrees(currentStateEKF.item(2)))

    vKF.append(currentStateEKF.item(3))
    theta_dotKF.append(math.degrees(currentStateEKF.item(4)))

    #Filter next step of data
    currentControlVector = [accInput[t], (steerAngleInput[t])]
    measurementVector = np.matrix([[XTruth[t]], [YTruth[t]], [math.radians(thetaTruth[t])], [velInput[t]], [math.radians(theta_dot[t])]])
    # print currentControlVector, measurementVector, wheelBase, timeStep[t]
    EKF.step(currentControlVector, measurementVector, wheelBase, timeStep[t])
    # print t

print 'theta truth', thetaTruth
print
print 'theta KF', thetaKF
'''Plotting'''
pltBool = 1
showImages = 0
saveImage = 1
if pltBool:
    plt.figure()
    plt.plot(XTruth, YTruth, 'b')
    plt.plot(XTruth[0], YTruth[0], 'go')
    plt.plot(XTruth[frameListLength-1], YTruth[frameListLength-1], 'ro')
    plt.plot(xKF, yKF, 'g')
    plt.plot(xKF[0], yKF[0], 'go')
    plt.plot(xKF[frameListLength-1], yKF[frameListLength-1], 'ro')
    title = 'GTA Position. Blue = Truth, Green = KF'
    plt.title(title)
    plt.grid()
    if saveImage:
        print 'saving image...'
        plt.savefig(title + '.png')

    plt.figure()
    plt.plot(time, XTruth, 'b')
    plt.plot(time, xKF, 'g')
    title = 'GTA X. Truth = Blue, KF = Green'
    plt.title(title)
    plt.grid()
    if saveImage:
        print 'saving image...'
        plt.savefig(title + '.png')


    plt.figure()
    plt.plot(time, YTruth, 'b')
    plt.plot(time, yKF, 'g')
    title = 'GTA Y. Truth = Blue, KF = Green'
    plt.title(title)
    plt.grid()
    if saveImage:
        print 'saving image...'
        plt.savefig(title + '.png')


    plt.figure()
    plt.plot(time, thetaTruth, 'b', time, thetaKF, 'g')
    title = 'GTA theta over time. Truth = Blue, KF = Green'
    plt.title(title)
    plt.grid()
    if saveImage:
        print 'saving image...'
        plt.savefig(title + '.png')


    plt.figure()
    plt.plot(time, velInput, 'b', time, vKF, 'g')
    title = 'GTA velocity over time. Truth = Blue, KF = Green'
    plt.title(title)
    plt.grid()
    if saveImage:
        print 'saving image...'
        plt.savefig(title + '.png')


    # plt.figure()
    # plt.plot(time, accInput, 'b')
    # plt.title('GTA computed acceleration over time')
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(time, steerAngleIn   put, 'b')
    # plt.title('GTA steering angle over time')
    # plt.grid()

    plt.figure()
    plt.plot(time, theta_dot, 'b', time, theta_dotKF, 'g')
    title = 'GTA theta_dot over time. Truth = Blue, KF = Green'
    plt.title(title)
    plt.grid()
    if saveImage:
        print 'saving image...'
        plt.savefig(title + '.png')

    if showImages:
        plt.show()
