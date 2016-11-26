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
xmlList = os.listdir('gtaOutput/gtaOutput1')
xmlList.sort()
frameListLength = len(xmlList) #number of frames taken in GTA
XTruth = [] #meters
YTruth = [] #meters
thetaTruth = [] #degrees
velInput = [] #m/s
steerAngleInput = [] #degrees
timeStep = [] #Time from previous frame to current frame
for i in range(frameListLength):
    data = XML_Reader.getXMLInfo('gtaOutput/gtaOutput1/' + xmlList[i]) #[posX, posY, theta, vel, steerAngle, dt]
    XTruth.append(data[0])
    YTruth.append(data[1])
    thetaTruth.append(data[2])
    velInput.append(data[3])
    steerAngleInput.append(math.radians(data[4]))
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

'''Plotting'''
pltBool = 1
if pltBool:
    plt.figure()
    plt.plot(XTruth, YTruth, 'b')
    plt.plot(XTruth[0], YTruth[0], 'go')
    plt.plot(XTruth[frameListLength-1], YTruth[frameListLength-1], 'ro')
    plt.title('GTA Position Truth')
    plt.grid()

    plt.figure()
    plt.plot(time, XTruth, 'b', time, YTruth, 'g')
    plt.title('GTA X and Y over time. X = Blue, Y = Green')
    plt.grid()

    plt.figure()
    plt.plot(time, thetaTruth, 'b')
    plt.title('GTA theta over time')
    plt.grid()

    plt.figure()
    plt.plot(time, velInput, 'b')
    plt.title('GTA velocity over time')
    plt.grid()

    plt.figure()
    plt.plot(time, accInput, 'b')
    plt.title('GTA computed acceleration over time')
    plt.grid()

    plt.figure()
    plt.plot(time, steerAngleInput, 'b')
    plt.title('GTA steering angle over time')
    plt.grid()

    plt.figure()
    plt.plot(time, theta_dot, 'b')
    plt.title('GTA theta_dot over time')
    plt.grid()

    plt.show()
