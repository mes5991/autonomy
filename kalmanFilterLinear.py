import numpy as np

class KalmanFilterLinear:
    def __init__(self, A, B, H, X, P, Q, R):
        self.A = A #State transition matrix
        self.B = B #Control matrix
        self.H = H #Observation matrix
        self.X = X #Initial state estimate
        self.P = P #Initial covariance matrix
        self.Q = Q #Estimated error in process
        self.R = R #Estimated error in measurements

    def getCurrentState(self):
        return self.X

    def step(self, controlVector, measurementVector):
        """PREDICTION"""
        predicted_X = (self.A * self.X) + (self.B * controlVector)
        predicted_P = (self.A * self.P * np.transpose(self.A)) + self.Q

        """OBSERVATION"""
        y = measurementVector - self.H * self.X
        S = (self.H * predicted_P * np.transpose(self.H)) + self.R

        """UPDATESTEP"""
        K = predicted_P * np.transpose(self.H) * np.linalg.inv(S)
        self.X = predicted_X + K * y
        size = self.P.shape[0]
        self.P = (np.eye(size) - (K * self.H)) * predicted_P
