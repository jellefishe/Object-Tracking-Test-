import numpy as np
from pdb import set_trace

class KalmanFilter():
    # Equations from https://towardsdatascience.com/an-intro-to-kalman-filters-for-autonomous-vehicles-f43dd2e2004b
    def __init__(self):

        # State Matrix
        self.X = np.array([[0], [0], [0], [0]])

        # State Transition Matrix
        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Process Covariance Matrix
        self.P = np.eye(self.A.shape[1])

        # B Matrix applies acceleration
        self.B = 0

        # Acceleration
        self.u = 0

        # error term
        self.w = 0

        # error term
        self.Q = np.eye(self.A.shape[1])
        
        # H matrix transforms format of P into format of K
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurement Noise
        self.R = np.eye(self.H.shape[0])

    def correct(self, Z):
        # calculate Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv((np.dot(np.dot(self.H, self.P), self.H.T)) + self.R))

        # calculate updated P
        self.P = np.dot(np.eye(self.A.shape[1]) - np.dot(K, self.H), self.P)

        # calculate updated X
        self.X = self.X + np.dot(K, (np.reshape(Z,(2,1)) - np.dot(self.H, self.X)))

        return self.X

    def predict(self):
        # calculate predicted x
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.u) + self.w

        # calculate predicted P
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.X
    