import numpy as np

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        "x_k = F*x_k-1 + B*u_k + w_k"
        "y_k = H*x_k + v_k"
        self.F = F  # state transition matrix 
        self.B = B  # control input matrix
        self.H = H  # observation matrix
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance
        self.x = x0.astype(np.float64)  # initial state estimate
        self.P = P0.astype(np.float64)  # initial covariance matrix
        self.H_list = [] # initial observation matrix list
        self.J_list = [] # initial measurement matrix list
        self.i = 0 #initial counter to loop through the lists

    def get_matrixs(self, Y_list, J_list):
        #get measurements from the main script
        self.H_list = Y_list
        self.J_list = J_list
        # print ("HLIST",self.H_list)

    def predict(self, u=None):
        # state prediction
        # print ("X",self.x)
        # print ("F",self.F)
        self.x = self.F @ self.x
        if u is not None:
            self.x += self.B @ u
        # covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        # innovation
        z = y - self.H @ self.x
        # innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # state update
        self.x += K @ z
        # covariance update
        self.P -= K @ self.H @ self.P

    
    def nextH(self):
        #up dating the H matrix
        self.H = self.H_list[self.i]
        self.i += 1

    def run_kf(self, measurements, controls=None):
        # run Kalman filter on a sequence of measurements and optional controls
        for i, y in enumerate(measurements):
            if controls is not None:
                self.predict(u=controls[i])
            else:
                self.predict()
                self.nextH()
            self.update(y)
            yield self.x



