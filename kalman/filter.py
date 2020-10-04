#!/usr/bin/env python3
"""
This file contains different variations of the Kalman Filter algorithm
__filter - base class, all filters should be inherited from this class
KF  - implements the usual linear Kalman filter
EKF - implements the extended Kalman filter
"""

import numpy as np


class __filter:

    def __init__(self, covariance_matrix, process_covariance, observation_covariance, initial_state=None):
        """Construct base filter"""
        self.P = covariance_matrix
        self.Q = process_covariance
        self.R = observation_covariance

        self.x = initial_state


class KF(__filter):

    def __init__(self, transition_matrix, observation_matrix, covariance_matrix, process_covariance,
                 observation_covariance, initial_state=None):
        """Constructor of the Kalman filter"""
        super(KF, self).__init__(covariance_matrix, process_covariance, observation_covariance, initial_state)
        
        self.F = transition_matrix
        self.H = observation_matrix

    def init(self, initial_state):
        self.x = initial_state

    def update(self, y):
        """Implements Kalman filter step"""
        F, H, P, Q, R = self.F, self.H, self.P, self.Q, self.R
        x = self.x

        x_est = F @ x
        P = F @ P @ F.T + Q

        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x_hat = x_est + K @ (y - H @ x_est)
        tmp = K @ H
        P = (np.eye(tmp.shape[0]) - tmp) @ P

        self.x = x_hat
        self.P = P


class EKF(__filter):

    def __init__(self, transition, observation, covariance_matrix, process_covariance,
                 observation_covariance, initial_state=None):
        """Constructor of the Extended Kalman filter"""
        super(EKF, self).__init__(covariance_matrix, process_covariance, observation_covariance, initial_state)
        self.F, self.Jf = transition
        self.H, self.Jh = observation

    def init(self, initial_state):
        self.x = initial_state

    def update(self, y):
        """Implements Extended Kalman filter step"""
        F, H, P, Q, R = self.F, self.H, self.P, self.Q, self.R
        x = self.x

        x_est = F(x)
        Fk = self.Jf(x_est)
        P = Fk @ P @ Fk.T + Q

        Hk = self.Jh(x_est)
        K = P @ Hk.T @ np.linalg.inv(Hk @ P @ Hk.T + R)
        x_hat = x_est + K @ (y - H @ x_est)
        tmp = K @ Hk
        P = (np.eye(tmp.shape[0]) - tmp) @ P

        self.x = x_hat
        self.P = P
