#!/usr/bin/env python3
"""
This file decsribes the linear Kalman Filter algorithm
"""

import numpy as np


class KF:

    def __init__(self, transition_matrix, observation_matrix, covariance_matrix, process_covariance,
                 observation_covariance, initial_state=None):
        """Constructor of the Kalman filter"""
        n = transition_matrix.shape[0]
        m = observation_matrix.shape[0]

        assert transition_matrix.shape == (n, n)
        assert observation_matrix.shape == (m, n)
        assert observation_covariance.shape == (m, m)
        assert covariance_matrix.shape == (n, n)
        assert process_covariance.shape == (n, n)

        self.F = transition_matrix
        self.H = observation_matrix
        self.P = covariance_matrix
        self.Q = process_covariance
        self.R = observation_covariance

        self.x = initial_state

    def init(self, initial_state):
        self.x = initial_state

    def update(self, y):
        """Implements Kalman filter logic"""
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
