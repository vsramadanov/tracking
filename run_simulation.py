#!/usr/bin/env python3
"""

"""
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import kalman.models
from simulation import Simulation
from kalman.core import KF
from trajectory.generators import TrajectoryGenerator
from sensors.gnss import GNSS

# Define target trajectory
T = 1
Tmax = 100
time = np.arange(0, Tmax, T)
vel = 2
rms = 20

x_init = np.array([0, 0, vel, vel])
w = np.where(time < 50, 0, 2*pi/100)

trajectory_generator = TrajectoryGenerator(time_series=time, x_initial=x_init, w=w)

# Define trajectory filter
P = np.array([
    [rms**2, 0, 0, 0],
    [0, rms**2, 0, 0],
    [0, 0, 1e4, 0],
    [0, 0, 0, 1e4],
])

Q = 0 * np.eye(4)
R = rms**2 * np.eye(2)
F = kalman.models.linear(T=T, depth=2, dim=2)

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])

kalman_filter = KF(transition_matrix=F,
                   covariance_matrix=P,
                   observation_matrix=H,
                   process_covariance=Q,
                   observation_covariance=R)

sensor = GNSS(drop_velocity=True)

sim = Simulation(time=time,
                 trajectory_generator=trajectory_generator,
                 trajectory_filter=kalman_filter,
                 sensor=sensor)

if __name__ == '__main__':
    sim.run()

    x_true = sim.trajectory_generator.trajectory[0, :]
    y_true = sim.trajectory_generator.trajectory[1, :]

    y_x = sim.observation[0, :]
    y_y = sim.observation[1, :]

    x_hat = sim.estimation[0, :]
    y_hat = sim.estimation[1, :]

    plt.plot(x_true, y_true)
    plt.plot(y_x, y_y)
    plt.plot(x_hat, y_hat)

    plt.axis('equal')
    plt.show()
