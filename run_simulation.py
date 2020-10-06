#!/usr/bin/env python3
"""

"""
import numpy as np
from numpy import pi
import argparse
import configparser
from importlib import import_module
import matplotlib.pyplot as plt

import kalman
from trajectory.generators import TrajectoryGenerator
from simulation import Simulation
from sensors.gnss import GNSS

# Define target trajectory
T = 1
Tmax = 100
time = np.arange(0, Tmax, T)
vel = 2
rms = 20

x_init = np.array([0, 0, vel, vel])
w = np.where(time < 50, 0, 2 * pi / 100)

trajectory_generator = TrajectoryGenerator(time_series=time, x_initial=x_init, w=w)
sensor = GNSS(drop_velocity=True)


def get_filter_model(name):

    loc = name.rfind('.')
    filter_lib_name, filter_name = name[:loc], name[loc + 1:]
    filter_lib = import_module(filter_lib_name)

    return getattr(filter_lib, filter_name)


config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='script for running simulation')
parser.add_argument('config', help='configuration file with filter description')


if __name__ == '__main__':

    args = parser.parse_args()

    config.read(args.config)

    filter_model = get_filter_model(config['filter']['model'])
    filter_args = {key: eval(val) for key, val in config['filter'].items() if key != 'model'}
    kalman_filter = filter_model(**filter_args)

    sim = Simulation(time=time,
                     trajectory_generator=trajectory_generator,
                     trajectory_filter=kalman_filter,
                     sensor=sensor)
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
