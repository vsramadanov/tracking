#!/usr/bin/env python3
"""

"""
import numpy as np
from numpy import pi
import argparse
import configparser
from importlib import import_module
import matplotlib.pyplot as plt

from simulation import Simulation


def get_from_module(name):
    loc = name.rfind('.')
    module_name, function_name = name[:loc], name[loc + 1:]
    module = import_module(module_name)

    return getattr(module, function_name)


def parse(expr):
    try:
        return eval(expr)
    except NameError:
        loc = expr.rfind('(')
        func = get_from_module(expr[:loc])
        return eval('func' + expr[loc:])


def get_config_args(config):
    return {key: parse(val) for key, val in config.items() if key != 'model'}


class Parameters:
    pass


config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='script for running simulation')
parser.add_argument('config', help='configuration file with filter description')

if __name__ == '__main__':

    args = parser.parse_args()

    config.read(args.config)

    pars = Parameters()
    for key, val in config['parameters'].items():
        setattr(pars, key, parse(val))

    time = np.arange(pars.start, pars.end, pars.step)

    trajectory_model = get_from_module(config['trajectory']['model'])
    trajectory_args = get_config_args(config['trajectory'])
    trajectory_generator = trajectory_model(**trajectory_args)

    filter_model = get_from_module(config['filter']['model'])
    filter_args = get_config_args(config['filter'])
    kalman_filter = filter_model(**filter_args)

    sensor_model = get_from_module(config['sensor']['model'])
    sensor_args = get_config_args(config['sensor'])
    sensor = sensor_model(**sensor_args)

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
