#!/usr/bin/env python3
"""

"""
import numpy as np
from numpy import concatenate as cat


class Simulation:
    """Class decription"""

    def __init__(self, time, trajectory_generator, trajectory_filter, sensor):
        """initialize simulation scene"""
        self.time = time

        self.trajectory_generator = trajectory_generator
        self.filter = trajectory_filter
        self.sensor = sensor
        self.estimation = None
        self.observation = None

    def run(self):
        trajectory = self.trajectory_generator.trajectory
        observation = self.sensor.observe(trajectory)
        time = self.time

        estimation = np.zeros(shape=(4, len(time)))
        estimation[:, 0] = cat((observation[:, 0], np.zeros(2)), axis=0)
        self.filter.init(estimation[:, 0])

        for k in range(1, len(time)):
            y = observation[:, k]
            self.filter.update(y)

            estimation[:, k] = self.filter.x

        self.estimation = estimation
        self.observation = observation
