#!/usr/bin/env python3
"""

"""

import numpy as np
from functools import cached_property

import kalman.models


class TrajectoryGenerator:
    """Class description"""

    def __init__(self, time_series, x_initial, w):
        """Initialize class"""
        assert time_series.shape == w.shape
        assert x_initial.shape == (4,)

        self.w = w
        self.x_initial = x_initial
        self.time_series = time_series
        self.n = len(time_series)

    @cached_property
    def trajectory(self):
        """Generates required trajectory"""
        F = lambda wk, T: kalman.models.linear(T, 2, 2) if wk == 0 else kalman.models.constant_turn_2d(T, wk)

        w, time_series = self.w, self.time_series
        x = np.zeros(shape=(4, len(time_series)))
        x[:, 0] = self.x_initial

        for k in range(1, self.n):
            T = time_series[k] - time_series[k - 1]
            x[:, k] = F(w[k], T) @ x[:, k - 1]

        return x
