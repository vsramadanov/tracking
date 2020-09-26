#!/usr/bin/env python3
"""

"""

import numpy as np
from functools import cached_property


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
        Fcirc = lambda wk, T: np.array([
            [1, 0, np.sin(wk * T) / wk, (np.cos(wk * T) - 1) / wk],
            [0, 1, (1 - np.cos(wk * T)) / wk, np.sin(wk * T) / wk],
            [0, 0, np.cos(wk * T), -np.sin(wk * T)],
            [0, 0, np.sin(wk * T), np.cos(wk * T)],
        ])
        Fst = lambda T: np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        F = lambda wk, T: Fst(T) if wk == 0 else Fcirc(wk, T)

        w, time_series = self.w, self.time_series
        x = np.zeros(shape=(4, len(time_series)))
        x[:, 0] = self.x_initial

        for k in range(1, self.n):
            T = time_series[k] - time_series[k - 1]
            x[:, k] = F(w[k], T) @ x[:, k - 1]

        return x
