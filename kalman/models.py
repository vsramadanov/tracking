"""
File contains different state transition models
"""
import numpy as np
import math
from numpy import concatenate as cat


def __generate_linear_row(T, depth, zeros, dim):
    """Generates linear state transition matrix rows"""
    core = np.eye(dim)
    if depth == 1:
        return cat((np.zeros((dim, dim * zeros)), core), axis=1)
    depth = depth - 1
    return cat((__generate_linear_row(T, depth, zeros, dim),
                T ** depth / math.factorial(depth) * core,), axis=1)


def __generate_linear_cols(T, depth, target, dim):
    """Carefully concatenates linear state transition matrix rows"""
    if depth == 1:
        return __generate_linear_row(T, depth, target, dim)
    return cat((__generate_linear_row(T, depth, target, dim),
                __generate_linear_cols(T, depth - 1, target + 1, dim)), axis=0)


def linear(T, depth, dim):
    """Returns transition state matrix for linear model"""
    return __generate_linear_cols(T, depth, 0, dim)


def constant_turn_2d(T, wk):
    """Returns 2d state transition matrix with constant turn wk [rad/sec]"""
    return np.array([
        [1, 0, np.sin(wk * T) / wk, (np.cos(wk * T) - 1) / wk],
        [0, 1, (1 - np.cos(wk * T)) / wk, np.sin(wk * T) / wk],
        [0, 0, np.cos(wk * T), -np.sin(wk * T)],
        [0, 0, np.sin(wk * T), np.cos(wk * T)],
    ])
