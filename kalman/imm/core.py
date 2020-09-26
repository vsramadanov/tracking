"""
TODO: File description
"""
import numpy as np


class IMM_KF:
    """Implements Interactive Multiply Model Kalman filter"""

    def __init__(self, models, transition_probabilities, weights=None, initial_state=None):
        """Initialize IMM KF with parameters"""
        assert transition_probabilities.shape == (len(models), len(models))
        assert len(weights) == len(models) or weights is None
        if weights is None:
            weights = np.ones(len(models)) / len(models)

        self.models = models
        self.N = len(models)
        self.W = weights
        self.mu = transition_probabilities

        self.x = initial_state

    def init(self, x_initial):
        for model in self.models:
            model.init(initial_state=x_initial)
        self.x = x_initial

    def update(self, y):
        """Updates IMM KF state with new observation"""
        Wk = np.sum(self.mu * self.W, axis=0)
        Wks = self.mu * np.tile(self.W, [self.N, 1]) / np.tile(Wk, [self.N, 1]).T

        x = np.zeros((4, self.N))
        P = np.zeros((4, 4, self.N))
        for s in range(self.N):
            for k, model in enumerate(self.models):
                x[:, s] = x[:, s] + model.x * Wks[k, s]
                P[:, :, s] = P[:, :, s] + (model.P + (x[:, s] - model.x) @
                                           (x[:, s] - model.x).T) * Wks[k, s]

        for model in self.models:
            model.update(y)


