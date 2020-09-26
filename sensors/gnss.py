"""

"""
import numpy as np


class GNSS:
    """TODO: Describe class"""
    def __init__(self, position_rms=10, velocity_rms=1, drop_velocity=False):
        """Init GNSS sensor"""
        self.position_rms = position_rms
        self.velocity_rms = velocity_rms
        self.drop_velocity = drop_velocity

    def observe(self, trajectory):
        """Corrupt trajectory using white gaussian noise"""
        N = trajectory.shape[1]
        if self.drop_velocity:
            return trajectory[:2, :] + self.position_rms * np.random.randn(2, N)
        else:
            raise NotImplemented
