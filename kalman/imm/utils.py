"""
TODO: file description
"""

from kalman.core import KF


def generate_linear_filters(transition_matrices, observation_matrix, covariance_matrix,
                            process_covariance, observation_covariance):
    """Returns a bunch of Kalman filters with different transition state matrix"""
    H, P, Q, R = observation_matrix, covariance_matrix, process_covariance, observation_covariance
    filter_parameters = [{'transition_matrix': F,
                          'observation_matrix': H,
                          'covariance_matrix': P,
                          'process_covariance': Q,
                          'observation_covariance': R
                          } for F in transition_matrices]
    return [KF(**args) for args in filter_parameters]
