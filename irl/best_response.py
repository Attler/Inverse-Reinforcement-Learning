"""
Find the best response policy for H with respect to R
Based on Hadfield-Menell et al. 2016

"""

from . import value_iteration
from . import maxent


def find_best_response(n_states, n_actions, transition_probabilities, reward, discount, feature_matrix,
                       threshold=1e-2, v=None, stochastic=True):

    # TODO
    """
    Find the best response policy under ACIRL

    :param n_states:
    :param n_actions:
    :param transition_probabilities:
    :param reward:
    :param discount:
    :param threshold:
    :param v:
    :param stochastic:
    :return:
    """

    print("Finding best response for H")

    expert_policy = find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=threshold, v=v, stochastic=stochastic)

    trajectories = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s])

    feature_expectations = maxent.find_feature_expectations(feature_matrix,
                                                     trajectories)

    expected_svf = maxent.find_expected_svf(n_states, r, n_actions, discount,
                                     transition_probability, trajectories)

    grad = feature_expectations - feature_matrix.T.dot(expected_svf)


    features_theta = maxent.find_expected_svf(n_states, reward, n_actions, discount, transition_probabilities, trajectories)





    return expert_policy