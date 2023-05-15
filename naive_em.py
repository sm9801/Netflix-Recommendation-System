"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    # class GaussianMixture(NamedTuple):
    #     """Tuple holding a gaussian mixture"""
    #     mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    #     var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    #     p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component
    K = mixture.mu.shape[0]
    X_row, d = X.shape
    posteriors = np.empty([X_row, K])
    likelihoods = np.empty([X_row, K])

    def calculate_marginal(x) :
        marginal = 0

        for row in range(K) :

            coefficient = 1 / ((2 * np.pi * mixture.var[row]) ** (d / 2))
            e = np.e ** ((1 / (-2 * mixture.var[row])) * (np.linalg.norm(x - mixture.mu[row]) ** 2))

            marginal += mixture.p[row] * coefficient * e

        return marginal

    for row in range(X_row) :

        for j in range(K) :

            p_j = mixture.p[j]
            mu_j = mixture.mu[j]
            var_j = mixture.var[j]

            coefficient = 1 / ((2 * np.pi * var_j) ** (d / 2))
            e = np.e ** ((1 / (-2 * var_j)) * (np.linalg.norm(X[row] - mu_j) ** 2))
            numerator = p_j * coefficient * e
            denominator = calculate_marginal(X[row])

            posterior = numerator / denominator

            posteriors[row, j] = posterior
            likelihoods[row, j] = numerator

    LL = np.sum(np.log(np.sum(likelihoods, axis = 1)))

    return [posteriors, LL]
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # """Tuple holding a gaussian mixture"""
    # mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    # var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    # p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component

    # n, d = X.shape
    #
    # n_hat = np.sum(post, axis=0)
    # p_hat = n_hat / n
    #
    # sum_px = np.matmul(np.transpose(post), X)
    #
    # mu_hat = (1 / n_hat) * sum_px
    # norm_xmu = np.linalg.norm(X - mu_hat) ** 2
    # sigma_hat = (1 / (n_hat * d)) * np.sum(np.matmul(post, norm_xmu))
    #
    # return GaussianMixture(mu_hat, sigma_hat, p_hat)

    n, d = X.shape

    n_hat = np.sum(post, axis = 0)
    p_hat = n_hat / n
    mu_hat = (1 / n_hat.reshape(-1, 1)) * np.matmul(np.transpose(post), X)
    norm_xmu = np.linalg.norm(X[:, None] - mu_hat, ord = 2, axis = 2) ** 2
    sigma_hat = (1 / (n_hat * d)) * np.sum(post * norm_xmu, axis = 0)

    return GaussianMixture(mu_hat, sigma_hat, p_hat)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    log_old = 0
    log_new = 0

    while log_old == 0 or ((log_new - log_old) > (10 ** (-6)) * np.abs(log_new)) :

        log_old = log_new

        post, log_new = estep(X, mixture)

        mixture = mstep(X, post, mixture)

    return mixture, post, log_new
    raise NotImplementedError
