import numpy as np


def generate_gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    kernel /= np.max(kernel)
    return kernel


def create_bins(num_bins_per_obs=10):
    bins_contrast = np.linspace(0, 1.0, num_bins_per_obs)
    bins_frequency = np.linspace(0, 1.0, num_bins_per_obs)

    bins = np.array([
        bins_contrast,
        bins_frequency
    ])

    return bins


def discretize_observation(observations, bins):
    binned_observations = []

    for idx, observation in enumerate(observations):
        bin_index = 0

        while bin_index < len(bins[idx]) and observation >= bins[idx][bin_index]:
            bin_index += 1

        discrete_observation = bin_index - 1
        binned_observations.append(discrete_observation)

    return tuple(binned_observations)
