import math
import warnings

import numpy as np
import torch

from .geometry import cartesian_to_spherical


def rand_sphere(size):
    """
    Randomly sample on the sphere uniformly
    See http://corysimon.github.io/articles/uniformdistn-on-sphere/
    Our convention is 0 < theta < pi ; -pi < phi < pi
    :param size: torch.size or list of ints, size of returned tensors
    :return: theta, phi
    """
    theta = torch.acos(1 - 2 * torch.rand(size))
    phi = 2 * math.pi * torch.rand(size) - math.pi
    return theta, phi


def fibonacci_sphere(samples: int = 1000):
    """
    Evenly sample on the sphere
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples: size of the array sampled
    :return : [samples, 3], torch.float, X, Y, and Z coordinates of the points in Euclidean space
    """
    increment = np.linspace(0, samples - 1, samples)
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    y = 1 - (increment / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y

    theta = phi * increment  # golden angle increment

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack([x, y, z], axis=1)


def rejection_sphere_sampling(pdf, n: int = 1000, device=None, iter_max: int = 100):
    """
    Sample from a probability density function using rejection sampling
    See https://en.wikipedia.org/wiki/Rejection_sampling
    :param pdf: a function that takes as input a 1D vector theta and a 1D vector phi and return their densities
    :param n: number of samples
    :param device: torch.device
    :param iter_max: This algorithm can fail when the density function has big spikes,
        iter_max is the maximal number of iterations allowed
    :return: torch.float, [n, 2], theta and phi arrays
    """
    warnings.warn('rejection_sphere_sampling is a very basic loop, and can be very slow when the pdf function is' +
                  ' slow or has big spikes')
    if device is None:
        device = 'cpu'

    # Approximate upper bound value of the pdf
    xyz = torch.tensor(fibonacci_sphere(samples=5000), device=device)
    _, theta, phi = cartesian_to_spherical(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    p_max = pdf(theta=theta, phi=phi).max()

    sample = torch.empty([0, 2], dtype=torch.float, device=device)
    n_iters = 0
    while sample.shape[0] < n:
        if n_iters > iter_max:
            raise RuntimeError('rejection_sphere_sampling did not succeed in time. Try to increase iter_max')
        # Rejection sampling
        y = torch.rand([2, n], device=device)  # Uniform distribution to create initial distribution
        theta = torch.acos(1 - 2 * y[0])
        phi = 2 * math.pi * y[1] - math.pi
        u = torch.rand([n], device=device)  # Uniform distribution to select valid and invalid samples

        valid = u < pdf(theta=theta, phi=phi) / p_max
        sample = torch.cat([sample,
                            torch.stack([theta[valid], phi[valid]], dim=1)],
                           dim=0)
        n_iters += 1

    sample = sample[:n]
    return sample
