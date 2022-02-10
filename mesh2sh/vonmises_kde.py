# Adaptation from https://github.com/williamjameshandley/spherical_kde
# For the rule of thumb : https://arxiv.org/pdf/1306.0517.pdf
# Exact risk improvement of bandwidth selectors for kernel density estimation with directional data
# Eduardo García-Portugués
import math

import scipy.optimize
import scipy.special
import torch

from .geometry import spherical_to_cartesian


def logsinh(x):
    """
    Compute log(sinh(x)), stably for large x.
    :param x : torch.tensor, argument to evaluate at, must be positive
    :return torch.tensor, log(sinh(x))
    """
    if torch.any(x < 0):
        raise ValueError("logsinh only valid for positive arguments")
    return x + torch.log(0.5 - torch.exp(-2 * x) / 2)


def maxlikelihood_kappa(data) -> float:
    """
    Estimate kappa if data follows a Von Mises Fisher distribution
    https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Estimation_of_parameters
    :param data : torch.tensor, [Nx3], xyz coordinates of points on the sphere
    :return float
    """
    r = data.mean(dim=0).square().sum(0).sqrt().item()

    def a_p(kappa):
        return scipy.special.iv(3 / 2, kappa) / scipy.special.iv(3 / 2 - 1, kappa) - r

    opti_kappa = scipy.optimize.brentq(a_p, 1e-4, 1e2)
    return opti_kappa


def h_rot(data):
    """
    Rule-of-thumb bandwidth hrot for Kernel Density Estimation using Von Mises Fisher kernels
    Typo in the original paper, see : https://github.com/egarpor/DirStats/blob/master/R/bw-pi.R
    :param data : torch.tensor, [Nx3], xyz coordinates of points on the sphere
    :return float
    """
    kappa = maxlikelihood_kappa(data)
    n = data.shape[0]

    num = 8 * math.sinh(kappa) ** 2
    den = (-2 * kappa * math.cosh(2 * kappa) + (1 + 4 * kappa ** 2) * math.sinh(2 * kappa)) * n
    return (num / den) ** (1 / 6)


class SphereKDE:
    """
    Spherical kernel density estimator, using Von Mises Fisher kernels
    Inspired by https://github.com/williamjameshandley/spherical_kde
    """

    def __init__(self, data_pts, chunk_matmul=10000):
        self.pts = data_pts
        self.device = data_pts.device
        self.chunk_matmul = chunk_matmul

    def __call__(self, sampling_pts, bandwidth):
        sampling = sampling_pts.to(self.device)
        kappa = torch.tensor(1 / (bandwidth ** 2), device=self.device)
        logc = torch.log(kappa / (4 * math.pi)) - logsinh(kappa)
        # kernels = logc + torch.matmul(sampling, self.pts.T) * kappa
        # pdf = torch.exp(torch.logsumexp(kernels, dim=1)) / self.pts.shape[0]
        pdf = torch.empty([sampling.shape[0]], device=sampling.device, dtype=sampling.dtype)
        for i in range(sampling.shape[0] // self.chunk_matmul + 1):  # Solve memory limitations
            chk_sampling = sampling[i * self.chunk_matmul:(i + 1) * self.chunk_matmul]
            kernels = logc + torch.matmul(chk_sampling, self.pts.T) * kappa
            pdf[i * self.chunk_matmul:(i + 1) * self.chunk_matmul] = \
                torch.exp(torch.logsumexp(kernels, dim=1)) / self.pts.shape[0]
        return pdf


def vonmisesfisher_kde(data_theta, data_phi, x_theta, x_phi, bandwidth=None):
    """
    Perform Von Mises-Fisher Kernel Density Estimation (used for spherical data)
    :param data_theta: 1D torch.float tensor, containing theta values for training data, between 0 and pi
    :param data_phi: 1D torch.float tensor, containing phi values for training data, between 0 and 2*pi
    :param x_theta: torch.float tensor, containing theta values for sampling points, between 0 and pi
    :param x_phi: torch.float tensor, containing phi values for sampling points, between 0 and 2*pi
    :param bandwidth: smoothing bandwith. If None, then uses rule-of-thumb
    :return: torch tensor of interpolated values, of the same shape as x_theta and x_phi
    """
    data_pts = torch.stack(spherical_to_cartesian(torch.ones_like(data_theta), data_theta, data_phi), dim=1)
    if bandwidth is None:
        bandwidth = h_rot(data_pts)
    assert x_theta.shape == x_phi.shape
    shape = x_theta.shape
    x_theta, x_phi = x_theta.flatten(), x_phi.flatten()
    sampling_pts = torch.stack(spherical_to_cartesian(torch.ones_like(x_theta), x_theta, x_phi), dim=1)
    pdf = SphereKDE(data_pts)(sampling_pts, bandwidth=bandwidth)
    return pdf.view(shape)
