# For spherical harmonics computation, we use pyshtools :
# https://shtools.github.io/SHTOOLS/index.html
import math

import numpy as np
import torch
from pyshtools.expand import MakeGridPoint, GLQGridCoord, SHExpandGLQ, SHGLQ
from scipy.interpolate import RBFInterpolator

from .geometry import spherical_to_cartesian
from .vonmises_kde import vonmisesfisher_kde

_NORM, _CSPHASE = 1, -1


# Almost same as MakeGridPoint, in pyshtools v4.9.1 optional arguments must be np.array, here it doesn't need to
def _sample_points(cilm, lat, lon, lmax, norm=1, csphase=1, dealloc=0):
    return np.vectorize(MakeGridPoint.pyfunc, excluded=[0, 3])(cilm, lat, lon, [lmax, norm, csphase, dealloc])


def mapping_shtools_to_compact(lmax):
    """
    pyshtools uses an output format to represent spherical harmonic coefficients that is not memory efficient.
    This function creates a mapping to represent the coefficients differently.

    Our representation Y(l, m), shape(lmax+1, lmax+1) :
        Y(0, 0) Y(1, 1) Y(2, 2) ...
        Y(1,-1) Y(1, 0) Y(2, 1) ...
        Y(2,-2) Y(2,-1) Y(2, 0) ...
        ...     ...     ...

    Example :
        mapping = mapping_shtools_to_compact(lmax)
        x, _ = SHExpandLSQ(d, phi, theta, lmax, [1, -1])  # pyshtools
        y = torch.tensor(x)[mapping[..., 0], mapping[..., 1], mapping[..., 2]]  # Compact
        z = torch.zeros([2, lmax+1, lmax+1])
        z[mapping[..., 0], mapping[..., 1], mapping[..., 2]] = y  # Back to pyshtools
    """
    mapping = torch.zeros([lmax + 1, lmax + 1, 3], dtype=torch.long)
    mapping[..., 0] = torch.tril(torch.ones([lmax + 1, lmax + 1], dtype=torch.long)) - torch.eye(lmax + 1,
                                                                                                 dtype=torch.long)
    linspace = torch.linspace(0, lmax, lmax + 1, dtype=torch.long)
    mapping[..., 1] = torch.triu(linspace.view(1, -1).expand(lmax + 1, lmax + 1)) \
        + torch.tril(linspace.view(-1, 1).expand(lmax + 1, lmax + 1) - torch.diag(linspace))
    mapping[..., 2] = torch.abs(linspace.view(1, -1).expand(lmax + 1, lmax + 1) -
                                linspace.view(-1, 1).expand(lmax + 1, lmax + 1))
    return mapping


def interpolate_on_grid(theta, phi, grid_theta, grid_phi, values, kernel='multiquadric',
                        smoothing=0., degree=None):
    """
    Interpolate values using Scipy's RBFInterpolator
    """
    x, y, z = spherical_to_cartesian(r=torch.ones_like(phi), theta=theta, phi=phi)
    pts = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).cpu().numpy()
    gx, gy, gz = spherical_to_cartesian(r=torch.ones_like(grid_phi), theta=grid_theta, phi=grid_phi)
    target = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1).cpu().numpy()
    data = values.flatten().cpu().numpy()
    interp_values = RBFInterpolator(pts, data, smoothing=smoothing, kernel=kernel,
                                    degree=degree)(target)
    interp_values = interp_values.reshape(*gx.shape)
    return interp_values


def approximate_sh_coefficients(values, theta, phi, lmax: int,
                                smoothing: float = 1e-7, kernel: str = 'quintic', degree: int = 2):
    """
    From V (values) data of dim K measured on the sphere on the V points (theta, phi),
    this returns the spherical harmonics coefficients that approximate the
    best the data at degree lmax, for each dimension K.
    :param values : torch.tensor, [V, K], the values to be approximated
    :param theta : torch.tensor, [V], polar angle between 0 and pi
    :param phi : torch.tensor, [V], azimuthal angle between -pi and pi
    :param lmax : integer, the maximum spherical harmonic degree of the output coefficients
    :param smoothing: Smoothing coefficient for interpolation, see Scipy's RBFInterpolator
    :param kernel: Kernel of interpolation, see Scipy's RBFInterpolator
    :param degree: Degree of interpolation, see Scipy's RBFInterpolator
    :return torch.float, shape [K, lmax+1, lmax+1]
        SH coefficients Y(l, m), shape(lmax+1, lmax+1) :
            Y(0, 0) Y(1, 1) Y(2, 2) ... Y(lmax, lmax)
            Y(1,-1) Y(1, 0) Y(2, 1) ...
            Y(2,-2) Y(2,-1) Y(2, 0) ...
            ...     ...     ...     ...
            Y(lmax, -lmax)              Y(lmax, 0)
    """
    device = theta.device
    assert theta.shape[0] == phi.shape[0] == values.shape[0], f"Need the same number of points, " \
                                                              f"got {theta.shape[0]}, {phi.shape[0]}, {values.shape[0]}"
    assert len(theta.shape) == 1 and len(phi.shape) == 1, f"Theta and phi must be 1D, got theta: {len(theta.shape)} " \
                                                          f"and phi: {len(phi.shape)}"

    # Grid in our standard for the interpolation
    latglq, longlq = GLQGridCoord(lmax, [False])
    grid_theta = torch.deg2rad(torch.tensor(latglq + 90.))
    grid_phi = torch.deg2rad(torch.tensor(longlq)) - math.pi
    st, sp = grid_theta.shape[0], grid_phi.shape[0]
    grid_theta = grid_theta.view(-1, 1).expand(st, sp).float().to(device)
    grid_phi = grid_phi.view(1, -1).expand(st, sp).float().to(device)

    # Original theta and phi in our standard for interpolation
    theta = theta.view(-1).float()
    phi = phi.view(-1).float()

    sph_harm = torch.zeros([values.shape[1], lmax + 1, lmax + 1], device=device)
    mapping = mapping_shtools_to_compact(lmax).to(device)
    for k in range(values.shape[1]):
        d = interpolate_on_grid(theta=theta, phi=phi, grid_theta=grid_theta, grid_phi=grid_phi, values=values[:, k],
                                kernel=kernel, smoothing=smoothing, degree=degree)
        zero, w = SHGLQ(lmax)
        cilm = SHExpandGLQ(d, w, zero, [_NORM, _CSPHASE, lmax])
        sph_harm[k] = torch.tensor(cilm, device=device)[mapping[..., 0], mapping[..., 1], mapping[..., 2]]

    return sph_harm


def sample_sh_values(sh_coeffs, theta, phi):
    """
    From K grids of spherical harmonics coefficients, this function evaluate
    the value of the spherical harmonics function at the positions (theta, phi)
    :param sh_coeffs : torch.tensor, [K, Lmax, Lmax], the SH coefficients
    :param theta : torch.tensor, [V], polar angle between 0 and pi
    :param phi : torch.tensor, [V], azimuthal angle between -pi and pi
    :return torch.float, shape [V, K]
    """
    device = sh_coeffs.device
    theta = torch.rad2deg(theta) - 90.  # Theta between -90 and 90 degrees
    phi = torch.rad2deg(phi + math.pi)  # Phi between 0 and 360 degrees
    assert theta.numel() == phi.numel(), f"Need the same number of points, got {theta.shape}, {phi.shape}"
    assert len(theta.shape) == 1 and len(phi.shape) == 1, f"Theta and phi must be 1D, got theta: {len(theta.shape)} " \
                                                          f"and phi: {len(phi.shape)}"

    theta = theta.view(-1).float().cpu().numpy()
    phi = phi.view(-1).float().cpu().numpy()
    lmax = sh_coeffs.shape[1] - 1
    mapping = mapping_shtools_to_compact(lmax).numpy()
    sh_coeffs = sh_coeffs.cpu().numpy()

    values = np.empty([len(theta), sh_coeffs.shape[0]], dtype=float)
    for k in range(sh_coeffs.shape[0]):
        cilm = np.zeros([2, lmax + 1, lmax + 1], dtype=sh_coeffs.dtype)
        cilm[mapping[..., 0], mapping[..., 1], mapping[..., 2]] = sh_coeffs[k]
        # See https://shtools.github.io/SHTOOLS/pymakegridpoint.html#parameters
        x = _sample_points(cilm, theta, phi, lmax, _NORM, _CSPHASE, 0)
        values[..., k] = x

    values = torch.from_numpy(values).float().to(device=device)
    return values


def approximate_sh_density(theta, phi, lmax: int):
    """
    Computes spherical harmonics for the density map. Instead of first projecting the density
    on the vertices points, and then approximating spherical harmonics coefficients, we use
    a spherical kernel density estimation
    :param theta : torch.tensor, [V], polar angle between 0 and pi
    :param phi : torch.tensor, [V], azimuthal angle between -pi and pi
    :param lmax : int, the maximum spherical harmonic degree of the output coefficients
    :return torch.float, shape [lmax+1, lmax+1]
    """
    device = theta.device
    mapping = mapping_shtools_to_compact(lmax).to(device)

    # lat : -90° -> 90° ; long : 0 -> 360°
    latglq, longlq = GLQGridCoord(lmax, [False])
    grid_theta = torch.deg2rad(torch.tensor(latglq + 90.))
    grid_phi = torch.deg2rad(torch.tensor(longlq))
    st, sp = grid_theta.shape[0], grid_phi.shape[0]
    grid_theta = grid_theta.view(-1, 1).expand(st, sp).float()
    grid_phi = grid_phi.view(1, -1).expand(st, sp).float()
    theta = theta.float()
    phi = phi.float() + math.pi
    density_glq = vonmisesfisher_kde(theta, phi, grid_theta, grid_phi)
    density_glq = density_glq.cpu().numpy()

    zero, w = SHGLQ(lmax)
    cilm = SHExpandGLQ(density_glq, w, zero, [_NORM, _CSPHASE, lmax])
    density = torch.tensor(cilm, device=device)[mapping[..., 0], mapping[..., 1], mapping[..., 2]]

    return density


def pdf_function_from_sh_coeffs(sh_coeffs):
    """
    Return a probability density function on the sphere from spherical harmonics coefficients
    :param sh_coeffs: SH coeffs
    :return: (theta: torch.float of shape [V], phi: torch.float of shape [V]) -> torch.float of shape [V]
    """
    if len(sh_coeffs.shape) == 2:
        sh_coeffs = sh_coeffs.unsqueeze(0)

    def pdf(theta, phi):
        return sample_sh_values(sh_coeffs, theta, phi).squeeze(-1)

    return pdf
