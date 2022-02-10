import torch
from math import pi, cos
import mesh2sh.rejection_sampling as rejection
import mesh2sh.spherical_harmonics as sh
import mesh2sh.cholesky_solver as chosolve
import mesh2sh.centroidal_voronoi_tessellation as voronoi
import mesh2sh.geometry as geometry
import mesh2sh.vonmises_kde as vonmi
import pytest


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.filterwarnings("ignore:rejection_sphere_sampling")
def test_rejection(device):

    def pdf(theta, phi):
        return (theta < pi/4).float() * (phi < -pi/2).float()

    x = rejection.rejection_sphere_sampling(pdf, n=100, device=device)

    assert x.numel() == 2*100
    assert x[:, 0].min() >= 0 and x[:, 0].max() < pi/4 and \
           x[:, 1].min() >= -pi and x[:, 1].max() < -pi / 2


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_spherical_harmonics(device):
    lmax = 100
    n_theta = 8
    n_phi = 16

    grid_theta = torch.linspace(0, pi, n_theta+2)[1:-1]
    grid_phi = torch.linspace(-pi, pi, n_phi+2)[1:-1]
    st, sp = grid_theta.shape[0], grid_phi.shape[0]
    theta = grid_theta.view(-1, 1).expand(st, sp).flatten()
    phi = grid_phi.view(1, -1).expand(st, sp).flatten()

    grid_val_theta = torch.sin(torch.linspace(0, pi, n_theta+2)[1:-1])
    grid_val_phi = torch.sin(torch.linspace(0, pi, n_phi+2)[1:-1])
    st, sp = grid_val_theta.shape[0], grid_val_phi.shape[0]
    grid_val_theta = grid_val_theta.view(-1, 1).expand(st, sp).flatten()
    grid_val_phi = grid_val_phi.view(1, -1).expand(st, sp).flatten()

    values = torch.stack([
        torch.sin(grid_val_theta) * torch.sin(2*grid_val_phi),
        torch.sin(grid_val_theta) * torch.sin(2 * grid_val_phi)
    ], dim=1)

    sph_harm = sh.approximate_sh_coefficients(values=values, theta=theta, phi=phi, lmax=lmax,
                                              smoothing=0, kernel='linear', degree=0)

    init_values = sh.sample_sh_values(sph_harm, theta=theta, phi=phi)

    assert torch.dist(init_values, values) < 5e-2


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_cholesky_solver(device):
    A = torch.tensor([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]], dtype=torch.float)
    b = torch.tensor([1, 1, 1, 1], dtype=torch.float)

    A = A.to_sparse()

    x = chosolve.solver_cholesky(A, b)

    b2 = torch.matmul(A, x.unsqueeze(1)).squeeze(1)

    assert torch.allclose(b, b2, rtol=0, atol=0)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_cvt(device):
    n_pts = 100
    points = torch.stack(rejection.rand_sphere([n_pts]), dim=1).float().to(device)

    def pdf(theta, phi):
        return torch.ones_like(theta)/n_pts

    v, f = voronoi.cvt(points, pdf, n_iters=5)

    assert v.shape[0]*2 - 4 == f.shape[0]

    # https://stackoverflow.com/questions/52366421
    points = torch.stack([*geometry.spherical_to_cartesian(torch.ones([points.shape[0]], device=points.device),
                                                           points[:, 0], points[:, 1])], dim=1)
    distance = torch.sqrt(torch.sum((points[:, None, :] - points[None, :, :]) ** 2, dim=2))
    distance = distance + torch.eye(n_pts, device=device) * 1e10
    min_distance_p = torch.min(distance, dim=1)[0]

    v = torch.stack([*geometry.spherical_to_cartesian(torch.ones([v.shape[0]], device=v.device),
                                                      v[:, 0], v[:, 1])], dim=1)
    distance = torch.sqrt(torch.sum((v[:, None, :] - v[None, :, :]) ** 2, dim=2))
    distance = distance + torch.eye(n_pts, device=device) * 1e10
    min_distance = torch.min(distance, dim=1)[0]

    assert min_distance.std() < min_distance_p.std()


@pytest.mark.parametrize("tpos,ppos,device", [[pi/2, 0, 'cpu'], [pi/4, -pi/4, 'cpu'], [3*pi/4, pi, 'cpu']])
def test_vonmiseskde(tpos, ppos, device):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    kdes = list()
    for i in [50, 300]:
        y = torch.randn([2, i], device='cpu') / 30
        y[0] = y[0] + (-cos(tpos) + 1) / 2
        y[1] = y[1] + (ppos + pi) / (2 * pi)
        theta = torch.acos(1 - 2 * y[0])
        phi = 2 * pi * y[1] - pi

        target_theta = torch.tensor([tpos, tpos + 0.2, tpos + 0.2, tpos - 0.2, tpos - 0.2],
                                    device=theta.device)
        target_phi = torch.tensor([ppos, ppos + 0.2, ppos - 0.2, ppos + 0.2, ppos - 0.2],
                                  device=theta.device)

        kdes.append(vonmi.vonmisesfisher_kde(theta, phi, target_theta, target_phi))

    assert kdes[0].max(dim=0)[1] == 0
    assert kdes[1].max(dim=0)[1] == 0
    assert kdes[0][0] < kdes[1][0]
    assert kdes[0][1:].sum() > kdes[1][1:].sum()
