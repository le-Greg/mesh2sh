import torch
from math import sqrt, pi, atan
import mesh2sh.geometry as geometry
import pytest


class BasicPolyhedron:
    def __init__(self, h1=1., h2=1., device='cpu'):
        self.h1 = h1
        self.h2 = h2
        self.device = device

        self.verts = torch.tensor([
            [0., 0., h1],
            [0., 0., -h2],
            [0, -1, 0],
            [sqrt(3)/2, 1./2, 0],
            [-sqrt(3) / 2, 1. / 2, 0]
        ], dtype=torch.float, device=self.device)

        self.faces = torch.tensor([
            [2, 3, 0],
            [3, 4, 0],
            [4, 2, 0],
            [3, 2, 1],
            [4, 3, 1],
            [2, 4, 1]
        ], dtype=torch.long, device=self.device)

    def to(self, device):
        self.device = device
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)

    def face_area(self):
        up = sqrt(12*self.h1**2+3)/4
        down = sqrt(12*self.h2**2+3)/4
        return torch.tensor([up, up, up, down, down, down], dtype=torch.float, device=self.device)

    def vertex_area(self):
        up = sqrt(12*self.h1**2+3)/4
        down = sqrt(12 * self.h2 ** 2 + 3) / 4
        middle = (2*up+2*down)/3
        return torch.tensor([up, down, middle, middle, middle], dtype=torch.float, device=self.device)

    def mesh_center(self):
        up = sqrt(12 * self.h1 ** 2 + 3) / 4
        down = sqrt(12 * self.h2 ** 2 + 3) / 4
        center_z = (self.h1*up-self.h2*down)/(up+down)/3
        return torch.tensor([0., 0., center_z], dtype=torch.float, device=self.device)

    def face_normals(self):
        pts = self.verts[self.faces]
        normals = torch.stack([
            pts[:, 2, 2]*(pts[:, 0, 0]+pts[:, 1, 0]),
            pts[:, 2, 2] * (pts[:, 0, 1] + pts[:, 1, 1]),
            torch.ones_like(pts[:, 0, 0]) / 2
        ], dim=1)
        normals[3:] *= -1
        normals = normals / normals.square().sum(1).sqrt().unsqueeze(1)
        return normals

    def verts_normals(self):
        return torch.tensor([
            [0, 0, 1],
            [0, 0, -1],
            [0, -1, 0],
            [sqrt(3) / 2, 1. / 2, 0],
            [-sqrt(3) / 2, 1. / 2, 0]
        ], dtype=torch.float, device=self.device)

    def laplace_beltrami_mat(self):
        low_h = sqrt(self.h1**2+1/4)
        low_low_cot = (low_h - 3/(4*low_h))/sqrt(3)
        low_middle_cot = sqrt(3)/(2*low_h)

        high_h = sqrt(self.h2**2+1/4)
        high_high_cot = (high_h - 3/(4*high_h))/sqrt(3)
        high_middle_cot = sqrt(3)/(2*high_h)

        mat = torch.zeros([5, 5], dtype=torch.float, device=self.device)
        mat[[2, 3, 4, 3, 4, 2], [3, 4, 2, 2, 3, 4]] = (low_low_cot + high_high_cot)/2
        mat[[0, 0, 0, 2, 3, 4], [2, 3, 4, 0, 0, 0]] = low_middle_cot
        mat[[1, 1, 1, 2, 3, 4], [2, 3, 4, 1, 1, 1]] = high_middle_cot
        mat[0, 0] = -3 * low_middle_cot
        mat[1, 1] = -3 * high_middle_cot
        mat[[2, 3, 4], [2, 3, 4]] = - low_middle_cot - low_low_cot - high_middle_cot - high_high_cot

        return mat


@pytest.mark.parametrize("h1,h2,device", [(1., 1., 'cpu'), (4.78, 3.92, 'cuda'),
                                          (2.36, 0., 'cuda'), (4.1, -1.7, 'cuda')])
def test_geometry(h1, h2, device):
    poly = BasicPolyhedron(h1=h1, h2=h2, device=device)

    assert torch.allclose(poly.face_area(),
                          geometry.face_area(poly.verts, poly.faces),
                          rtol=0, atol=1e-6)
    assert torch.allclose(poly.vertex_area(),
                          geometry.vertex_area(poly.verts, poly.faces),
                          rtol=0, atol=1e-6)
    assert torch.allclose(poly.mesh_center(),
                          geometry.mesh_center(poly.verts, poly.faces),
                          rtol=0, atol=1e-6)
    assert torch.allclose(poly.face_normals(),
                          geometry.compute_face_normals(poly.verts, poly.faces),
                          rtol=0, atol=1e-6)
    assert torch.allclose(poly.verts_normals(),
                          geometry.compute_vertex_normals(poly.verts, poly.faces),
                          rtol=0, atol=1e-6)
    assert torch.allclose(poly.laplace_beltrami_mat(),
                          geometry.laplacian_beltrami_matrix(poly.verts, poly.faces).to_dense(),
                          rtol=0, atol=1e-6)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_polar_euclidean(device):
    x = torch.randn([100], device=device)
    y = torch.randn([100], device=device)
    z = torch.randn([100], device=device)

    r, t, p = geometry.cartesian_to_spherical(x, y, z)
    x2, y2, z2 = geometry.spherical_to_cartesian(r, t, p)

    assert torch.allclose(x, x2, rtol=0, atol=1e-6)
    assert torch.allclose(y, y2, rtol=0, atol=1e-6)
    assert torch.allclose(z, z2, rtol=0, atol=1e-6)

    x = torch.tensor([1., 2, 0, 0, -5, 0, 0, 1.312938193], device=device)
    y = torch.tensor([1., 0, 3, 0, 0, -6, 0, -2.842818271], device=device)
    z = torch.tensor([1., 0, 0, 4, 0, 0, -7, 3.522882813], device=device)

    r = torch.tensor([sqrt(3), 2, 3, 4, 5, 6, 7, 4.7133985695], device=device)
    t = torch.tensor([atan(sqrt(2)), pi / 2, pi / 2, 0, pi / 2, pi / 2, pi, 0.726628014], device=device)
    p = torch.tensor([pi / 4, 0, pi / 2, 0, pi, -pi / 2, 0, -1.138136716], device=device)

    r2, t2, p2 = geometry.cartesian_to_spherical(x, y, z)
    x2, y2, z2 = geometry.spherical_to_cartesian(r, t, p)

    assert torch.allclose(x, x2, rtol=0, atol=1e-6)
    assert torch.allclose(y, y2, rtol=0, atol=1e-6)
    assert torch.allclose(z, z2, rtol=0, atol=1e-6)
    assert torch.allclose(r, r2, rtol=0, atol=1e-6)
    assert torch.allclose(t, t2, rtol=0, atol=1e-6)
    assert torch.allclose(p, p2, rtol=0, atol=1e-6)
