import torch
import trimesh
import mesh2sh.sphere_parametrization as sph_param
import mesh2sh.metrics as metrics
import mesh2sh.geometry as geometry
import pytest


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_cmcf(device):
    mesh = trimesh.load_mesh("data/spot_color.obj", "obj", maintain_order=True, process=False)

    assert mesh.is_watertight  # Watertight (or a closed manifold mesh)
    assert mesh.euler_number == 2  # Genus 0 (or topology of a sphere)

    verts = torch.from_numpy(mesh.vertices).to(device=device).float()
    faces = torch.from_numpy(mesh.faces).to(device=device).long()

    proj, feat = sph_param.cmcf(verts, faces, max_iters=100, step_size=0.05)
    sphere_verts = torch.stack([*geometry.spherical_to_cartesian(torch.ones([proj.shape[0]], device=proj.device),
                                                                 proj[:, 0], proj[:, 1])], dim=1)

    est_verts = sph_param.reverse_cmcf(sphere_verts=proj, features_vector=feat, faces=faces,
                                       n_iters=100, step_size=0.0005)

    assert 1-metrics.sphericity(sphere_verts, faces) < 0.01
    assert metrics.AngularDistortion(verts, faces)(sphere_verts) < 0.05
    assert metrics.AngularDistortion(est_verts, faces)(sphere_verts) < 0.05
    assert metrics.AreaDistortion(est_verts, faces)(verts) < 0.1
