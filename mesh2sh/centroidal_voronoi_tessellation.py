import torch
from scipy.spatial import SphericalVoronoi, ConvexHull

from .geometry import spherical_to_cartesian, cartesian_to_spherical


def polygons_weighted_center(vertices, polygons_indices, pdf):
    """
    Calculates polygon weighted center with respect to a density function, using the formula described in
    the paper, but on the sphere. Return each polygon new center
    :param vertices: torch.float, [Vx3]
    :param polygons_indices: list of list of ints, list of every polygons indices
    :param pdf: a function that takes as input theta vector and phi vector and return their densities
    :return: torch.float [Px3]
    """
    device = vertices.device
    # Find polygons centers on the sphere
    centers = torch.stack([vertices[i].sum(0).float() / len(i) for i in polygons_indices], dim=0)
    centers = centers / centers.square().sum(1, keepdims=True).sqrt()
    all_pts = torch.cat([vertices, centers], dim=0)  # [V+C, 3] = [X, 3]
    # Compute densities for vertices and for centers
    _, theta, phi = cartesian_to_spherical(all_pts[..., 0], all_pts[..., 1], all_pts[..., 2])
    densities = pdf(theta=theta, phi=phi).to(device)  # [X]
    # Triangulate every polygons : associate each triplex on indices together
    tri_idx = torch.cat([
        torch.stack([
            torch.tensor(p, dtype=torch.long, device=device),
            torch.tensor(p, dtype=torch.long, device=device).roll(-1, dims=0),
            vertices.shape[0] + t * torch.ones(len(p), dtype=torch.long, device=device)
        ], dim=1) for t, p in enumerate(polygons_indices)
    ], dim=0)  # [V=T, 3] torch.long
    # Compute triangle weighted center
    tri_densities = densities[tri_idx]  # [T, 3]
    tri_pts = all_pts[tri_idx]
    tri_centroid = torch.matmul(tri_densities, (torch.eye(3) + torch.ones([3, 3])).to(device))
    tri_centroid = (tri_centroid.unsqueeze(2) * tri_pts).sum(1)
    tri_centroid = tri_centroid / (4 * tri_densities.sum(1).unsqueeze(1))  # [T, 3]
    # Compute weighted area
    tri_area = torch.cross(tri_pts[:, 0] - tri_pts[:, 1], tri_pts[:, 0] - tri_pts[:, 2], dim=1)
    tri_area = tri_area.square().sum(1).sqrt() / 2.
    weighted_area = tri_densities.mean(1) * tri_area  # [T]
    # Find polygons centroid
    r = torch.tensor([0] + [len(p) for p in polygons_indices], device=device).cumsum(dim=0)
    wa = [weighted_area[r[i]:r[i + 1]] for i in range(len(polygons_indices))]
    tc = [tri_centroid[r[i]:r[i + 1]] for i in range(len(polygons_indices))]
    polygon_centroids = torch.stack(
        [(wai.unsqueeze(1) * tci).sum(0) / wai.unsqueeze(1).sum(0) for wai, tci in zip(wa, tc)],
        dim=0)
    # Align centroids on the sphere, still keeps euclidean space
    polygon_centroids = polygon_centroids / polygon_centroids.square().sum(1, keepdims=True).sqrt()

    return polygon_centroids


def cvt(spherical_verts, pdf, n_iters):
    """
    Weighted Centroidal Voronoi Tesselation on a sphere, using Lloyd algorithm
    :param spherical_verts: torch.float, [Vx2], theta and phi vectors, concatenated in this order
    :param pdf: a function that takes as input theta vector and phi vector and return their densities
    :param n_iters: number of iteration of lloyd relaxation
    :return: verts (torch.float, [Vx2]), faces (torch.long, [Fx3])
    """
    device = spherical_verts.device
    verts = torch.stack(spherical_to_cartesian(r=torch.ones_like(spherical_verts[:, 0]),
                                               theta=spherical_verts[:, 0],
                                               phi=spherical_verts[:, 1]), dim=1)

    for it in range(n_iters):
        # The Voronoi diagram of the k sites is computed.
        vor = SphericalVoronoi(verts.cpu().numpy())
        # Each cell of the Voronoi diagram is integrated, and the centroid is computed.
        centroids = polygons_weighted_center(torch.tensor(vor.vertices, device='cpu', dtype=torch.float),
                                             vor.regions, pdf=pdf)
        # Each site is then moved to the centroid of its Voronoi cell.
        verts = centroids

    # Get the Delaunay triangulation from Voronoi diagram, Voronoi seeds are vertices of Delaunay
    hull = ConvexHull(verts.cpu().numpy())
    faces = torch.tensor(hull.simplices, device=device, dtype=torch.long)
    verts = verts.float().to(device)
    # Flip faces that are not facing outward (counterclockwise)
    pts = verts[faces]
    outward = torch.bmm(pts[:, 0].unsqueeze(1),
                        torch.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0], dim=1).unsqueeze(2),
                        ).squeeze(2).squeeze(1)
    outward = outward < 0
    faces[outward] = faces[outward][:, [0, 2, 1]]
    # Reproject on the sphere
    _, theta, phi = cartesian_to_spherical(verts[..., 0], verts[..., 1], verts[..., 2])
    verts = torch.stack([theta, phi], dim=1)

    return verts, faces
