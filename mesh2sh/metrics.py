import torch
from math import pi


def deformation_scale(current_vertices, initial_vertices, mass_matrix):
    """
    :param current_vertices: Vx3, vertices position after parametrization
    :param initial_vertices: Vx3, vertices position of the original mesh
    :param mass_matrix: VxV, sparse COO mass matrix
    :return: float
    """
    x, v, m = current_vertices, initial_vertices, mass_matrix
    i, j = m._indices()  # 1-D arrays
    values = m._values()  # 1-D array

    # Batched dot products :
    difference_norm = torch.bmm((x[i] - v[i]).unsqueeze(1),  # difference in position of first vertex
                                (x[j] - v[j]).unsqueeze(2)  # difference in position of second vertex (can be the same)
                                ).squeeze(2).squeeze(1) * values  # Edge/vertex weight
    old_norm = torch.bmm(x[i].unsqueeze(1),  # absolute position of first vertex
                         x[j].unsqueeze(2)  # absolute position of second vertex
                         ).squeeze(2).squeeze(1) * values  # Edge/vertex weight
    return torch.sqrt(difference_norm.sum() / old_norm.sum())


def radial_deviation(vertices, faces):
    """
    Radial deviation, measures the dispersion of points relative to a sphere
    """
    pts = vertices[faces]
    f_area = torch.cross(pts[:, 0] - pts[:, 1], pts[:, 0] - pts[:, 2], dim=1).square().sum(1).sqrt() / 2.
    area = f_area.sum()
    vertex_areas = torch.zeros(vertices.shape[0], device=vertices.device)
    vertex_areas = vertex_areas.index_add(0, faces.flatten(),
                                          f_area.unsqueeze(1).expand(-1, 3).flatten() / 3.)
    # vertex_areas = torch.bincount(faces.flatten(), weights=f_area.unsqueeze(1).expand(-1, 3).flatten() / 3.)
    center = (vertices * vertex_areas.unsqueeze(1)).sum(0) / area

    # radius of each vertex from the mesh center of mass
    r = torch.sqrt((vertices - center).square().sum(1))
    # weighted radius std and mean
    mean = (r * vertex_areas).sum() / area
    std = torch.sqrt(torch.abs((r*r*vertex_areas).sum() / area - mean*mean))
    # = relative standard deviation of vertices' weighted radial distances from center of mass
    return std/mean


def sphericity(vertices, faces):
    """
    Sphericity measure how much a discrete mesh is similar to a sphere
    definition : (36*PI*V²)^(1/3)/A with V = volume and A = area of the mesh
    """
    pts = vertices[faces]
    f_area = torch.cross(pts[:, 0] - pts[:, 1], pts[:, 0] - pts[:, 2], dim=1).square().sum(1).sqrt() / 2.
    f_volume = torch.abs(torch.bmm(pts[:, 0].unsqueeze(1),
                                   torch.cross(pts[:, 1], pts[:, 2], dim=1).unsqueeze(2)
                                   ) / 6.).squeeze(2).squeeze(1)
    s = torch.pow(pi*36*f_volume.sum()**2, 1./3) / f_area.sum()
    return s


class CrossLengthRatio:
    """
    Compute the mean absolute error cross-length ratio between an initial set of vertices
    and the updated parametrization.
    For an oriented quadrilateral face IMJK, lcr = (IM*JK)/(MJ*KI)
    This is a little biased, because we want to avoid division by zero when 2 vertices
    are at the same position
    """
    def __init__(self, initial_verts, faces):
        self.faces = faces
        pts = initial_verts[faces]
        # length 01, 12, 20 for each face, Fx3
        lengths = (pts[:, [1, 2, 0]] - pts).square().sum(2).sqrt()
        # lcr ratio for edges 01, 12, 20 of each face, Fx3
        face_lcr_ratio = lengths[:, [1, 2, 0]] / (lengths[:, [2, 0, 1]] + 1e-8)
        # converts to not unique edges indexes and ratio, 3Fx2 and 3F
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]].view(-1, 2)
        edges_lcr_ratio = face_lcr_ratio.flatten()
        # find complementary edges
        unique_edge = edges.sort(1)[0]
        unique_edge = unique_edge[:, 0]*unique_edge.shape[0]+unique_edge[:, 1]
        unique_edge, order = unique_edge.sort(0)
        self.order = order
        # Product of lcr value 2-by-2
        edges_lcr_ratio = edges_lcr_ratio[order].view(-1, 2).prod(1)
        self.initial_lcr = edges_lcr_ratio

    def __call__(self, updated_verts, full=False):
        pts = updated_verts[self.faces]
        lengths = (pts[:, [1, 2, 0]] - pts).square().sum(2).sqrt()
        face_lcr_ratio = lengths[:, [1, 2, 0]] / (lengths[:, [2, 0, 1]] + 1e-8)
        edges_lcr_ratio = face_lcr_ratio.flatten()
        edges_lcr_ratio = edges_lcr_ratio[self.order].view(-1, 2).prod(1)
        # Mean absolute error of length cross ratio for each edge
        mae = torch.abs(edges_lcr_ratio - self.initial_lcr)
        if full:
            return mae
        return mae.mean()


class AngularDistortion:
    """
    Compute the angular distortion between the original mesh and the projection
    For a triangular face ABC, with angles a, b, c, and its projection a', b', c':
    angular distortion = abs(theta - theta')/theta with theta = max(a, b, c)
    This is a little biased, because we want to avoid division by zero when 2 vertices
    are at the same position
    """
    def __init__(self, initial_verts, faces):
        self.faces = faces
        self.initial_angles = self.angles_for_each_face(initial_verts).max(1)[0]

    def angles_for_each_face(self, verts):
        verts = verts[self.faces]
        # Vectors 01, 12, 20
        vectors = verts[:, [1, 2, 0]] - verts
        # Length normalization
        vectors = vectors / (vectors.square().sum(2).sqrt()[..., None] + 1e-8)  # Remove 1e-8 if no collapsing
        # Batched dot products :
        dot = torch.bmm(vectors.view(-1, 3).unsqueeze(1),
                        -vectors[:, [2, 0, 1]].view(-1, 3).unsqueeze(2)
                        ).squeeze(2).squeeze(1)
        angles = torch.arccos(dot).view(-1, 3)
        return angles

    def __call__(self, updated_verts, full=False):
        updated_angles = self.angles_for_each_face(updated_verts).max(1)[0]

        result = torch.abs(updated_angles - self.initial_angles) / self.initial_angles
        if full:
            return result
        return result.mean()


class AreaDistortion:
    """
    Area distortion metric
    """
    def __init__(self, initial_verts, faces):
        self.faces = faces
        self.initial_vertex_areas, self.initial_area = self.vertex_area(initial_verts)

    def vertex_area(self, verts):
        pts = verts[self.faces]
        f_area = torch.cross(pts[:, 0] - pts[:, 1], pts[:, 0] - pts[:, 2], dim=1).square().sum(1).sqrt() / 2.
        vertex_areas = torch.zeros(verts.shape[0], device=verts.device)
        vertex_areas = vertex_areas.index_add(
            0, self.faces.flatten(), f_area.repeat_interleave(3, dim=0)
        )
        return vertex_areas, f_area.sum()

    def __call__(self, updated_verts, full=False):
        updated_vertex_areas, updated_area = self.vertex_area(updated_verts)
        ratio = torch.log((updated_vertex_areas * self.initial_area + 1e-8)
                          / (self.initial_vertex_areas * updated_area + 1e-8))
        if full:
            return ratio
        return (ratio * self.initial_vertex_areas).abs().sum() / self.initial_vertex_areas.sum()
