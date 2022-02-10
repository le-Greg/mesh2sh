import torch


def face_area(vertices, faces):
    """
    Computes area of each faces using cross product
    Args:
        vertices (torch.tensor): Vx3, xyz coordinates of vertices
        faces (torch.tensor): Fx3, index of vertices for each face
    Returns:
        (torch.tensor): F, area of each face
    """
    pts = vertices[faces]
    x = torch.cross(pts[:, 0] - pts[:, 1], pts[:, 0] - pts[:, 2], dim=1)
    x = x.square().sum(1).sqrt()
    area = x * 0.5
    return area


def vertex_area(vertices, faces):
    """
    Compute area vertex-wise
    """
    f_area = face_area(vertices=vertices, faces=faces)
    v_area = torch.bincount(faces.flatten(), weights=f_area.unsqueeze(1).expand(-1, 3).flatten() / 3.)
    return v_area


def mesh_center(vertices, faces):
    """
    Compute center of the surface of the mesh
    Args:
        vertices (torch.tensor): Vx3, xyz coordinates of vertices
        faces (torch.tensor): Fx3, index of vertices for each face
    Returns:
        (torch.tensor):  3, xyz coordinates of the center
    """
    f_area = face_area(vertices, faces).unsqueeze(1)
    f_center = vertices[faces].sum(1) / 3
    center = (f_center * f_area).sum(0) / f_area.sum()
    return center


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical coordinates to cartesian
    r, theta and phi are 1-D tensors
    """
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts spherical coordinates to cartesian
    x, y and z are 1-D tensors
    """
    xy = x**2 + y**2
    r = torch.sqrt(xy + z**2)
    theta = torch.atan2(torch.sqrt(xy), z)
    phi = torch.atan2(y, x)
    return r, theta, phi


def project_onto_sphere(vertices):
    """
    Project our vertices onto 2D spherical surfaces
    The vertices are the result of the cMCF algorithm
    0 < theta < pi ; -pi < phi < pi
    :param vertices: Vx3 torch tensor, XYZ coordinates of the vertices
    :return: Vx2 torch tensor, Theta and Phi coordinates of the vertices
    """
    _, theta, phi = cartesian_to_spherical(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    return torch.stack([theta, phi], dim=1)


def compute_face_normals(vertices, faces):
    """
    Compute each face normal
    """
    pts = vertices[faces]
    cross_product = torch.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 1], dim=1)
    normals = cross_product / cross_product.square().sum(1).sqrt()[..., None]
    return normals


def compute_vertex_normals(vertices, faces):
    """
    Compute each vertex normal
    The code is inspired by pytorch3d's _compute_vertex_normals in structures/meshes.py, v0.6.0
    This function is probably not differentiable though
    """
    pts = vertices[faces]
    cross_product = torch.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 1], dim=1)
    vertex_normals = torch.zeros(vertices.shape, device=vertices.device)
    vertex_normals = vertex_normals.index_add(
        0, faces.flatten(), cross_product.repeat_interleave(3, dim=0)
    )
    vertex_normals = vertex_normals / vertex_normals.square().sum(1).sqrt()[..., None]
    return vertex_normals


def laplacian_beltrami_matrix(vertices, faces):
    """
    Return the Laplacian-Beltrami matrix
    See https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians
    How we compute cotangent :
        -> Cross product ABxAC = AB*AC*sin(BAC)
        -> Dot product AB*AC = AB*AC*cos(BAC)
        -> cot(BAC) = dot_product(AB, AC)/cross_product(AB, AC)
    We compute the 3 cotangents of each faces. Then to sum opposing angles of each edge, we use the
    properties of sparse tensors to sum overlapping values with coalesce()
    :param vertices: torch.float, [V, 3]
    :param faces: torch.long, [F, 3]
    :return: torch.float, [V, V] sparse COO tensor
    """
    # First we get the cotangents
    pts = vertices[faces]
    vectors = pts[:, [1, 2, 0]] - pts  # [V, 3] vectors 01, 21, 20
    dot = torch.bmm(vectors.view(-1, 3).unsqueeze(1),
                    -vectors[:, [2, 0, 1]].view(-1, 3).unsqueeze(2)
                    ).view(faces.shape[0], 3)
    len_cross = torch.cross(vectors[:, 0], -vectors[:, 1], dim=1).square().sum(1).sqrt().expand(3, -1).T
    half_cot_alpha = dot / len_cross / 2.  # cotangent / 2

    # Then we get their position in the matrix
    indices = faces[:, [[1, 2], [2, 0], [0, 1]]].view(-1, 2).T  # Vertices i and j for i != j
    ind_diag = torch.cat([faces[:, [1, 2, 0]].view(-1).expand(2, -1),
                          faces[:, [2, 0, 1]].view(-1).expand(2, -1)], dim=1)  # Vertices i and j for i == j
    indices = torch.cat([indices, indices[[1, 0], :], ind_diag], dim=1)
    values = half_cot_alpha.view(-1).repeat(2)  # Values for i != j
    values_diag = - half_cot_alpha.view(-1).repeat(2)  # Values for i == j
    values = torch.cat([values, values_diag], dim=0)

    # We fill the sparse matrix, and add overlapping values
    laplacian = torch.sparse_coo_tensor(indices=indices, values=values, size=[vertices.shape[0], vertices.shape[0]])
    laplacian = laplacian.coalesce()

    return laplacian
