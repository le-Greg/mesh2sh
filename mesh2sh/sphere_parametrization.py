import torch

from .cholesky_solver import solver_cholesky
from .geometry import face_area, vertex_area, mesh_center, project_onto_sphere, \
    compute_vertex_normals, spherical_to_cartesian, laplacian_beltrami_matrix
from .metrics import radial_deviation


def stiffness_matrix(vertices, faces):
    """
    The stiffness matrix used by cMCF algorithm
    :return: sparse torch tensor, shape [V, V]
    """
    return - laplacian_beltrami_matrix(vertices=vertices, faces=faces)


def mass_matrix(vertices, faces):
    """
    The mass matrix used by cMCF algorithm
    :return: sparse torch tensor, shape [V, V]
    """
    device = vertices.device
    indices = torch.empty([2, len(faces), 3, 3], device=device, dtype=torch.long)

    # Gets indices i,j for each face mass matrix
    indices[0] = faces.unsqueeze(1).expand(faces.shape[0], 3, 3)
    indices[1] = faces.unsqueeze(2).expand(faces.shape[0], 3, 3)
    # Get area of each faces, shape : F
    f_area = face_area(vertices, faces)
    # Create mass matrices for each face, shape : Fx3x3
    values = (torch.ones(3, 3, device=device) + torch.eye(3, device=device)).expand(faces.shape[0], 3, 3)
    values = values * f_area[..., None, None] / 12.
    # We use sparse matrix to overlap same indexes, then coalesce to correct it
    mass_mat = torch.sparse_coo_tensor(indices=indices.view(2, -1),
                                       values=values.view(-1),
                                       size=(vertices.shape[0], vertices.shape[0]),
                                       device=device)
    mass_mat = mass_mat.coalesce()

    return mass_mat


def cmcf(vertices, faces, max_iters: int, step_size=0.05, threshold_rd=1e-3):
    """
    cMCF implementation in Python using Pytorch
    See : https://arxiv.org/pdf/1203.6819.pdf
    :param vertices: torch.float [V, 3], vertices coordinates of the mesh in euclidean space
    :param faces: torch.long [F, 3], index of vertices for each face
    :param max_iters: int, maximal number of iterations before stopping. It can stop earlier
    :param step_size: float, step size, must be between 0 and 1
    :param threshold_rd: float, maximal radial deviation allowed for early stopping
    :return: (x_proj, vector)
        x_proj: torch.float [V, 2], vertices coordinates on the sphere, along theta and phi
        vector: torch.float [V, 2], features vector containing curvature density and area distortion
    """
    # Normalization
    vertices = vertices - mesh_center(vertices, faces)
    vertices = vertices / face_area(vertices, faces).sum().sqrt()
    # Modified vertices through the flow
    x = vertices.clone()
    # The stiffness matrix L is unchanged throughout the flow so we can initialize it once.
    L = stiffness_matrix(vertices, faces)

    for i in range(max_iters):
        # Get current mass matrix
        D = mass_matrix(x, faces)
        # Set the new constraint vector: b = D * x
        b = torch.matmul(D, x)
        # Set the system matrix: M = D + t * L
        M = D + step_size * L

        x = solver_cholesky(M, b, return_dense=True).float()

        # Unit area scaling
        x = x - mesh_center(x, faces)
        x = x / face_area(x, faces).sum().sqrt()

        # Early stop, if the mesh is spherical enough
        if radial_deviation(x, faces).item() <= threshold_rd:
            break

    # Flow is done, now we prepare our curvature vector :
    # Project XYZ points onto 2D theta/phi sphere
    x_proj = project_onto_sphere(x)
    # Area distortion
    # TODO : Use sphere area instead of euclidean ?
    original_v_area, sphere_v_area = vertex_area(vertices, faces), vertex_area(x, faces)
    area_dist = torch.log(sphere_v_area / original_v_area)  # Area distortion [V]
    # Curvature vectors for each points
    curv = torch.matmul(L, vertices)
    v_normals = compute_vertex_normals(vertices=vertices, faces=faces)
    normals_dot_curv = torch.matmul(v_normals.unsqueeze(1), curv.unsqueeze(2)).squeeze(2).squeeze(1)
    curv = curv.square().sum(1).sqrt()
    curv[normals_dot_curv < 0] *= -1
    curv = curv / (original_v_area + 1e-10)  # Curvature density [V]
    # Result vector
    vector = torch.stack([curv, area_dist], dim=1)  # [V, 2]
    return x_proj, vector


def reverse_cmcf(sphere_verts, features_vector, faces, n_iters: int, step_size=0.05):
    """
    Reverse cMCF, to get the vertices to a desired curvature instead of minimizing it
    :param sphere_verts: torch.float [V, 2], vertices coordinates on the sphere, along theta and phi
    :param features_vector: torch.float [V, 2], features vector containing curvature density and area distortion
    :param faces: torch.long [F, 3], index of vertices for each face
    :param n_iters: int, number of iterations
    :param step_size: float, step size, must be between 0 and 1
    :return: torch.float [V, 3], vertices coordinates
    """
    # XYZ representation
    verts_x, verts_y, verts_z = spherical_to_cartesian(
        r=torch.ones([sphere_verts.shape[0]], device=sphere_verts.device),
        theta=sphere_verts[:, 0],
        phi=sphere_verts[:, 1]
    )
    vertices = torch.stack([verts_x, verts_y, verts_z], dim=1)
    # Normalization
    vertices = vertices - mesh_center(vertices, faces)
    vertices = vertices / face_area(vertices, faces).sum().sqrt()
    # Modified vertices through the flow
    x = vertices.clone()
    # The stiffness matrix L is unchanged throughout the flow so we can initialize it once.
    L = stiffness_matrix(vertices, faces)
    # Get our target curvature
    curv_density = features_vector[:, 0]  # Curvature density
    area_dist = features_vector[:, 1]  # Area distortion
    original_vertex_area = vertex_area(x, faces) / torch.exp(area_dist)
    curv = (curv_density * original_vertex_area).unsqueeze(1)

    for i in range(n_iters):
        # Get current mass matrix
        D = mass_matrix(x, faces)
        # Target curvature vector N = H*n = Curvature(scalar) * normal vector
        N = compute_vertex_normals(x, faces) * curv
        # Set the new constraint vector: b = D * x + t * N
        b = torch.matmul(D, x) + step_size * N
        # Set the system matrix: M = D + t * L
        M = D + step_size * L

        x = solver_cholesky(M, b, return_dense=True).float()

        # Unit area scaling
        x = x - mesh_center(x, faces)
        x = x / face_area(x, faces).sum().sqrt()

    return x
