import time
import trimesh
import numpy as np
import skimage.measure
import plyfile
import torch








# =========================================
# 12 edges of a cube
# =========================================
CUBE_EDGES = [
    ((0, 0, 0), (1, 0, 0)),
    ((0, 1, 0), (1, 1, 0)),
    ((0, 0, 1), (1, 0, 1)),
    ((0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0)),
    ((1, 0, 0), (1, 1, 0)),
    ((0, 0, 1), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 1)),
    ((1, 1, 0), (1, 1, 1)),
]


# =========================================
# evaluate sdf
# =========================================
def evaluate_sdf(model, latent, points, device):
    with torch.no_grad():
        points_t = torch.tensor(points, dtype=torch.float32, device=device)
        latent_expand = latent.expand(points_t.shape[0], -1)

        sdf = model(points_t, latent_expand)
        sdf = sdf.squeeze(-1).detach().cpu().numpy()

    return sdf


# =========================================
# finite difference gradient
# =========================================
def estimate_gradient(model, latent, point, device, eps=1e-3):
    grads = []

    for i in range(3):
        p1 = point.copy()
        p2 = point.copy()

        p1[i] += eps
        p2[i] -= eps

        sdf1 = evaluate_sdf(model, latent, [p1], device)[0]
        sdf2 = evaluate_sdf(model, latent, [p2], device)[0]

        g = (sdf1 - sdf2) / (2 * eps)
        grads.append(g)

    grad = np.array(grads)

    norm = np.linalg.norm(grad)
    if norm > 1e-8:
        grad /= norm

    return grad


# =========================================
# solve QEF
# =========================================
def solve_qef(points, normals):
    A = np.array(normals)
    b = np.sum(A * np.array(points), axis=1)

    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        x = np.mean(points, axis=0)

    return x


# =========================================
# export ply
# =========================================
def export_ply(vertices, faces, filename):
    vertex_data = np.array(
        [(v[0], v[1], v[2]) for v in vertices],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )

    face_data = np.array(
        [([f[0], f[1], f[2]],) for f in faces],
        dtype=[("vertex_indices", "i4", (3,))]
    )

    ply = PlyData([
        PlyElement.describe(vertex_data, "vertex"),
        PlyElement.describe(face_data, "face")
    ])

    ply.write(filename)


# =========================================
# main create mesh dual contouring
# =========================================
def create_mesh(
    model,
    latent,
    output_file,
    N=128,
    bbox_min=-1.0,
    bbox_max=1.0,
    device="cuda"
):
    print(f"[DC] reconstructing mesh with resolution={N}")

    xs = np.linspace(bbox_min, bbox_max, N)

    sdf_grid = np.zeros((N, N, N), dtype=np.float32)

    # =====================================
    # sample SDF grid
    # =====================================
    print("[DC] sampling sdf grid...")
    for i, x in enumerate(xs):
        pts = []
        indices = []

        for j, y in enumerate(xs):
            for k, z in enumerate(xs):
                pts.append([x, y, z])
                indices.append((j, k))

        sdf_vals = evaluate_sdf(model, latent, pts, device)

        for idx, (j, k) in enumerate(indices):
            sdf_grid[i, j, k] = sdf_vals[idx]

    vertices = []
    faces = []
    vertex_map = {}

    print("[DC] dual contouring...")

    for i in range(N - 2):
        for j in range(N - 2):
            for k in range(N - 2):

                cube = sdf_grid[i:i+2, j:j+2, k:k+2]

                if cube.min() > 0 or cube.max() < 0:
                    continue

                intersections = []
                normals = []

                for edge in CUBE_EDGES:
                    p1_idx, p2_idx = edge

                    v1 = cube[p1_idx]
                    v2 = cube[p2_idx]

                    if v1 * v2 > 0:
                        continue

                    p1 = np.array([
                        xs[i + p1_idx[0]],
                        xs[j + p1_idx[1]],
                        xs[k + p1_idx[2]]
                    ])

                    p2 = np.array([
                        xs[i + p2_idx[0]],
                        xs[j + p2_idx[1]],
                        xs[k + p2_idx[2]]
                    ])

                    t = v1 / (v1 - v2 + 1e-8)
                    p = (1 - t) * p1 + t * p2

                    n = estimate_gradient(model, latent, p, device)

                    intersections.append(p)
                    normals.append(n)

                if len(intersections) == 0:
                    continue

                vertex = solve_qef(intersections, normals)

                vid = len(vertices)
                vertices.append(vertex)
                vertex_map[(i, j, k)] = vid

    # =====================================
    # create simple faces
    # =====================================
    print("[DC] generating faces...")

    for i in range(N - 2):
        for j in range(N - 2):
            for k in range(N - 2):

                keys = [
                    (i, j, k),
                    (i+1, j, k),
                    (i, j+1, k),
                    (i+1, j+1, k)
                ]

                if all(key in vertex_map for key in keys):
                    v0 = vertex_map[keys[0]]
                    v1 = vertex_map[keys[1]]
                    v2 = vertex_map[keys[2]]
                    v3 = vertex_map[keys[3]]

                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])

    export_ply(vertices, faces, output_file)

    print(f"[DC] mesh saved → {output_file}")




def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
