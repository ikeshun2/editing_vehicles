import time
import trimesh
import numpy as np
import skimage.measure
import plyfile
import torch


def create_mesh(
    model, latent, output_file, N=512, max_batch=30000, offset=None, scale=None
):
    start = time.time()
    ply_filename = output_file

    model.eval()

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    # from 0 to N^3-1 all tensor index
    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    # 前三列为坐标，后一列为预测的sdf值（默认初始化为0）
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    latent_code = latent.float().cuda().reshape(1, -1)

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        pred_sdf = model(latent_code, sample_subset)
        pred_sdf = pred_sdf.squeeze().detach().cpu()

        samples[head : min(head + max_batch, num_samples), 3] = pred_sdf

        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    import trimesh

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_tuple = np.array(
        [([f[0], f[1], f[2]],) for f in faces],
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

    # =========================
    # ★ スムージング追加 ★
    # =========================
    try:
        mesh = trimesh.load(ply_filename_out)

        trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=5,
            lamb=0.1
        )

        mesh.export(ply_filename_out)
        print(f"[SMOOTH] applied smoothing to {ply_filename_out}")

    except Exception as e:
        print(f"[SMOOTH ERROR] {e}")
