import numpy as np
import torch
from plyfile import PlyData, PlyElement


# =========================================
# SDF評価（バッチ）
# =========================================
def evaluate_sdf(model, latent, points, device, batch_size=8192):
    sdf_all = []

    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            pts = torch.tensor(points[i:i+batch_size], dtype=torch.float32, device=device)

            # DeepSDF仕様：latentはrepeat内部でされる
            sdf = model(latent, pts)

            sdf_all.append(sdf.squeeze(-1).cpu().numpy())

    return np.concatenate(sdf_all)


# =========================================
# autograd gradient（高速＆安定）
# =========================================
def estimate_gradient(model, latent, point, device):
    x = torch.tensor(point, dtype=torch.float32, device=device, requires_grad=True)

    sdf = model(latent, x.unsqueeze(0))
    sdf.backward()

    grad = x.grad.detach().cpu().numpy()

    norm = np.linalg.norm(grad)
    if norm > 1e-8:
        grad /= norm

    return grad


# =========================================
# QEF（安定版）
# =========================================
def solve_qef(points, normals):
    A = np.array(normals)
    b = np.sum(A * np.array(points), axis=1)

    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except:
        x = np.mean(points, axis=0)

    # 暴走防止
    center = np.mean(points, axis=0)
    if np.linalg.norm(x - center) > 0.3:
        x = center

    return x


# =========================================
# PLY出力（安全版）
# =========================================
def export_ply(vertices, faces, filename):

    vertices = np.array(vertices, dtype=np.float32)

    clean_faces = []
    for f in faces:
        if isinstance(f, (list, tuple)) and len(f) == 3:
            clean_faces.append([int(f[0]), int(f[1]), int(f[2])])

    if len(clean_faces) == 0:
        print("[WARN] No faces generated.")
        return

    vertex_data = np.array(
        [(v[0], v[1], v[2]) for v in vertices],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )

    face_data = np.array(
        [(f,) for f in clean_faces],
        dtype=[("vertex_indices", "i4", (3,))]
    )

    PlyData([
        PlyElement.describe(vertex_data, "vertex"),
        PlyElement.describe(face_data, "face")
    ]).write(filename)


# =========================================
# メイン：Dual Contouring（実用安定版）
# =========================================
def create_mesh(
    model,
    latent,
    output_file,
    N=96,
    bbox_min=-1.0,
    bbox_max=1.0,
    device="cuda"
):

    print(f"[DC] resolution={N}")

    xs = np.linspace(bbox_min, bbox_max, N)

    # =========================================
    # SDFサンプリング
    # =========================================
    print("[DC] sampling sdf...")

    grid_points = np.array(
        [[x, y, z] for x in xs for y in xs for z in xs],
        dtype=np.float32
    )

    sdf_vals = evaluate_sdf(model, latent, grid_points, device)
    sdf_grid = sdf_vals.reshape(N, N, N)

    vertices = []
    vertex_map = {}

    # =========================================
    # 頂点生成
    # =========================================
    print("[DC] generating vertices...")

    MARGIN = 2

    for i in range(MARGIN, N - 1 - MARGIN):
        for j in range(MARGIN, N - 1 - MARGIN):
            for k in range(MARGIN, N - 1 - MARGIN):

                cube = sdf_grid[i:i+2, j:j+2, k:k+2]

                # 符号変化なし
                if cube.min() > 0 or cube.max() < 0:
                    continue

                intersections = []
                normals = []

                # 12エッジ（簡略）
                for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:

                    v1 = cube[0,0,0]
                    v2 = cube[dx,dy,dz]

                    if v1 * v2 > 0:
                        continue

                    p1 = np.array([xs[i], xs[j], xs[k]])
                    p2 = np.array([xs[i+dx], xs[j+dy], xs[k+dz]])

                    t = v1 / (v1 - v2 + 1e-8)
                    p = (1 - t) * p1 + t * p2

                    n = estimate_gradient(model, latent, p, device)

                    intersections.append(p)
                    normals.append(n)

                if len(intersections) < 3:
                    continue

                v = solve_qef(intersections, normals)

                vid = len(vertices)
                vertices.append(v)
                vertex_map[(i,j,k)] = vid

    # =========================================
    # face生成（安定版）
    # =========================================
    print("[DC] generating faces...")

    faces = []

    for (i,j,k), v0 in vertex_map.items():

        # X-Y面
        if (i+1,j,k) in vertex_map and (i,j+1,k) in vertex_map:
            v1 = vertex_map[(i+1,j,k)]
            v2 = vertex_map[(i,j+1,k)]
            faces.append([v0, v1, v2])

        # Y-Z面
        if (i,j+1,k) in vertex_map and (i,j,k+1) in vertex_map:
            v1 = vertex_map[(i,j+1,k)]
            v2 = vertex_map[(i,j,k+1)]
            faces.append([v0, v1, v2])

        # X-Z面
        if (i+1,j,k) in vertex_map and (i,j,k+1) in vertex_map:
            v1 = vertex_map[(i+1,j,k)]
            v2 = vertex_map[(i,j,k+1)]
            faces.append([v0, v1, v2])

    # =========================================
    # 出力
    # =========================================
    export_ply(vertices, faces, output_file)

    print(f"[DC] saved → {output_file}")