#!/usr/bin/env python3
"""ORENのSDF (query_result.npz) から障害物の表面 (SDF=0 の等値面) を三角形メッシュにして保存する。

入力 : points (N,3) 規則格子 / sdf1 (N,) [/ grad_sdf (N,3) があれば法線に使う]
出力 : <out>.ply (頂点色つき, バイナリ) / <out>.obj / <out>.stl  (--format で選択。座標は OREN のワールド座標 [m])

  python3 npz_to_mesh.py query_result.npz --out mesh_map
  python3 npz_to_mesh.py query_result.npz --out mesh_map --level 0.0 --smooth 1 --min-faces 200 --format ply obj stl

頂点色 (PLY): --color up のとき、法線が +Z から max-angle 以内 (床・段の上面など = CHOMP の除外フィルタ対象の向き) を青、
              それ以外を灰色にする。--color normal で法線の向きを RGB に、--color none で色なし。
注意: 格子の外側 (範囲の端) は面が閉じないので、端に板状の面が出ることがある。--trim-border n で端 n セルの面を捨てる。
      SDF が未観測領域の外挿値を含む場合、その領域の面は信用できない (--bbox で観測範囲に切り出す)。
"""
import argparse, os, struct, sys
import numpy as np


def load_grid(path, pk, sk, gk, bbox, nan_fill):
    d = np.load(path)
    P, s = d[pk].astype(np.float64), d[sk].astype(np.float64)
    ax = [np.unique(np.round(P[:, i], 6)) for i in range(3)]
    shape = tuple(len(a) for a in ax)
    if int(np.prod(shape)) != len(s):
        raise SystemExit(f"points is not a regular grid: axes {shape} -> {np.prod(shape)} != {len(s)}")
    # 並び順を確実にするため、座標から添字を求めて詰め直す (meshgrid の ij/xy どちらでも正しく読める)
    ix = [np.searchsorted(ax[i], np.round(P[:, i], 6)) for i in range(3)]
    S = np.full(shape, np.nan)
    S[ix[0], ix[1], ix[2]] = s
    G = None
    if gk and gk in d:
        G = np.full(shape + (3,), np.nan)
        G[ix[0], ix[1], ix[2]] = d[gk].astype(np.float64)
    if bbox:
        lo, hi = np.array(bbox[:3]), np.array(bbox[3:])
        sl = tuple(slice(int(np.searchsorted(ax[i], lo[i] - 1e-9)), int(np.searchsorted(ax[i], hi[i] + 1e-9))) for i in range(3))
        ax = [ax[i][sl[i]] for i in range(3)]
        S = S[sl]
        G = None if G is None else G[sl]
    n_nan = int(np.isnan(S).sum())
    if n_nan:
        print(f"[info] {n_nan} NaN cells -> filled with {nan_fill}")
        S = np.where(np.isnan(S), nan_fill, S)
    return ax, S, G


def trilinear(ax, V, P):
    """格子 V (nx,ny,nz[,3]) を点 P (M,3) で三線形補間 (範囲外は境界に丸める)"""
    idx, w = [], []
    for i in range(3):
        a = ax[i]
        t = np.clip((P[:, i] - a[0]) / (a[1] - a[0]), 0, len(a) - 1 - 1e-9)
        i0 = np.floor(t).astype(int)
        idx.append(i0); w.append(t - i0)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ww = (w[0] if dx else 1 - w[0]) * (w[1] if dy else 1 - w[1]) * (w[2] if dz else 1 - w[2])
                v = V[idx[0] + dx, idx[1] + dy, idx[2] + dz]
                out = out + (ww[:, None] * v if v.ndim == 2 else ww * v)
    return out


def components(faces, n_vert):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_vert, n_vert))
    n, lab = connected_components(g, directed=False)
    return n, lab


def smooth_laplacian(V, F, iters, lam=0.5):
    """軽い Laplacian 平滑化 (頂点を隣接頂点の平均へ lam だけ寄せる)。形が縮むので回数は少なく。"""
    from scipy.sparse import coo_matrix
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iters):
        V = V + lam * ((A @ V) / deg[:, None] - V)
    return V


def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # 面積で重み付け
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def write_ply(path, V, F, C=None):
    with open(path, "wb") as f:
        h = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
        if C is not None:
            h += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        h += f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n"
        f.write(h.encode())
        if C is None:
            f.write(V.astype("<f4").tobytes())
        else:
            rec = np.empty(len(V), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = V, C
            f.write(rec.tobytes())
        rec = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
        rec["n"], rec["i"] = 3, F
        f.write(rec.tobytes())


def write_obj(path, V, F):
    with open(path, "w") as f:
        np.savetxt(f, V, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, F + 1, fmt="f %d %d %d")


def write_stl(path, V, F):
    fn = face_normals(V, F).astype("<f4")
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = fn, V[F].astype("<f4")
    with open(path, "wb") as f:
        f.write(b"OREN SDF zero level set".ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", default="mesh_map", help="出力の接頭辞")
    ap.add_argument("--format", nargs="+", default=["ply"], choices=["ply", "obj", "stl"])
    ap.add_argument("--points-key", default="points")
    ap.add_argument("--sdf-key", default="sdf1")
    ap.add_argument("--grad-key", default="grad_sdf", help="あれば法線に使う (無ければメッシュから計算)")
    ap.add_argument("--bbox", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    ap.add_argument("--level", type=float, default=0.0, help="等値面の値 [m]。0=障害物の表面")
    ap.add_argument("--nan-fill", type=float, default=1.0, help="欠損セルの値 (正=自由空間)")
    ap.add_argument("--step", type=int, default=1, help="格子を間引く (2 で半分の解像度)")
    ap.add_argument("--smooth", type=int, default=0, help="Laplacian 平滑化の回数 (0=なし。1〜3 程度)")
    ap.add_argument("--min-faces", type=int, default=0, help="これより面の少ない連結成分 (ノイズの小片) を捨てる")
    ap.add_argument("--trim-border", type=int, default=0, help="格子の端から n セル以内の面を捨てる (端の板状の面を消す)")
    ap.add_argument("--color", choices=["up", "normal", "none"], default="up")
    ap.add_argument("--max-angle", type=float, default=30.0, help="--color up の +Z からの角度 [deg]")
    a = ap.parse_args()

    from skimage.measure import marching_cubes
    ax, S, G = load_grid(a.npz, a.points_key, a.sdf_key, a.grad_key, a.bbox, a.nan_fill)
    if a.step > 1:
        ax = [v[::a.step] for v in ax]; S = S[::a.step, ::a.step, ::a.step]
        G = None if G is None else G[::a.step, ::a.step, ::a.step]
    sp = [float(v[1] - v[0]) for v in ax]
    print(f"[grid] {S.shape} step={np.round(sp, 3)} x[{ax[0][0]:.2f},{ax[0][-1]:.2f}] y[{ax[1][0]:.2f},{ax[1][-1]:.2f}] z[{ax[2][0]:.2f},{ax[2][-1]:.2f}]  sdf range [{S.min():.2f}, {S.max():.2f}]")
    if not (S.min() < a.level < S.max()):
        raise SystemExit(f"[error] level {a.level} is outside the SDF range: no surface")
    V, F, _, _ = marching_cubes(S, level=a.level, spacing=tuple(sp))
    V = V + np.array([ax[0][0], ax[1][0], ax[2][0]])  # 添字→ワールド座標
    # skimage の既定の向きで、面の法線は SDF が増える側 (自由空間側) を向く (合成マップで床 +Z・柱の側面が外向きと確認済み)
    print(f"[mesh] marching cubes: {len(V)} vertices, {len(F)} faces")

    if a.trim_border > 0:
        lo = np.array([ax[i][0] for i in range(3)]) + a.trim_border * np.array(sp)
        hi = np.array([ax[i][-1] for i in range(3)]) - a.trim_border * np.array(sp)
        # 面の全頂点が、同じ側の境界帯の中にあるものだけ捨てる (内部の面は残す)
        drop = np.zeros(len(F), bool)
        for i in range(3):
            drop |= np.all(V[F][:, :, i] < lo[i], axis=1) | np.all(V[F][:, :, i] > hi[i], axis=1)
        F = F[~drop]
        print(f"[mesh] trim-border: removed {int(drop.sum())} faces")
    if a.min_faces > 0 and len(F):
        n, lab = components(F, len(V))
        cnt = np.bincount(lab[F[:, 0]], minlength=n)
        keep = cnt[lab[F[:, 0]]] >= a.min_faces
        print(f"[mesh] components: {n}, removed {int((~keep).sum())} faces in pieces < {a.min_faces} faces")
        F = F[keep]
    if len(F) == 0:
        raise SystemExit("[error] no faces left")
    used = np.unique(F)  # 未使用頂点を落として詰め直す
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    if a.smooth > 0:
        V = smooth_laplacian(V, F, a.smooth)
        print(f"[mesh] laplacian smoothing x{a.smooth}")

    # 法線: 解析勾配があれば SDF の増加方向 (=自由空間側) を使う。無ければメッシュ法線。
    if G is not None:
        gv = trilinear(ax, G, V)
        gn = np.linalg.norm(gv, axis=1, keepdims=True)
        N = np.where(gn > 1e-6, gv / (gn + 1e-12), vertex_normals(V, F))  # 勾配が0/欠損の頂点はメッシュ法線で代用
    else:
        N = vertex_normals(V, F)
    C = None
    if a.color == "up":
        up = N[:, 2] >= np.cos(np.radians(a.max_angle))
        C = np.where(up[:, None], np.array([70, 130, 220], np.uint8), np.array([170, 170, 170], np.uint8)).astype(np.uint8)
        print(f"[color] upward-facing (within {a.max_angle:.0f} deg of +Z): {up.mean() * 100:.1f}% of vertices (blue)")
    elif a.color == "normal":
        C = ((N * 0.5 + 0.5) * 255).astype(np.uint8)

    base = os.path.splitext(a.out)[0] if a.out.lower().endswith((".ply", ".obj", ".stl")) else a.out
    for fmt in a.format:
        p = f"{base}.{fmt}"
        {"ply": lambda: write_ply(p, V, F, C), "obj": lambda: write_obj(p, V, F), "stl": lambda: write_stl(p, V, F)}[fmt]()
        print(f"[saved] {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    area = float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    print(f"[mesh] {len(V)} vertices, {len(F)} faces, area {area:.2f} m^2, bounds min {np.round(V.min(0), 2)} max {np.round(V.max(0), 2)}")


if __name__ == "__main__":
    main()
