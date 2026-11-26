"""oren_query.py が作った query_result.npz（3次元格子のSDF）を3次元で可視化する。

出力（--out-prefix で名前を変えられる。ライブラリが無いものは自動でスキップ）:
  <prefix>_slices.png  水平断面（複数の高さ）と垂直断面（x, y）。SDF=0 を黒線で表示
  <prefix>_surface.ply SDF=0 の等値面（メッシュ）。MeshLab / CloudCompare / Blender で開く
                       （scikit-image が必要）
  <prefix>_points.ply  SDF の色付き点群（表面の近く |SDF|<=--shell のみ）。CloudCompare などで開く
  <prefix>.html        回転・拡大できる対話表示（plotly が入っている場合のみ）

使い方:
  python visualize_sdf3d.py query_result.npz --bbox -5.1 -5.1 -0.4 6.1 5.1 2.1
  # 床を除いて障害物だけ見たい場合は、bbox の ZMIN を床より上げる
  python visualize_sdf3d.py query_result.npz --bbox -5.1 -5.1 0.2 6.1 5.1 2.1

注意: SDF=0 の面には、床・天井（学習範囲に入っていれば）も含まれる。学習範囲の外の値は
      学習されていないので、必ず --bbox で学習の bound_min / bound_max（+0.1 m 程度）を指定する。
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def load_volume(path, key, bbox):
    d = np.load(path)
    pts, sdf = d["points"].astype(np.float64), d[key].astype(np.float64)
    if bbox is not None:
        x0, y0, z0, x1, y1, z1 = bbox
        k = ((pts[:, 0] >= x0 - 1e-6) & (pts[:, 0] <= x1 + 1e-6) &
             (pts[:, 1] >= y0 - 1e-6) & (pts[:, 1] <= y1 + 1e-6) &
             (pts[:, 2] >= z0 - 1e-6) & (pts[:, 2] <= z1 + 1e-6))
        pts, sdf = pts[k], sdf[k]
    ax = [np.unique(np.round(pts[:, i], 4)) for i in range(3)]
    vol = np.full([len(a) for a in ax], np.nan)
    idx = [np.searchsorted(ax[i], np.round(pts[:, i], 4)) for i in range(3)]
    vol[idx[0], idx[1], idx[2]] = sdf
    return ax, vol


def save_slices(ax, vol, path, n_z, lim):
    xs, ys, zs = ax
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    zi = np.unique(np.linspace(0, len(zs) - 1, n_z).round().astype(int))
    cols = max(len(zi), 2)
    fig, axs = plt.subplots(2, cols, figsize=(3.6 * cols, 7.2), squeeze=False)
    for j, k in enumerate(zi):
        a = axs[0, j]
        im = a.imshow(vol[:, :, k].T, origin="lower", cmap="RdBu", norm=norm,
                      extent=(xs[0], xs[-1], ys[0], ys[-1]))
        if np.nanmin(vol[:, :, k]) < 0 < np.nanmax(vol[:, :, k]):
            a.contour(xs, ys, vol[:, :, k].T, levels=[0.0], colors="k", linewidths=0.8)
        a.set_title(f"z = {zs[k]:.2f} m", fontsize=9)
        a.set_aspect("equal")
    cx, cy = len(xs) // 2, len(ys) // 2
    for j, (name, sl, a1, a2) in enumerate([
            (f"x = {xs[cx]:.2f} m (y-z)", vol[cx, :, :], ys, zs),
            (f"y = {ys[cy]:.2f} m (x-z)", vol[:, cy, :], xs, zs)]):
        a = axs[1, j]
        a.imshow(sl.T, origin="lower", cmap="RdBu", norm=norm,
                 extent=(a1[0], a1[-1], a2[0], a2[-1]), aspect="equal")
        if np.nanmin(sl) < 0 < np.nanmax(sl):
            a.contour(a1, a2, sl.T, levels=[0.0], colors="k", linewidths=0.8)
        a.set_title(name, fontsize=9)
    for j in range(2, cols):
        axs[1, j].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.6, label="SDF [m]  (blue: free, red: occupied)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_ply(path, verts, faces=None, colors=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, v in enumerate(verts):
            line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"
            if colors is not None:
                line += " %d %d %d" % tuple(colors[i])
            f.write(line + "\n")
        if faces is not None:
            for t in faces:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def sdf_colors(v, lim):
    c = plt.get_cmap("RdBu")(TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)(v))[:, :3]
    return (c * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--key", default="sdf1")
    ap.add_argument("--bbox", type=float, nargs=6, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    ap.add_argument("--n-z", type=int, default=5, help="水平断面の枚数")
    ap.add_argument("--lim", type=float, default=1.5, help="カラーバーの範囲 ±[m]")
    ap.add_argument("--shell", type=float, default=0.3, help="点群PLYに残す |SDF| の上限 [m]")
    ap.add_argument("--out-prefix", default="sdf3d")
    a = ap.parse_args()

    ax, vol = load_volume(a.npz, a.key, a.bbox)
    xs, ys, zs = ax
    print(f"[volume] {vol.shape}  x[{xs[0]:.2f},{xs[-1]:.2f}] y[{ys[0]:.2f},{ys[-1]:.2f}] "
          f"z[{zs[0]:.2f},{zs[-1]:.2f}]  SDF[{np.nanmin(vol):.2f},{np.nanmax(vol):.2f}]  欠損={int(np.isnan(vol).sum())}")

    save_slices(ax, vol, f"{a.out_prefix}_slices.png", a.n_z, a.lim)
    print(f"[save] {a.out_prefix}_slices.png")

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    m = np.isfinite(vol) & (np.abs(vol) <= a.shell)
    P = np.stack([X[m], Y[m], Z[m]], axis=1)
    write_ply(f"{a.out_prefix}_points.ply", P, colors=sdf_colors(vol[m], a.shell))
    print(f"[save] {a.out_prefix}_points.ply  ({len(P)} 点, |SDF|<={a.shell})")

    verts = faces = None
    try:
        from skimage.measure import marching_cubes
        v0 = np.nan_to_num(vol, nan=np.nanmax(vol))  # 欠損は「空き」扱い
        step = [float(np.median(np.diff(q))) for q in ax]
        verts, faces, _, _ = marching_cubes(v0, level=0.0, spacing=step)
        verts += np.array([xs[0], ys[0], zs[0]])
        write_ply(f"{a.out_prefix}_surface.ply", verts, faces=faces)
        print(f"[save] {a.out_prefix}_surface.ply  (頂点 {len(verts)}, 面 {len(faces)})")
    except ImportError:
        print("[skip] scikit-image が無いので等値面メッシュは作りません (pip install scikit-image)")
    except ValueError as e:
        print(f"[skip] 等値面なし: {e}")

    try:
        import plotly.graph_objects as go
        traces = [go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=np.nan_to_num(vol).ravel(),
                            isomin=-a.lim, isomax=a.lim, opacity=0.08, surface_count=12,
                            colorscale="RdBu", caps=dict(x_show=False, y_show=False, z_show=False))]
        if verts is not None:
            traces.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="gray", opacity=0.6))
        fig = go.Figure(traces)
        fig.update_layout(scene=dict(aspectmode="data"))
        fig.write_html(f"{a.out_prefix}.html")
        print(f"[save] {a.out_prefix}.html")
    except ImportError:
        print("[skip] plotly が無いので HTML は作りません (pip install plotly)")


if __name__ == "__main__":
    main()
