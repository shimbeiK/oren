"""query_result.npz から、指定座標の周辺の SDF 勾配 (∇SDF) を可視化する。

勾配の取得元 (優先順):
  1) npz に勾配配列がある場合:  --grad-key grad_sdf (N,3)   … モデルの解析勾配。こちらを推奨
  2) 無い場合: SDF 格子 (--key, 既定 sdf1) から中心差分で計算 (格子間隔が粗いと近似になる)

出力 (--out-prefix):
  <prefix>_slice.png  中心高さの水平断面: SDF の色 + 勾配の矢印(向き=最も増える方向) + SDF=0 の線
  <prefix>_3d.png     周辺の3D矢印 (表面近くの点だけ)
  <prefix>_grad.ply   頂点=位置, 法線=勾配方向, 色=|∇SDF| (CloudCompare で矢印/法線表示)
  stdout              |∇SDF| の統計 (正しい SDF なら障害物近傍の自由空間で約1)

使い方:
  python visualize_gradient.py query_result.npz --center 1.0 0.5 1.0 --radius 1.5
  python visualize_gradient.py query_result.npz --center 1.0 0.5 1.0 --radius 1.5 --grad-key grad_sdf
注意: SDF の向きは「自由空間で正、物体の内側/裏側で負」。矢印は SDF が増える方向 = 物体から遠ざかる向き。
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def to_volume(pts, vals):
    ax = [np.unique(np.round(pts[:, i], 4)) for i in range(3)]
    shape = [len(a) for a in ax]
    vol = np.full(shape + list(vals.shape[1:]), np.nan)
    idx = [np.searchsorted(ax[i], np.round(pts[:, i], 4)) for i in range(3)]
    vol[idx[0], idx[1], idx[2]] = vals
    return ax, vol


def central_diff(ax, vol):
    """格子上の SDF から ∇SDF を計算 (端は片側差分, NaN は NaN のまま)"""
    g = [np.gradient(vol, ax[i], axis=i) for i in range(3)]
    return np.stack(g, axis=-1)


def write_ply(path, p, n, mag):
    c = plt.get_cmap("viridis")(np.clip(mag / 1.5, 0, 1))[:, :3]
    c = (c * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(p))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for a, b, cc in zip(p, n, c):
            f.write("%.4f %.4f %.4f %.4f %.4f %.4f %d %d %d\n" % (*a, *b, *cc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--center", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--radius", type=float, default=1.5, help="中心からの距離 [m] (球)。--box で直方体にも")
    ap.add_argument("--box", type=float, nargs=3, default=None, metavar=("HX", "HY", "HZ"),
                    help="指定時は半径の代わりに中心±(HX,HY,HZ) の直方体")
    ap.add_argument("--key", default="sdf1", help="SDF のキー名")
    ap.add_argument("--grad-key", default=None, help="勾配 (N,3) のキー名。無ければ SDF から差分計算")
    ap.add_argument("--stride", type=int, default=2, help="矢印の間引き (格子点おき)")
    ap.add_argument("--shell", type=float, default=0.5, help="3D図で |SDF|<=shell の点だけ描く [m]")
    ap.add_argument("--lim", type=float, default=1.0, help="色の範囲 ±lim")
    ap.add_argument("--out-prefix", default="grad_view")
    a = ap.parse_args()

    d = np.load(a.npz)
    pts = d["points"].astype(np.float64)
    sdf = d[a.key].astype(np.float64)
    ax, vol = to_volume(pts, sdf)
    if a.grad_key:
        g = d[a.grad_key].astype(np.float64).reshape(-1, 3)
        _, gvol = to_volume(pts, g)
        src = f"npz の {a.grad_key} (解析勾配)"
    else:
        gvol = central_diff(ax, vol)
        src = "SDF 格子の中心差分"
    print(f"勾配の取得元: {src}")

    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    c = np.array(a.center)
    if a.box is None:
        sel = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= a.radius ** 2
    else:
        h = np.array(a.box)
        sel = (np.abs(X - c[0]) <= h[0]) & (np.abs(Y - c[1]) <= h[1]) & (np.abs(Z - c[2]) <= h[2])
    ok = sel & np.isfinite(vol) & np.isfinite(gvol).all(-1)
    if not ok.any():
        raise SystemExit("指定範囲に有効な格子点がありません。--center/--radius と npz の範囲を確認してください")
    P = np.stack([X[ok], Y[ok], Z[ok]], 1)
    S = vol[ok]
    G = gvol[ok]
    M = np.linalg.norm(G, axis=1)
    near = np.abs(S) < 1.0
    print(f"範囲内の点数={len(P)}  SDF: min={S.min():.3f} max={S.max():.3f}")
    print(f"|∇SDF|: 中央値={np.median(M):.3f}  p10={np.percentile(M,10):.3f}  p90={np.percentile(M,90):.3f}"
          f"  (|SDF|<1 の点: 中央値={np.median(M[near]) if near.any() else float('nan'):.3f})")

    # ---- 水平断面 (中心に最も近い高さ) ----
    kz = int(np.argmin(np.abs(ax[2] - c[2])))
    zc = ax[2][kz]
    s2 = vol[:, :, kz]
    g2 = gvol[:, :, kz, :]
    mask2 = ok[:, :, kz]
    fig, axp = plt.subplots(figsize=(8, 7))
    im = axp.pcolormesh(ax[0], ax[1], s2.T, cmap="RdBu", norm=TwoSlopeNorm(0, -a.lim, a.lim), shading="nearest")
    if np.nanmin(s2) < 0 < np.nanmax(s2):
        axp.contour(ax[0], ax[1], s2.T, levels=[0], colors="k", linewidths=1.5)
    ii, jj = np.where(mask2)
    sel2 = (ii % a.stride == 0) & (jj % a.stride == 0)
    ii, jj = ii[sel2], jj[sel2]
    gx, gy = g2[ii, jj, 0], g2[ii, jj, 1]
    n = np.hypot(gx, gy)
    n[n < 1e-9] = 1.0
    axp.quiver(ax[0][ii], ax[1][jj], gx / n, gy / n, np.hypot(gx, gy), cmap="viridis", clim=(0, 1.5),
               angles="xy", scale_units="xy", scale=1 / (np.diff(ax[0]).mean() * a.stride * 0.8), width=0.003)
    axp.plot(*c[:2], "m*", ms=14)
    axp.set_aspect("equal")
    axp.set_xlim(c[0] - (a.radius if a.box is None else a.box[0]), c[0] + (a.radius if a.box is None else a.box[0]))
    axp.set_ylim(c[1] - (a.radius if a.box is None else a.box[1]), c[1] + (a.radius if a.box is None else a.box[1]))
    axp.set_title(f"SDF と ∇SDF の向き (z={zc:.2f} m, 矢印色=|∇SDF|, 黒線=SDF 0)")
    axp.set_xlabel("x [m]"); axp.set_ylabel("y [m]")
    fig.colorbar(im, ax=axp, label="SDF [m]")
    fig.savefig(a.out_prefix + "_slice.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 3D 矢印 ----
    idx = np.where(np.abs(S) <= a.shell)[0][:: max(1, a.stride ** 2)]
    fig = plt.figure(figsize=(8, 7))
    a3 = fig.add_subplot(111, projection="3d")
    if len(idx):
        Gn = G[idx] / np.maximum(M[idx, None], 1e-9)
        cs = a3.quiver(P[idx, 0], P[idx, 1], P[idx, 2], Gn[:, 0], Gn[:, 1], Gn[:, 2],
                       length=np.diff(ax[0]).mean() * a.stride, normalize=False,
                       colors=plt.get_cmap("RdBu")(TwoSlopeNorm(0, -a.shell, a.shell)(S[idx])), linewidth=0.7)
    a3.scatter(*c, c="m", marker="*", s=120)
    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z")
    a3.set_title(f"∇SDF (|SDF|<={a.shell} m, 色=SDF符号, 向き=遠ざかる方向)  n={len(idx)}")
    fig.savefig(a.out_prefix + "_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    write_ply(a.out_prefix + "_grad.ply", P, G / np.maximum(M[:, None], 1e-9), M)
    print("出力:", a.out_prefix + "_slice.png", a.out_prefix + "_3d.png", a.out_prefix + "_grad.ply")


if __name__ == "__main__":
    main()