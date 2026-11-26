"""OREN の学習済み .pth を読み込んで3次元SDFを問い合わせる / 2つの .pth を比較する。

使い方（OREN のリポジトリを入れた環境、GPU ありで実行）:
  # 1つのモデルの動作確認
  python oren_query.py --config <学習に使った yaml> --ckpt final.pth

  # 2回の走行の比較
  python oren_query.py --config <学習に使った yaml> --ckpt run1/final.pth --ckpt2 run2/final.pth

注意:
  - config は学習時と同じものを使うこと（八分木と残差ネットワークの構成が合わないと読み込めない）。
  - このスクリプトは trainer.py の Trainer.query_sdf と同じ呼び出し
    (OrenEvaluator.forward_model) を使っているが、実機では未検証。
"""
import argparse

import numpy as np

from oren import torch
from oren.evaluator_oren import OrenEvaluator
from oren.model import SdfNetwork
from oren.trainer_config import TrainerConfig


def load_model(cfg, ckpt_path):
    model = SdfNetwork(cfg.model).to(cfg.device)
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()
    evaluator = OrenEvaluator(
        batch_size=cfg.batch_size,
        clean_mesh=cfg.clean_mesh,
        model_cfg=cfg.model,
        model=model,
        device=cfg.device,
    )
    return model, evaluator, state


def query(model, evaluator, cfg, pts, grad=False, chunk=50000):
    """pts: (N, 3) numpy（八分木の座標系）。戻り値: sdf (N,), grad (N, 3) or None"""
    sdf, g = [], []
    for i in range(0, len(pts), chunk):
        p = torch.as_tensor(pts[i : i + chunk], dtype=torch.float32, device=cfg.device)
        out = evaluator.forward_model(
            model, p, get_grad=grad, auto_grad=True, prior_only=False, device=cfg.device
        )
        if i == 0 and grad:
            gd = out.get("grad")
            if isinstance(gd, dict):
                print("[debug] grad の中身:",
                      {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in gd.items()})
        sdf.append(out["sdf"].detach().cpu().numpy().reshape(-1))
        if grad:
            gd = out["grad"]
            if isinstance(gd, dict):
                if "sdf" not in gd:
                    raise KeyError(f"grad の中に 'sdf' がありません。keys={list(gd.keys())}")
                gd = gd["sdf"]
            g.append(gd.detach().cpu().numpy().reshape(-1, 3))
    return np.concatenate(sdf), (np.concatenate(g) if grad else None)


def observed_bbox(state, margin):
    """最小サイズのボクセル（表面近傍）の中心から、観測された範囲を求める。"""
    v = state["octree.voxels"]
    c = state["octree.voxel_centers"]
    used = (v != 0).any(dim=1)
    smallest = v[used][:, 3].min()
    c = c[used & (v[:, 3] == smallest)].cpu().numpy()
    return c.min(0) - margin, c.max(0) + margin


def make_grid(lo, hi, step):
    axes = [np.arange(l, h + 1e-6, step) for l, h in zip(lo, hi)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt2", default=None)
    ap.add_argument("--step", type=float, default=0.2, help="問い合わせ格子の間隔 [m]")
    ap.add_argument("--margin", type=float, default=0.35, help="観測範囲に足す余白 [m]")
    ap.add_argument("--near", type=float, default=0.5, help="差分を評価する |SDF1| の上限 [m]")
    ap.add_argument("--out", default="query_result.npz")
    args = ap.parse_args()

    # trainer.py の main() と同じ方法で config を読む
    cfg = TrainerConfig.get_argparser().parse_args(["--config", args.config])

    model1, ev1, state1 = load_model(cfg, args.ckpt)

    # 1) 動作確認: 最小ボクセルの中心でのSDF（表面近傍なので小さい値になるはず）
    v = state1["octree.voxels"]
    c = state1["octree.voxel_centers"]
    used = (v != 0).any(dim=1)
    small = used & (v[:, 3] == v[used][:, 3].min())
    centers = c[small].cpu().numpy()
    s, g = query(model1, ev1, cfg, centers, grad=True)
    print(f"[check] 最小ボクセル中心 {len(centers)} 点: |sdf| 平均={np.abs(s).mean():.3f} 最大={np.abs(s).max():.3f}")
    print(f"[check] 勾配ノルム 平均={np.linalg.norm(g, axis=1).mean():.3f}（Euclidean SDF なら 1 付近）")

    # 2) 観測範囲の格子でSDFを問い合わせる
    lo, hi = observed_bbox(state1, args.margin)
    print(f"[grid] 範囲 min={lo} max={hi} 間隔={args.step} m")
    pts = make_grid(lo, hi, args.step)
    sdf1, _ = query(model1, ev1, cfg, pts)
    result = {"points": pts, "sdf1": sdf1}

    # 3) 2つ目があれば差分
    if args.ckpt2:
        model2, ev2, _ = load_model(cfg, args.ckpt2)
        sdf2, _ = query(model2, ev2, cfg, pts)
        delta = sdf2 - sdf1
        near = np.abs(sdf1) < args.near
        print(f"[diff] 格子点 {len(pts)} 点中、|sdf1|<{args.near} の点: {near.sum()}")
        d = np.abs(delta[near])
        print(f"[diff] |Δ| 平均={d.mean():.3f} 中央値={np.median(d):.3f} 95%={np.percentile(d, 95):.3f} 最大={d.max():.3f}")
        result.update(sdf2=sdf2, delta=delta, near_mask=near)

    np.savez(args.out, **result)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()