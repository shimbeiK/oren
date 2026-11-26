#!/usr/bin/env python3
"""ORENの3次元SDF(query_result.npz)を使ったオフラインCHOMP経路計画 (障害物コストに除外フィルタ付き)。

入力  : oren_query.py が作る npz (points: (N,3) 規則格子, sdf1: (N,) [, grad_sdf: (N,3)])
出力  : path.csv, result.json, chomp_result.png (+ 任意で filter_excluded.ply)

障害物コスト除外フィルタ (YAML: obstacle_filter):
    次の2条件を同時に満たす評価点を、障害物コスト(値と勾配)の計算から除外する。
      (1) distance_min <= d(x) <= distance_max          (既定 0.0 <= d <= 0.2 m, 障害物内部 d<0 は対象外)
      (2) (∇d/|∇d|)·direction >= cos(max_angle_deg)       (既定 +Z から ±30°以内 = ∂d/∂z/|∇d| >= 0.866)
    元の SDF / 勾配の配列は一切変更しない (評価点ごとにマスクを作るだけ)。

ロボット: 球の集合 (基準位置からのオフセット + 半径)。球の中心で d(x)・∇d を評価する。
    surface距離 dist = d(center) - radius  に CHOMP の障害物コストをかける。

使い方:  python chomp_sdf3d.py --config chomp_config.yaml [--no-filter] [--start x y z --goal x y z]
"""
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml


# ----------------------------------------------------------------------------------------------------------------------
# SDF volume
# ----------------------------------------------------------------------------------------------------------------------
class SdfVolume:
    """規則格子上の SDF と勾配を三線形補間する。配列は読み取り専用として扱う。"""

    def __init__(self, ax, sdf, grad):
        self.ax = ax  # [x_axis, y_axis, z_axis]
        self.h = np.array([float(np.mean(np.diff(a))) for a in ax])
        self.origin = np.array([a[0] for a in ax])
        self.shape = np.array([len(a) for a in ax])
        self.sdf = sdf  # (nx,ny,nz)
        self.grad = grad  # (nx,ny,nz,3)
        self.sdf.setflags(write=False)
        self.grad.setflags(write=False)
        self.n_oob = 0  # 補間時に格子外へ出たサンプル数 (診断用)
        self.oob_example = None

    @staticmethod
    def from_npz(path, points_key="points", sdf_key="sdf1", grad_key=None, bbox=None, nan_fill=1.0):
        d = np.load(path)
        pts = d[points_key].astype(np.float64)
        sdf = d[sdf_key].astype(np.float64).reshape(-1)
        grad = d[grad_key].astype(np.float64).reshape(-1, 3) if grad_key else None
        if bbox is not None:
            b = np.asarray(bbox, dtype=np.float64)
            keep = np.all((pts >= b[:3] - 1e-6) & (pts <= b[3:] + 1e-6), axis=1)
            pts, sdf = pts[keep], sdf[keep]
            grad = grad[keep] if grad is not None else None
        ax = [np.unique(np.round(pts[:, i], 4)) for i in range(3)]
        for i, a in enumerate(ax):
            if len(a) < 2 or not np.allclose(np.diff(a), np.diff(a).mean(), atol=1e-3):
                raise ValueError(f"axis {i} is not a regular grid (n={len(a)})")
        shape = tuple(len(a) for a in ax)
        idx = tuple(np.searchsorted(ax[i], np.round(pts[:, i], 4)) for i in range(3))
        vol = np.full(shape, np.nan)
        vol[idx] = sdf
        n_nan = int(np.isnan(vol).sum())
        if n_nan:
            print(f"[map] {n_nan} NaN/missing cells ({100.0 * n_nan / vol.size:.1f}%) -> filled with {nan_fill}")
            vol[np.isnan(vol)] = nan_fill
        if grad is not None:
            gv = np.zeros(shape + (3,))
            gv[idx] = grad
            gv[~np.isfinite(gv)] = 0.0
        else:
            h = [float(np.mean(np.diff(a))) for a in ax]
            gv = np.stack([np.gradient(vol, h[i], axis=i) for i in range(3)], axis=-1)
        return SdfVolume(ax, vol, gv)

    def _lookup(self, vol, P):
        u = (P - self.origin) / self.h
        hi = (self.shape - 1).astype(np.float64)
        oob = np.any((u < -1e-9) | (u > hi + 1e-9), axis=1)
        if oob.any():
            self.n_oob += int(oob.sum())
            self.oob_example = P[oob][0]
        u = np.clip(u, 0.0, hi - 1e-9)
        i0 = np.floor(u).astype(np.int64)
        f = u - i0
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    w = wx * wy * wz
                    v = vol[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                    out = out + (w[:, None] * v if v.ndim == 2 else w * v)
        return out

    def query(self, P):
        """P: (M,3) -> (d (M,), grad (M,3))  (勾配は SDF 自身の勾配。正規化しない)"""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        return self._lookup(self.sdf, P), self._lookup(self.grad, P)


# ----------------------------------------------------------------------------------------------------------------------
# Obstacle filter
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ObstacleFilter:
    enabled: bool = True
    distance_min: float = 0.0  # 障害物内部 (d<0) は除外しない
    distance_max: float = 0.2
    max_angle_deg: float = 30.0
    direction: tuple = (0.0, 0.0, 1.0)
    distance_reference: str = "center"  # center: d(center) / surface: d(center)-radius

    def __post_init__(self):
        v = np.asarray(self.direction, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("obstacle_filter.direction must be non-zero")
        self._dir = v / n
        self._cos = math.cos(math.radians(self.max_angle_deg))
        if self.distance_reference not in ("center", "surface"):
            raise ValueError("distance_reference must be 'center' or 'surface'")

    def mask(self, d, grad):
        """True = 障害物コストから除外する評価点。d: (...,), grad: (...,3)。入力は変更しない。"""
        if not self.enabled:
            return np.zeros(np.shape(d), dtype=bool)
        gn = np.linalg.norm(grad, axis=-1)
        valid = gn > 1e-8  # 勾配が定義できない点は除外しない(安全側)
        cosang = np.where(valid, (grad @ self._dir) / np.where(valid, gn, 1.0), -1.0)
        return valid & (d >= self.distance_min) & (d <= self.distance_max) & (cosang >= self._cos)


# ----------------------------------------------------------------------------------------------------------------------
# CHOMP
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ChompParams:
    n_waypoints: int = 60
    max_iterations: int = 300
    eta: float = 50.0
    smoothness_weight: float = 1.0
    obstacle_weight: float = 10.0
    tolerance: float = 1e-6
    restarts: int = 0
    restart_noise: float = 0.3
    seed: int = 0
    # 初期経路: straight=直線 / astar=格子A* / auto=直線で失敗(衝突)したときだけA*
    init_method: str = "auto"
    init_resolution: float = 0.1  # A* の格子間隔 [m]
    init_clearance: float = 0.05  # A* で要求する球表面の最小距離 [m] (見つからなければ自動で 0.05→0.02→0 に下げて再探索)
    # 接触の禁止: 最終経路の最小クリアランス(フィルタ後)がこの値以上でなければ「失敗」とし、path.csv は出力しない
    required_clearance: float = 0.02
    # 要求を満たさないとき、障害物の重みを repair_weight_factor 倍にして最大 max_repairs 回やり直す
    max_repairs: int = 5
    repair_weight_factor: float = 3.0


@dataclass
class Robot:
    offsets: np.ndarray  # (K,3) 車体座標(x前,y左,z上)での評価点オフセット (基準位置から)
    radii: np.ndarray  # (K,) 各評価点の半径 (箱では点の被覆半径)
    margin: float = 0.15  # CHOMP の ε (表面からの安全距離)
    is_box: bool = False  # True: 向き(yaw)で評価点が回転する
    yaw_mode: str = "free"  # free: yaw も最適化 / tangent: yaw=経路の接線方向 (勾配は yaw に流さない近似)
    yaw_scale: float = 0.5  # [m/rad] yaw を長さ単位にそろえる係数 (平滑化の重み付け)


def make_box_points(size, center_offset, spacing):
    """直方体の表面を格子状にサンプルした点 (K,3)。車体座標で、center_offset が箱の中心。"""
    size = np.asarray(size, dtype=np.float64)
    n = [max(2, int(math.ceil(v / spacing)) + 1) for v in size]
    axes = [np.linspace(-v / 2, v / 2, k) for v, k in zip(size, n)]
    pts = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    on_surface = np.any(np.isclose(np.abs(pts), size / 2, atol=1e-9), axis=1)
    return pts[on_surface] + np.asarray(center_offset, dtype=np.float64)


@dataclass
class Result:
    xi: np.ndarray
    cost: float
    cost_obs: float
    cost_smooth: float
    iterations: int
    history: list = field(default_factory=list)


class Chomp:
    """状態 xi: (N, D) = [x, y(, z)] (+ 箱で yaw=free のとき最後の列に yaw_scale*yaw)"""

    def __init__(self, vol, robot, obs_filter, params, dof=2, base_z=0.0):
        self.vol, self.robot, self.flt, self.p = vol, robot, obs_filter, params
        self.dof, self.base_z = dof, base_z
        self.pos = dof  # 位置の次元
        self.has_yaw = bool(robot.is_box and robot.yaw_mode == "free")
        self.D = self.pos + (1 if self.has_yaw else 0)
        n = params.n_waypoints - 2
        A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        self.A = A
        self.Ainv = np.linalg.inv(A)

    # --- geometry ---
    def _world(self, xi):
        if self.pos == 2:
            return np.column_stack([xi[:, :2], np.full(len(xi), self.base_z)])
        return xi[:, :3]

    def _yaw(self, xi):
        if not self.robot.is_box:
            return np.zeros(len(xi))
        if self.has_yaw:
            return xi[:, -1] / self.robot.yaw_scale
        if len(xi) < 2:  # 1点だけ (診断用) のときは接線が定義できない
            return np.zeros(len(xi))
        t = np.gradient(xi[:, :2], axis=0)  # tangent mode
        return np.unwrap(np.arctan2(t[:, 1], t[:, 0]))

    def _points(self, W, yaw):
        """基準位置 W (N,3)・向き yaw (N,) -> 評価点のワールド座標 (N,K,3) と dP/dyaw (N,K,3)"""
        off = self.robot.offsets
        c, s_ = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        Rx = c * off[None, :, 0] - s_ * off[None, :, 1]
        Ry = s_ * off[None, :, 0] + c * off[None, :, 1]
        Z = np.broadcast_to(off[None, :, 2], Rx.shape)
        P = W[:, None, :] + np.stack([Rx, Ry, Z], axis=-1)
        dP = np.stack([-Ry, Rx, np.zeros_like(Rx)], axis=-1)
        return P, dP

    def _eval_full(self, W, yaw):
        P, dP = self._points(W, yaw)
        d, g = self.vol.query(P.reshape(-1, 3))
        d = d.reshape(P.shape[:2])
        g = g.reshape(P.shape)
        dist = d - self.robot.radii[None, :]
        ref = d if self.flt.distance_reference == "center" else dist
        excluded = self.flt.mask(ref, g)
        return dist, g, excluded, d, dP

    def eval_samples(self, xi):
        """全 waypoint x 評価点 で (dist, grad, excluded, d_center) を返す"""
        dist, g, excl, d, _ = self._eval_full(self._world(xi), self._yaw(xi))
        return dist, g, excl, d

    def _cost_terms(self, dist, excluded):
        eps = self.robot.margin
        c = np.where(dist < 0, -dist + eps / 2, np.where(dist <= eps, (dist - eps) ** 2 / (2 * eps), 0.0))
        dc = np.where(dist < 0, -1.0, np.where(dist <= eps, (dist - eps) / eps, 0.0))
        c = np.where(excluded, 0.0, c)  # フィルタ: 該当点は値も勾配も 0
        dc = np.where(excluded, 0.0, dc)
        return c, dc

    # --- objective ---
    def smooth_cost(self, xi):
        return 0.5 * float(np.sum(np.diff(xi, axis=0) ** 2))

    def obstacle_cost_and_grad(self, xi):
        N = xi.shape[0]
        P_ = self.pos
        dist, g, excl, _, dP = self._eval_full(self._world(xi), self._yaw(xi))
        c, dc = self._cost_terms(dist, excl)
        C = c.sum(axis=1)  # (N,)
        G = (dc[:, :, None] * g).sum(axis=1)[:, :P_]  # (N,pos) 作業空間勾配 (評価点の和)
        xp = xi[:, :P_]
        v = 0.5 * (xp[2:] - xp[:-2])
        a = xp[2:] - 2 * xp[1:-1] + xp[:-2]
        sp = np.linalg.norm(v, axis=1) + 1e-9
        vh = v / sp[:, None]
        Gi, Ci = G[1:-1], C[1:-1]
        Gp = Gi - vh * np.sum(vh * Gi, axis=1, keepdims=True)  # (I - v^ v^T) G
        ap = a - vh * np.sum(vh * a, axis=1, keepdims=True)
        kappa = ap / (sp[:, None] ** 2)
        grad = np.zeros((N - 2, self.D))
        grad[:, :P_] = sp[:, None] * (Gp - Ci[:, None] * kappa)
        if self.has_yaw:  # yaw の勾配: sum_k dc_k * (g_k . dP_k/dyaw) を長さ単位に換算
            Gth = (dc * np.einsum("nkj,nkj->nk", g, dP)).sum(axis=1)
            grad[:, -1] = sp * Gth[1:-1] / self.robot.yaw_scale
        cost = float(np.sum(Ci * sp))
        return cost, grad

    def total(self, xi):
        co, _ = self.obstacle_cost_and_grad(xi)
        cs = self.smooth_cost(xi)
        return self.p.smoothness_weight * cs + self.p.obstacle_weight * co, co, cs

    # --- optimisation ---
    def optimise(self, xi0):
        p = self.p
        xi = xi0.copy()
        b = np.zeros_like(xi[1:-1])
        b[0] -= xi[0]
        b[-1] -= xi[-1]
        cost, co, cs = self.total(xi)
        step, hist, stall = 1.0, [cost], 0
        it = 0
        for it in range(1, p.max_iterations + 1):
            _, go = self.obstacle_cost_and_grad(xi)
            grad = p.smoothness_weight * (self.A @ xi[1:-1] + b) + p.obstacle_weight * go
            delta = -(self.Ainv @ grad) / p.eta
            accepted = False
            for _ in range(8):
                xn = xi.copy()
                xn[1:-1] = xi[1:-1] + step * delta
                cn, con, csn = self.total(xn)
                if cn <= cost:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            rel = (cost - cn) / max(abs(cost), 1e-12)
            xi, cost, co, cs = xn, cn, con, csn
            step = min(step * 1.5, 1.0)
            hist.append(cost)
            stall = stall + 1 if rel < p.tolerance else 0
            if stall >= 5:
                break
        return Result(xi, cost, co, cs, it, hist)

    def initial(self, start, goal, rng=None, noise=0.0):
        N = self.p.n_waypoints
        s = np.linspace(0.0, 1.0, N)[:, None]
        xi = start[None, :] * (1 - s) + goal[None, :] * s
        if noise > 0 and rng is not None:
            for k in (1, 2, 3):  # 低周波の摂動 (端点は固定)
                xi = xi + noise / k * rng.standard_normal(xi.shape[1])[None, :] * np.sin(k * np.pi * s)
        return xi

    def with_yaw_from_tangent(self, xp, start, goal):
        """位置だけの経路 xp (N,pos) に、接線方向の yaw 列を足す (free のとき)。端点は start/goal の yaw。"""
        if not self.has_yaw:
            return xp
        t = np.gradient(xp[:, :2], axis=0)
        yaw = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        y0, y1 = start[-1] / self.robot.yaw_scale, goal[-1] / self.robot.yaw_scale
        ramp = np.linspace(0, 1, len(yaw))
        yaw = yaw + (y0 - yaw[0]) * (1 - ramp) + (y1 - yaw[-1]) * ramp  # 端点を一致させる
        return np.column_stack([xp, yaw * self.robot.yaw_scale])

    def astar_init(self, start, goal):
        """球が(フィルタ後に)衝突しない格子セルだけを通る A* 経路を返す。見つからなければ None。"""
        import heapq

        D = self.pos
        res = self.p.init_resolution
        lo = np.array([a[0] for a in self.vol.ax])[:D]
        hi = np.array([a[-1] for a in self.vol.ax])[:D]
        start, goal = start[:D], goal[:D]
        axes = [np.arange(lo[i], hi[i] + 1e-9, res) for i in range(D)]
        shape = tuple(len(a) for a in axes)
        grids = np.meshgrid(*axes, indexing="ij")
        Pg = np.stack([g.ravel() for g in grids], axis=1)
        free = np.zeros(len(Pg), dtype=bool)
        yaws = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4] if self.robot.is_box else [0.0]  # 箱は向きに依らず通れる格子だけ
        for i in range(0, len(Pg), 20000):
            Wc = self._world(np.column_stack([Pg[i:i + 20000]]))
            ok = np.ones(len(Wc), dtype=bool)
            for yw in yaws:
                dist, _, excl, _, _ = self._eval_full(Wc, np.full(len(Wc), yw))
                ok &= np.all(excl | (dist >= self.p.init_clearance), axis=1)
            free[i:i + 20000] = ok
        free = free.reshape(shape)

        def cell(p):
            return tuple(int(np.clip(round((p[i] - lo[i]) / res), 0, shape[i] - 1)) for i in range(D))

        def snap(c):
            if free[c]:
                return c
            best, bd = None, 1e9
            for r in range(1, 8):  # 近傍の自由セルへ
                for off in np.ndindex(*(2 * r + 1,) * D):
                    n = tuple(c[i] + off[i] - r for i in range(D))
                    if all(0 <= n[i] < shape[i] for i in range(D)) and free[n]:
                        dd = sum((n[i] - c[i]) ** 2 for i in range(D))
                        if dd < bd:
                            best, bd = n, dd
                if best is not None:
                    return best
            return None

        cs, cg = snap(cell(start)), snap(cell(goal))
        if cs is None or cg is None:
            return None
        nbr = [o for o in np.ndindex(*(3,) * D) if any(x != 1 for x in o)]
        nbr = [tuple(x - 1 for x in o) for o in nbr]
        gscore, parent, heap = {cs: 0.0}, {}, [(0.0, cs)]
        while heap:
            _, c = heapq.heappop(heap)
            if c == cg:
                break
            for o in nbr:
                n = tuple(c[i] + o[i] for i in range(D))
                if not all(0 <= n[i] < shape[i] for i in range(D)) or not free[n]:
                    continue
                ng = gscore[c] + math.sqrt(sum(x * x for x in o))
                if ng < gscore.get(n, 1e18):
                    gscore[n], parent[n] = ng, c
                    h = math.sqrt(sum((n[i] - cg[i]) ** 2 for i in range(D)))
                    heapq.heappush(heap, (ng + h, n))
        if cg not in parent and cg != cs:
            return None
        path, c = [cg], cg
        while c != cs:
            c = parent[c]
            path.append(c)
        pts = np.array([[lo[i] + c[i] * res for i in range(D)] for c in path[::-1]])
        pts = np.vstack([start[None, :], pts, goal[None, :]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, s[-1], self.p.n_waypoints)
        return np.column_stack([np.interp(t, s, pts[:, i]) for i in range(D)])

    def plan(self, start, goal, warm=None, verbose=True):
        """warm: 前回の経路 (N,D)。与えると端点だけ start/goal に合わせてそこから最適化する (リアルタイム用)。"""
        rng = np.random.default_rng(self.p.seed)
        self.astar_path = None
        method = self.p.init_method
        cands = []

        self.candidates = cands  # 各候補 (res.init_xi = 最適化前の初期経路, res.xi = 最適化後)

        def run(xi0, tag):
            res = self.optimise(xi0)
            res.init_xi = np.array(xi0, copy=True)
            res.collision = self.min_clearance(res.xi)[0]
            res.tag = tag
            cands.append(res)
            return res

        req = self.p.required_clearance
        feasible = lambda: [c for c in cands if c.collision >= req]
        if warm is not None:
            w = np.asarray(warm, dtype=np.float64).copy()
            if len(w) != self.p.n_waypoints:  # 点数が違えば弧長で再サンプル
                seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
                sa = np.concatenate([[0.0], np.cumsum(seg)])
                tt = np.linspace(0, sa[-1], self.p.n_waypoints)
                w = np.column_stack([np.interp(tt, sa, w[:, i]) for i in range(w.shape[1])])
            ramp = np.linspace(0, 1, len(w))[:, None]
            w = w + (start - w[0])[None, :] * (1 - ramp) + (goal - w[-1])[None, :] * ramp  # 端点のずれを線形に配分
            run(w, "warm")
            if feasible():
                return min(feasible(), key=lambda c: c.cost)
        if method in ("straight", "auto"):
            run(self.initial(start, goal), "straight")
        if method == "astar" or (method == "auto" and not feasible()):
            xp = None
            c0 = self.p.init_clearance
            for cl_try in dict.fromkeys([c0, min(c0, 0.05), min(c0, 0.02), 0.0]):  # 重複除去しつつ順に緩める
                self.p.init_clearance = cl_try
                xp = self.astar_init(start, goal)
                if xp is not None:
                    if cl_try != c0:
                        print(f"[info] A*: no path with clearance {c0:.2f} m; found with {cl_try:.2f} m")
                    break
            self.p.init_clearance = c0
            if xp is None:
                print("[warn] A* initial path not found")
            else:
                self.astar_path = np.array(xp, copy=True)  # A* の生の経路 (位置のみ・最適化前)
                run(self.with_yaw_from_tangent(xp, start, goal), "astar")
        for r in range(self.p.restarts):
            run(self.initial(start, goal, rng, self.p.restart_noise), f"restart{r + 1}")
        if not cands:
            run(self.initial(start, goal), "straight")
        best = min(feasible() or cands, key=lambda c: c.cost) if feasible() else max(cands, key=lambda c: c.collision)
        # 接触が残る/余裕が足りない場合: 障害物の重みを上げて、その経路からやり直す
        w0 = self.p.obstacle_weight
        try:
            for k in range(self.p.max_repairs):
                if best.collision >= req:
                    break
                self.p.obstacle_weight = w0 * self.p.repair_weight_factor ** (k + 1)
                r2 = self.optimise(best.xi)
                r2.init_xi, r2.tag = best.xi.copy(), f"repair{k + 1}(w={self.p.obstacle_weight:g})"
                r2.collision = self.min_clearance(r2.xi)[0]
                cands.append(r2)
                if r2.collision > best.collision:
                    best = r2
        finally:
            self.p.obstacle_weight = w0
        best.feasible = bool(best.collision >= req)
        if verbose:
            print("[plan] candidates: " + ", ".join(f"{c.tag}(cost={c.cost:.3f},clear={c.collision:.3f})" for c in cands) + f" -> {best.tag}")
        return best

    # --- evaluation of a path ---
    def densify(self, xi, spacing=0.02):
        seg = np.linalg.norm(np.diff(xi, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
        t = np.linspace(0, s[-1], n)
        return np.column_stack([np.interp(t, s, xi[:, i]) for i in range(xi.shape[1])])

    def min_clearance(self, xi):
        """(フィルタ後に残る点での最小 surface 距離, フィルタ無しの最小 surface 距離, 除外サンプル数, 全サンプル数)"""
        dense = self.densify(xi)
        dist, _, excl, _ = self.eval_samples(dense)
        active = np.where(excl, np.inf, dist)
        return float(active.min()), float(dist.min()), int(excl.sum()), int(excl.size)


# ----------------------------------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------------------------------
def build_from_config(cfg, args):
    m = cfg["map"]
    vol = SdfVolume.from_npz(
        args.npz or m["npz"],
        points_key=m.get("points_key", "points"),
        sdf_key=m.get("sdf_key", "sdf1"),
        grad_key=m.get("grad_key"),
        bbox=m.get("bbox"),
        nan_fill=m.get("nan_fill", 1.0),
    )
    pl = cfg["planning"]
    dof = int(pl.get("dof", 2))
    base_z = float(pl.get("base_z", 0.0))
    rc = cfg["robot"]
    margin = float(rc.get("margin", 0.15))
    if rc.get("type", "spheres") == "box":
        b = rc["box"]
        spacing = float(b.get("sample_spacing", 0.1))
        pts = make_box_points(b["size"], b.get("center_offset", [0.0, 0.0, 0.0]), spacing)
        # 点の被覆半径: 点間隔の半分の対角 + 追加の余裕。点を球とみなして隙間を埋める
        rad = b.get("point_radius")
        rad = spacing * 0.5 * math.sqrt(2) if rad is None else float(rad)
        robot = Robot(pts, np.full(len(pts), rad), margin, True, rc.get("yaw_mode", "free"), float(rc.get("yaw_scale", 0.5)))
    else:
        sph = np.asarray(rc["spheres"], dtype=np.float64).reshape(-1, 4)
        robot = Robot(sph[:, :3], sph[:, 3], margin)
    f = cfg.get("obstacle_filter", {})
    flt = ObstacleFilter(
        enabled=False if args.no_filter else bool(f.get("enabled", True)),
        distance_min=float(f.get("distance_min", 0.0)),
        distance_max=float(f.get("distance_max", 0.2)),
        max_angle_deg=float(f.get("max_angle_deg", 30.0)),
        direction=tuple(f.get("direction", [0.0, 0.0, 1.0])),
        distance_reference=f.get("distance_reference", "center"),
    )
    c = cfg.get("chomp", {})
    params = ChompParams(**{k: c[k] for k in c if k in ChompParams.__dataclass_fields__})
    start = np.asarray(args.start or pl["start"], dtype=np.float64)
    goal = np.asarray(args.goal or pl["goal"], dtype=np.float64)
    if dof == 2:
        start, goal = start[:2], goal[:2]
    planner = Chomp(vol, robot, flt, params, dof, base_z)
    if planner.has_yaw:  # 状態の最後の列に yaw_scale*yaw。未指定なら start→goal の方向
        yaw_s = pl.get("start_yaw_deg")
        yaw_g = pl.get("goal_yaw_deg")
        head = math.atan2(goal[1] - start[1], goal[0] - start[0])
        ys = math.radians(yaw_s) if yaw_s is not None else head
        yg = math.radians(yaw_g) if yaw_g is not None else head
        k = robot.yaw_scale
        start = np.append(start, ys * k)
        goal = np.append(goal, yg * k)
    return vol, planner, start, goal


def export_filter_ply(vol, flt, path, radius_ref=0.0):
    X, Y, Z = np.meshgrid(*vol.ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    d, g = vol.sdf.ravel(), vol.grad.reshape(-1, 3)
    m = flt.mask(d - (radius_ref if flt.distance_reference == "surface" else 0.0), g)
    Q = P[m]
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(Q)}\nproperty float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for q in Q:
            fh.write("%.4f %.4f %.4f 255 60 60\n" % tuple(q))
    return int(m.sum())


def plot(planner, vol, res, start, goal, path_png, dense):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    W = planner._world(dense)
    dd = planner.densify(res.xi, 0.1)
    dist, _, excl, _ = planner.eval_samples(dd)
    Wc = planner._world(dd)
    P, _ = planner._points(Wc, planner._yaw(dd))
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # top view: slice at mean sphere height
    zc = float(np.mean(P[..., 2]))
    start, goal = start[:planner.pos], goal[:planner.pos]
    kz = int(np.argmin(np.abs(vol.ax[2] - zc)))
    im = axs[0].pcolormesh(vol.ax[0], vol.ax[1], vol.sdf[:, :, kz].T, cmap="RdBu", norm=TwoSlopeNorm(0, -1, 1), shading="nearest")
    for c_ in getattr(planner, "candidates", []):
        if getattr(c_, "tag", "") == getattr(res, "tag", None) and hasattr(c_, "init_xi") and c_.tag != "warm":
            Wi = planner._world(c_.init_xi)
            axs[0].plot(Wi[:, 0], Wi[:, 1], "--", color="tab:green", lw=1.5, label=f"initial ({c_.tag})")
    axs[0].plot(W[:, 0], W[:, 1], "k-", lw=2, label="optimised")
    axs[0].legend(loc="upper left", fontsize=8)
    axs[0].plot(*W[0, :2], "go", ms=10)
    axs[0].plot(*W[-1, :2], "m*", ms=14)
    rmax = float(planner.robot.radii.max())
    if planner.robot.is_box:  # 箱の外形 (評価点の凸包ではなく軸平行の外形) を向きつきで描く
        off = planner.robot.offsets[:, :2]
        lo, hi = off.min(0), off.max(0)
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])
        yw = planner._yaw(dd)
        for i in range(0, len(Wc), max(1, len(Wc) // 12)):
            c, s_ = np.cos(yw[i]), np.sin(yw[i])
            R = np.array([[c, -s_], [s_, c]])
            q = corners @ R.T + Wc[i, :2]
            axs[0].plot(q[:, 0], q[:, 1], "k-", lw=0.7)
    else:
        for q in Wc[:: max(1, len(Wc) // 12)]:
            axs[0].add_patch(plt.Circle(q[:2], rmax, fill=False, ec="k", lw=0.6))
    axs[0].set_aspect("equal")
    axs[0].set_title(f"top view (SDF slice z={vol.ax[2][kz]:.2f} m)")
    fig.colorbar(im, ax=axs[0], label="SDF [m]")
    # side view: slice at mean y
    ky = int(np.argmin(np.abs(vol.ax[1] - float(W[:, 1].mean()))))
    axs[1].pcolormesh(vol.ax[0], vol.ax[2], vol.sdf[:, ky, :].T, cmap="RdBu", norm=TwoSlopeNorm(0, -1, 1), shading="nearest")
    flat = P.reshape(-1, 3)
    ex = excl.reshape(-1)
    axs[1].scatter(flat[~ex, 0], flat[~ex, 2], s=10, c="k", label="obstacle cost applied")
    axs[1].scatter(flat[ex, 0], flat[ex, 2], s=26, marker="x", c="orange", label="excluded by filter")
    axs[1].set_aspect("equal")
    axs[1].legend(loc="upper right")
    axs[1].set_title(f"side view (SDF slice y={vol.ax[1][ky]:.2f} m); sphere centres along path")
    fig.savefig(path_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz")
    ap.add_argument("--start", type=float, nargs="+")
    ap.add_argument("--goal", type=float, nargs="+")
    ap.add_argument("--no-filter", action="store_true", help="フィルタを無効化 (比較用)")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = args.out_dir or cfg.get("output", {}).get("dir", "chomp_out")
    os.makedirs(out, exist_ok=True)

    vol, planner, start, goal = build_from_config(cfg, args)
    sdf_hash = hash(vol.sdf.tobytes())
    f = planner.flt
    print(f"[filter] enabled={f.enabled} {f.distance_min}<=d<={f.distance_max} m, angle<={f.max_angle_deg} deg "
          f"around {tuple(np.round(f._dir, 3))}, ref={f.distance_reference}")
    # 診断: SDF 格子の範囲と、スタート姿勢でのロボット評価点の範囲 (範囲外だと値が境界で代用され、結果が信頼できない)
    lo = np.array([v[0] for v in vol.ax]); hi = np.array([v[-1] for v in vol.ax])
    print(f"[grid] x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]  step={[round(float(v[1]-v[0]), 3) for v in vol.ax]}")
    for name, st in (("start", start), ("goal", goal)):
        xs = st[None, :]
        Pp, _ = planner._points(planner._world(xs), planner._yaw(xs))
        Pp = Pp.reshape(-1, 3)
        print(f"[robot@{name}] points x[{Pp[:,0].min():.2f},{Pp[:,0].max():.2f}] y[{Pp[:,1].min():.2f},{Pp[:,1].max():.2f}] "
              f"z[{Pp[:,2].min():.2f},{Pp[:,2].max():.2f}] (radius max {planner.robot.radii.max():.2f})")
        if np.any(Pp < lo) or np.any(Pp > hi):
            print(f"[warn] robot at {name} sticks out of the SDF grid: enlarge the npz range or check base_z / robot offsets")
    for name, st in (("start", start), ("goal", goal)):
        p3 = planner._world(st[None, :])[0]
        if np.any(p3[:2] < lo[:2]) or np.any(p3[:2] > hi[:2]):
            print(f"[error] {name} {np.round(p3, 2)} is outside the SDF grid x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}]: "
                  f"values there are clamped to the border and meaningless. Re-create the npz with a larger range.")
            return 2
    s0 = planner.min_clearance(np.vstack([start, start]))
    g0 = planner.min_clearance(np.vstack([goal, goal]))
    endpoint_bad = False
    for name, c in (("start", s0), ("goal", g0)):
        if c[0] < 0:
            print(f"[error] {name} is in contact (clearance {c[0]:.3f} m, ignoring excluded samples): "
                  f"no contact-free path can start/end here. Fix the {name} position, the robot box (floor clearance), base_z or the filter.")
            endpoint_bad = True
            # どの評価点が最も悪いか: 位置・d・勾配の z 成分・除外されたか (原因の切り分け用)
            st = (start if name == "start" else goal)[None, :]
            Pp, _ = planner._points(planner._world(st), planner._yaw(st))
            Pp = Pp.reshape(-1, 3)
            d_, g_ = vol.query(Pp)
            dist_ = d_ - planner.robot.radii
            gn = np.linalg.norm(g_, axis=1) + 1e-12
            ex_ = planner.flt.mask(d_ if planner.flt.distance_reference == "center" else dist_, g_)
            for k in np.argsort(dist_)[:5]:
                print(f"   worst@{name}: p=({Pp[k,0]:.2f},{Pp[k,1]:.2f},{Pp[k,2]:.2f}) d={d_[k]:.3f} dist={dist_[k]:.3f} "
                      f"dd/dz/|g|={g_[k,2]/gn[k]:.2f} excluded={bool(ex_[k])}")

    if endpoint_bad:
        return 2
    res = planner.plan(start, goal)
    vol.n_oob = 0  # 最適化途中の試行ステップで格子外に出た分は数えない (最終経路だけ数える)
    cl_f, cl_raw, n_ex, n_all = planner.min_clearance(res.xi)
    dense = planner.densify(res.xi)
    length = float(np.sum(np.linalg.norm(np.diff(dense[:, :planner.pos], axis=0), axis=1)))
    stats = dict(
        success=bool(cl_f >= planner.p.required_clearance), required_clearance_m=planner.p.required_clearance, iterations=res.iterations, cost=res.cost, cost_obstacle=res.cost_obs,
        cost_smooth=res.cost_smooth, path_length_m=length, min_clearance_filtered_m=cl_f,
        min_clearance_unfiltered_m=cl_raw, excluded_samples=n_ex, total_samples=n_all,
        out_of_grid_samples=vol.n_oob, filter_enabled=f.enabled, init_used=getattr(res, 'tag', ''),
    )
    assert hash(vol.sdf.tobytes()) == sdf_hash  # 元のSDFは不変
    W = planner._world(res.xi)
    hdr = "x,y,z"
    if planner.robot.is_box:  # 箱は向きも出力 (tangent のときは接線方向から計算)
        W = np.column_stack([W, planner._yaw(res.xi)])
        hdr += ",yaw_rad"
    pcsv = os.path.join(out, "path.csv")
    if os.path.exists(pcsv):
        os.remove(pcsv)  # 前回の結果が残って「使える経路」と誤認されないように
    if stats["success"]:
        np.savetxt(pcsv, W, delimiter=",", header=hdr, comments="")
    else:  # 接触が残る経路は path.csv として出さない (確認用に別名で保存)
        np.savetxt(os.path.join(out, "path_INFEASIBLE.csv"), W, delimiter=",", header=hdr, comments="")
        print("[error] no contact-free path found: path.csv is NOT written (see path_INFEASIBLE.csv for inspection only)")
    if cfg.get("output", {}).get("save_init", True):
        # 初期経路を全候補ぶん保存 (init_<tag>.csv) + A* の生経路 (init_astar_raw.csv)
        def _save(name, xi_):
            Wi = planner._world(xi_)
            h = "x,y,z"
            if planner.robot.is_box:
                Wi = np.column_stack([Wi, planner._yaw(xi_)]); h += ",yaw_rad"
            np.savetxt(os.path.join(out, name), Wi, delimiter=",", header=h, comments="")
        saved = []
        for c_ in getattr(planner, "candidates", []):
            if hasattr(c_, "init_xi"):
                _save(f"init_{c_.tag}.csv", c_.init_xi); saved.append(f"init_{c_.tag}.csv")
        if getattr(planner, "astar_path", None) is not None:
            _save("init_astar_raw.csv", planner.astar_path); saved.append("init_astar_raw.csv")
        stats["saved_init_paths"] = saved
    json.dump(stats, open(os.path.join(out, "result.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    if vol.n_oob:
        print(f"[warn] {vol.n_oob} samples were outside the SDF grid (values clamped to the border), e.g. {np.round(vol.oob_example, 3)}")
    if cfg.get("output", {}).get("plot", True):
        plot(planner, vol, res, start, goal, os.path.join(out, "chomp_result.png"), dense)
    if cfg.get("output", {}).get("export_filter_ply", False):
        n = export_filter_ply(vol, f, os.path.join(out, "filter_excluded.ply"))
        print(f"[filter] excluded grid cells: {n} -> filter_excluded.ply")
    return 0 if stats["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
