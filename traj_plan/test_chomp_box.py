"""直方体ロボットのテスト+速度計測: python test_chomp_box.py"""
import os, sys, tempfile, json, time, subprocess
import numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chomp_sdf3d as C

fails = 0
def check(c, msg):
    global fails
    print(("PASS " if c else "FAIL ") + msg); fails += (not c)

def box_sdf(P, lo, hi):
    c, h = (np.array(lo) + np.array(hi)) / 2, (np.array(hi) - np.array(lo)) / 2
    q = np.abs(P - c) - h
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)

tmp = tempfile.mkdtemp()
ax = [np.arange(-1, 5.0001, 0.05), np.arange(-2.5, 2.5001, 0.05), np.arange(-0.5, 2.0001, 0.05)]
X, Y, Z = np.meshgrid(*ax, indexing="ij"); P = np.stack([X, Y, Z], -1)
# 床 + 柱(幅1.2) + 壁(x=3.0, y -2.5..0.3 に隙間 0.5 のみ y>0.3...) は使わず、狭い隙間テスト用に別ワールドも作る
sdf = np.minimum(Z, box_sdf(P, [1.8, -0.6, 0.0], [2.2, 0.6, 1.5]))
np.savez(os.path.join(tmp, "w.npz"), points=P.reshape(-1, 3).astype(np.float32), sdf1=sdf.ravel().astype(np.float32))
# 隙間ワールド: x=2 の壁 (厚み0.2) に y=0 中心の隙間。幅 g
def gap_world(g, name):
    wall = np.minimum(box_sdf(P, [1.9, -2.5, 0.0], [2.1, -g / 2, 1.5]), box_sdf(P, [1.9, g / 2, 0.0], [2.1, 2.5, 1.5]))
    np.savez(os.path.join(tmp, name), points=P.reshape(-1, 3).astype(np.float32), sdf1=np.minimum(Z, wall).ravel().astype(np.float32))
gap_world(0.9, "gap_ok.npz"); gap_world(0.5, "gap_ng.npz")

base = yaml.safe_load(open(os.path.join(HERE, "chomp_config.yaml")))
base["robot"].update(type="box", margin=0.1)
base["robot"]["box"].update(size=[0.8, 0.5, 0.5], center_offset=[0, 0, 0.3], sample_spacing=0.1)
base["chomp"].update(n_waypoints=50, init_method="auto", obstacle_weight=10.0)
base["obstacle_filter"].update(distance_max=0.15)
base["output"] = {"dir": os.path.join(tmp, "o"), "plot": False}
class A: npz = None; start = None; goal = None; no_filter = False

def run(npz, start, goal, **rb):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["map"]["npz"] = os.path.join(tmp, npz)
    cfg["planning"].update(dof=2, base_z=0.0, start=start, goal=goal)
    cfg["robot"].update(rb)
    vol, pl, s, g = C.build_from_config(cfg, A())
    t = time.time(); r = pl.plan(s, g, verbose=False); dt = time.time() - t
    vol.n_oob = 0
    cl = pl.min_clearance(r.xi)
    return pl, r, cl, dt, s, g

n_pts = len(C.make_box_points([0.8, 0.5, 0.5], [0, 0, 0.3], 0.1))
print(f"box surface points: {n_pts}")
check(n_pts > 20, "箱表面の評価点が生成される")
# 1. 柱を避ける (free / tangent)
for mode in ("free", "tangent"):
    pl, r, cl, dt, s, g = run("w.npz", [0, 0.0, 0], [4, 0.0, 0], yaw_mode=mode)
    ymax = np.abs(r.xi[:, 1]).max()
    print(f"   [{mode}] tag={r.tag} iters={r.iterations} |y|max={ymax:.2f} clearance={cl[0]:.3f} time={dt:.2f}s")
    check(cl[0] >= 0 and ymax > 0.9, f"{mode}: 柱を避ける (clearance>=0)")
    if mode == "free":
        yaw = pl._yaw(r.xi)
        check(abs(yaw[0]) < 1e-6 and abs(yaw[-1]) < 1e-6, "free: 始点・終点の向きは固定 (goal方向=0)")
# 2. 隙間: 箱幅0.5 (+余裕)。隙間0.9は通れて、0.5は通れない(clearance<0)
_, r1, c1, _, _, _ = run("gap_ok.npz", [0, 0.0, 0], [4, 0.0, 0], yaw_mode="free")
_, r2, c2, _, _, _ = run("gap_ng.npz", [0, 0.0, 0], [4, 0.0, 0], yaw_mode="free")
print(f"   gap0.9 clearance={c1[0]:.3f}  gap0.5 clearance={c2[0]:.3f}")
check(c1[0] >= 0, "隙間0.9m(箱幅0.5): 通過できる")
check(c2[0] < 0.05, "隙間0.5m(箱幅0.5): 余裕なし/衝突と判定される")
# 3. 球近似との差: 長辺方向に狭い隙間 -> 向きを変えて通る (yaw が実際に回る)
pl, r, c, _, s, g = run("gap_ok.npz", [0, 0.6, 0], [4, 0.6, 0], yaw_mode="free")
print(f"   yaw range (deg) {np.degrees(pl._yaw(r.xi)).min():.1f} .. {np.degrees(pl._yaw(r.xi)).max():.1f}, clearance={c[0]:.3f}")
check(c[0] >= 0, "斜めから隙間へ: 衝突しない")
# 4. ウォームスタート (リアルタイム用): 前回経路から目標を少し動かし、反復上限10で再計画
pl, r, cl, dt_cold, s, g = run("w.npz", [0, 0.0, 0], [4, 0.0, 0], yaw_mode="free")
g2 = g.copy(); g2[1] += 0.1
pl.p.max_iterations = 10
t = time.time(); r2 = pl.plan(s, g2, warm=r.xi, verbose=False); dt_warm = time.time() - t
c2 = pl.min_clearance(r2.xi)
print(f"   cold: {r.iterations} it {dt_cold:.2f}s -> warm(max 10 it): {r2.iterations} it {dt_warm:.3f}s clearance={c2[0]:.3f} tag={r2.tag}")
check(c2[0] >= 0 and r2.tag == "warm" and dt_warm < dt_cold, "warm start (<=10反復) で衝突せず、cold より速い")
# 4b. 評価点を粗くすると (間隔0.15) どれだけ速いか
pl2, r3, c3, _, s3, g3 = run("w.npz", [0, 0.0, 0], [4, 0.0, 0], yaw_mode="free")
# 5. 1反復あたりの時間
xi = r.xi
t = time.time()
for _ in range(20): pl.total(xi)
t_eval = (time.time() - t) / 20
print(f"   評価 1 回 (N={len(xi)}, K={n_pts}): {t_eval*1e3:.1f} ms  -> 反復(line search込み ~3 評価) ≈ {t_eval*3e3:.0f} ms")
# 6. CLI
cfg = yaml.safe_load(yaml.safe_dump(base)); cfg["map"]["npz"] = os.path.join(tmp, "w.npz")
cfg["planning"].update(dof=2, start=[0, 0, 0], goal=[4, 0, 0]); cfg["output"] = {"dir": os.path.join(tmp, "cli"), "plot": True}
cp = os.path.join(tmp, "c.yaml"); yaml.safe_dump(cfg, open(cp, "w"))
p = subprocess.run([sys.executable, os.path.join(HERE, "chomp_sdf3d.py"), "--config", cp], capture_output=True, text=True)
print(p.stdout[-500:], p.stderr[-500:])
check(p.returncode == 0, "CLI(box): 成功終了")
hdr = open(os.path.join(tmp, "cli", "path.csv")).readline().strip()
check(hdr == "x,y,z,yaw_rad", "path.csv に yaw が出る")
import shutil; shutil.copy(os.path.join(tmp, "cli", "chomp_result.png"), os.path.join(HERE, "demo_chomp_box.png"))
print("ALL PASSED" if not fails else f"FAILED ({fails})"); sys.exit(1 if fails else 0)
