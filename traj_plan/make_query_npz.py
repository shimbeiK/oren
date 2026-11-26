#!/usr/bin/env python3
"""稼働中の OREN (/query_sdf サービス) から規則格子の SDF を取得して query_result.npz に保存する。

使い方 (OREN のマッピングノードを起動した状態で):
  python3 make_query_npz.py --min -1 -3 -0.3 --max 6 3 1.5 --res 0.05 --out query_result.npz
  python3 make_query_npz.py ... --grad        # 解析勾配も保存 (grad_sdf)。リクエストが重くなる

出力キー: points (N,3) float32 / sdf1 (N,) float32 / grad_sdf (N,3) (--grad 時)
座標は OREN のマップ座標系 (odom と同じ) のまま。
"""
import argparse, time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from oren_msgs.srv import QuerySdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--max", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--res", type=float, default=0.05, help="格子間隔 [m]")
    ap.add_argument("--chunk", type=int, default=20000, help="1リクエストの点数")
    ap.add_argument("--service", default="/query_sdf")
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--prior", action="store_true", help="事前分布(SDF prior)だけを問い合わせる")
    ap.add_argument("--out", default="query_result.npz")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    ax = [np.arange(lo, hi + 1e-9, a.res) for lo, hi in zip(a.min, a.max)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    n = len(P)
    print(f"grid {[len(v) for v in ax]} = {n} points, chunks of {a.chunk}")

    rclpy.init()
    node = Node("make_query_npz")
    cli = node.create_client(QuerySdf, a.service)
    if not cli.wait_for_service(timeout_sec=10.0):
        raise SystemExit(f"service {a.service} not available (OREN のマッピングノードは起動していますか)")

    sdf = np.full(n, np.nan, dtype=np.float32)
    grad = np.full((n, 3), np.nan, dtype=np.float32) if a.grad else None
    t0 = time.time()
    for i in range(0, n, a.chunk):
        req = QuerySdf.Request()
        req.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in P[i:i + a.chunk]]
        req.return_grad = bool(a.grad)
        req.prior = bool(a.prior)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=a.timeout)
        res = fut.result()
        m = len(P[i:i + a.chunk])
        if res is None or len(res.sdf) != m:
            print(f"[warn] chunk {i}: no/short response ({0 if res is None else len(res.sdf)}/{m}); left as NaN")
            continue
        sdf[i:i + m] = np.asarray(res.sdf, dtype=np.float32)
        if a.grad and len(res.grad) == m:
            grad[i:i + m] = np.asarray([[g.x, g.y, g.z] for g in res.grad], dtype=np.float32)
        print(f"  {min(i + m, n)}/{n}  ({time.time() - t0:.0f}s)", end="\r", flush=True)
    print()
    out = dict(points=P, sdf1=sdf)
    if a.grad:
        out["grad_sdf"] = grad
    np.savez(a.out, **out)
    print(f"saved {a.out}: NaN={int(np.isnan(sdf).sum())}/{n}, SDF range [{np.nanmin(sdf):.2f}, {np.nanmax(sdf):.2f}]")
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
