#!/usr/bin/env python3
"""
/lidar_points を購読し、自機領域内の点を除去して
/filtered_points として同じレートで配信するノード。

パラメータ:
    input_topic   (string)   入力トピック            既定: /lidar_points
    output_topic  (string)   出力トピック            既定: /filtered_points
    ego_min       (double[]) 自機除去範囲の最小 [x,y,z] [m]
    ego_max       (double[]) 自機除去範囲の最大 [x,y,z] [m]
    log_interval  (double)   統計ログの表示間隔 [s]（0で非表示）
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class EgoFilterNode(Node):

    def __init__(self):
        super().__init__("ego_filter_node")

        # パラメータ
        self.declare_parameter("input_topic", "/lidar_points")
        self.declare_parameter("output_topic", "/points_filtered")
        self.declare_parameter("ego_min", [-2.0, -0.40, -1.0])
        self.declare_parameter("ego_max", [0.2, 0.40, 1.125])
        self.declare_parameter("log_interval", 5.0)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.ego_min = np.array(self.get_parameter("ego_min").value, dtype=np.float64)
        self.ego_max = np.array(self.get_parameter("ego_max").value, dtype=np.float64)
        self.log_interval = float(self.get_parameter("log_interval").value)

        if self.ego_min.shape != (3,) or self.ego_max.shape != (3,):
            raise ValueError("ego_min / ego_max は [x, y, z] の3要素で指定してください")

        # 入力: BEST_EFFORT で購読（RELIABLE / BEST_EFFORT どちらの配信元にも接続できる）
        self.sub = self.create_subscription(
            PointCloud2, input_topic, self.callback, qos_profile_sensor_data
        )

        # 出力: RELIABLE（RViz の既定設定でそのまま表示できる）
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, pub_qos)

        # 統計
        self.count = 0
        self.sum_input = 0
        self.sum_removed = 0
        self.last_log = self.get_clock().now()

        self.get_logger().info(f"入力: {input_topic}  →  出力: {output_topic}")
        self.get_logger().info(
            "自機除去範囲: "
            f"X[{self.ego_min[0]}, {self.ego_max[0]}] "
            f"Y[{self.ego_min[1]}, {self.ego_max[1]}] "
            f"Z[{self.ego_min[2]}, {self.ego_max[2]}] m"
        )

    def callback(self, cloud: PointCloud2):
        field_names = {f.name for f in cloud.fields}
        if not {"x", "y", "z"} <= field_names:
            self.get_logger().error(
                "PointCloud2 に x, y, z フィールドがありません",
                throttle_duration_sec=5.0,
            )
            return

        # 構造化配列として取得（全フィールド・point_step を保持）
        points_data = point_cloud2.read_points(cloud, skip_nans=False)
        if points_data.ndim > 1:
            points_data = points_data.reshape(-1)

        n_input = len(points_data)

        if n_input > 0:
            xyz = np.column_stack(
                (points_data["x"], points_data["y"], points_data["z"])
            ).astype(np.float64)

            finite = np.isfinite(xyz).all(axis=1)
            inside = np.all((xyz >= self.ego_min) & (xyz <= self.ego_max), axis=1)
            keep = ~(finite & inside)

            filtered = points_data[keep]
            kept_xyz = xyz[keep]
        else:
            filtered = points_data
            kept_xyz = np.empty((0, 3))

        # 出力メッセージ作成（ヘッダ・フィールド構成は入力と同じ）
        out = PointCloud2()
        out.header = cloud.header
        out.fields = cloud.fields
        out.is_bigendian = cloud.is_bigendian
        out.point_step = cloud.point_step
        out.height = 1
        out.width = len(filtered)
        out.row_step = out.point_step * out.width
        out.data = filtered.tobytes() if len(filtered) > 0 else b""
        out.is_dense = bool(np.isfinite(kept_xyz).all()) if len(filtered) > 0 else True

        self.pub.publish(out)

        # 統計ログ
        self.count += 1
        self.sum_input += n_input
        self.sum_removed += n_input - len(filtered)

        if self.log_interval > 0:
            now = self.get_clock().now()
            elapsed = (now - self.last_log).nanoseconds * 1e-9
            if elapsed >= self.log_interval:
                hz = self.count / elapsed
                ratio = (self.sum_removed / self.sum_input * 100.0) if self.sum_input else 0.0
                self.get_logger().info(
                    f"{hz:.1f} Hz | 平均入力 {self.sum_input // max(self.count, 1)} 点 | "
                    f"自機除去 {ratio:.2f}%"
                )
                self.count = 0
                self.sum_input = 0
                self.sum_removed = 0
                self.last_log = now


def main(args=None):
    rclpy.init(args=args)
    node = EgoFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
