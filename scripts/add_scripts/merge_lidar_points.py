#!/usr/bin/env python3
"""
ROS 2 bag 点群統合・フィルタ処理

入力:
  /velodyne_points
  /lidar_points

処理:
  1. センサー原点から30 mを超える点を除去
  2. URDFの外部パラメータを用いてbase_link座標系へ変換
  3. base_link座標系で自機領域を除去
  4. 両センサーの処理結果を/points_filteredへ出力
  5. 元の全トピックをそのままコピー

対応ストレージ:
  - SQLite3
  - MCAP

タイムスタンプと各センサーのメッセージレートは維持する。
出力ストレージ形式は入力bagと同じにする。

使用例:
  python3 merge_lidar_points.py input_bag output_bag
"""

import argparse
import copy
import math
import os
from pathlib import Path

import numpy as np
import rosbag2_py
import yaml

from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


# ============================================================
# 設定
# ============================================================

VELODYNE_TOPIC = "/velodyne_points"
LIDAR_TOPIC = "/lidar_points"
FILTERED_TOPIC = "/points_filtered"

FILTER_TOPICS = {
    VELODYNE_TOPIC,
    LIDAR_TOPIC,
}

MAX_DISTANCE = 30.0

# Velodyneのpitch角
# URDFのvelodyne_pitch_angleに合わせること
VELODYNE_PITCH = math.radians(-12.7)

# 自機領域（base_link座標系）
EGO_MIN = np.array([-2.0, -0.50, -1.0])
EGO_MAX = np.array([0.2, 0.50, 1.125])


# ============================================================
# Bagストレージ形式の取得
# ============================================================

def get_storage_id(bag_path: str) -> str:
    """metadata.yamlからbagのストレージ形式を取得する。"""

    metadata_path = Path(bag_path) / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.yamlが見つかりません: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    storage_id = metadata.get("rosbag2_bagfile_information", {}).get(
        "storage_identifier"
    )

    if storage_id not in ("sqlite3", "mcap"):
        raise ValueError(
            f"未対応のストレージ形式です: {storage_id}"
        )

    return storage_id


# ============================================================
# 座標変換
# ============================================================

def rotation_matrix_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """URDFの固定軸RPYから回転行列を生成する。"""

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])

    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return rz @ ry @ rx


# base_link -> velodyne
VELODYNE_TRANSLATION = np.array([
    0.0913,
    0.0,
    0.925,
])

VELODYNE_ROTATION = rotation_matrix_rpy(
    roll=0.0,
    pitch=VELODYNE_PITCH,
    yaw=math.pi,
)

# base_link -> hesai_lidar
HESAI_TRANSLATION = np.array([
    0.25,
    0.0,
    1.0,
])

HESAI_ROTATION = rotation_matrix_rpy(
    roll=math.pi,
    pitch=0.0,
    yaw=-0.105,
)


def get_sensor_transform(topic_name: str):
    """トピックに対応するbase_linkへの変換を返す。"""

    if topic_name == VELODYNE_TOPIC:
        return VELODYNE_ROTATION, VELODYNE_TRANSLATION

    if topic_name == LIDAR_TOPIC:
        return HESAI_ROTATION, HESAI_TRANSLATION

    raise ValueError(
        f"未対応の点群トピック: {topic_name}"
    )


# ============================================================
# 点群フィルタ
# ============================================================

def filter_pointcloud(
    msg: PointCloud2,
    topic_name: str,
) -> PointCloud2:
    """
    センサー座標系で距離制限を行い、
    base_linkへ変換後に自機除去する。
    """

    points = point_cloud2.read_points(
        msg,
        skip_nans=False,
    ).reshape(-1).copy()

    x_sensor = points["x"].astype(np.float64)
    y_sensor = points["y"].astype(np.float64)
    z_sensor = points["z"].astype(np.float64)

    # NaN / Infを除去
    valid = (
        np.isfinite(x_sensor) &
        np.isfinite(y_sensor) &
        np.isfinite(z_sensor)
    )

    # センサー原点から30 m以内
    distance_squared = (
        x_sensor * x_sensor +
        y_sensor * y_sensor +
        z_sensor * z_sensor
    )

    valid &= distance_squared <= MAX_DISTANCE * MAX_DISTANCE

    points = points[valid]

    # センサー座標系からbase_linkへ変換
    rotation, translation = get_sensor_transform(topic_name)

    xyz_sensor = np.column_stack((
        points["x"].astype(np.float64),
        points["y"].astype(np.float64),
        points["z"].astype(np.float64),
    ))

    xyz_base = xyz_sensor @ rotation.T + translation

    points["x"] = xyz_base[:, 0]
    points["y"] = xyz_base[:, 1]
    points["z"] = xyz_base[:, 2]

    x = xyz_base[:, 0]
    y = xyz_base[:, 1]
    z = xyz_base[:, 2]

    # base_link座標系で自機領域内の点を除去
    inside_ego = (
        (x >= EGO_MIN[0]) & (x <= EGO_MAX[0]) &
        (y >= EGO_MIN[1]) & (y <= EGO_MAX[1]) &
        (z >= EGO_MIN[2]) & (z <= EGO_MAX[2])
    )

    points = points[~inside_ego]

    # PointCloud2を更新
    filtered = copy.deepcopy(msg)

    filtered.header.frame_id = "base_link"
    filtered.height = 1
    filtered.width = len(points)
    filtered.row_step = filtered.width * filtered.point_step
    filtered.data = points.tobytes()
    filtered.is_dense = True

    return filtered


# ============================================================
# Bag処理
# ============================================================

def process_bag(
    input_bag: str,
    output_bag: str,
) -> None:

    if not os.path.exists(input_bag):
        raise FileNotFoundError(
            f"入力bagが見つかりません: {input_bag}"
        )

    if os.path.exists(output_bag):
        raise FileExistsError(
            f"出力先が既に存在します: {output_bag}"
        )

    # 入力bagのストレージ形式を取得
    storage_id = get_storage_id(input_bag)

    print(f"Input storage: {storage_id}")
    print(f"Output storage: {storage_id}")

    # Reader
    reader = rosbag2_py.SequentialReader()

    reader.open(
        rosbag2_py.StorageOptions(
            uri=input_bag,
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topics = reader.get_all_topics_and_types()

    topic_type_map = {
        topic.name: topic.type
        for topic in topics
    }

    for topic_name in FILTER_TOPICS:
        if topic_name not in topic_type_map:
            print(
                f"警告: 入力bagに {topic_name} がありません"
            )

        elif topic_type_map[topic_name] != "sensor_msgs/msg/PointCloud2":
            raise TypeError(
                f"{topic_name} の型がPointCloud2ではありません: "
                f"{topic_type_map[topic_name]}"
            )

    # Writer
    writer = rosbag2_py.SequentialWriter()

    writer.open(
        rosbag2_py.StorageOptions(
            uri=output_bag,
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    # 元の全トピックを登録
    for topic in topics:
        writer.create_topic(topic)

    # フィルタ後点群の出力トピックを登録
    if any(
        topic_name in topic_type_map
        for topic_name in FILTER_TOPICS
    ):

        # 既存トピックIDの最大値を取得
        max_topic_id = max(
            (topic.id for topic in topics),
            default=0,
        )

        new_topic_id = max_topic_id + 1

        # QoS設定を取得
        source_topic = (
            VELODYNE_TOPIC
            if VELODYNE_TOPIC in topic_type_map
            else LIDAR_TOPIC
        )

        source_metadata = next(
            topic for topic in topics
            if topic.name == source_topic
        )

        filtered_metadata = rosbag2_py.TopicMetadata(
            id=new_topic_id,
            name=FILTERED_TOPIC,
            type="sensor_msgs/msg/PointCloud2",
            serialization_format=source_metadata.serialization_format,
            offered_qos_profiles=source_metadata.offered_qos_profiles,
        )

        writer.create_topic(filtered_metadata)

    # 処理数
    counts = {
        VELODYNE_TOPIC: 0,
        LIDAR_TOPIC: 0,
    }

    # Bag読み込み・書き込み
    while reader.has_next():

        topic_name, data, timestamp = reader.read_next()

        # 元のメッセージをそのままコピー
        writer.write(
            topic_name,
            data,
            timestamp,
        )

        # 対象点群を処理
        if topic_name in FILTER_TOPICS:

            msg = deserialize_message(
                data,
                PointCloud2,
            )

            filtered_msg = filter_pointcloud(
                msg,
                topic_name,
            )

            # 共通トピックへ出力
            writer.write(
                FILTERED_TOPIC,
                serialize_message(filtered_msg),
                timestamp,
            )

            counts[topic_name] += 1

    del reader
    del writer

    print("\n========== Processing Result ==========")
    print(f"Output bag: {output_bag}")
    print(f"Storage: {storage_id}")
    print(f"{VELODYNE_TOPIC}: {counts[VELODYNE_TOPIC]} messages")
    print(f"{LIDAR_TOPIC}: {counts[LIDAR_TOPIC]} messages")
    print(f"Output topic: {FILTERED_TOPIC}")
    print("Output frame_id: base_link")
    print("=======================================")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 bagのVelodyne/Hesai点群をbase_linkへ変換し、"
            "30 m制限と自機除去を適用する"
        )
    )

    parser.add_argument(
        "input_bag",
        help="入力bagディレクトリ",
    )

    parser.add_argument(
        "output_bag",
        help="出力bagディレクトリ",
    )

    args = parser.parse_args()

    process_bag(
        args.input_bag,
        args.output_bag,
    )


if __name__ == "__main__":
    main()