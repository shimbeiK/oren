
#!/usr/bin/env python3
"""
ROS 2 bag 前処理

入力:
  /tf
  /tf_static
  /velodyne_points
  /lidar_points
  /odom

出力:
  /tf              : 約33 Hzに間引き
  /tf_static       : そのまま保持
  /odom            : そのまま保持
  /points_filtered : Velodyne + Hesaiの統合・フィルタ済み点群

点群処理:
  1. センサー原点から30 m以内の点を抽出
  2. センサー座標系からbase_link座標系へ変換
  3. base_link座標系で自機領域を除去
  4. 2つのLiDARの点群を時間同期して統合
  5. /points_filteredへ出力

統合条件:
  - 2つの点群のタイムスタンプ差が50 ms以内
  - 統合後の座標系はbase_link
  - 共通フィールドはx, y, z, intensity
  - 統合後のタイムスタンプは新しい方を使用

その他:
  - 元の点群トピックは出力しない
  - 点群の元タイムスタンプを基準に同期
  - SQLite3 / MCAPに対応
  - ストレージ形式はmetadata.yamlから自動判別

使用例:
  python3 filter_and_extract_bag.py input_bag output_bag
  python3 filter_and_extract_bag.py input_bag output_bag --storage-id mcap
"""

import argparse
import copy
import math
import os
from pathlib import Path
from collections import deque

import numpy as np
import rosbag2_py
import yaml

from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


# ============================================================
# 設定
# ============================================================

TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
ODOM_TOPIC = "/odom"

VELODYNE_TOPIC = "/velodyne_points"
LIDAR_TOPIC = "/lidar_points"

FILTERED_TOPIC = "/points_filtered"

KEEP_TOPICS = {
    TF_TOPIC,
    TF_STATIC_TOPIC,
    ODOM_TOPIC,
    VELODYNE_TOPIC,
    LIDAR_TOPIC,
}

POINTCLOUD_TOPICS = {
    VELODYNE_TOPIC,
    LIDAR_TOPIC,
}

# /tfの間引き周期
THROTTLE_PERIOD_NS = {
    TF_TOPIC: 30_000_000,  # 約33 Hz
    ODOM_TOPIC: 20_000_000,   # 33 Hz
}

# 点群の最大距離 [m]
MAX_DISTANCE = 30.0

# 点群の同期許容時間 [ns]
SYNC_SLOP_NS = 100_000_000  # 50 ms

# センサーごとの最大バッファ数
MAX_BUFFER_SIZE = 50

# Velodyneのpitch角
# URDFのvelodyne_pitch_angleに合わせること
VELODYNE_PITCH = math.radians(-12.7)

# 自機領域（base_link座標系）
EGO_MIN = np.array([
    -2.0,
    -0.70,
    -1.0,
])

EGO_MAX = np.array([
    0.3,
    0.70,
    2.0,
])


# ============================================================
# Bagストレージ形式
# ============================================================

def get_storage_id(bag_path: str, requested_id: str) -> str:
    """入力bagのストレージ形式を取得する。"""

    if requested_id != "auto":
        if requested_id not in ("sqlite3", "mcap"):
            raise ValueError(
                f"未対応のストレージ形式です: {requested_id}"
            )

        return requested_id

    metadata_path = Path(bag_path) / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.yamlが見つかりません: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    storage_id = metadata.get(
        "rosbag2_bagfile_information", {}
    ).get("storage_identifier")

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
    """センサー座標系からbase_linkへの変換を返す。"""

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
) -> np.ndarray:
    """
    点群をbase_link座標系に変換し、自機領域を除去する。

    戻り値:
      Nx4のfloat32配列 (x, y, z, intensity)
    """

    points = point_cloud2.read_points(
        msg,
        skip_nans=False,
    ).reshape(-1)

    names = points.dtype.names

    x = points["x"].astype(np.float64)
    y = points["y"].astype(np.float64)
    z = points["z"].astype(np.float64)

    if "intensity" in names:
        intensity = points["intensity"].astype(np.float32)
    else:
        intensity = np.zeros(
            len(points),
            dtype=np.float32,
        )

    # NaN / Inf除去
    valid = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(z)
    )

    # センサー原点から30 m以内
    distance_squared = (
        x * x +
        y * y +
        z * z
    )

    valid &= distance_squared <= MAX_DISTANCE ** 2

    x = x[valid]
    y = y[valid]
    z = z[valid]
    intensity = intensity[valid]

    # センサー座標系からbase_linkへ変換
    rotation, translation = get_sensor_transform(topic_name)

    xyz_sensor = np.column_stack((
        x,
        y,
        z,
    ))

    xyz_base = (
        xyz_sensor @ rotation.T
        + translation
    )

    x = xyz_base[:, 0]
    y = xyz_base[:, 1]
    z = xyz_base[:, 2]

    # base_link座標系で自機領域を除去
    inside_ego = (
        (x >= EGO_MIN[0]) & (x <= EGO_MAX[0]) &
        (y >= EGO_MIN[1]) & (y <= EGO_MAX[1]) &
        (z >= EGO_MIN[2]) & (z <= EGO_MAX[2])
    )

    keep = ~inside_ego

    # x, y, z, intensityの共通形式
    result = np.column_stack((
        x[keep],
        y[keep],
        z[keep],
        intensity[keep],
    ))

    return result.astype(np.float32)


# ============================================================
# 点群統合
# ============================================================

def merge_pointclouds(
    velodyne_points: np.ndarray,
    hesai_points: np.ndarray,
    header,
) -> PointCloud2:
    """VelodyneとHesaiの点群を1つのPointCloud2に統合する。"""

    merged_points = np.concatenate((
        velodyne_points,
        hesai_points,
    ), axis=0)

    merged = PointCloud2()

    merged.header = copy.deepcopy(header)
    merged.header.frame_id = "base_link"

    merged.height = 1
    merged.width = len(merged_points)

    merged.fields = [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="intensity",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]

    merged.is_bigendian = False
    merged.point_step = 16
    merged.row_step = merged.width * merged.point_step
    merged.data = merged_points.tobytes()
    merged.is_dense = True

    return merged


# ============================================================
# 点群同期・出力
# ============================================================

def try_merge_pointclouds(
    pointcloud_buffers,
    writer,
    output_counts,
):
    """
    2つのセンサーの点群を時刻で比較し、
    同期許容時間内の点群を統合して出力する。
    """

    velodyne_buffer = pointcloud_buffers[VELODYNE_TOPIC]
    hesai_buffer = pointcloud_buffers[LIDAR_TOPIC]

    while velodyne_buffer and hesai_buffer:

        v_time = velodyne_buffer[0][0]
        h_time = hesai_buffer[0][0]

        time_diff = abs(v_time - h_time)

        if time_diff <= SYNC_SLOP_NS:

            v_timestamp, v_points, v_header = (
                velodyne_buffer.popleft()
            )

            h_timestamp, h_points, h_header = (
                hesai_buffer.popleft()
            )

            # 新しい方のタイムスタンプを使用
            if v_timestamp >= h_timestamp:
                output_timestamp = v_timestamp
                output_header = v_header
            else:
                output_timestamp = h_timestamp
                output_header = h_header

            merged_msg = merge_pointclouds(
                v_points,
                h_points,
                output_header,
            )

            writer.write(
                FILTERED_TOPIC,
                serialize_message(merged_msg),
                output_timestamp,
            )

            output_counts[FILTERED_TOPIC] += 1

        else:
            # 古い方の点群を破棄
            if v_time < h_time:
                velodyne_buffer.popleft()
            else:
                hesai_buffer.popleft()


def limit_pointcloud_buffers(pointcloud_buffers):
    """点群バッファが増え続けないように制限する。"""

    for topic_name in POINTCLOUD_TOPICS:

        buffer = pointcloud_buffers[topic_name]

        while len(buffer) > MAX_BUFFER_SIZE:
            buffer.popleft()

            print(
                f"警告: {topic_name} の未同期点群を破棄しました"
            )


# ============================================================
# Bag処理
# ============================================================

def process_bag(
    input_bag: str,
    output_bag: str,
    requested_storage_id: str,
) -> None:

    if not os.path.isdir(input_bag):
        raise FileNotFoundError(
            f"入力bagが見つかりません: {input_bag}"
        )

    if os.path.exists(output_bag):
        raise FileExistsError(
            f"出力先が既に存在します: {output_bag}"
        )

    # ストレージ形式を取得
    storage_id = get_storage_id(
        input_bag,
        requested_storage_id,
    )

    print(f"Input storage : {storage_id}")
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

    all_topics = reader.get_all_topics_and_types()

    topic_info = {
        topic.name: topic
        for topic in all_topics
    }

    missing = KEEP_TOPICS - topic_info.keys()

    if missing:
        print(
            "警告: 入力bagに存在しないトピック:",
            sorted(missing),
        )

    selected_topics = KEEP_TOPICS & topic_info.keys()

    if not selected_topics:
        raise RuntimeError(
            "対象トピックが入力bagにありません"
        )

    # 点群トピックの型を確認
    selected_pointclouds = POINTCLOUD_TOPICS & selected_topics

    for topic_name in selected_pointclouds:

        if topic_info[topic_name].type != "sensor_msgs/msg/PointCloud2":
            raise TypeError(
                f"{topic_name} の型がPointCloud2ではありません: "
                f"{topic_info[topic_name].type}"
            )

    # 2つの点群が必要
    if selected_pointclouds != POINTCLOUD_TOPICS:
        raise RuntimeError(
            "点群統合には両方のトピックが必要です。\n"
            f"検出された点群トピック: {sorted(selected_pointclouds)}"
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

    # 元の点群トピックは登録しない
    output_topics = selected_topics - POINTCLOUD_TOPICS

    for topic_name in output_topics:
        writer.create_topic(topic_info[topic_name])

    # 追加する点群トピックのID
    max_topic_id = max(
        (topic.id for topic in all_topics),
        default=0,
    )

    new_topic_id = max_topic_id + 1

    # 点群出力トピックのQoS設定
    source_metadata = topic_info[VELODYNE_TOPIC]

    filtered_metadata = rosbag2_py.TopicMetadata(
        id=new_topic_id,
        name=FILTERED_TOPIC,
        type="sensor_msgs/msg/PointCloud2",
        serialization_format=source_metadata.serialization_format,
        offered_qos_profiles=source_metadata.offered_qos_profiles,
    )

    writer.create_topic(filtered_metadata)

    # 点群バッファ
    pointcloud_buffers = {
        VELODYNE_TOPIC: deque(),
        LIDAR_TOPIC: deque(),
    }

    # 間引き管理
    next_time = {
        topic: None
        for topic in THROTTLE_PERIOD_NS
    }

    # メッセージ数管理
    input_counts = {
        topic: 0
        for topic in selected_topics
    }

    output_counts = {
        topic: 0
        for topic in output_topics
    }

    output_counts[FILTERED_TOPIC] = 0

    pointcloud_input_counts = {
        VELODYNE_TOPIC: 0,
        LIDAR_TOPIC: 0,
    }

    # Bag読み込み・書き込み
    while reader.has_next():

        topic_name, data, timestamp = reader.read_next()

        if topic_name not in selected_topics:
            continue

        input_counts[topic_name] += 1

        # ----------------------------------------------------
        # 点群処理
        # ----------------------------------------------------

        if topic_name in POINTCLOUD_TOPICS:

            msg = deserialize_message(
                data,
                PointCloud2,
            )

            filtered_points = filter_pointcloud(
                msg,
                topic_name,
            )

            pointcloud_buffers[topic_name].append((
                timestamp,
                filtered_points,
                msg.header,
            ))

            pointcloud_input_counts[topic_name] += 1

            # 同期できる点群があれば統合
            try_merge_pointclouds(
                pointcloud_buffers,
                writer,
                output_counts,
            )

            # バッファサイズ制限
            limit_pointcloud_buffers(
                pointcloud_buffers,
            )

            continue

        # ----------------------------------------------------
        # 指定トピックの間引き
        # ----------------------------------------------------

        if topic_name in THROTTLE_PERIOD_NS:

            next_allowed = next_time[topic_name]

            if (
                next_allowed is not None
                and timestamp < next_allowed
            ):
                continue

            next_time[topic_name] = (
                timestamp + THROTTLE_PERIOD_NS[topic_name]
            )

        # ----------------------------------------------------
        # その他のトピックをそのまま出力
        # ----------------------------------------------------

        writer.write(
            topic_name,
            data,
            timestamp,
        )

        output_counts[topic_name] += 1

    # Readerの終了時点で残った未同期点群は出力しない
    remaining_velodyne = len(
        pointcloud_buffers[VELODYNE_TOPIC]
    )

    remaining_hesai = len(
        pointcloud_buffers[LIDAR_TOPIC]
    )

    del reader
    del writer

    # --------------------------------------------------------
    # 処理結果
    # --------------------------------------------------------

    print("\n========== Processing Result ==========")

    print(f"Input bag : {input_bag}")
    print(f"Output bag: {output_bag}")
    print(f"Storage   : {storage_id}")

    print("\n[PointCloud counts]")

    print(
        f"{VELODYNE_TOPIC:24s} "
        f"{pointcloud_input_counts[VELODYNE_TOPIC]:10d}"
    )

    print(
        f"{LIDAR_TOPIC:24s} "
        f"{pointcloud_input_counts[LIDAR_TOPIC]:10d}"
    )

    print(
        f"{FILTERED_TOPIC:24s} "
        f"{output_counts[FILTERED_TOPIC]:10d}"
    )

    print("\n[Other topic counts]")

    for topic_name in sorted(output_topics):

        print(
            f"{topic_name:24s} "
            f"{input_counts[topic_name]:10d} -> "
            f"{output_counts[topic_name]:10d}"
        )

    print("\n[Unmatched pointclouds]")

    print(
        f"Velodyne: {remaining_velodyne}"
    )

    print(
        f"Hesai   : {remaining_hesai}"
    )

    print("\n[Output topics]")

    for topic_name in sorted(output_topics):
        print(f"  {topic_name}")

    print(f"  {FILTERED_TOPIC}")

    print("\nPointCloud frame_id: base_link")
    print("PointCloud fields : x, y, z, intensity")
    print(f"Sync slop         : {SYNC_SLOP_NS / 1e6:.1f} ms")

    print("=======================================")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 bagから指定トピックを抽出し、"
            "2つのLiDAR点群を統合・自機除去する"
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

    parser.add_argument(
        "--storage-id",
        choices=["auto", "sqlite3", "mcap"],
        default="auto",
        help="bagのストレージ形式 (default: auto)",
    )

    args = parser.parse_args()

    process_bag(
        args.input_bag,
        args.output_bag,
        args.storage_id,
    )


if __name__ == "__main__":
    main()
