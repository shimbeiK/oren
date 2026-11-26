
#!/usr/bin/env python3

import argparse
import copy
import os
import sys

import numpy as np
import rosbag2_py

from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import PointCloud2


INPUT_TOPIC = "/lidar_points"

REMOVED_TOPIC = "/points_removed"
FILTERED_TOPIC = "/points_filtered"

# 自機除去範囲 [m]
EGO_MIN = np.array(
    [-2.0, -0.50, -1.0],
    dtype=np.float64,
)

EGO_MAX = np.array(
    [0.2, 0.50, 1.125],
    dtype=np.float64,
)


def get_reader(input_path, storage_id):
    """rosbag2のSequentialReaderを作成する。"""

    reader = rosbag2_py.SequentialReader()

    reader.open(
        rosbag2_py.StorageOptions(
            uri=input_path,
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    return reader


def get_writer(output_path, storage_id, topics):
    """出力用のSequentialWriterを作成する。"""

    if os.path.exists(output_path):
        raise RuntimeError(
            f"出力先が既に存在します: {output_path}"
        )

    writer = rosbag2_py.SequentialWriter()

    writer.open(
        rosbag2_py.StorageOptions(
            uri=output_path,
            storage_id=storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    # 元のトピックをすべて登録
    for topic in topics:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=0,
                name=topic.name,
                type=topic.type,
                serialization_format=topic.serialization_format,
                offered_qos_profiles=topic.offered_qos_profiles,
            )
        )

#     # 自機領域内の点群
#     writer.create_topic(
#         rosbag2_py.TopicMetadata(
#             id=0,
#             name=REMOVED_TOPIC,
#             type="sensor_msgs/msg/PointCloud2",
#             serialization_format="cdr",
#         )
#     )

    # 自機領域外の点群
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=0,
            name=FILTERED_TOPIC,
            type="sensor_msgs/msg/PointCloud2",
            serialization_format="cdr",
        )
    )

    return writer


def read_points(cloud):
    """PointCloud2からXYZ座標を取得する。"""

    field_names = [
        field.name
        for field in cloud.fields
    ]

    for name in ("x", "y", "z"):
        if name not in field_names:
            raise RuntimeError(
                f"PointCloud2に必須フィールド {name} がありません"
            )

    points_data = __import__(
        "sensor_msgs_py.point_cloud2",
        fromlist=["read_points"],
    ).read_points(
        cloud,
        skip_nans=False,
    )

    points = np.column_stack(
        (
            points_data["x"],
            points_data["y"],
            points_data["z"],
        )
    ).astype(np.float64)

    return points_data, points


def create_split_cloud(original_cloud, selected_points):
    """
    元のPointCloud2のフィールド構成を維持して、
    選択された点群を作成する。
    """

    cloud = copy.deepcopy(original_cloud)

    # 点群を非組織化形式にする
    cloud.height = 1
    cloud.width = len(selected_points)

    # PointCloud2の各点のバイト列をそのまま保持
    cloud.data = selected_points.tobytes()

    cloud.row_step = (
        cloud.point_step * cloud.width
    )

    # XYZの有限値を確認
    if len(selected_points) > 0:
        points = np.column_stack(
            (
                selected_points["x"],
                selected_points["y"],
                selected_points["z"],
            )
        )

        cloud.is_dense = bool(
            np.isfinite(points).all()
        )
    else:
        cloud.is_dense = True
        cloud.data = b""

    return cloud


def split_ego_points(cloud):
    """
    点群を自機領域内と領域外に分割する。

    内側:
        /points_removed

    外側:
        /points_filtered
    """

    points_data, points = read_points(cloud)

    # NaNやInfを含まない点
    finite = np.isfinite(points).all(axis=1)

    # 自機領域内の点を判定
    inside = np.all(
        (points >= EGO_MIN)
        & (points <= EGO_MAX),
        axis=1,
    )

    # 無効な座標は自機領域内とは判定しない
    removed_mask = finite & inside

    # 自機領域外の点
    filtered_mask = ~removed_mask

    removed_points = points_data[removed_mask]
    filtered_points = points_data[filtered_mask]

    removed_cloud = create_split_cloud(
        cloud,
        removed_points,
    )

    filtered_cloud = create_split_cloud(
        cloud,
        filtered_points,
    )

    # デバッグ用統計情報
    finite_points = points[finite]

    if len(finite_points) > 0:
        min_xyz = np.min(
            finite_points,
            axis=0,
        )

        max_xyz = np.max(
            finite_points,
            axis=0,
        )
    else:
        min_xyz = np.array(
            [np.nan, np.nan, np.nan]
        )

        max_xyz = np.array(
            [np.nan, np.nan, np.nan]
        )

    stats = {
        "input": len(points),
        "finite": int(np.count_nonzero(finite)),
        "removed": len(removed_points),
        "filtered": len(filtered_points),
        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
    }

    return removed_cloud, filtered_cloud, stats


def print_stats(index, cloud, stats):
    """処理結果を表示する。"""

    print(
        f"\n[Scan {index}]"
    )

    print(
        f"frame_id: {cloud.header.frame_id}"
    )

    print(
        f"入力点数: {stats['input']}"
    )

    print(
        f"有限座標点数: {stats['finite']}"
    )

    print(
        f"自機領域内: {stats['removed']}"
    )

    print(
        f"自機領域外: {stats['filtered']}"
    )

    print(
        "XYZ最小値: "
        f"X={stats['min_xyz'][0]:.4f}, "
        f"Y={stats['min_xyz'][1]:.4f}, "
        f"Z={stats['min_xyz'][2]:.4f}"
    )

    print(
        "XYZ最大値: "
        f"X={stats['max_xyz'][0]:.4f}, "
        f"Y={stats['max_xyz'][1]:.4f}, "
        f"Z={stats['max_xyz'][2]:.4f}"
    )

    if stats["input"] > 0:
        ratio = (
            stats["removed"]
            / stats["input"]
            * 100.0
        )
    else:
        ratio = 0.0

    print(
        f"自機領域内の割合: {ratio:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "lidar_pointsを自機領域内と領域外に分割して"
            "別々のトピックとして出力する"
        )
    )

    parser.add_argument(
        "input",
        help="入力rosbagディレクトリ",
    )

    parser.add_argument(
        "output",
        help="出力rosbagディレクトリ",
    )

    parser.add_argument(
        "--storage-id",
        default="mcap",
        choices=[
            "mcap",
            "sqlite3",
        ],
        help="rosbag2のストレージ形式",
    )

    args = parser.parse_args()

    try:
        # 入力bagを開く
        reader = get_reader(
            args.input,
            args.storage_id,
        )

        topics = reader.get_all_topics_and_types()

        topic_types = {
            topic.name: topic.type
            for topic in topics
        }

        # 入力トピックの確認
        if INPUT_TOPIC not in topic_types:
            raise RuntimeError(
                f"必要なトピックがありません: {INPUT_TOPIC}"
            )

        if topic_types[INPUT_TOPIC] != "sensor_msgs/msg/PointCloud2":
            raise RuntimeError(
                f"{INPUT_TOPIC}の型がPointCloud2ではありません"
            )

        # 出力トピック名の重複確認
        for name in (REMOVED_TOPIC, FILTERED_TOPIC):
            if name in topic_types:
                raise RuntimeError(
                    f"出力先トピックが入力bagに存在します: {name}"
                )

        cloud_type = get_message(
            topic_types[INPUT_TOPIC]
        )

        # 出力bagを作成
        writer = get_writer(
            args.output,
            args.storage_id,
            topics,
        )

        # 入力bagを再度開く
        reader = get_reader(
            args.input,
            args.storage_id,
        )

        processed = 0

        total_input = 0
        total_finite = 0
        total_removed = 0
        total_filtered = 0

        print("点群の分割処理を開始します")

        print("\n自機除去範囲:")
        print(
            f"X: {EGO_MIN[0]} ～ {EGO_MAX[0]} m"
        )
        print(
            f"Y: {EGO_MIN[1]} ～ {EGO_MAX[1]} m"
        )
        print(
            f"Z: {EGO_MIN[2]} ～ {EGO_MAX[2]} m"
        )

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            # lidar_points以外はそのままコピー
            if topic != INPUT_TOPIC:
                writer.write(
                    topic,
                    data,
                    timestamp,
                )
                continue

            # PointCloud2を復元
            msg = deserialize_message(
                data,
                cloud_type,
            )

            # 点群を内側と外側に分割
            (
                removed_cloud,
                filtered_cloud,
                stats,
            ) = split_ego_points(msg)

        #     # 元のlidar_pointsも保持
        #     writer.write(
        #         INPUT_TOPIC,
        #         data,
        #         timestamp,
        #     )

        #     # 自機領域内の点群
        #     writer.write(
        #         REMOVED_TOPIC,
        #         serialize_message(removed_cloud),
        #         timestamp,
        #     )

            # 自機領域外の点群
            writer.write(
                FILTERED_TOPIC,
                serialize_message(filtered_cloud),
                timestamp,
            )

            processed += 1

            total_input += stats["input"]
            total_finite += stats["finite"]
            total_removed += stats["removed"]
            total_filtered += stats["filtered"]

            # 50スキャンごとに統計を表示
            if processed <= 5 or processed % 50 == 0:
                print_stats(
                    processed,
                    msg,
                    stats,
                )

        print("\n========== 処理完了 ==========")

        print(
            f"処理スキャン数: {processed}"
        )

        print(
            f"入力点数: {total_input}"
        )

        print(
            f"有限座標点数: {total_finite}"
        )

        print(
            f"自機領域内の点数: {total_removed}"
        )

        print(
            f"自機領域外の点数: {total_filtered}"
        )

        if total_input > 0:
            ratio = (
                total_removed
                / total_input
                * 100.0
            )
        else:
            ratio = 0.0

        print(
            f"自機領域内の割合: {ratio:.2f}%"
        )

        print(
            f"出力先: {args.output}"
        )

    except Exception as e:
        print(
            f"エラー: {e}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
