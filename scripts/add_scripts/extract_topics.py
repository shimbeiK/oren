#!/usr/bin/env python3

import argparse
import os

import rosbag2_py


KEEP_TOPICS = {
    "/tf",
    "/tf_static",
    # "/velodyne_points",
    "/points_filtered",
    "/odom",
}

THROTTLE_PERIOD_NS = {
    "/tf": 30_000_000,   # 33 Hz
    "/odom": 20_000_000,   # 50 Hz
    # "/velodyne_points": 200_000_000, # 5 Hz
    # "/lidar_points": 125_000_000, # 8 Hz
}


def main():
    parser = argparse.ArgumentParser(
        description="ROS 2 bagから指定トピックを抽出し、/tfと/odomを間引く"
    )
    parser.add_argument("input_bag", help="入力bagディレクトリ")
    parser.add_argument("output_bag", help="出力bagディレクトリ")
    parser.add_argument(
        "--storage-id",
        default="sqlite3",
        help="bagのストレージ形式 (default: sqlite3)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_bag):
        parser.error(f"入力bagが見つかりません: {args.input_bag}")

    if os.path.exists(args.output_bag):
        parser.error(f"出力先がすでに存在します: {args.output_bag}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=args.input_bag,
            storage_id=args.storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    all_topics = reader.get_all_topics_and_types()
    topic_info = {topic.name: topic for topic in all_topics}

    missing = KEEP_TOPICS - topic_info.keys()
    if missing:
        print("警告: 入力bagに存在しないトピック:", sorted(missing))

    selected_topics = KEEP_TOPICS & topic_info.keys()
    if not selected_topics:
        parser.error("対象トピックが入力bagにありません")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(
            uri=args.output_bag,
            storage_id=args.storage_id,
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    for topic in all_topics:
        if topic.name in selected_topics:
            writer.create_topic(topic)

    next_time = {topic: None for topic in THROTTLE_PERIOD_NS}
    input_counts = {topic: 0 for topic in selected_topics}
    output_counts = {topic: 0 for topic in selected_topics}

    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()

        if topic_name not in selected_topics:
            continue

        input_counts[topic_name] += 1

        if topic_name in THROTTLE_PERIOD_NS:
            next_allowed = next_time[topic_name]

            if next_allowed is not None and timestamp < next_allowed:
                continue

            next_time[topic_name] = (
                timestamp + THROTTLE_PERIOD_NS[topic_name]
            )

        writer.write(topic_name, data, timestamp)
        output_counts[topic_name] += 1

    print("\n処理結果")
    print("-" * 52)

    for topic in sorted(selected_topics):
        print(
            f"{topic:24s} "
            f"{input_counts[topic]:10d} -> "
            f"{output_counts[topic]:10d} messages"
        )

    print(f"\n出力bag: {args.output_bag}")


if __name__ == "__main__":
    main()
