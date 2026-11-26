
#!/usr/bin/env python3
"""
ROS 2 bag LiDAR点群統合・フィルタ処理

入力:
  /velodyne_points
  /lidar_points

処理:
  1. header.stampを使用した近似時間同期（許容差20 ms）
  2. 各センサーの原点から30 mを超える点を除去
  3. URDFの外部パラメータを用いてbase_link座標系へ変換
  4. base_link座標系で自機領域を除去
  5. LiDAR反射強度フィルタ
  6. VelodyneとHesaiの点群を統合
  7. 統合後の点群にVoxel Grid Filterを適用
  8. /points_filteredへ出力
  9. 元の全トピックをそのままコピー

対応ストレージ:
  - SQLite3
  - MCAP

同期できなかった点群は統計情報に記録し、統合出力しない。

出力点群:
  - frame_id: base_link
  - fields: x, y, z, intensity
  - datatype: FLOAT32

使用例:
  python3 merge_and_filter_lidar_points.py input_bag output_bag
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
from sensor_msgs.msg import PointCloud2, PointField
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

# 近似時間同期の許容時間差 [ms]
SYNC_TOLERANCE_MS = 50.0
SYNC_TOLERANCE_NS = int(SYNC_TOLERANCE_MS * 1_000_000)


# Velodyneのpitch角
VELODYNE_PITCH = math.radians(-12.7)


# 自機領域（base_link座標系）
EGO_MIN = np.array([-2.0, -0.50, -1.0])
EGO_MAX = np.array([0.25, 0.50, 2.125])


# --- Voxel Grid Filter ---

ENABLE_VOXEL_FILTER = True

# ボクセルの一辺 [m]
VOXEL_SIZE = 0.05


# --- Intensity Filter ---

ENABLE_INTENSITY_FILTER = True

# PointCloud2の反射強度フィールド名
INTENSITY_FIELD = "intensity"

# 反射強度の許容範囲
INTENSITY_MIN = 4.0
INTENSITY_MAX = None


# ============================================================
# 出力点群のデータ形式
# ============================================================

OUTPUT_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("intensity", "<f4"),
])

OUTPUT_FIELDS = [
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

OUTPUT_POINT_STEP = OUTPUT_DTYPE.itemsize


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

    storage_id = metadata.get(
        "rosbag2_bagfile_information", {}
    ).get("storage_identifier")

    if storage_id not in ("sqlite3", "mcap"):
        raise ValueError(
            f"未対応のストレージ形式です: {storage_id}"
        )

    return storage_id


# ============================================================
# タイムスタンプ
# ============================================================

def stamp_to_ns(stamp) -> int:
    """ROS Timeをナノ秒に変換する。"""

    return (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
    )


def ns_to_stamp(timestamp_ns: int):
    """ナノ秒をROS Timeのsec/nanosecに変換する。"""

    sec, nanosec = divmod(
        int(timestamp_ns),
        1_000_000_000,
    )

    return sec, nanosec


def get_sync_timestamp(
    msg: PointCloud2,
    bag_timestamp: int,
) -> int:
    """
    同期用タイムスタンプを取得する。

    header.stampが0の場合はbagの記録時刻を使用する。
    """

    header_timestamp = stamp_to_ns(msg.header.stamp)

    if header_timestamp == 0:
        return int(bag_timestamp)

    return header_timestamp


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
# 点群の統一形式への変換
# ============================================================

def create_output_array(
    xyz: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    """XYZとIntensityから統一形式の点群配列を生成する。"""

    output = np.empty(
        len(xyz),
        dtype=OUTPUT_DTYPE,
    )

    output["x"] = xyz[:, 0]
    output["y"] = xyz[:, 1]
    output["z"] = xyz[:, 2]
    output["intensity"] = intensity

    return output


# ============================================================
# センサーごとの前処理
# ============================================================

def preprocess_pointcloud(
    msg: PointCloud2,
    topic_name: str,
) -> np.ndarray:
    """
    センサーごとの点群前処理。

    処理:
      1. NaN / Inf除去
      2. センサー原点から30 m以内に制限
      3. base_linkへ座標変換
      4. 自機領域除去
      5. Intensity Filter

    戻り値:
      XYZ + Intensityの統一形式の配列
    """

    points = point_cloud2.read_points(
        msg,
        skip_nans=False,
    ).reshape(-1)

    if len(points) == 0:
        return np.empty(0, dtype=OUTPUT_DTYPE)

    field_names = points.dtype.names or ()

    required_fields = ("x", "y", "z")

    for field_name in required_fields:
        if field_name not in field_names:
            raise ValueError(
                f"{topic_name}: 必須フィールドがありません: "
                f"{field_name}"
            )

    has_intensity = INTENSITY_FIELD in field_names

    if ENABLE_INTENSITY_FILTER and not has_intensity:
        raise ValueError(
            f"{topic_name}: 反射強度フィールドがありません: "
            f"{INTENSITY_FIELD}"
        )

    # センサー座標系のXYZ
    xyz_sensor = np.column_stack((
        points["x"].astype(np.float64),
        points["y"].astype(np.float64),
        points["z"].astype(np.float64),
    ))

    # NaN / Inf除去
    valid = np.all(
        np.isfinite(xyz_sensor),
        axis=1,
    )

    # センサー原点から30 m以内
    distance_squared = np.sum(
        xyz_sensor * xyz_sensor,
        axis=1,
    )

    valid &= (
        distance_squared <= MAX_DISTANCE * MAX_DISTANCE
    )

    xyz_sensor = xyz_sensor[valid]

    if has_intensity:
        intensity = points[INTENSITY_FIELD].astype(
            np.float64
        )[valid]
    else:
        intensity = np.zeros(
            len(xyz_sensor),
            dtype=np.float64,
        )

    # センサー座標系からbase_linkへ変換
    rotation, translation = get_sensor_transform(
        topic_name
    )

    xyz_base = (
        xyz_sensor @ rotation.T
        + translation
    )

    x = xyz_base[:, 0]
    y = xyz_base[:, 1]
    z = xyz_base[:, 2]

    # 自機領域内の点を除去
    inside_ego = (
        (x >= EGO_MIN[0]) & (x <= EGO_MAX[0]) &
        (y >= EGO_MIN[1]) & (y <= EGO_MAX[1]) &
        (z >= EGO_MIN[2]) & (z <= EGO_MAX[2])
    )

    valid = ~inside_ego

    xyz_base = xyz_base[valid]
    intensity = intensity[valid]

    # Intensity Filter
    if ENABLE_INTENSITY_FILTER:

        valid = np.isfinite(intensity)

        if INTENSITY_MIN is not None:
            valid &= intensity >= INTENSITY_MIN

        if INTENSITY_MAX is not None:
            valid &= intensity <= INTENSITY_MAX

        xyz_base = xyz_base[valid]
        intensity = intensity[valid]

    # Intensityの非有限値を安全な値に置換
    intensity = np.where(
        np.isfinite(intensity),
        intensity,
        0.0,
    )

    return create_output_array(
        xyz_base,
        intensity,
    )


# ============================================================
# 点群統合
# ============================================================

def merge_pointclouds(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    """2つの統一形式点群を結合する。"""

    if len(points_a) == 0:
        return points_b.copy()

    if len(points_b) == 0:
        return points_a.copy()

    return np.concatenate(
        (points_a, points_b)
    )


# ============================================================
# Voxel Grid Filter
# ============================================================

def voxel_grid_filter(
    points: np.ndarray,
) -> np.ndarray:
    """
    統合後の点群にVoxel Grid Filterを適用する。

    同一ボクセルに属する点を1点に集約する。
    XYZ座標とIntensityは平均値を使用する。
    """

    if not ENABLE_VOXEL_FILTER:
        return points

    if len(points) == 0:
        return points

    if VOXEL_SIZE <= 0.0:
        raise ValueError(
            f"VOXEL_SIZEは0より大きい値にしてください: "
            f"{VOXEL_SIZE}"
        )

    # XYZ座標
    xyz = np.column_stack((
        points["x"].astype(np.float64),
        points["y"].astype(np.float64),
        points["z"].astype(np.float64),
    ))

    # ボクセルインデックス
    voxel_indices = np.floor(
        xyz / VOXEL_SIZE
    ).astype(np.int64)

    # ボクセルごとにグループ化
    _, first_indices, inverse = np.unique(
        voxel_indices,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    # NumPyのバージョンによるinverseの形状差に対応
    inverse = np.asarray(inverse).reshape(-1)

    if len(inverse) != len(points):
        raise RuntimeError(
            "Voxel Grid Filter: inverseの要素数が"
            "点群数と一致しません。"
        )

    voxel_count = len(first_indices)

    # 出力配列
    filtered = np.empty(
        voxel_count,
        dtype=OUTPUT_DTYPE,
    )

    # ボクセルごとの点数
    counts = np.bincount(
        inverse,
        minlength=voxel_count,
    )

    # XYZ座標の平均値
    for field_index, field_name in enumerate(
        ("x", "y", "z")
    ):

        coordinate_sum = np.bincount(
            inverse,
            weights=xyz[:, field_index],
            minlength=voxel_count,
        )

        filtered[field_name] = (
            coordinate_sum / counts
        )

    # Intensityの平均値
    intensity = points["intensity"].astype(
        np.float64
    )

    intensity_sum = np.bincount(
        inverse,
        weights=intensity,
        minlength=voxel_count,
    )

    filtered["intensity"] = (
        intensity_sum / counts
    )

    return filtered


# ============================================================
# PointCloud2生成
# ============================================================

def create_pointcloud2(
    points: np.ndarray,
    reference_msg: PointCloud2,
    timestamp_ns: int,
) -> PointCloud2:
    """統合点群からPointCloud2を生成する。"""

    filtered = PointCloud2()

    filtered.header = copy.deepcopy(
        reference_msg.header
    )

    filtered.header.frame_id = "base_link"

    sec, nanosec = ns_to_stamp(timestamp_ns)

    filtered.header.stamp.sec = sec
    filtered.header.stamp.nanosec = nanosec

    filtered.height = 1
    filtered.width = len(points)

    filtered.fields = OUTPUT_FIELDS

    filtered.is_bigendian = False
    filtered.point_step = OUTPUT_POINT_STEP

    filtered.row_step = (
        filtered.width * filtered.point_step
    )

    filtered.data = points.tobytes()

    filtered.is_dense = True

    return filtered


# ============================================================
# 近似時間同期
# ============================================================

class ApproximateSynchronizer:
    """
    2つのLiDAR点群を近似時間同期する。

    各センサーのメッセージはタイムスタンプ順に保持する。

    最も古いメッセージについて、相手側センサーの最新時刻が
    許容時間差を超えた時点でマッチングを確定する。

    これにより、後から到着するメッセージを待ってから
    最も近いメッセージを選択できる。
    """

    def __init__(self, tolerance_ns: int):

        self.tolerance_ns = tolerance_ns

        self.pending = {
            VELODYNE_TOPIC: [],
            LIDAR_TOPIC: [],
        }

        self.latest_timestamp = {
            VELODYNE_TOPIC: None,
            LIDAR_TOPIC: None,
        }

        self.matched_count = 0

        self.unmatched_count = {
            VELODYNE_TOPIC: 0,
            LIDAR_TOPIC: 0,
        }

    def add(
        self,
        topic_name: str,
        msg: PointCloud2,
        bag_timestamp: int,
    ):
        """点群を同期キューに追加する。"""

        sync_timestamp = get_sync_timestamp(
            msg,
            bag_timestamp,
        )

        entry = (
            sync_timestamp,
            int(bag_timestamp),
            msg,
        )

        queue = self.pending[topic_name]

        # タイムスタンプ順に挿入
        insert_index = len(queue)

        while (
            insert_index > 0
            and queue[insert_index - 1][0] > sync_timestamp
        ):
            insert_index -= 1

        queue.insert(
            insert_index,
            entry,
        )

        latest = self.latest_timestamp[topic_name]

        if latest is None:
            self.latest_timestamp[topic_name] = sync_timestamp
        else:
            self.latest_timestamp[topic_name] = max(
                latest,
                sync_timestamp,
            )

    def drain(self, force: bool = False):
        """
        同期可能なペアを取り出す。

        戻り値:
          [(velodyne_entry, hesai_entry), ...]
        """

        matched_pairs = []

        while True:

            velodyne_queue = self.pending[VELODYNE_TOPIC]
            lidar_queue = self.pending[LIDAR_TOPIC]

            if not velodyne_queue and not lidar_queue:
                break

            # 最も古いメッセージを選択
            if not lidar_queue:
                earliest_topic = VELODYNE_TOPIC

            elif not velodyne_queue:
                earliest_topic = LIDAR_TOPIC

            elif velodyne_queue[0][0] <= lidar_queue[0][0]:
                earliest_topic = VELODYNE_TOPIC

            else:
                earliest_topic = LIDAR_TOPIC

            other_topic = (
                LIDAR_TOPIC
                if earliest_topic == VELODYNE_TOPIC
                else VELODYNE_TOPIC
            )

            earliest_queue = self.pending[earliest_topic]
            other_queue = self.pending[other_topic]

            earliest_entry = earliest_queue[0]
            earliest_timestamp = earliest_entry[0]

            other_latest = self.latest_timestamp[other_topic]

            # 相手側の最新時刻が許容範囲を超えるまで待つ
            if not force:

                if other_latest is None:
                    break

                if (
                    other_latest
                    <= earliest_timestamp + self.tolerance_ns
                ):
                    break

            # 相手側キューから最も近い時刻を検索
            candidates = []

            for index, entry in enumerate(other_queue):

                difference = abs(
                    entry[0] - earliest_timestamp
                )

                if difference <= self.tolerance_ns:
                    candidates.append(
                        (difference, entry[0], index)
                    )

            # 対応するメッセージが見つかった場合
            if candidates:

                _, _, best_index = min(candidates)

                matched_entry = other_queue.pop(
                    best_index
                )

                earliest_queue.pop(0)

                if earliest_topic == VELODYNE_TOPIC:

                    pair = (
                        earliest_entry,
                        matched_entry,
                    )

                else:

                    pair = (
                        matched_entry,
                        earliest_entry,
                    )

                matched_pairs.append(pair)
                self.matched_count += 1

            else:

                # 同期できないメッセージを破棄
                earliest_queue.pop(0)

                self.unmatched_count[earliest_topic] += 1

        return matched_pairs

    def flush(self):
        """終了時に残ったキューを処理する。"""

        return self.drain(force=True)


# ============================================================
# 同期済み点群の統合・出力
# ============================================================

def process_synchronized_pair(
    velodyne_entry,
    lidar_entry,
    writer,
    output_timestamp: int,
):
    """
    同期済みの2つの点群を処理して出力する。
    """

    velodyne_timestamp, _, velodyne_msg = velodyne_entry
    lidar_timestamp, _, lidar_msg = lidar_entry

    # センサーごとの前処理
    velodyne_points = preprocess_pointcloud(
        velodyne_msg,
        VELODYNE_TOPIC,
    )

    lidar_points = preprocess_pointcloud(
        lidar_msg,
        LIDAR_TOPIC,
    )

    # 2つの点群を統合
    merged_points = merge_pointclouds(
        velodyne_points,
        lidar_points,
    )

    # 統合後にVoxel Grid Filterを適用
    filtered_points = voxel_grid_filter(
        merged_points
    )

    # 2つのセンサーのタイムスタンプの中間時刻
    synchronized_timestamp = (
        velodyne_timestamp + lidar_timestamp
    ) // 2

    # PointCloud2を生成
    filtered_msg = create_pointcloud2(
        filtered_points,
        velodyne_msg,
        synchronized_timestamp,
    )

    # 出力
    writer.write(
        FILTERED_TOPIC,
        serialize_message(filtered_msg),
        output_timestamp,
    )

    return (
        len(velodyne_points),
        len(lidar_points),
        len(merged_points),
        len(filtered_points),
    )


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

    # ストレージ形式
    storage_id = get_storage_id(input_bag)

    print(f"Input storage: {storage_id}")
    print(f"Output storage: {storage_id}")

    print(f"Sync tolerance: +/- {SYNC_TOLERANCE_MS} ms")

    print(f"Voxel filter: {ENABLE_VOXEL_FILTER}")

    if ENABLE_VOXEL_FILTER:
        print(f"Voxel size: {VOXEL_SIZE} m")

    print(f"Intensity filter: {ENABLE_INTENSITY_FILTER}")

    if ENABLE_INTENSITY_FILTER:
        print(f"Intensity field: {INTENSITY_FIELD}")
        print(f"Intensity min: {INTENSITY_MIN}")
        print(f"Intensity max: {INTENSITY_MAX}")

    # --------------------------------------------------------
    # Reader
    # --------------------------------------------------------

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

        elif (
            topic_type_map[topic_name]
            != "sensor_msgs/msg/PointCloud2"
        ):

            raise TypeError(
                f"{topic_name} の型がPointCloud2ではありません: "
                f"{topic_type_map[topic_name]}"
            )

    # 両方のセンサーが存在することを確認
    available_topics = FILTER_TOPICS.intersection(
        topic_type_map.keys()
    )

    if len(available_topics) != 2:
        raise ValueError(
            "VelodyneとHesaiの両方の点群トピックが必要です。"
        )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

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

    # 統合点群トピックを登録
    max_topic_id = max(
        (topic.id for topic in topics),
        default=0,
    )

    source_metadata = next(
        topic for topic in topics
        if topic.name == VELODYNE_TOPIC
    )

    filtered_metadata = rosbag2_py.TopicMetadata(
        id=max_topic_id + 1,
        name=FILTERED_TOPIC,
        type="sensor_msgs/msg/PointCloud2",
        serialization_format=source_metadata.serialization_format,
        offered_qos_profiles=source_metadata.offered_qos_profiles,
    )

    writer.create_topic(filtered_metadata)

    # --------------------------------------------------------
    # 同期器
    # --------------------------------------------------------

    synchronizer = ApproximateSynchronizer(
        tolerance_ns=SYNC_TOLERANCE_NS,
    )

    # --------------------------------------------------------
    # 処理統計
    # --------------------------------------------------------

    input_counts = {
        VELODYNE_TOPIC: 0,
        LIDAR_TOPIC: 0,
    }

    output_pair_count = 0

    total_velodyne_points = 0
    total_lidar_points = 0
    total_merged_points = 0
    total_filtered_points = 0

    last_input_timestamp = 0
    last_output_timestamp = 0

    # --------------------------------------------------------
    # Bag読み込み・書き込み
    # --------------------------------------------------------

    while reader.has_next():

        topic_name, data, timestamp = reader.read_next()

        timestamp = int(timestamp)

        last_input_timestamp = max(
            last_input_timestamp,
            timestamp,
        )

        # 元の全メッセージをそのままコピー
        writer.write(
            topic_name,
            data,
            timestamp,
        )

        # 対象外トピックはそのままスキップ
        if topic_name not in FILTER_TOPICS:
            continue

        input_counts[topic_name] += 1

        # PointCloud2を復元
        msg = deserialize_message(
            data,
            PointCloud2,
        )

        # 同期キューへ追加
        synchronizer.add(
            topic_name,
            msg,
            timestamp,
        )

        # 同期可能なペアを取得
        matched_pairs = synchronizer.drain()

        # 同期済みペアを処理
        for velodyne_entry, lidar_entry in matched_pairs:

            # 現在までに読み込んだbag時刻を出力時刻とする
            # これにより出力レコードの時刻が過去に戻らない
            output_timestamp = max(
                timestamp,
                last_output_timestamp,
            )

            (
                velodyne_points_count,
                lidar_points_count,
                merged_points_count,
                filtered_points_count,
            ) = process_synchronized_pair(
                velodyne_entry,
                lidar_entry,
                writer,
                output_timestamp,
            )

            last_output_timestamp = output_timestamp

            output_pair_count += 1

            total_velodyne_points += velodyne_points_count
            total_lidar_points += lidar_points_count
            total_merged_points += merged_points_count
            total_filtered_points += filtered_points_count

    # --------------------------------------------------------
    # 残った同期キューを処理
    # --------------------------------------------------------

    remaining_pairs = synchronizer.flush()

    for velodyne_entry, lidar_entry in remaining_pairs:

        output_timestamp = max(
            last_input_timestamp,
            last_output_timestamp,
        )

        (
            velodyne_points_count,
            lidar_points_count,
            merged_points_count,
            filtered_points_count,
        ) = process_synchronized_pair(
            velodyne_entry,
            lidar_entry,
            writer,
            output_timestamp,
        )

        last_output_timestamp = output_timestamp

        output_pair_count += 1

        total_velodyne_points += velodyne_points_count
        total_lidar_points += lidar_points_count
        total_merged_points += merged_points_count
        total_filtered_points += filtered_points_count

    # Reader / Writerを閉じる
    del reader
    del writer

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print("\n========== Processing Result ==========")

    print(f"Output bag: {output_bag}")
    print(f"Storage: {storage_id}")

    print("\n--- Input Messages ---")

    print(
        f"{VELODYNE_TOPIC}: "
        f"{input_counts[VELODYNE_TOPIC]} messages"
    )

    print(
        f"{LIDAR_TOPIC}: "
        f"{input_counts[LIDAR_TOPIC]} messages"
    )

    print("\n--- Synchronization ---")

    print(
        f"Matched pairs: "
        f"{synchronizer.matched_count}"
    )

    print(
        f"Unmatched Velodyne: "
        f"{synchronizer.unmatched_count[VELODYNE_TOPIC]}"
    )

    print(
        f"Unmatched Hesai: "
        f"{synchronizer.unmatched_count[LIDAR_TOPIC]}"
    )

    print("\n--- Point Counts ---")

    print(
        f"Velodyne after preprocessing: "
        f"{total_velodyne_points}"
    )

    print(
        f"Hesai after preprocessing: "
        f"{total_lidar_points}"
    )

    print(
        f"Total before Voxel Filter: "
        f"{total_merged_points}"
    )

    print(
        f"Total after Voxel Filter: "
        f"{total_filtered_points}"
    )

    print("\n--- Output ---")

    print(f"Output topic: {FILTERED_TOPIC}")
    print(f"Output messages: {output_pair_count}")
    print("Output frame_id: base_link")
    print("Output fields: x, y, z, intensity")

    print("=======================================")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 bagのVelodyne/Hesai点群を近似時間同期し、"
            "base_linkへ変換・統合後にフィルタ処理する"
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
