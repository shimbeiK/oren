## 前処理
1. extract_topics.pyで必要なトピックを抽出し、周波数を調整する
```bash
python3 scripts/extract_topics.py '/home/robolab/bag/ssd/20261029_111603' '/home/robolab/bag/ssd/exted' --storage-id mcap
```

2. merge_lidar_points.pyで/lidar_pointsから自機除去する
```bash
python3 scripts/merge_lidar_points.py '/home/robolab/bag/ssd/exted' '/home/robolab/bag/ssd/merged'
```

```bash
ros2 launch oren_ros mapping.launch.py bag_path:='/home/robolab/bag/ssd/remove_myself_part2_extract_with_odom' trainer_config_path:='/home/robolab/oren/oren_ros/configs/trainer-kenkyushitsu.yaml' rviz:=false
```

統合版
```bash
python3 scripts/filter_and_extract_bag.py '/home/robolab/bag/20260826_151631' '/home/robolab/bag/ssd/merged_16F'
```

実機対応
パラメータ調整+launchファイルの修正（/clockをなくし、bag読み込みを消した）
```bash
ros2 launch oren_ros online_mapping.launch.py trainer_config_path:='/home/robolab/oren/oren_ros/configs/trainer-kenkyushitsu.yaml' rviz:=false params_file:='/home/robolab/oren/oren_ros/configs/ros2-params_dul.yaml'
```

```bash
ros2 launch oren_ros mapping.launch.py bag_path:= trainer_config_path:='/home/robolab/oren/oren_ros/configs/trainer-kenkyushitsu.yaml' rviz:=false
```