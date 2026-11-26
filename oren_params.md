# OREN パラメータ早見表

対象: `ros2-params.yaml`（ROS ノード用）と `trainer-ros.yaml`（学習用）、および今回の上書き設定（`kenkyusitsu`）。
単位はすべてメートル [m]、時間は秒 [s]。

**根拠の凡例**（確かさの目安）

| 印 | 意味 |
|---|---|
| ✅ | 実際のコード（`trainer.py` / `criterion.py` / `trainer_config.py`）や、実行ログ・チェックポイントの実測で確認した |
| 📖 | OREN の README に書かれている |
| 🔶 | 名前や周辺の情報からの推測（未確認） |
| ❓ | 分からない。末尾の「確認してほしいこと」を参照 |

---

## 1. ROS ノード設定（`ros2-params.yaml`）

### 1-1. `oren_mapping_node`（LiDAR/深度を受け取って建図する）

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `trainer_config_path` | `""` | 学習用 yaml のパス。空のときは、起動時の `trainer_config_path` 引数（既定は `trainer-ros.yaml`）が使われると思われる | 📖🔶 |
| `modality` | `lidar` | 入力の種類。`lidar` か `depth` | 📖 |
| `use_odom` | `false` | `true`: センサーと `odom_topic` を同期して姿勢を得る。`false`: センサー時刻の姿勢を tf2 から引く | 📖 |
| `odom_msg_type` | `odometry` | オドメトリのメッセージ型の選択。選択肢は不明 | 🔶 |
| `depth_scale` | `0.002` | 深度画像の値を [m] に直す係数。`modality: depth` のときだけ使うと思われる | 🔶 |
| `world_frame` | `odom` | 地図を作る座標系。tf2 の参照先（`use_odom: false` のとき） | 📖 |
| `sensor_frame` | `""` | センサーの座標系。空のときの挙動（点群ヘッダの frame_id を使うか）は不明 | 📖 |
| `tf_lookup_timeout_sec` | `0.5` | tf2 の参照を待つ最大時間 | 🔶 |
| `sync_tolerance` | `0.01` | センサーとオドメトリを同期するときの時刻差の許容値。`use_odom: false` では使われないと思われる | 📖🔶 |
| `no_data_timeout_sec` | `10.0` | この時間データが来なかったら、入力が終了したとみなして終了処理に入ると思われる | 🔶 |
| `frame_queue_maxsize` | `32` | 受信したフレームを貯めておくキューの上限。満杯になると最も古いフレームを捨てる（警告 `frame queue full; dropped oldest frame`） | ✅ |
| `idle_block_sec` | `0.0` | フレームが無いときの待ち時間の設定と思われる | ❓ |
| `lidar_topic.path` | `/points_filtered` | LiDAR 点群のトピック名 | 📖 |
| `lidar_topic.qos_preset` | `""` | QoS のプリセット。空なら、以下の項目で個別に指定 | 📖 |
| `lidar_topic.qos_history` / `qos_depth` | `keep_last` / `10` | 受信側のバッファ方式と深さ（ROS 2 の標準の QoS） | 📖 |
| `lidar_topic.qos_reliability` | `best_effort` | 取りこぼしを許す（センサーデータの標準） | 📖 |
| `lidar_topic.qos_durability` | `volatile` | 過去のメッセージを受け取らない | 📖 |
| `odom_topic.*` | `/odom` ほか | オドメトリ用の同じ構成。`use_odom: true` のときだけ使われると思われる | 📖🔶 |

### 1-2. `sdf_query_node`（学習中のSDFを問い合わせて配信する）

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `resolution` | `0.1` | グリッドの1セルの大きさ | 📖 |
| `x_cells`, `y_cells` | `101`, `101` | セル数。幅は `101 × 0.1 = 10.1 m`（配信された `info` の `length_x` と一致） | ✅ |
| `x`, `y` | `0.0`, `0.0` | グリッドの中心の位置と思われる | 🔶❓ |
| `z` | `0.7` | **水平断面の高さ**（README の query plane height に当たると思われる） | 📖🔶 |
| `publish_grid_map` | `true` | `grid_map_msgs/GridMap` を配信する | 📖 |
| `publish_point_cloud` | `false` | `PointCloud2` でも配信する | 📖 |
| `publish_gradient` | `true` | 勾配のレイヤー（`gradient_x/y/z`）を加える | ✅ |
| `query_prior` | `false` | `true` で、八分木の事前値のSDFを問い合わせると思われる（`false` では最終値） | 🔶❓ |
| `publish_rate` | `10.0` | 配信の周波数 [Hz]。高いほど GPU の負荷が増える | 📖🔶 |
| `attached_to_frame` | `false` | `true` でグリッドを `attached_frame` に追従させる。`false` は `world_frame` に固定 | 📖🔶 |
| `attached_frame` | `base_sensor` | 追従させる座標系 | 📖 |
| `world_frame` | `odom` | グリッドを置く座標系 | 📖 |
| `query_service` | `query_sdf` | 任意の点でSDFを問い合わせるサービス名と思われる（型は `oren_msgs` 側） | 🔶❓ |
| `map_topic` | `sdf_grid_map` | `GridMap` のトピック名。レイヤーは `sdf`、`gradient_x/y/z` | ✅ |
| `point_cloud_topic` | `sdf_point_cloud` | `PointCloud2` のトピック名 | 🔶 |
| `trigger_service` | `sdf_trigger_query` | 問い合わせを手動で起動するサービスと思われる | 🔶❓ |

---

## 2. 学習設定（`trainer-ros.yaml`）

### 2-1. 全般・データ

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `identifier` | `…TrainerConfig` | 設定のクラス名（読み込み用）。調整しない | 🔶 |
| `seed` | `12345` | 乱数の種（torch / numpy / random） | ✅ |
| `log_dir` | `logs` | ログの出力先の親フォルダ | ✅ |
| `exp_name` | `oren_ros` | 実験名。`<log_dir>/<exp_name>/<日時>/` に出力 | ✅ |
| `device` | `cuda` | 計算に使うデバイス | ✅ |
| `data.dataset_name` | `ros` | データの種類（ROS のストリーム入力） | ✅ |
| `data.dataset_args.min_depth`, `max_depth` | `0.8`, `30.0` | 使う距離の範囲。範囲外の点は使われないと思われる | 🔶 |
| `data.apply_bound` | `true` | `false` のとき、範囲 `bound_*` の外のサンプルを、サンプル生成の後に損失から除外する（`valid_mask`）。`true` は、データ側で点を範囲に切り取る動作と思われる | ✅（false 側）🔶（true 側） |
| `data.bound_min`, `bound_max` | `[-40,-40,-1]`, `[45,10,3]` | 関心のある範囲 `[x, y, z]`。学習はこの範囲の中だけで行う | ✅ |
| `data.start_frame`, `end_frame` | `0`, `-1` | 処理するフレームの範囲（`-1` は最後まで）。ストリーム入力では意味が薄い | ✅ |
| `bound_min`, `bound_max`（最上位） | 上と同じ | メッシュ・断面画像の範囲、学習の除外範囲。データセットの範囲があれば、**0.1 m の余白つきで自動的に上書きされる** | ✅ |

### 2-2. キーフレーム（`key_frame_set`）

過去のフレームのうち、学習に使うものを選ぶ仕組み（H2-Mapping の選び方に基づく）。

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `insert_method` | `intersection` | キーフレームに登録する条件の方式 | ❓ |
| `insert_interval` | `50` | 登録の間隔（フレーム数）と思われる | 🔶❓ |
| `insert_ratio` | `0.7` | 登録の判定に使う割合（既存のキーフレームとの重なり）と思われる | 🔶❓ |
| `frame_selection` | `multiple_max_set_coverage` | 学習に使うキーフレームの選び方（見えているボクセルの被覆が大きくなるように選ぶ） | 📖🔶 |
| `selection_window_size` | `8` | 1回の学習で選ぶキーフレームの数と思われる | 🔶❓ |
| `frame_weight` | `uniform` | キーフレームからのサンプリングの重み（`uniform` は均等） | 🔶 |

### 2-3. 八分木（`model.octree_cfg`）

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `resolution` | `0.7` | **最小ボクセルの一辺 [m]** | ✅ |
| `tree_depth` | `8` | 木の深さ。八分木の範囲は一辺 `resolution × 2^8`（0.7 のとき 179.2 m、原点が中心）。yaml コメントの「112m」は実測と合わない | ✅ |
| `semi_sparse_depth` | `5` | 粗い階層を密に確保する深さと思われる | 🔶❓ |
| `init_voxel_num` | `80000` | **ボクセルの最大数（確保する容量）**。テンソルの行数に一致。`KeyFrameSet` の容量にも使われる | ✅ |
| `insertion_threshold` | `3` | ボクセルを作るのに必要な点の数と思われる | 🔶❓ |
| `skip_insertion_if_exists` | `true` | すでにあるボクセルには挿入処理をしない（高速化。yaml コメント） | 🔶 |
| `residual_feature_dim` | `3` | 頂点ごとの残差特徴量の次元 | ✅ |
| `residual_num_levels` | `1` | 残差特徴量を持つ階層の数と思われる | 🔶❓ |
| `gradient_augmentation` | `true` | 頂点ごとにSDFの勾配も持ち、補間に使う（勾配つき八分木補間） | 📖✅ |
| `independent_smallest_leaf_vertex` | `true` | 最小の葉ボクセルの頂点を、隣と共有せず独立に持つと思われる | 🔶❓ |

### 2-4. 残差ネットワーク（`model.residual_net_cfg`）

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `mlp_activation` | `LeakyReLU` | 活性化関数 | ✅ |
| `hidden_dims` | `[32, 32]` | 隠れ層の幅（チェックポイントは 4→32→32→1） | ✅ |
| `input_feature_dim` | `1` | 入力の追加の次元。入力は `residual_feature_dim(3) + 1 = 4` になっている。追加の1次元の中身は不明 | ✅❓ |
| `output_sdf_scale` | `0.1` | 残差の出力にかける倍率（補正量を小さく抑える） | 🔶 |

### 2-5. 損失（`criterion`）

学習のサンプルは、1本のレイにつき、**自由空間 `n_stratified` 点、表面近傍の摂動 `n_perturbed` 点、表面 1 点**。各項は、モデルの最終出力と、八分木だけの事前値（`*_prior_*`）の両方に適用される。

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `boundary_loss_weight`（と `_prior_`） | `100` | 表面の点で SDF = 0 になるようにする | ✅ |
| `perturbation_loss_weight`（と `_prior_`） | `100` | 表面近傍の摂動点の値を、正解の上限と `sigma_s` の下限の間に収める | ✅ |
| `eikonal_loss_surface_weight`（と `_prior_`） | `30` | 表面の点で、勾配の大きさが1になるようにする | ✅ |
| `eikonal_loss_perturbation_weight`（と `_prior_`） | `50` | 摂動点での同じ項 | ✅ |
| `eikonal_loss_space_weight`（と `_prior_`） | `100` | 自由空間の点での同じ項 | ✅ |
| `projection_loss_weight`（と `_prior_`） | `100` | 自由空間の点の値を、正解 `gt_sdf_stratified` に近づける | ✅ |
| `*_loss_type` | `L1` | 損失の種類（`L1` / `L2`） | ✅ |
| `perturbation_loss_exp_penalty` | `1.0` | 下限を割ったときの指数ペナルティの強さ | ✅ |
| `heat_loss_weight` | `0.0` | ヒート損失（0 で無効） | ✅ |
| `heat_loss_lambda` | `4.0` | ヒート損失の係数。`heat_loss_weight` が 0 なら無関係 | ✅ |
| `sign_loss_free_weight`, `sign_loss_occ_weight`, `sign_loss_temperature` | `0.0`, `0.0`, `100.0` | 符号の損失。**現在の `Criterion.forward` からは呼ばれていないので、値を変えても効果がない** | ✅ |

### 2-6. 学習の進め方・最適化

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `num_init_frames` | `3` | 最初のこの枚数のフレームだけ、`init_frame_iterations` 回ずつ最適化する | ✅ |
| `init_frame_iterations` | `10` | 上記の最適化の回数 | ✅ |
| `num_iterations_per_frame` | `1` | それ以降の、1フレームあたりの最適化の回数 | ✅ |
| `num_rays_total` | `20480` | 1回の最適化で使うレイの本数 | ✅ |
| `extra_surface_sample` | `false` | 表面の追加サンプル（yaml コメント: 効果なし） | ✅ |
| `frame_downsample` | `10` | 追加サンプルの割合 `1/10`。`extra_surface_sample: false` なら無関係 | ✅ |
| `batch_size` | `204800` | 1回に流す点の数（メモリと速度のため分割する大きさ） | ✅ |
| `lr` | `0.005` | Adam の学習率 | ✅ |
| `grad_method` | `finite_difference` | 勾配の求め方（`autodiff` / `finite_difference`） | ✅ |
| `finite_difference_eps` | `0.03` | 有限差分の刻み幅 [m] | ✅ |
| `final_iterations` | `0` | 全フレーム処理した後の追加の最適化回数（0 で無し） | ✅ |

### 2-7. サンプリング（`sample_rays`）

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `n_stratified` | `1` | 1本のレイあたりの、自由空間のサンプル数 | ✅ |
| `n_perturbed` | `2` | 1本のレイあたりの、表面近傍の摂動サンプル数 | ✅ |
| `depth_min`, `depth_max` | `0.8`, `40.0` | サンプルに使う距離の範囲 | 🔶❓ |
| `surface_margin` | `1.44` | 表面の近傍の幅と思われる。Replica 用・屋外用のどちらの設定でも `3 × sigma_s` | 🔶❓ |
| `sigma_s` | `0.48` | 摂動の幅と思われる。`perturb_eta` として、摂動損失の下限にも使われる | ✅🔶 |

### 2-8. 出力・保存

| パラメータ | 値 | 役割 | 根拠 |
|---|---|---|---|
| `save_mesh` | `true` | 終了時にメッシュ（`mesh.ply`、`mesh_prior.ply`）を保存 | ✅ |
| `mesh_resolution` | `0.1` | メッシュ・断面画像を作る格子の間隔。出力の細かさだけで、地図の質には影響しない | ✅ |
| `mesh_iso_value` | `0.0` | メッシュにする SDF の値（0 が表面） | ✅ |
| `clean_mesh` | `true` | メッシュの後処理（細部は不明） | ✅❓ |
| `ckpt_interval` | `-1` | `N > 0` で、N エポックごとに `epoch_XXXX.pth` を保存（`-1` は保存しない）。エポック数は処理したフレーム数 | ✅ |
| `profiling`, `profiling_verbose` | `false` | 処理時間とメモリの計測、その詳細表示 | ✅ |

**yaml に書かれていない既定値**: `final_evaluate: true`（終了時にメッシュと断面画像を出力）、`final_save_model: true`（`final.pth` を保存）、`final_save_profiling_stats: true`、`save_slice: true`、`slice_center: None`（断面の位置。`None` は範囲の中心）、`frozen_model_path: null`（`trainer.py` では使われていない）。

---

## 3. 今回の上書き設定（`kenkyusitsu`）

`base_config: trainer-ros.yaml` を土台に、次の項目だけを上書きしていた（最後に確認した内容。マージ後の実際の値は、実行後の `bak/config.yaml` で確認する）。

| パラメータ | ベース | 上書き後 |
|---|---|---|
| `data.apply_bound` | `true` | `false` |
| `data.dataset_args.min_depth` / `max_depth` | `0.8` / `30.0` | `0.3` / `10.0` |
| `data.bound_min` / `bound_max`（最上位も同じ） | `[-40,-40,-1]` / `[45,10,3]` | `[-5,-5,-0.3]` / `[6,5,2]` |
| `model.octree_cfg.resolution` | `0.7` | `0.2` |
| `model.octree_cfg.init_voxel_num` | `80000` | `200000` |
| `sample_rays.depth_min` / `depth_max` | `0.8` / `40.0` | `0.3` / `10.0` |
| `sample_rays.surface_margin` | `1.44` | `1.0`（`sigma_s` は `0.48` のまま） |
| `final_iterations` | `0` | `100` |
| `slice_center` | `None` | `[0, 0, 0.4]`（以前は `[0,0,1.0]`） |

## 4. 運用上のメモ

- **学習範囲の外は学習されない**: `apply_bound: false` でも、範囲の外のサンプルは損失から除外される（✅）。経路計画では、範囲の外を通行不可にする。
- **断面の高さは合わせる**: `slice_center` の z（学習終了時の画像）と、`sdf_query_node` の `z`（動作中の配信）は独立の設定。比較したいときは同じ値にする。
- **`surface_margin = 3 × sigma_s`** の関係は、Replica 用と屋外用のどちらの設定でも保たれている。変えるなら両方を一緒に。
- **フレームが捨てられていないか**: 警告 `frame queue full` が出ていたら、`play_rate` を下げる。

---

## 5. 確認してほしいこと（❓の解消）

| 見るファイル | 確認する項目 |
|---|---|
| `oren/key_frame_set.py` | `insert_method`、`insert_interval`、`insert_ratio`、`selection_window_size`、`frame_weight` の意味 |
| `oren/octree_config.py`（と `oren/model.py` 周辺） | `semi_sparse_depth`、`insertion_threshold`、`residual_num_levels`、`independent_smallest_leaf_vertex` の意味 |
| `oren/residual_net.py` | `input_feature_dim` の追加1次元の中身、`output_sdf_scale` の使われ方 |
| `oren/utils/sampling.py` | `depth_min` / `depth_max`、`surface_margin`、`sigma_s`、`n_stratified` の正解値（`gt_sdf_stratified`）の作り方 |
| `oren_ros/` のノードのソース | `sensor_frame: ""` の挙動、`idle_block_sec`、`no_data_timeout_sec`、`sync_tolerance`、`odom_msg_type` の選択肢、`depth_scale`、`trainer_config_path: ""` と launch の関係 |
| `oren_ros/` の `sdf_query_node` と `oren_msgs/` | `x` / `y` / `z` の意味（`attached_to_frame: true` のとき）、`query_prior`、`trigger_service`、`query_service` のメッセージ型 |
| データセット（`dataset_name: ros`）のソース | `min_depth` / `max_depth` の扱い、`apply_bound: true` のときの動作 |