# ORENリアルタイム運用: 変更点まとめ
 
対象環境: ROS 2 + OREN（八分木SDFマッピング）、Rocinante、RTX 3060 Laptop 6GB。
状態の凡例: ✅ 動作確認済み / 🟡 適用したが効果未確認 / 📄 提案のみ（未適用・未検証）
 
---
 
## 1. 保存マップ（`final.pth`）の読み込み ✅
 
### 症状
`init_model_path` で `final.pth` を読み込むと、観測が入った瞬間に地図が壊れる（学習率を 0.0 にしても同じ）。
 
### 原因
オクトリーの本体は C++ オブジェクト `self.sso`（`erl_geometry`）で、`state_dict` に入らない。
読み込み時に `sso` は空で作られ、最初の `insert_voxels` が新フレームのボクセルだけで
`voxels / structure / vertex_indices` を**上書き**する。ボクセル番号が振り直され、
読み込んだパラメータ（`sdf_priors` など）との対応が崩れていた。
 
調べて外れた仮説: Adam の初期状態・初期化フェーズの反復・座標のずれ（`[align]` の |SDF| 中央値 0.04〜0.07 m で整合）・`ever_inserted` フラグ。
 
### 修正
| ファイル | 内容 |
|---|---|
| `oren/oren/semi_sparse_octree.py` | `insert_voxels` に渡したキーを順番どおり `_insert_history` に記録。`get_extra_state` で保存し、`set_extra_state`（`load_state_dict` から自動で呼ばれる）で同じ順に再生して C++ の木を再構築。再生後に4つのバッファを読み込み値と比較し `restore_report` に結果を残す |
| `oren_ros/ros_trainer.py` | `init_model_path` を読み込み。`strict=False` で読み込み、欠落キーを警告。履歴の無い古い `final.pth` は `NO octree insert history` エラーを出す。再生結果をログ出力 |
| `oren/oren/trainer.py` | `align_check_frames`（既定 0 = 動作変更なし）と `_log_alignment`。挿入前後の |SDF| 中央値を `[align] before-insert / after-insert` として出力 |
 
### 運用上の注意
- **旧形式の `final.pth` は復元できない**（履歴が入っていない）。新コードで一度マッピングし直し、Ctrl-C で保存した `final.pth` を使う。
- 成功時のログ: `octree restored by replaying N insert calls (M keys); buffers identical to checkpoint`
- `REPLAY MISMATCH` が出たら、C++ の挿入が非決定的。差のある行がログに出る。
- 履歴は全呼び出しをそのまま保存するので、チェックポイントが大きくなる。`skip_insertion_if_exists` が有効なら小さく済む。
### 追加パラメータ（`ros2-params*.yaml` の `oren_mapping_node` → `ros__parameters`）
| 名前 | 既定 | 内容 |
|---|---|---|
| `init_model_path` | `""` | 読み込む `.pth`。launch の引数で渡す（空なら最初から作る） |
| `init_skip_init_phase` | `true` | 再開時に初期化フェーズ（`init_frame_iterations` 回の多反復）を飛ばす |
| `init_lr_scale` | `1.0` | 再開時の学習率の倍率（問題が無ければ 1.0 のままでよい） |
| `init_align_check_frames` | `20` | 再開時に座標整合を確認するフレーム数 |
 
YAML の配列は型を揃える（`[5, 15]` ではなく `[5.0, 15.0]`）。
 
---
 
## 2. 処理速度（GIL の競合） ✅
 
- 原因: 学習スレッドと rclpy の spin スレッドの GIL 取り合い。
- 修正: `mapping_node.py` の `main()` に `sys.setswitchinterval(2e-4)`。約 0.55 → 3.3〜4.7 it/s。
- 計測用に入れた `torch.cuda.synchronize()` は削除済み（`trainer.py` には無い）。
- `get_points` の `.to(device)` が暗黙の同期になっている（GPU 待ちは壁時間の約16%）。
---
 
## 3. `sdf_query_node` の表示ずれ 🟡
 
- 症状: `attached_to_frame: true` でオフセット（x, y, z）を変えると、中身はずれるが RViz 上の位置はずれない。
- 原因: `_make_grid_map` で添付モードのとき `info.pose.position` を 0 にしていた。
- 修正: `else 0.0` を `else self._x`（y, z も同様）にする。または `if not … else …` ごと消して `self._x` 等を直接代入。
  添付しない場合・オフセット 0 の場合は動作が変わらない。
- z（断面の高さ）を RViz が平面の高さに使うかは未確認。ずれる場合は `publish_point_cloud: true`（点群は z も正しい位置に出る）か、オフセット用の静的 TF フレームを使う。
- 📄 未適用の提案: `attached_to_frame: false` のとき `Point` リストをキャッシュ、範囲外セルを NaN にする。
### `/sdf_grid_map` を軽くする設定
`publish_gradient: false`、グリッドを小さく・粗く（例 0.2 m × 51×51）、必要最小限の `publish_rate`。
グリッドの範囲は学習範囲（`bound_min/bound_max`）とは独立なので、手動で整合させる。
 
---
 
## 4. LiDAR フィルタ（新規 C++ パッケージ `lidar_filter`） 🟡
 
`/lidar_points` 系 → `/lidar_points_filtered`（入出力名は yaml で変更）。
 
処理順（既定）: 距離範囲・NaN → **自機除去** → 孤立点除去 → 反射強度 → 距離適応ボクセル。
 
| パラメータ | 内容 |
|---|---|
| `min_range / max_range` | 範囲外と NaN を最初に捨てる |
| `self_filter.boxes` | 自機除去。センサ座標（点群の `frame_id`）の直方体 `[xmin,xmax,ymin,ymax,zmin,zmax]` を連結。例の初期値は**要調整** |
| `outlier.radius / min_neighbors` | セル+隣接26セルに自分を含め `min_neighbors+1` 点未満なら除去 |
| `intensity.min / before_outlier` | `intensity <= min` を除去。順序の切り替え |
| `voxel.band_edges / sizes` | 例 `[5.0, 15.0]` / `[0.05, 0.15, 0.30]`（5 m 未満 5 cm、15 m 以上 30 cm） |
| `intensity_field` | 強度フィールド名（Livox の uint8 反射率なら `reflectivity`） |
 
- 代表点は重心に最も近い実在点。stamp・frame_id・フィールド構成は入力のまま。
- 単体テスト（ROS 不要）と合成13万点でのベンチ（サンドボックスの低速2コアで約20 ms）は確認済み。ROS ノード部はスタブでの構文確認のみ。
- 実機ログ例: 入力 230,400 点 → 自機除去で約 42,000 点（18%）が消えており、**自機除去の箱が大きすぎる可能性**。孤立点除去は 1 点しか消さず、Hesai のように密な点群では `radius` を小さく・`min_neighbors` を大きくしないと効かない。
- RViz に出ない件: QoS（Best Effort / Reliable）、TF、表示設定の確認項目を案内済み（結果は未確認）。
- OREN の点数: 1 回の学習で使うのは `num_rays_total`（20480）本なので、点数を減らしても学習1回の重さはほぼ変わらない。メモリと前処理に効く。
---
 
## 5. 可視化スクリプト 🟡（合成データで確認）
 
| ファイル | 内容 |
|---|---|
| `visualize_sdf3d.py` | `query_result.npz` の3次元可視化（断面・等値面メッシュ・点群）。`--bbox`、`--lim`（色範囲の固定） |
| `visualize_gradient.py` | 指定座標周辺の ∇SDF。水平/垂直断面、SDF=0 面の法線、PLY 出力。`--center`、`--radius`/`--box`、`--grad-key`（解析勾配）、`--lim` |
| `diagnose_grid.py` | 高さごとの SDF 統計 |
 
- 色の範囲は固定（既定 ±1.0）。RViz の GridMap 表示は `Auto Compute` を外し、Min/Max を固定する。
- 解析勾配を使うには、`oren_query.py` の保存時に `out["grad"]["sdf"]` を `grad_sdf` として保存する。
---
 
## 6. launch ファイル
 
- `live.launch.py`: ライブ用。引数 `trainer_config_path`、`params_file`、`use_sim_time`（false）、`rviz`、`visualize_sdf`、`init_model_path`。
  空文字パラメータは `ParameterValue(..., value_type=str)` で渡す。
- `bringup_all.launch.py` / `start_rocinante_motor.launch.py`: ロボット側の起動を1ファイルに統合（元のコマンドは先頭にコメントで保存）。
---
 
## 7. 未解決・提案のみ
 
- 📄 消失への感度が低い: 古いキーフレームが消えない。レンジ画像による自由空間チェックで矛盾したキーフレーム点の `valid_mask` を False にする案（`key_frame_set.py` の対応が必要）。
- `odom` が TF に無い件: 起動直後の TF 警告は起動準備中のもの。継続する場合は `ros2 node list`、`/tf` の publisher を確認。
- 論文値との比較: OREN は RTX 3090 で約 14 fps、Jetson Orin で約 7 fps（fps の定義は不明）。
## 8. 触ったファイル一覧
 
| リポジトリ側（要反映） | 新規（別パッケージ・スクリプト） |
|---|---|
| `oren/oren/semi_sparse_octree.py` | `lidar_filter/`（C++ パッケージ） |
| `oren/oren/trainer.py` | `visualize_sdf3d.py`、`visualize_gradient.py`、`diagnose_grid.py` |
| `oren_ros/ros_trainer.py` | `live.launch.py`、`bringup_all.launch.py`、`start_rocinante_motor.launch.py` |
| `oren_ros/node/mapping_node.py`（`setswitchinterval`） | |
| `oren_ros/node/sdf_query_node.py`（`info.pose` の3行） | |
| `ros2-params*.yaml`（新パラメータ） | |
 