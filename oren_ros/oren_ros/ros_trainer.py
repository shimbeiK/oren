# """Wrapper around `Trainer` that owns the ROS interface (services + shutdown).

# Mirrors the `GuiTrainer` pattern: builds a `Trainer`, attaches to its existing callback hooks
# (`training_iteration_end_callback`, `training_frame_start_callback`), and registers ROS 2 services on a shared node.

# Threading: `Trainer.train()` runs on the main thread (CUDA + heavy work).
# Subscriber and service callbacks run on a `MultiThreadedExecutor` spinning in a background daemon thread
# (set up by the entry-point script). Service handlers enqueue requests into a `queue.Queue` and block on a `Future`;
# the main thread drains the queue between iterations and while idle (via `data_stream.on_idle`), so all model access
# is serialized.
# """

# import queue
# import threading
# from concurrent.futures import Future
# from typing import Callable

# import torch, os
# from geometry_msgs.msg import Vector3
# from rclpy.callback_groups import ReentrantCallbackGroup
# from rclpy.node import Node

# from oren.frame import Frame
# from oren.trainer import Trainer
# from oren.trainer_config import TrainerConfig
# from oren_msgs.srv import QuerySdf, SaveMesh, SaveModel

# from oren_ros.ros_dataloader import RosDataLoader
# import torch  # ファイルの先頭に

# class RosTrainer:
#     def __init__(self, trainer_cfg: TrainerConfig, ros_node: Node):
#         self.ros_node = ros_node

#         # Build the data loader directly so cfg.data.dataset_args never holds a Node.
#         loader = RosDataLoader(ros_node=ros_node, **trainer_cfg.data.dataset_args)
#         self.trainer = Trainer(trainer_cfg, data_stream=loader)

#         # Service request queue + shutdown signal.
#         self._service_queue: queue.Queue[tuple[Callable[[Trainer], object], Future]] = queue.Queue()
#         self._shutdown = threading.Event()

#         # Trainer callbacks (run on the main / training thread).
#         self.trainer.training_iteration_end_callback = self._drain_services
#         self.trainer.training_frame_start_callback = self._on_frame_start
#         # Idle drain while __getitem__ is blocked waiting for data.
#         loader.on_idle = lambda: self._drain_services(self.trainer)

#         # new!
#         ros_node.declare_parameter("init_model_path", "")
#         p = (ros_node.get_parameter("init_model_path").value or "").strip()
#         if p:
#             if not os.path.isfile(p):
#                 raise FileNotFoundError(f"init_model_path not found: {p}")
#             state = torch.load(p, map_location=trainer_cfg.device)
#             self.trainer.model.load_state_dict(state)
#             ros_node.get_logger().info(f"[RosTrainer] loaded {p}")
#         else:
#             ros_node.get_logger().info("[RosTrainer] init_model_path is empty: building the map from scratch")
#         # Service registration. Reentrant group so a handler waiting on Future.result()
#         # doesn't block other service / subscriber callbacks.
#         srv_cbg = ReentrantCallbackGroup()
#         ros_node.create_service(QuerySdf, "query_sdf", self._srv_query_sdf, callback_group=srv_cbg)
#         ros_node.create_service(SaveMesh, "save_mesh", self._srv_save_mesh, callback_group=srv_cbg)
#         ros_node.create_service(SaveModel, "save_model", self._srv_save_model, callback_group=srv_cbg)

#         ros_node.get_logger().info("[RosTrainer] services ready: /query_sdf /save_mesh /save_model")

#     # ---------------------- main entry point ----------------------------------

#     def run(self) -> None:
#         try:
#             self.trainer.train()
#         except KeyboardInterrupt:
#             self.ros_node.get_logger().info("[RosTrainer] KeyboardInterrupt — shutting down.")
#             self._shutdown.set()
#             # Make sure the data loader stops blocking too.
#             if hasattr(self.trainer.data_stream, "shutdown"):
#                 self.trainer.data_stream.shutdown()
#             raise

#     # ---------------------- callbacks (main thread) ---------------------------

#     def _drain_services(self, trainer: Trainer) -> None:
#         while True:
#             try:
#                 fn, fut = self._service_queue.get_nowait()
#             except queue.Empty:
#                 return
#             try:
#                 fut.set_result(fn(trainer))
#             except Exception as e:  # noqa: BLE001 — propagate to the service caller
#                 fut.set_exception(e)

#     def _on_frame_start(self, trainer: Trainer, frame: Frame) -> bool:
#         return not self._shutdown.is_set()

#     # ---------------------- service plumbing (executor thread) ----------------

#     def _submit(self, fn: Callable[[Trainer], object], timeout: float = 60.0):
#         fut: Future = Future()
#         self._service_queue.put((fn, fut))
#         return fut.result(timeout=timeout)

#     def _srv_query_sdf(self, request, response):
#         try:
#             points = torch.tensor(
#                 [[p.x, p.y, p.z] for p in request.points],
#                 dtype=torch.float32,
#             )
#             if points.shape[0] == 0:
#                 response.sdf = []
#                 response.grad = []
#                 return response
#             out = self._submit(lambda t: t.query_sdf(points, return_grad=request.return_grad, prior_only=request.prior))
#             sdf_t: torch.Tensor = out["sdf"].detach().cpu().reshape(-1)  # if request.prior, this equals sdf_prior
#             response.sdf = sdf_t.tolist()
#             if request.return_grad:
#                 grad = out["grad"]
#                 grad = grad["sdf_prior"] if request.prior else grad["sdf"]
#                 grad = grad.detach().cpu().reshape(-1, 3)
#                 response.grad = [Vector3(x=float(g[0]), y=float(g[1]), z=float(g[2])) for g in grad]
#             else:
#                 response.grad = []
#         except Exception as e:
#             self.ros_node.get_logger().error(f"[RosTrainer] query_sdf failed: {e}")
#             response.sdf = []
#             response.grad = []
#         return response

#     def _srv_save_mesh(self, request, response):
#         try:
#             bound_min = list(request.bound_min) if request.use_bound else None
#             bound_max = list(request.bound_max) if request.use_bound else None
#             grid_resolution = request.resolution if request.resolution > 0.0 else None
#             self._submit(
#                 lambda t: t.save_mesh(
#                     request.path,
#                     prior=request.prior,
#                     bound_min=bound_min,
#                     bound_max=bound_max,
#                     grid_resolution=grid_resolution,
#                     iso_value=request.iso_value,
#                 )
#             )
#             response.success = True
#             response.message = f"mesh saved: {request.path} (prior={request.prior})"
#         except Exception as e:
#             response.success = False
#             response.message = repr(e)
#         return response

#     def _srv_save_model(self, request, response):
#         try:
#             self._submit(lambda t: t.save_model(request.path))
#             response.success = True
#             response.message = f"model saved: {request.path}"
#         except Exception as e:
#             response.success = False
#             response.message = repr(e)
#         return response

"""Wrapper around `Trainer` that owns the ROS interface (services + shutdown).

Mirrors the `GuiTrainer` pattern: builds a `Trainer`, attaches to its existing callback hooks
(`training_iteration_end_callback`, `training_frame_start_callback`), and registers ROS 2 services on a shared node.

Threading: `Trainer.train()` runs on the main thread (CUDA + heavy work).
Subscriber and service callbacks run on a `MultiThreadedExecutor` spinning in a background daemon thread
(set up by the entry-point script). Service handlers enqueue requests into a `queue.Queue` and block on a `Future`;
the main thread drains the queue between iterations and while idle (via `data_stream.on_idle`), so all model access
is serialized.
"""

import os
import queue
import threading
from concurrent.futures import Future
from typing import Callable

import torch
from geometry_msgs.msg import Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from oren.frame import Frame
from oren.trainer import Trainer
from oren.trainer_config import TrainerConfig
from oren_msgs.srv import QuerySdf, SaveMesh, SaveModel

from oren_ros.ros_dataloader import RosDataLoader


class RosTrainer:
    def __init__(self, trainer_cfg: TrainerConfig, ros_node: Node):
        self.ros_node = ros_node

        # Build the data loader directly so cfg.data.dataset_args never holds a Node.
        loader = RosDataLoader(ros_node=ros_node, **trainer_cfg.data.dataset_args)
        self.trainer = Trainer(trainer_cfg, data_stream=loader)

        # Service request queue + shutdown signal.
        self._service_queue: queue.Queue[tuple[Callable[[Trainer], object], Future]] = queue.Queue()
        self._shutdown = threading.Event()

        # Trainer callbacks (run on the main / training thread).
        self.trainer.training_iteration_end_callback = self._drain_services
        self.trainer.training_frame_start_callback = self._on_frame_start
        # Idle drain while __getitem__ is blocked waiting for data.
        loader.on_idle = lambda: self._drain_services(self.trainer)

        # --- warm start from a saved model (final.pth etc.) -----------------------------------
        ros_node.declare_parameter("init_model_path", "")
        ros_node.declare_parameter("init_skip_init_phase", True)  # 再開時は初期化フェーズ(多反復)を飛ばす
        ros_node.declare_parameter("init_lr_scale", 1.0)  # 再開時の学習率の倍率 (例 0.1)
        ros_node.declare_parameter("init_align_check_frames", 20)  # 再開時に座標整合を確認するフレーム数
        p = (ros_node.get_parameter("init_model_path").value or "").strip()
        if p:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"init_model_path not found: {p}")
            state = torch.load(p, map_location=trainer_cfg.device)
            # strict=False so that checkpoints saved before octree history was recorded can still be inspected;
            # missing/unexpected keys are reported below.
            res = self.trainer.model.load_state_dict(state, strict=False)
            ros_node.get_logger().info(f"[RosTrainer] loaded {p}")
            if res.missing_keys or res.unexpected_keys:
                ros_node.get_logger().warn(
                    f"[RosTrainer] load_state_dict: missing={list(res.missing_keys)} unexpected={list(res.unexpected_keys)}"
                )
            if "octree._extra_state" in res.missing_keys:
                ros_node.get_logger().error(
                    "[RosTrainer] this checkpoint has NO octree insert history (saved by the old code): the C++ octree "
                    "cannot be restored and the loaded map will be scrambled on the first insertion. "
                    "Re-create the map once with the new code and use that final.pth."
                )
            report = getattr(self.trainer.model.octree, "restore_report", "")
            if report:
                (ros_node.get_logger().error if "MISMATCH" in report or "NOT" in report else ros_node.get_logger().info)(
                    f"[RosTrainer] {report}"
                )

            # Diagnostics: plain (non-buffer, non-parameter) attributes of the octree are NOT stored in state_dict.
            # If allocation counters live there, they are lost on load.
            try:
                oc = self.trainer.model.octree
                plain = {
                    k: (tuple(v.shape) if torch.is_tensor(v) else v)
                    for k, v in vars(oc).items()
                    if not k.startswith("_") and (torch.is_tensor(v) or isinstance(v, (int, float, bool)))
                }
                ros_node.get_logger().info(f"[RosTrainer] octree plain attrs (not in state_dict): {plain}")
                ros_node.get_logger().info(
                    f"[RosTrainer] state_dict keys: {list(self.trainer.model.state_dict().keys())}"
                )
            except Exception as e:  # noqa: BLE001
                ros_node.get_logger().warn(f"[RosTrainer] diagnostics failed: {e!r}")

            tcfg = self.trainer.cfg
            ros_node.get_logger().info(
                f"[RosTrainer] num_init_frames={tcfg.num_init_frames} "
                f"init_frame_iterations={tcfg.init_frame_iterations} "
                f"num_iterations_per_frame={tcfg.num_iterations_per_frame} lr={tcfg.lr}"
            )
            if bool(ros_node.get_parameter("init_skip_init_phase").value):
                self.trainer.epoch = max(self.trainer.epoch, tcfg.num_init_frames)
                ros_node.get_logger().info(f"[RosTrainer] init phase skipped (epoch={self.trainer.epoch})")
            scale = float(ros_node.get_parameter("init_lr_scale").value)
            if scale != 1.0:
                for g in self.trainer.optimizer.param_groups:
                    g["lr"] *= scale
                ros_node.get_logger().info(
                    f"[RosTrainer] lr scaled x{scale}: {[g['lr'] for g in self.trainer.optimizer.param_groups]}"
                )
            self.trainer.align_check_frames = int(ros_node.get_parameter("init_align_check_frames").value)
        else:
            ros_node.get_logger().info("[RosTrainer] init_model_path is empty: building the map from scratch")

        # Service registration. Reentrant group so a handler waiting on Future.result()
        # doesn't block other service / subscriber callbacks.
        srv_cbg = ReentrantCallbackGroup()
        ros_node.create_service(QuerySdf, "query_sdf", self._srv_query_sdf, callback_group=srv_cbg)
        ros_node.create_service(SaveMesh, "save_mesh", self._srv_save_mesh, callback_group=srv_cbg)
        ros_node.create_service(SaveModel, "save_model", self._srv_save_model, callback_group=srv_cbg)

        ros_node.get_logger().info("[RosTrainer] services ready: /query_sdf /save_mesh /save_model")

    # ---------------------- main entry point ----------------------------------

    def run(self) -> None:
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            self.ros_node.get_logger().info("[RosTrainer] KeyboardInterrupt — shutting down.")
            self._shutdown.set()
            # Make sure the data loader stops blocking too.
            if hasattr(self.trainer.data_stream, "shutdown"):
                self.trainer.data_stream.shutdown()
            raise

    # ---------------------- callbacks (main thread) ---------------------------

    def _drain_services(self, trainer: Trainer) -> None:
        while True:
            try:
                fn, fut = self._service_queue.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn(trainer))
            except Exception as e:  # noqa: BLE001 — propagate to the service caller
                fut.set_exception(e)

    def _on_frame_start(self, trainer: Trainer, frame: Frame) -> bool:
        return not self._shutdown.is_set()

    # ---------------------- service plumbing (executor thread) ----------------

    def _submit(self, fn: Callable[[Trainer], object], timeout: float = 60.0):
        fut: Future = Future()
        self._service_queue.put((fn, fut))
        return fut.result(timeout=timeout)

    def _srv_query_sdf(self, request, response):
        try:
            points = torch.tensor(
                [[p.x, p.y, p.z] for p in request.points],
                dtype=torch.float32,
            )
            if points.shape[0] == 0:
                response.sdf = []
                response.grad = []
                return response
            out = self._submit(lambda t: t.query_sdf(points, return_grad=request.return_grad, prior_only=request.prior))
            sdf_t: torch.Tensor = out["sdf"].detach().cpu().reshape(-1)  # if request.prior, this equals sdf_prior
            response.sdf = sdf_t.tolist()
            if request.return_grad:
                grad = out["grad"]
                grad = grad["sdf_prior"] if request.prior else grad["sdf"]
                grad = grad.detach().cpu().reshape(-1, 3)
                response.grad = [Vector3(x=float(g[0]), y=float(g[1]), z=float(g[2])) for g in grad]
            else:
                response.grad = []
        except Exception as e:
            self.ros_node.get_logger().error(f"[RosTrainer] query_sdf failed: {e}")
            response.sdf = []
            response.grad = []
        return response

    def _srv_save_mesh(self, request, response):
        try:
            bound_min = list(request.bound_min) if request.use_bound else None
            bound_max = list(request.bound_max) if request.use_bound else None
            grid_resolution = request.resolution if request.resolution > 0.0 else None
            self._submit(
                lambda t: t.save_mesh(
                    request.path,
                    prior=request.prior,
                    bound_min=bound_min,
                    bound_max=bound_max,
                    grid_resolution=grid_resolution,
                    iso_value=request.iso_value,
                )
            )
            response.success = True
            response.message = f"mesh saved: {request.path} (prior={request.prior})"
        except Exception as e:
            response.success = False
            response.message = repr(e)
        return response

    def _srv_save_model(self, request, response):
        try:
            self._submit(lambda t: t.save_model(request.path))
            response.success = True
            response.message = f"model saved: {request.path}"
        except Exception as e:
            response.success = False
            response.message = repr(e)
        return response