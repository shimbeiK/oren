# from erl_geometry import SemiSparseOctreeF, find_voxel_indices, morton_encode

# from oren import torch
# from oren.octree_config import OctreeConfig
# from oren.semi_sparse_octree_base import SemiSparseOctreeBase


# class SemiSparseOctree(SemiSparseOctreeBase):
#     def __init__(self, cfg: OctreeConfig):
#         super(SemiSparseOctree, self).__init__(cfg)

#         sso_setting = SemiSparseOctreeF.Setting()
#         sso_setting.resolution = cfg.resolution
#         sso_setting.tree_depth = cfg.tree_depth
#         sso_setting.semi_sparse_depth = cfg.semi_sparse_depth
#         sso_setting.init_voxel_num = cfg.init_voxel_num
#         sso_setting.independent_smallest_leaf_vertex = cfg.independent_smallest_leaf_vertex
#         sso_setting.cache_voxel_centers = True
#         self.sso = SemiSparseOctreeF(sso_setting)
#         self.key_offset = 1 << (self.cfg.tree_depth - 1)

#     @torch.no_grad()
#     def points_to_voxels(self, points: torch.Tensor):
#         """
#         Converts points to voxel coordinates.
#         Args:
#             points: (..., 3) point cloud in world coordinates
#         Returns:
#             voxels: (..., 3) voxel coordinates
#         """
#         voxels = torch.div(points, self.cfg.resolution, rounding_mode="floor").long()
#         voxels += self.key_offset
#         return voxels

#     @torch.no_grad()
#     def insert_voxels(self, voxels: torch.Tensor):
#         self.ever_inserted = True
#         svo_idx = self.sso.insert_keys(voxels.cpu().to(torch.uint32))  # on CPU

#         # (N, 4) [x, y, z, voxel_size], (x, y, z) is the center coordinate

#         self.voxels = self.sso.voxels_tensor.long().to(self.sdf_priors.device)
#         self.voxel_centers = self.sso.voxel_centers_tensor.to(self.sdf_priors.device)
#         self.vertex_indices = self.sso.vertices_tensor.to(self.sdf_priors.device)
#         self.structure = self.sso.children_tensor.to(self.sdf_priors.device)

#         return svo_idx

#     @torch.no_grad()
#     def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool, level: int = 1) -> torch.Tensor:
#         if are_voxels:
#             voxels = points
#         else:
#             voxels = self.points_to_voxels(points)
#         morton_codes = morton_encode(voxels.to(torch.uint32))
#         voxel_indices = find_voxel_indices(
#             codes=morton_codes,
#             dims=3,
#             n_levels=self.cfg.tree_depth - level,
#             children=self.structure,
#         ).long()
#         mask = ((voxels < 0) | (voxels >= (1 << self.cfg.tree_depth))).any(dim=-1)
#         voxel_indices[mask] = -1  # Out of bounds
#         return voxel_indices

#     @property
#     def little_endian_vertex_order(self):
#         return True  # e.g. 1 -> (1, 0, 0), a vertex on the x-axis

from erl_geometry import SemiSparseOctreeF, find_voxel_indices, morton_encode

from oren import torch
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree_base import SemiSparseOctreeBase


class SemiSparseOctree(SemiSparseOctreeBase):
    def __init__(self, cfg: OctreeConfig):
        super(SemiSparseOctree, self).__init__(cfg)

        self.sso = self._new_sso()
        self.key_offset = 1 << (self.cfg.tree_depth - 1)

        # The C++ tree (`self.sso`) is not part of state_dict. To make checkpoints restorable, every batch of keys that is
        # inserted is recorded here and saved through get_extra_state(); set_extra_state() replays it on load.
        self._insert_history: list = []
        self.restore_report: str = ""

    def _new_sso(self):
        cfg = self.cfg
        sso_setting = SemiSparseOctreeF.Setting()
        sso_setting.resolution = cfg.resolution
        sso_setting.tree_depth = cfg.tree_depth
        sso_setting.semi_sparse_depth = cfg.semi_sparse_depth
        sso_setting.init_voxel_num = cfg.init_voxel_num
        sso_setting.independent_smallest_leaf_vertex = cfg.independent_smallest_leaf_vertex
        sso_setting.cache_voxel_centers = True
        return SemiSparseOctreeF(sso_setting)

    @torch.no_grad()
    def points_to_voxels(self, points: torch.Tensor):
        """
        Converts points to voxel coordinates.
        Args:
            points: (..., 3) point cloud in world coordinates
        Returns:
            voxels: (..., 3) voxel coordinates
        """
        voxels = torch.div(points, self.cfg.resolution, rounding_mode="floor").long()
        voxels += self.key_offset
        return voxels

    def _sync_buffers_from_sso(self):
        # (N, 4) [x, y, z, voxel_size], (x, y, z) is the center coordinate
        device = self.sdf_priors.device
        self.voxels = self.sso.voxels_tensor.long().to(device)
        self.voxel_centers = self.sso.voxel_centers_tensor.to(device)
        self.vertex_indices = self.sso.vertices_tensor.to(device)
        self.structure = self.sso.children_tensor.to(device)

    @torch.no_grad()
    def insert_voxels(self, voxels: torch.Tensor):
        self.ever_inserted = True
        keys = voxels.cpu().to(torch.uint32)
        self._insert_history.append(keys.clone())  # exact sequence of inserts => deterministic replay
        svo_idx = self.sso.insert_keys(keys)  # on CPU
        self._sync_buffers_from_sso()
        return svo_idx

    # ---------------------------------------------------------------- checkpoint support

    def get_extra_state(self):
        return {"version": 1, "insert_history": self._insert_history}

    def set_extra_state(self, state):
        """Rebuild the C++ tree by replaying the recorded inserts (called by load_state_dict)."""
        history = state.get("insert_history", []) if isinstance(state, dict) else []
        names = ("voxels", "voxel_centers", "vertex_indices", "structure")
        loaded = {n: getattr(self, n).detach().clone() for n in names}  # buffers as loaded from the checkpoint

        self.sso = self._new_sso()
        self._insert_history = []
        self.ever_inserted = False
        for keys in history:
            keys = keys.cpu().to(torch.uint32)
            self._insert_history.append(keys.clone())
            self.sso.insert_keys(keys)
        if not history:
            self.restore_report = "octree history is empty: C++ tree NOT restored"
            return
        self._sync_buffers_from_sso()
        self.ever_inserted = True

        # Verify that the replayed tree is identical to the saved one (same voxel indices => parameters still match).
        problems = []
        for n in names:
            new = getattr(self, n).detach().cpu()
            old = loaded[n].cpu()
            m = min(new.shape[0], old.shape[0])
            if m == 0 or not torch.equal(new[:m].to(old.dtype), old[:m]):
                bad = int((new[:m].to(old.dtype) != old[:m]).any(dim=-1).sum()) if m > 0 else -1
                problems.append(f"{n}: {bad} rows differ (new={tuple(new.shape)}, loaded={tuple(old.shape)})")
        if problems:
            self.restore_report = "REPLAY MISMATCH -> " + "; ".join(problems)
        else:
            self.restore_report = (
                f"octree restored by replaying {len(history)} insert calls "
                f"({sum(int(k.shape[0]) for k in history)} keys); buffers identical to checkpoint"
            )

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool, level: int = 1) -> torch.Tensor:
        if are_voxels:
            voxels = points
        else:
            voxels = self.points_to_voxels(points)
        morton_codes = morton_encode(voxels.to(torch.uint32))
        voxel_indices = find_voxel_indices(
            codes=morton_codes,
            dims=3,
            n_levels=self.cfg.tree_depth - level,
            children=self.structure,
        ).long()
        mask = ((voxels < 0) | (voxels >= (1 << self.cfg.tree_depth))).any(dim=-1)
        voxel_indices[mask] = -1  # Out of bounds
        return voxel_indices

    @property
    def little_endian_vertex_order(self):
        return True  # e.g. 1 -> (1, 0, 0), a vertex on the x-axis