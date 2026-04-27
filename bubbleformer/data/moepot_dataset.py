"""
This module contains dataset class for Time Series Forecasting
using MoE-Pot model on BubbleML dataset. It returns data that
supports the autoregressive training loop and also allows for
PDE class token prediction on different bubbleml datasets.
Classes:
    BubbleForecast: Dataset class for BubbleML dataset
Author: Sheikh Md Shakeel Hassan
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py as h5
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# -----------------------------
# Optional: explicit mapping docstring / enum
# -----------------------------

DATASET_CLASS_NAMES = {
    0: "subcooled-fc72",
    1: "subcooled-r515b",
    2: "subcooled-ln2",
    3: "saturated-fc72",
    4: "saturated-r515b",
    5: "saturated-ln2",
}

# -----------------------------
# Helpers
# -----------------------------

def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_json_path(h5_path: str) -> str:
    # Your repo convention: same stem, .json next to .hdf5
    if h5_path.endswith(".hdf5"):
        return h5_path[:-5] + ".json"
    if h5_path.endswith(".h5"):
        return h5_path[:-3] + ".json"
    return h5_path + ".json"


def _combo_to_dataset_idx(setup: str, liquid: str) -> int:
    """
    6-way classification from (setup, liquid):
      setup ∈ {subcooled, saturated}
      liquid ∈ {fc72, r515b, ln2}
    """
    setup = (setup or "").strip().lower()
    liquid = (liquid or "").strip().lower()

    valid_setups = ["subcooled", "saturated"]
    valid_liquids = ["fc72", "r515b", "ln2"]

    if setup not in valid_setups:
        raise ValueError(f"Unknown setup={setup!r}. Expected one of {valid_setups}.")
    if liquid not in valid_liquids:
        raise ValueError(f"Unknown liquid={liquid!r}. Expected one of {valid_liquids}.")

    # index = setup_id * 3 + liquid_id
    setup_id = valid_setups.index(setup)       # 0=subcooled, 1=saturated
    liquid_id = valid_liquids.index(liquid)    # 0=fc72, 1=r515b, 2=ln2
    return setup_id * 3 + liquid_id

class MOEPotBubbleForecast(Dataset):
    """
    MOE-Pot mixed dataset loader
      - multiple datasets mixed, optionally weighted
      - returns (x, y, msk, idx_cls) where:
          x: (T_in, Cmax, H, W)
          y: (T_out, Cmax, H, W)
          msk: (H, W, 1, Cmax) in original, 
                but here returned as (1, Cmax, H, W) for convenience
          idx_cls: LongTensor([dataset_idx]) derived 
                from JSON setup/liquid combination (6 classes)
                mapping:
                    0: subcooled-fc72
                    1: subcooled-r515b
                    2: subcooled-ln2
                    3: saturated-fc72
                    4: saturated-r515b
                    5: saturated-ln2
    """

    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        norm: str = "none",
        downsample_factor: int = 1,
        time_window: int = 10,            # == t_in
        t_ar: int = 1,                    # == t_ar for train
        start_time: int = 0,
    ):
        super().__init__()

        self.filenames = filenames

        self.input_fields = input_fields
        self.output_fields = output_fields
        self.norm = norm
        self.downsample_factor = downsample_factor

        self.time_window = time_window
        self.t_ar = t_ar
        self.start_time = start_time

        self.num_sources = len(self.filenames)

        self.json_paths = [_infer_json_path(fn) for fn in self.filenames]
        self.json_meta = [_read_json(p) for p in self.json_paths]

        # dataset_idx per source is determined by (setup, liquid) from JSON
        self.source_class_idx = []
        for meta in self.json_meta:
            setup = meta.get("setup", "")
            liquid = meta.get("liquid", "")
            self.source_class_idx.append(_combo_to_dataset_idx(setup, liquid))

        # For field-based: each field shape usually (T, H, W) in your BubbleForecast.
        self.traj_lens = []
        self.num_trajs = []
        self.pred_channels = []  # per source, like BubbleFormer orig_size[-1] adjustment

        for i in range(self.num_sources):
            if self.scatter_storage:
                # scatter_storage implies each sample is a separate file, so "trajectory length" is from that file.
                # We'll assume each scatter sample is one trajectory.
                # But without enumerating files, we treat each "source" as one trajectory handle.
                # (If you want many scatter samples, pass many "filenames" as roots OR create an index list externally.)
                sample = self._read_scatter_sample(i, sample_idx=0)  # just to infer shapes
                # sample should be (H, W, T, C) or (T, C, H, W) after conversion below; we infer T
                H, W, T, C = sample.shape
                self.traj_lens.append(T)
                self.num_trajs.append(1)
                self.pred_channels.append(C)
            else:
                h5f = self.data[i]
                if self.data_key is not None:
                    arr = h5f[self.data_key]
                    # Expect (N, H, W, T, C) or (H, W, T, C)
                    if arr.ndim == 5:
                        # multiple trajectories; we treat each trajectory as a sample trajectory
                        self.num_trajs.append(arr.shape[0])
                        self.traj_lens.append(arr.shape[3])  # T
                        self.pred_channels.append(arr.shape[4])  # C
                    elif arr.ndim == 4:
                        self.num_trajs.append(1)
                        self.traj_lens.append(arr.shape[2])  # T
                        self.pred_channels.append(arr.shape[3])  # C
                    else:
                        raise ValueError(f"{self.data_key} has unsupported ndim={arr.ndim} in {self.filenames[i]}")
                else:
                    # field-based (BubbleML style): each field is (T, H, W)
                    if self.input_fields is None:
                        raise ValueError("input_fields must be provided when data_key=None.")
                    first_field = self.input_fields[0]
                    self.num_trajs.append(1)
                    self.traj_lens.append(h5f[first_field].shape[0])  # T
                    # channel count = number of fields used (union)
                    fields_union = list(set((self.input_fields or []) + (self.output_fields or [])))
                    self.pred_channels.append(len(fields_union))

        # Determine unified Cmax across sources (BubbleFormer n_channels = max)
        self.cmax = max(self.pred_channels)

        # Determine t_test behavior
        # If train=False and t_test==0, use "rest of trajectory after t_in" (bounded by available length)
        self.t_test = t_test

        # Build weighted cumulative sizes (like BubbleFormer)
        # Length per source = number of trajectories * "number of possible windows"
        self.samples_per_source = []
        for ns, T in zip(self.num_trajs, self.traj_lens):
            if self.train:
                # number of possible start indices for window length (t_in + t_ar)
                usable = max(T - (self.time_window + self.t_ar) + 1, 1)
            else:
                # one sample per trajectory for eval (full target chunk)
                usable = 1
            self.samples_per_source.append(ns * usable)

        self.weighted_sizes = [s * w for s, w in zip(self.samples_per_source, self.data_weights)]
        self.cumulative_sizes = np.cumsum(self.weighted_sizes).astype(int)

        # Normalization terms (your BubbleForecast style)
        self.diff_terms: Dict[int, torch.Tensor] = {}
        self.div_terms: Dict[int, torch.Tensor] = {}
        if self.norm != "none":
            self._compute_norm_terms()

    # -----------------------------
    # Normalization (your style)
    # -----------------------------

    def _compute_norm_terms(self) -> None:
        """
        Compute per-source, per-channel normalization terms for x.
        We compute stats on INPUT channels only, across a small subset to keep it cheap.
        """
        for src in range(self.num_sources):
            # sample a limited set of frames to estimate stats robustly
            # We use up to first 500 timesteps worth of frames, similar spirit to BubbleFormer.
            # For field-based, union fields become channels.
            x_full = self._load_full_trajectory(src, traj_idx=0)  # (T, C, H, W)
            T = x_full.shape[0]
            Tsub = min(T, 500)
            x_sub = x_full[:Tsub]  # (Tsub, C, H, W)

            # Flatten over T,H,W for each channel
            flat = x_sub.permute(1, 0, 2, 3).contiguous().view(x_sub.shape[1], -1)  # (C, N)

            if self.norm == "std":
                diff = flat.mean(dim=1)
                div = flat.std(dim=1)
            elif self.norm == "minmax":
                diff = flat.min(dim=1).values
                div = flat.max(dim=1).values - diff
            elif self.norm == "tanh":
                mn = flat.min(dim=1).values
                mx = flat.max(dim=1).values
                diff = (mx + mn) / 2.0
                div = (mx - mn) / 2.0
            else:
                raise ValueError(f"Unknown normalization type: {self.norm}")

            self.diff_terms[src] = diff.float()
            self.div_terms[src] = (div.float() + 1e-8)

    # -----------------------------
    # Data reading
    # -----------------------------

    def _read_scatter_sample(self, src: int, sample_idx: int) -> torch.Tensor:
        """
        scatter_storage: data_{idx}.hdf5 files in scatter_root/src or scatter_root directly.
        Returns raw sample as (H, W, T, C).
        """
        if self.scatter_root is None:
            # if filenames are roots
            root = self.filenames[src]
        else:
            root = self.scatter_root

        path = os.path.join(root, f"data_{sample_idx}.hdf5")
        with h5.File(path, "r") as f:
            arr = f[self.scatter_key][:]
        x = torch.from_numpy(arr).float()
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        # expect H,W,T,C
        return x

    def _load_full_trajectory(self, src: int, traj_idx: int = 0) -> torch.Tensor:
        """
        Load one full trajectory for a source and return (T, C, H, W).
        Supports:
          - data_key="data": (N,H,W,T,C) or (H,W,T,C)
          - field-based: (T,H,W) per field
        """
        if self.scatter_storage:
            raw = self._read_scatter_sample(src, traj_idx)  # treat traj_idx as sample id
            # raw: (H, W, T, C) -> (T, C, H, W)
            x = raw.permute(2, 3, 0, 1).contiguous()
            return x

        h5f = self.data[src]

        if self.data_key is not None:
            arr = h5f[self.data_key]
            if arr.ndim == 5:
                raw = torch.from_numpy(arr[traj_idx]).float()   # (H,W,T,C)
            else:
                raw = torch.from_numpy(arr[:]).float()          # (H,W,T,C)
            if raw.ndim == 3:
                raw = raw.unsqueeze(-1)
            x = raw.permute(2, 3, 0, 1).contiguous()  # (T,C,H,W)
            return x

        # field-based: channels = union(fields)
        fields_union = list(set((self.input_fields or []) + (self.output_fields or [])))
        # Keep a deterministic order to avoid set() randomness
        fields_union = sorted(fields_union)

        chans = []
        for fld in fields_union:
            a = torch.from_numpy(h5f[fld][...]).float()  # (T,H,W)
            chans.append(a.unsqueeze(1))                  # (T,1,H,W)
        x = torch.cat(chans, dim=1)                      # (T,C,H,W)
        return x

    # -----------------------------
    # Preprocess: resize, channel-pad, downsample, normalize
    # -----------------------------

    def _preprocess(self, x: torch.Tensor, src: int, start_t: int, t_len: int) -> torch.Tensor:
        """
        x: full trajectory (T,C,H,W)
        returns: (t_len, Cmax, H', W') after slice + resize + pad + optional normalize + downsample
        """
        x = x[start_t : start_t + t_len]  # (t_len, C, H, W)

        # resize to res
        x = _resize_hw(x, self.res, mode=self.normalize_mode)

        # pad channels to Cmax
        x = _pad_to_cmax(x, self.cmax)

        # normalize (channelwise)
        if self.norm != "none":
            diff = self.diff_terms[src].view(1, -1, 1, 1)  # (1,C,1,1) but C is pred_channels
            div = self.div_terms[src].view(1, -1, 1, 1)
            # pad diff/div to cmax so broadcast works
            if diff.shape[1] != self.cmax:
                d = torch.zeros((1, self.cmax, 1, 1), dtype=diff.dtype)
                v = torch.ones((1, self.cmax, 1, 1), dtype=div.dtype)
                c = min(diff.shape[1], self.cmax)
                d[:, :c] = diff[:, :c]
                v[:, :c] = div[:, :c]
                diff, div = d, v

            x = (x - diff) / div

        # downsample
        x = _downsample_hw(x, self.downsamples[src])

        return x

    # -----------------------------
    # Dataset API
    # -----------------------------

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) else 0

    def _map_global_to_source(self, idx: int) -> Tuple[int, int]:
        """
        Map global idx -> (source_idx, local_idx), respecting data_weights.
        """
        src = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        prev = 0 if src == 0 else self.cumulative_sizes[src - 1]
        local = idx - prev

        # Undo weighting to pick actual sample within the source
        w = int(self.data_weights[src])
        local //= w
        return src, int(local)

    def __getitem__(self, idx: int):
        src, local = self._map_global_to_source(idx)

        # Decide which trajectory + which window within that trajectory
        if self.train:
            # local indexes windows across all trajectories within this source
            T = self.traj_lens[src]
            ns = self.num_trajs[src]
            windows_per_traj = max(T - (self.time_window + self.t_ar) + 1, 1)

            traj_idx = local // windows_per_traj
            win_idx = local % windows_per_traj

            traj_idx = int(traj_idx % ns)

            # If start_time > 0, further constrain randomization by shifting windowing,
            # but keep behavior close to BubbleFormer: random start.
            # We'll randomize start each call (like BubbleFormer), but keep within valid range.
            # If you want deterministic indexing, replace this with win_idx.
            max_start = max(T - (self.time_window + self.t_ar) + 1, 1)
            start_t = np.random.randint(max_start)

            full = self._load_full_trajectory(src, traj_idx=traj_idx)

            x = self._preprocess(full, src, start_t, self.time_window)  # (t_in, Cmax, H, W)
            y = self._preprocess(full, src, start_t + self.time_window, self.t_ar)  # (t_ar, Cmax, H, W)

            # BubbleFormer train mask is ones with shape (H,W,1,C), here we return (1,C,H,W)
            msk = torch.ones((1, self.cmax, x.shape[-2], x.shape[-1]), dtype=torch.float32)

        else:
            # eval: return x as first time_window, y as next t_test chunk
            # and a target mask that marks positions corresponding to original resolution
            # BubbleFormer uses get_target_mask to mark grid points; here we approximate with all-ones
            # after resize/downsample. If you want exact, you can compute using original H,W.
            traj_idx = local % self.num_trajs[src]
            full = self._load_full_trajectory(src, traj_idx=int(traj_idx))

            T = full.shape[0]
            t_test = self.t_test if self.t_test > 0 else max(T - self.time_window, 1)
            t_test = min(t_test, max(T - self.time_window, 1))

            x = self._preprocess(full, src, 0, self.time_window)
            y = self._preprocess(full, src, self.time_window, t_test)

            msk = torch.ones((1, self.cmax, x.shape[-2], x.shape[-1]), dtype=torch.float32)

        # dataset idx_cls is derived from JSON (setup, liquid) => 6-class id
        idx_cls = torch.LongTensor([self.source_class_idx[src]])

        if self.return_fluid_params:
            # return the full JSON dict as tensor is ambiguous; you can customize this list.
            # Here: return a small, stable set if present; else return an empty tensor.
            meta = self.json_meta[src]
            # Feel free to replace these with your preferred conditioning vector
            keys = [
                "inv_reynolds", "cpgas", "mugas", "rhogas", "thcogas",
                "stefan", "prandtl",
            ]
            vec = []
            for k in keys:
                if k in meta:
                    vec.append(float(meta[k]))
            fluid_params_tensor = torch.tensor(vec, dtype=torch.float32) if len(vec) else torch.zeros((0,), dtype=torch.float32)
            return x, y, msk, idx_cls, fluid_params_tensor

        return x, y, msk, idx_cls


