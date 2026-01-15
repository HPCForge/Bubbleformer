"""
This module contains dataset class for Time Series Forecasting
Classes:
    BubbleForecast: Dataset class for BubbleML dataset
Author: Sheikh Md Shakeel Hassan
"""
from typing import List, Optional, Tuple, Dict, Union
import json

import numpy as np
import h5py as h5
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class BubbleForecast(Dataset):
    """
    Dataset class for time series forecasting on the BubbleML dataset
    """
    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        norm: str = "none",
        downsample_factor: int = 1,
        time_window: int = 16,
        start_time: int = 50,
        return_fluid_params: bool = False,
    ):
        super().__init__()
        self.filenames = filenames
        if input_fields is not None:
            self.input_fields = input_fields
        else:
            self.input_fields = ["dfun", "temperature", "velx", "vely"]
        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["dfun", "temperature", "velx", "vely"]
        self.norm = norm
        self.downsample_factor = downsample_factor
        self.time_window = time_window
        self.start_time = start_time
        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []

        for h5_file in self.data:
            print(h5_file["dfun"].shape)
            self.num_trajs.append(1)
            self.traj_lens.append(h5_file[self.input_fields[0]].shape[0])

        self.input_num_fields = len(self.input_fields)
        self.output_num_fields = len(self.output_fields)
        self.fields = list(set(self.input_fields + self.output_fields))
        self.diff_terms = {k:[] for k in self.fields}
        self.div_terms = {k:[] for k in self.fields}

        self.return_fluid_params = return_fluid_params
        if self.return_fluid_params:
            fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
            self.fluid_params = []
            for fluid_params_file in fluid_params_files:
                with open(fluid_params_file, "r", encoding="utf-8") as f:
                    fluid_params = json.load(f)
                self.fluid_params.append(fluid_params)

    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            total_len += num_traj * (traj_len - self.start_time - 2 * self.time_window + 1)
        return total_len

    def normalize(
            self,
            diff_terms: Optional[Dict] = None,
            div_terms: Optional[Dict] = None,
        ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Calculate channel-wise normalization constants and store in a Dictionary
        Open each File object in self.data['files'] and calculate the channelwise
        mean and std of the data
        """
        if diff_terms is None and div_terms is None:
            diff_terms = {k:[] for k in self.fields}
            div_terms = {k:[] for k in self.fields}
            for field in self.fields:
                for _, h5_file in enumerate(self.data):
                    if self.norm == "std":
                        field_data = h5_file[field][...]
                        diff_terms[field].append(field_data.mean())
                        div_terms[field].append(field_data.std())
                    elif self.norm == "minmax":
                        field_data = h5_file[field][...]
                        diff_terms[field].append(field_data.min())
                        div_terms[field].append(field_data.max() - field_data.min())
                    elif self.norm == "tanh":
                        field_data = h5_file[field][...]
                        diff_terms[field].append(
                            (field_data.max() + field_data.min()) / 2.0
                        )
                        div_terms[field].append(
                            (field_data.max() - field_data.min()) / 2.0
                        )
                    elif self.norm == "none":
                        diff_terms[field].append(0.0)
                        div_terms[field].append(1.0)
                    else:
                        raise ValueError(f"Unknown normalization type: {self.norm}")

                diff_terms[field] = np.mean(diff_terms[field]).item()
                div_terms[field] = np.mean(div_terms[field]).item() + 1e-8

        self.diff_terms = diff_terms
        self.div_terms = div_terms

        return self.diff_terms, self.div_terms


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples_per_traj = [
            x * (y - self.start_time - 2 * self.time_window + 1)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        start = idx + self.start_time - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)

        inp_slice = slice(start, start + self.time_window)
        out_slice = slice(start + self.time_window, start + 2 * self.time_window)

        inp_data = []
        out_data = []

        for field in self.input_fields:
            data_item = torch.tensor(self.data[file_idx][field][inp_slice])
            if self.downsample_factor > 1:
                _, h, w = data_item.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                data_item = F.interpolate(
                    data_item.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="nearest"
                ).squeeze(1)
            inp_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )
        for field in self.output_fields:
            data_item = torch.tensor(self.data[file_idx][field][out_slice])
            if self.downsample_factor > 1:
                _, h, w = data_item.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                data_item = F.interpolate(
                    data_item.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="nearest"
                ).squeeze(1)
            out_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )

        inp_data = torch.stack(inp_data)                                   # (in_C, T, H, W)
        out_data = torch.stack(out_data)                                   # (out_C, T, H, W)

        if self.return_fluid_params:
            fluid_params = self.fluid_params[file_idx]
            fluid_params_tensor = torch.tensor(
                [
                    fluid_params["inv_reynolds"],
                    fluid_params["cpgas"],
                    fluid_params["mugas"],
                    fluid_params["rhogas"],
                    fluid_params["thcogas"],
                    fluid_params["stefan"],
                    fluid_params["prandtl"],
                    fluid_params["heater"]["nucWaitTime"],
                    fluid_params["heater"]["wallTemp"],
                ],
                dtype=torch.float32,
            )
            return inp_data.float().permute(1, 0, 2, 3), \
                    out_data.float().permute(1, 0, 2, 3), \
                    fluid_params_tensor

        return inp_data.float().permute(1, 0, 2, 3), out_data.float().permute(1, 0, 2, 3)



class VariableInputBubbleForecast(Dataset):
    """
    Random-chunk dataset for BubbleML:

      - On EVERY __getitem__ call, we sample:
          * a random trajectory
          * a random prediction start time t_pred_start
          * a random input length L in [pred_window, max_input_window_for_this_sample]

      - Input times:  [t_pred_start - L, ..., t_pred_start - 1]
      - Target times: [t_pred_start, ..., t_pred_start + pred_window - 1]

      - Input and target are disjoint and contiguous.
      - There is NO fixed mapping idx -> (file, t_pred_start), so each epoch
        sees genuinely new chunks.

    This is much less prone to overfitting than the deterministic-index version.
    """

    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        norm: str = "none",
        downsample_factor: int = 1,
        max_input_window: int = 100,
        pred_window: int = 5,
        start_time: int = 0,
        return_fluid_params: bool = False,
        samples_per_epoch: int = 10000,  # how many random chunks per epoch
    ):
        super().__init__()

        self.filenames = filenames
        self.input_fields = input_fields or ["dfun", "temperature", "velx", "vely"]
        self.output_fields = output_fields or ["dfun", "temperature", "velx", "vely"]
        self.norm = norm
        self.downsample_factor = downsample_factor
        self.max_input_window = max_input_window
        self.pred_window = pred_window
        self.start_time = start_time
        self.return_fluid_params = return_fluid_params
        self.samples_per_epoch = samples_per_epoch

        # ---- Load HDF5 files ----
        self.data = [h5.File(fname, "r") for fname in filenames]
        self.num_trajs = []
        self.traj_lens = []

        for h5_file in self.data:
            # one trajectory per file
            self.num_trajs.append(1)
            self.traj_lens.append(h5_file[self.input_fields[0]].shape[0])

        self.fields = list(set(self.input_fields + self.output_fields))
        self.diff_terms = {k: [] for k in self.fields}
        self.div_terms = {k: [] for k in self.fields}

        # ---- Fluid params (optional) ----
        if self.return_fluid_params:
            self.fluid_params = []
            for fname in filenames:
                params_file = fname.replace(".hdf5", ".json")
                with open(params_file, "r", encoding="utf-8") as f:
                    self.fluid_params.append(json.load(f))

        # ---- Precompute per-file valid prediction range ----
        # For each file i, valid t_pred_start ∈ [min_pred_start_i, max_pred_start_i]
        # where:
        #   - we need at least pred_window timesteps BEFORE t_pred_start
        #     so that L >= pred_window is always possible
        #   - we need pred_window timesteps AFTER t_pred_start for the target
        self.valid_files = []
        self.pred_start_ranges = []  # list of (min_pred_start, max_pred_start)

        for i, traj_len in enumerate(self.traj_lens):
            min_pred_start = self.start_time + self.pred_window
            max_pred_start = traj_len - self.pred_window

            if max_pred_start >= min_pred_start:
                self.valid_files.append(i)
                self.pred_start_ranges.append((min_pred_start, max_pred_start))

        if len(self.valid_files) == 0:
            raise ValueError("No trajectories have enough length for the given pred_window/start_time.")

    def __len__(self) -> int:
        """
        We decouple 'index' from the underlying time indices.

        Each epoch will draw 'samples_per_epoch' random (file, t_pred_start, L)
        tuples. Increase this if you want more coverage per epoch.
        """
        return self.samples_per_epoch

    def normalize(
        self,
        diff_terms: Optional[Dict] = None,
        div_terms: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Same normalization logic as your base class, but kept here for clarity.
        Should be called on the TRAIN dataset, then re-used on VAL.
        """
        if diff_terms is None and div_terms is None:
            diff_terms = {k: [] for k in self.fields}
            div_terms = {k: [] for k in self.fields}

            for field in self.fields:
                for h5_file in self.data:
                    field_data = h5_file[field][...]

                    if self.norm == "std":
                        diff_terms[field].append(field_data.mean())
                        div_terms[field].append(field_data.std())
                    elif self.norm == "minmax":
                        diff_terms[field].append(field_data.min())
                        div_terms[field].append(field_data.max() - field_data.min())
                    elif self.norm == "tanh":
                        diff_terms[field].append((field_data.max() + field_data.min()) / 2.0)
                        div_terms[field].append((field_data.max() - field_data.min()) / 2.0)
                    elif self.norm == "none":
                        diff_terms[field].append(0.0)
                        div_terms[field].append(1.0)
                    else:
                        raise ValueError(f"Unknown normalization type: {self.norm}")

                diff_terms[field] = np.mean(diff_terms[field]).item()
                div_terms[field] = np.mean(div_terms[field]).item() + 1e-8

        self.diff_terms = diff_terms
        self.div_terms = div_terms
        return diff_terms, div_terms

    def __getitem__(self, idx: int):
        """
        Ignore 'idx' for indexing into the trajectories; use it only to
        satisfy the Dataset interface. We randomly sample:

          - file_idx from self.valid_files
          - t_pred_start from that file's valid range
          - L in [pred_window, max_L_for_sample]

        Then return (input, target[, fluid_params]) with shapes:

          input:  (T_in=L,   C_in,  H, W)
          target: (T_out=PW, C_out, H, W)
        """
        # 1) random file
        file_pos = np.random.randint(0, len(self.valid_files))
        file_idx = self.valid_files[file_pos]
        min_ps, max_ps = self.pred_start_ranges[file_pos]
        traj_len = self.traj_lens[file_idx]

        # 2) random prediction start time
        t_pred_start = np.random.randint(min_ps, max_ps + 1)

        # 3) max possible input length for this sample
        max_L_for_sample = min(
            self.max_input_window,
            t_pred_start - self.start_time  # total history available back to start_time
        )

        # ensure L >= pred_window (your L_burn constraint downstream)
        min_L_for_sample = self.pred_window
        if max_L_for_sample < min_L_for_sample:
            # Should almost never happen given how we defined min_ps/max_ps,
            # but we keep this for safety.
            max_L_for_sample = min_L_for_sample

        L = np.random.randint(min_L_for_sample, max_L_for_sample + 1)

        inp_start = t_pred_start - L
        inp_end = t_pred_start          # exclusive
        out_start = t_pred_start
        out_end = t_pred_start + self.pred_window

        inp_slice = slice(inp_start, inp_end)
        out_slice = slice(out_start, out_end)

        inp_data = []
        out_data = []

        # ---- Build input ----
        for field in self.input_fields:
            data_item = torch.tensor(self.data[file_idx][field][inp_slice])
            if self.downsample_factor > 1:
                _, h, w = data_item.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                data_item = F.interpolate(
                    data_item.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="nearest",
                ).squeeze(1)
            inp_data.append((data_item - self.diff_terms[field]) / self.div_terms[field])

        # ---- Build target ----
        for field in self.output_fields:
            data_item = torch.tensor(self.data[file_idx][field][out_slice])
            if self.downsample_factor > 1:
                _, h, w = data_item.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                data_item = F.interpolate(
                    data_item.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="nearest",
                ).squeeze(1)
            out_data.append((data_item - self.diff_terms[field]) / self.div_terms[field])

        inp_data = torch.stack(inp_data)  # (C_in, L, H, W)
        out_data = torch.stack(out_data)  # (C_out, pred_window, H, W)

        if self.return_fluid_params:
            fluid = self.fluid_params[file_idx]
            fluid_tensor = torch.tensor(
                [
                    fluid["inv_reynolds"],
                    fluid["cpgas"],
                    fluid["mugas"],
                    fluid["rhogas"],
                    fluid["thcogas"],
                    fluid["stefan"],
                    fluid["prandtl"],
                    fluid["heater"]["nucWaitTime"],
                    fluid["heater"]["wallTemp"],
                ],
                dtype=torch.float32,
            )
            return (
                inp_data.float().permute(1, 0, 2, 3),   # (T_in=L, C, H, W)
                out_data.float().permute(1, 0, 2, 3),   # (T_out=PW, C, H, W)
                fluid_tensor,
            )

        return (
            inp_data.float().permute(1, 0, 2, 3),
            out_data.float().permute(1, 0, 2, 3),
        )

def collate_random_variable(
    batch: List[Union[Tuple[torch.Tensor, torch.Tensor],
                      Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
):
    """
    batch: list of:
        (inp, out) or (inp, out, fluid)
      where:
        inp:  (T_in_i, C, H, W)   # variable T_in_i
        out:  (T_out,  C, H, W)   # fixed pred_window
        fluid (optional): (F,)

    Returns:
        x:       (B, T_in_batch, C, H, W)
        y:       (B, T_out,      C, H, W)
        fluids:  (B, F)   if present
    """
    has_fluid = (len(batch[0]) == 3)

    if has_fluid:
        inps, outs, fluids = zip(*batch)   # tuples of tensors
    else:
        inps, outs = zip(*batch)

    # inps are (T_i, C, H, W); outs are (T_out, C, H, W)
    T_list = [x.shape[0] for x in inps]
    T_batch = min(T_list)   # enforce SAME T per batch, no padding

    # Crop all inputs to last T_batch timesteps
    inps_trimmed = [x[-T_batch:] for x in inps]   # still (T_batch, C, H, W)

    x = torch.stack(inps_trimmed, dim=0)  # (B, T_batch, C, H, W)
    y = torch.stack(outs, dim=0)          # (B, T_out,   C, H, W)

    if has_fluid:
        fluids = torch.stack(fluids, dim=0)   # (B, F)
        return x, y, fluids

    return x, y
