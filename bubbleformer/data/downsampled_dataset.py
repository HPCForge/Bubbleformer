from typing import List, Optional, Tuple, Dict
import json

import numpy as np
import h5py as h5
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class DownsampledBubbleForecast(Dataset):
    """
    Dataset class for time series forecasting on the BubbleML dataset.
    This downsamples the full dataset and stores it in cpu memory. This can be
    used to accelerate data loading on a networked file system.
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
        self.norm = norm
        self.downsample_factor = downsample_factor
        self.time_window = time_window
        self.start_time = start_time
        
        if input_fields is not None:
            self.input_fields = input_fields
        else:
            self.input_fields = ["dfun", "temperature", "velx", "vely"]
        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["dfun", "temperature", "velx", "vely"]
        self.fields = list(set(self.input_fields + self.output_fields))
        
        self.data = self._load_data()
        self.num_trajs = [1 for _ in range(len(self.filenames))]
        self.traj_lens = [d[self.input_fields[0]].shape[0] for d in self.data]

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

    def _load_data(self):
        hdf5_files = [h5.File(filename, "r") for filename in self.filenames]
        data = []
        i = 0
        for hdf5_file in hdf5_files:
            print("file", i)
            i += 1
            d = {}
            for field in self.fields:
                field_data = torch.tensor(hdf5_file[field][...])
                if self.downsample_factor > 1:
                    _, h, w = field_data.shape
                    new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                    field_data = F.interpolate(
                        field_data.unsqueeze(1),
                        size=(new_h, new_w),
                        mode="nearest"
                    ).squeeze(1)
                d[field] = field_data
            data.append(d)
        return data

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
            inp_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )
        for field in self.output_fields:
            data_item = torch.tensor(self.data[file_idx][field][out_slice])
            out_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )

        inp_data = torch.stack(inp_data) # (in_C, T, H, W)
        out_data = torch.stack(out_data) # (out_C, T, H, W)

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


class VariableInputDownsampledBubbleForecast(Dataset):
    """
    Dataset class that combines:
    1. Downsampling and preloading data to CPU memory (for faster access on networked file systems)
    2. Variable input window functionality (random input lengths per sample)

    On each __getitem__ call, randomly samples:
      - A random trajectory
      - A random prediction start time
      - A random input length L in [pred_window, max_input_window]

    Use with collate_random_variable from dataset.py to handle variable-length batching.
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
        samples_per_epoch: int = 10000,
    ):
        super().__init__()
        self.filenames = filenames
        self.norm = norm
        self.downsample_factor = downsample_factor
        self.max_input_window = max_input_window
        self.pred_window = pred_window
        self.start_time = start_time
        self.return_fluid_params = return_fluid_params
        self.samples_per_epoch = samples_per_epoch

        if input_fields is not None:
            self.input_fields = input_fields
        else:
            self.input_fields = ["dfun", "temperature", "velx", "vely"]
        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["dfun", "temperature", "velx", "vely"]
        self.fields = list(set(self.input_fields + self.output_fields))

        # Load and downsample data into memory
        self.data = self._load_data()
        self.num_trajs = [1 for _ in range(len(self.filenames))]
        self.traj_lens = [d[self.input_fields[0]].shape[0] for d in self.data]

        self.diff_terms = {k: [] for k in self.fields}
        self.div_terms = {k: [] for k in self.fields}

        if self.return_fluid_params:
            fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
            self.fluid_params = []
            for fluid_params_file in fluid_params_files:
                with open(fluid_params_file, "r", encoding="utf-8") as f:
                    fluid_params = json.load(f)
                self.fluid_params.append(fluid_params)

        # Precompute per-file valid prediction range
        # For each file, valid t_pred_start in [min_pred_start, max_pred_start]
        self.valid_files = []
        self.pred_start_ranges = []

        for i, traj_len in enumerate(self.traj_lens):
            min_pred_start = self.start_time + self.pred_window
            max_pred_start = traj_len - self.pred_window

            if max_pred_start >= min_pred_start:
                self.valid_files.append(i)
                self.pred_start_ranges.append((min_pred_start, max_pred_start))

        if len(self.valid_files) == 0:
            raise ValueError("No trajectories have enough length for the given pred_window/start_time.")

    def _load_data(self):
        hdf5_files = [h5.File(filename, "r") for filename in self.filenames]
        data = []
        for hdf5_file in hdf5_files:
            d = {}
            for field in self.fields:
                field_data = torch.tensor(hdf5_file[field][...])
                if self.downsample_factor > 1:
                    _, h, w = field_data.shape
                    new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                    field_data = F.interpolate(
                        field_data.unsqueeze(1),
                        size=(new_h, new_w),
                        mode="nearest"
                    ).squeeze(1)
                d[field] = field_data
            data.append(d)
        return data

    def __len__(self) -> int:
        return self.samples_per_epoch

    def normalize(
            self,
            diff_terms: Optional[Dict] = None,
            div_terms: Optional[Dict] = None,
        ) -> Tuple[Dict, Dict]:
        """
        Calculate channel-wise normalization constants.
        Should be called on the TRAIN dataset, then re-used on VAL.
        """
        if diff_terms is None and div_terms is None:
            diff_terms = {k: [] for k in self.fields}
            div_terms = {k: [] for k in self.fields}
            for field in self.fields:
                for data_dict in self.data:
                    field_data = data_dict[field]
                    if self.norm == "std":
                        diff_terms[field].append(field_data.mean().item())
                        div_terms[field].append(field_data.std().item())
                    elif self.norm == "minmax":
                        diff_terms[field].append(field_data.min().item())
                        div_terms[field].append((field_data.max() - field_data.min()).item())
                    elif self.norm == "tanh":
                        diff_terms[field].append(
                            ((field_data.max() + field_data.min()) / 2.0).item()
                        )
                        div_terms[field].append(
                            ((field_data.max() - field_data.min()) / 2.0).item()
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

    def __getitem__(self, idx: int):
        """
        Randomly sample:
          - file_idx from valid_files
          - t_pred_start from that file's valid range
          - L in [pred_window, max_L_for_sample]

        Returns (input, target[, fluid_params]) with shapes:
          input:  (T_in=L,   C_in,  H, W)
          target: (T_out=pred_window, C_out, H, W)
        """
        # 1) random file
        file_pos = np.random.randint(0, len(self.valid_files))
        file_idx = self.valid_files[file_pos]
        min_ps, max_ps = self.pred_start_ranges[file_pos]

        # 2) random prediction start time
        t_pred_start = np.random.randint(min_ps, max_ps + 1)

        # 3) max possible input length for this sample
        max_L_for_sample = min(
            self.max_input_window,
            t_pred_start - self.start_time
        )

        min_L_for_sample = self.pred_window
        if max_L_for_sample < min_L_for_sample:
            max_L_for_sample = min_L_for_sample

        L = np.random.randint(min_L_for_sample, max_L_for_sample + 1)

        inp_start = t_pred_start - L
        inp_end = t_pred_start
        out_start = t_pred_start
        out_end = t_pred_start + self.pred_window

        inp_slice = slice(inp_start, inp_end)
        out_slice = slice(out_start, out_end)

        inp_data = []
        out_data = []

        for field in self.input_fields:
            data_item = self.data[file_idx][field][inp_slice].clone()
            inp_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )
        for field in self.output_fields:
            data_item = self.data[file_idx][field][out_slice].clone()
            out_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )

        inp_data = torch.stack(inp_data)  # (C_in, L, H, W)
        out_data = torch.stack(out_data)  # (C_out, pred_window, H, W)

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
            return (
                inp_data.float().permute(1, 0, 2, 3),
                out_data.float().permute(1, 0, 2, 3),
                fluid_params_tensor,
            )

        return (
            inp_data.float().permute(1, 0, 2, 3),
            out_data.float().permute(1, 0, 2, 3),
        )
