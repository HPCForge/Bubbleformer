"""
This module contains dataset class for Time Series Forecasting
Classes:
    BubbleForecast: Dataset class for BubbleML dataset
Author: Sheikh Md Shakeel Hassan
"""
from typing import List, Optional, Tuple, Dict
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


class BubbleForecastSeparateWindows(Dataset):
    """
    Dataset class for time series forecasting on BubbleML where
    the input and output time windows are independent parameters.
    """

    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        norm: str = "none",
        downsample_factor: int = 1,
        input_window: int = 16,     # <------ NEW
        output_window: int = 16,    # <------ NEW
        start_time: int = 50,
        return_fluid_params: bool = False,
    ):
        super().__init__()

        self.filenames = filenames

        self.input_fields = input_fields or ["dfun", "temperature", "velx", "vely"]
        self.output_fields = output_fields or ["dfun", "temperature", "velx", "vely"]

        self.norm = norm
        self.downsample_factor = downsample_factor

        # ---- NEW WINDOW PARAMETERS ----
        self.input_window = input_window
        self.output_window = output_window
        # --------------------------------

        self.start_time = start_time

        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []

        for h5_file in self.data:
            self.num_trajs.append(1)
            self.traj_lens.append(h5_file[self.input_fields[0]].shape[0])

        self.fields = list(set(self.input_fields + self.output_fields))

        self.diff_terms = {k: [] for k in self.fields}
        self.div_terms = {k: [] for k in self.fields}

        # fluid params
        self.return_fluid_params = return_fluid_params
        if self.return_fluid_params:
            self.fluid_params = []
            for fname in filenames:
                params_file = fname.replace(".hdf5", ".json")
                with open(params_file, "r") as f:
                    self.fluid_params.append(json.load(f))

    def __len__(self):
        """
        For each trajectory of length L, valid starting points are:

            L - start_time - (input_window + output_window) + 1
        """
        total = 0
        for num_traj, traj_len in zip(self.num_trajs, self.traj_lens):
            valid = traj_len - self.start_time - (self.input_window + self.output_window) + 1
            valid = max(valid, 0)
            total += num_traj * valid
        return total

    def normalize(self, diff_terms=None, div_terms=None):
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
                        raise ValueError(f"Unknown normalization type {self.norm}")

                diff_terms[field] = np.mean(diff_terms[field]).item()
                div_terms[field] = np.mean(div_terms[field]).item() + 1e-8

        self.diff_terms = diff_terms
        self.div_terms = div_terms
        return diff_terms, div_terms

    def __getitem__(self, idx: int):
        # same logic as original
        samples_per_traj = [
            n * (L - self.start_time - (self.input_window + self.output_window) + 1)
            for n, L in zip(self.num_trajs, self.traj_lens)
        ]
        cum = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cum, idx, side="right")

        offset = cum[file_idx - 1] if file_idx > 0 else 0
        start = idx + self.start_time - offset

        inp_slice = slice(start, start + self.input_window)
        out_slice = slice(start + self.input_window,
                          start + self.input_window + self.output_window)

        inp_data = []
        out_data = []

        # -------- load input tensors --------
        for field in self.input_fields:
            arr = torch.tensor(self.data[file_idx][field][inp_slice])

            if self.downsample_factor > 1:
                _, h, w = arr.shape
                arr = F.interpolate(
                    arr.unsqueeze(1),
                    size=(h // self.downsample_factor, w // self.downsample_factor),
                    mode="nearest",
                ).squeeze(1)

            inp_data.append((arr - self.diff_terms[field]) / self.div_terms[field])

        # -------- load output tensors --------
        for field in self.output_fields:
            arr = torch.tensor(self.data[file_idx][field][out_slice])

            if self.downsample_factor > 1:
                _, h, w = arr.shape
                arr = F.interpolate(
                    arr.unsqueeze(1),
                    size=(h // self.downsample_factor, w // self.downsample_factor),
                    mode="nearest",
                ).squeeze(1)

            out_data.append((arr - self.diff_terms[field]) / self.div_terms[field])

        inp_data = torch.stack(inp_data)  # (C_in, T_in, H, W)
        out_data = torch.stack(out_data)  # (C_out, T_out, H, W)

        if self.return_fluid_params:
            params = self.fluid_params[file_idx]
            params_tensor = torch.tensor(
                [
                    params["inv_reynolds"],
                    params["cpgas"],
                    params["mugas"],
                    params["rhogas"],
                    params["thcogas"],
                    params["stefan"],
                    params["prandtl"],
                    params["heater"]["nucWaitTime"],
                    params["heater"]["wallTemp"],
                ],
                dtype=torch.float32,
            )
            return inp_data.permute(1, 0, 2, 3), out_data.permute(1, 0, 2, 3), params_tensor

        return inp_data.permute(1, 0, 2, 3), out_data.permute(1, 0, 2, 3)





class VariableInputBubbleForecast(BubbleForecast):
    """
    Dataset class for time series forecasting on the BubbleML dataset
    with:
      - variable-length input window (random per sample, up to max_input_window)
      - fixed-length prediction window immediately after the input

    For each sample:
        - choose a prediction start time t_pred_start in the trajectory
        - choose an input length L uniformly in [pred_window, max_input_window_for_this_sample]
        - input times:  [t_pred_start - L, ..., t_pred_start - 1]
        - target times: [t_pred_start, ..., t_pred_start + pred_window - 1]

    All slices come from the same trajectory.
    """

    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        norm: str = "none",
        downsample_factor: int = 1,
        max_input_window: int = 100,   # maximum input context length
        pred_window: int = 5,          # fixed prediction window length
        start_time: int = 0,           # earliest time index allowed for input/prediction
        return_fluid_params: bool = False,
    ):
        super().__init__(
            filenames=filenames,
            input_fields=input_fields,
            output_fields=output_fields,
            norm=norm,
            downsample_factor=downsample_factor,
            time_window=max_input_window,
            start_time=start_time,
            return_fluid_params=return_fluid_params,
        )

        self.max_input_window = max_input_window
        self.pred_window = pred_window

    def __len__(self):
        """
        Valid prediction start times t_pred_start lie in:
            [start_time + pred_window, traj_len - pred_window]

        Why?
          - We need at least pred_window timesteps BEFORE t_pred_start
            so that we can always sample L >= pred_window.
          - We need pred_window timesteps AFTER t_pred_start for the target.
        """
        total_len = 0
        for num_traj, traj_len in zip(self.num_trajs, self.traj_lens):
            min_pred_start = self.start_time + self.pred_window
            max_pred_start = traj_len - self.pred_window
            if max_pred_start >= min_pred_start:
                valid_pred_starts = max_pred_start - min_pred_start + 1
            else:
                valid_pred_starts = 0
            total_len += num_traj * valid_pred_starts
        return total_len

    def __getitem__(self, idx: int):
        # samples per trajectory using the new valid_pred_starts definition
        samples_per_traj = []
        for num_traj, traj_len in zip(self.num_trajs, self.traj_lens):
            min_pred_start = self.start_time + self.pred_window
            max_pred_start = traj_len - self.pred_window
            if max_pred_start >= min_pred_start:
                valid_pred_starts = max_pred_start - min_pred_start + 1
            else:
                valid_pred_starts = 0
            samples_per_traj.append(num_traj * valid_pred_starts)

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        if file_idx >= len(self.traj_lens):
            raise IndexError("Index out of range in VariableInputBubbleForecast")

        traj_len = self.traj_lens[file_idx]

        prev_cum = cumulative_samples[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - prev_cum  # 0-based index within this trajectory

        # Valid t_pred_start ∈ [start_time + pred_window, traj_len - pred_window]
        min_pred_start = self.start_time + self.pred_window
        max_pred_start = traj_len - self.pred_window
        if max_pred_start < min_pred_start:
            raise IndexError("No valid prediction starts for this trajectory")

        t_pred_start = min_pred_start + local_idx

        if t_pred_start > max_pred_start:
            raise IndexError(
                f"Invalid t_pred_start={t_pred_start} for traj_len={traj_len}, "
                f"pred_window={self.pred_window}"
            )

        # Max possible input length for this sample
        max_L_for_sample = min(
            self.max_input_window,
            t_pred_start - self.start_time  # total history available back to start_time
        )

        # We now enforce L >= pred_window
        min_L_for_sample = self.pred_window
        if max_L_for_sample < min_L_for_sample:
            raise IndexError(
                f"Not enough history before t_pred_start={t_pred_start} "
                f"to ensure L >= pred_window={self.pred_window}"
            )

        # Randomly choose input length L in [pred_window, max_L_for_sample]
        L = np.random.randint(min_L_for_sample, max_L_for_sample + 1)

        inp_start = t_pred_start - L
        inp_end = t_pred_start          # exclusive
        out_start = t_pred_start
        out_end = t_pred_start + self.pred_window

        inp_slice = slice(inp_start, inp_end)
        out_slice = slice(out_start, out_end)

        inp_data = []
        out_data = []

        # ---------- Build input tensor ----------
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
            inp_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )

        # ---------- Build target tensor ----------
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
            out_data.append(
                (data_item - self.diff_terms[field]) / self.div_terms[field]
            )

        inp_data = torch.stack(inp_data)   # (C_in, T_in=L, H, W)
        out_data = torch.stack(out_data)   # (C_out, T_out=pred_window, H, W)

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
            inp_data.float().permute(1, 0, 2, 3),  # (T_in=L, C, H, W)
            out_data.float().permute(1, 0, 2, 3),  # (T_out=pred_window, C, H, W)
        )

