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