#!/usr/bin/env python3
"""
Script to downsample BubbleML HDF5 files using nearest neighbor interpolation.
Reads paths from the poolboiling.yaml config file and saves to a new directory.
"""

import os
import yaml
import h5py as h5
import torch
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# Configuration
#CONFIG_PATH = Path(__file__).parent / "config" / "data_cfg" / "poolboiling.yaml"
CONFIG_PATH = "/data/homezvol3/srachaba/Projects/Bubbleformer/bubbleformer/config/data_cfg/poolboiling.yaml"
SOURCE_BASE = "/share/crsp/lab/amowli/share/BubbleML_2"
DEST_BASE = "/share/crsp/lab/amowli/share/BubbleML_2_downsampled64"
DOWNSAMPLE_FACTOR = 8


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_all_paths(config: dict) -> list:
    """Extract all unique HDF5 file paths from the config."""
    paths = []
    for key in ["train_paths", "val_paths", "test_paths"]:
        if key in config and config[key]:
            paths.extend(config[key])
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def downsample_file(src_path: str) -> str:
    """Downsample a single HDF5 file and save to the destination directory."""
    #print("mine processing file: ", src_path)
    try:
        rel_path = src_path.replace(SOURCE_BASE, "").lstrip("/")
        dest_path = os.path.join(DEST_BASE, rel_path)

        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(dest_path):
            return f"SKIP: {src_path} -> already exists"

        with h5.File(src_path, "r") as src_file:
            with h5.File(dest_path, "w") as dest_file:
                for key in src_file.keys():
                    data = src_file[key][...]
                    print(f"shape: {data.shape}. path: {src_path}") 
                    if len(data.shape) == 3:
                        t, height, width = data.shape
                        new_h, new_w = height // DOWNSAMPLE_FACTOR, width // DOWNSAMPLE_FACTOR

                        tensor_data = torch.tensor(data, dtype=torch.float32)
                        downsampled = F.interpolate(
                            tensor_data.unsqueeze(1),
                            size=(new_h, new_w),
                            mode="nearest"
                        ).squeeze(1)

                        dest_file.create_dataset(key, data=downsampled.numpy())
                    else:
                        dest_file.create_dataset(key, data=data)

                for attr_name, attr_val in src_file.attrs.items():
                    dest_file.attrs[attr_name] = attr_val

        # Copy the corresponding JSON file if it exists
        json_src = src_path.replace(".hdf5", ".json")
        json_dest = dest_path.replace(".hdf5", ".json")
        if os.path.exists(json_src) and not os.path.exists(json_dest):
            shutil.copy2(json_src, json_dest)

        return f"OK: {src_path} -> {dest_path}"

    except Exception as e:
        return f"ERROR: {src_path} -> {str(e)}"


def main():
    print(f"Loading config from: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    all_paths = get_all_paths(config)

    paths_to_process = [p for p in all_paths if p.startswith(SOURCE_BASE)]

    print(f"Found {len(paths_to_process)} files to process")
    print(f"Source: {SOURCE_BASE}")
    print(f"Destination: {DEST_BASE}")
    print(f"Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"Using all available CPU cores")

    os.makedirs(DEST_BASE, exist_ok=True)

    print("\nProcessing files...")
    print()
    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(downsample_file, path): path for path in paths_to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downsampling"):
            result = future.result()
            tqdm.write(result)  # Print without breaking progress bar
            results.append(result)

    print("\n=== Summary ===")
    ok_count = sum(1 for r in results if r.startswith("OK"))
    skip_count = sum(1 for r in results if r.startswith("SKIP"))
    error_count = sum(1 for r in results if r.startswith("ERROR"))

    print(f"Completed: {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")

    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if r.startswith("ERROR"):
                print(f"  {r}")


if __name__ == "__main__":
    main()
