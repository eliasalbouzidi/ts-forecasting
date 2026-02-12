"""
Download and preprocess NOAA OISSTv2 Sea Surface Temperature data.

Downloads yearly NetCDF from NOAA PSL, extracts 11 regional patches
(60x60 grid points each, eastern tropical Pacific), and saves as CSVs
compatible with the ProbTS LTSF pipeline.

Requirements (local machine only):
    pip install xarray netCDF4 requests tqdm

Step 1 — Run locally to download & preprocess:
    python probts/utils/download_sst.py --data_path ./datasets
    python probts/utils/download_sst.py --data_path ./datasets --year 2020
    python probts/utils/download_sst.py --data_path ./datasets --patch_id 0

Step 2 — Transfer to VM via scp:
    scp -r datasets/sst/2019 user@vm-host:~/ts-forecasting/datasets/sst/

Step 3 — Train on the VM:
    python run.py --config config/default/dyffusion.yaml \\
                  --config config/sst/dyffusion_sst.yaml
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# 11 patches of 60x60 (15°x15° at 0.25° res) in the eastern tropical Pacific
PATCH_DEFS = [
    (-15.0,  0.0, 200.0, 215.0),
    (-15.0,  0.0, 215.0, 230.0),
    (-15.0,  0.0, 230.0, 245.0),
    (-15.0,  0.0, 245.0, 260.0),
    (  0.0, 15.0, 200.0, 215.0),
    (  0.0, 15.0, 215.0, 230.0),
    (  0.0, 15.0, 230.0, 245.0),
    (  0.0, 15.0, 245.0, 260.0),
    ( 15.0, 30.0, 200.0, 215.0),
    ( 15.0, 30.0, 215.0, 230.0),
    ( 15.0, 30.0, 230.0, 245.0),
]


def download_nc(year: int, download_dir: str) -> str:
    """Download the yearly SST NetCDF from NOAA PSL."""
    import requests

    filename = f"sst.day.mean.{year}.nc"
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):
        print(f"  [skip] {filename} already exists")
        return filepath

    url = f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/{filename}"
    print(f"  Downloading {url} (~500 MB) ...")

    os.makedirs(download_dir, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(filepath, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))

    return filepath


def extract_and_save(nc_path: str, output_dir: str, patch_ids: list = None):
    """Read NetCDF, extract patches, save as CSVs."""
    import xarray as xr

    print(f"  Reading {nc_path} ...")
    ds = xr.open_dataset(nc_path)
    sst = ds["sst"]
    lat = ds["lat"].values
    lon = ds["lon"].values
    times = pd.DatetimeIndex(ds["time"].values)

    if patch_ids is None:
        patch_ids = list(range(len(PATCH_DEFS)))

    os.makedirs(output_dir, exist_ok=True)

    for pid in patch_ids:
        lat_s, lat_e, lon_s, lon_e = PATCH_DEFS[pid]
        lat_idx = np.where((lat >= lat_s) & (lat < lat_e))[0][:60]
        lon_idx = np.where((lon >= lon_s) & (lon < lon_e))[0][:60]

        patch = sst.values[:, lat_idx][:, :, lon_idx]  # (T, 60, 60)
        patch = np.nan_to_num(patch, nan=0.0)
        patch = np.clip(patch, -10, 50)
        flat = patch.reshape(len(times), -1)            # (T, 3600)

        cols = [f"px_{i:04d}" for i in range(flat.shape[1])]
        df = pd.DataFrame(flat, columns=cols)
        df.insert(0, "date", times)

        fpath = os.path.join(output_dir, f"sst_patch_{pid:02d}.csv")
        df.to_csv(fpath, index=False)
        print(f"  Saved {fpath}  ({len(df)} days, {flat.shape[1]} pixels)")

    ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NOAA OISSTv2 dataset")
    parser.add_argument("--data_path", type=str, default="./datasets",
                        help="Root datasets directory (default: ./datasets)")
    parser.add_argument("--year", type=int, default=2019,
                        help="Year to download (default: 2019)")
    parser.add_argument("--patch_id", type=int, default=None,
                        help="Single patch to extract (0-10). Default: all 11")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_path, "sst", "raw")
    out_dir = os.path.join(args.data_path, "sst", str(args.year))

    patches = [args.patch_id] if args.patch_id is not None else None

    print(f"=== Downloading NOAA OISSTv2 ({args.year}) ===")
    nc_path = download_nc(args.year, raw_dir)

    print(f"=== Extracting patches ===")
    extract_and_save(nc_path, out_dir, patches)

    print(f"\nDone! CSVs are in {out_dir}")
    print(f"To train: python run.py --config config/default/dyffusion.yaml --config config/sst/dyffusion_sst.yaml")
