"""
Download and preprocess NOAA OISSTv2 Sea Surface Temperature data.

Downloads yearly NetCDF files from NOAA PSL for every year in 1981-2025,
extracts a single regional patch (patch 00, 60x60 grid points in the
eastern tropical Pacific), and concatenates all years into one CSV
compatible with the ProbTS LTSF pipeline.

To save disk space the script works **progressively**: it downloads one
year's NetCDF (~500 MB), extracts the patch, appends rows to the output
CSV, then deletes the NetCDF before moving on to the next year.

A small JSON progress file tracks which years are done, so the script
is **fully resumable** — if interrupted (Ctrl-C, crash, etc.), just
re-run the same command and it picks up where it left off.

Requirements (local machine only):
    pip install xarray netCDF4 requests tqdm

Step 1 — Run locally to download & preprocess:
    python probts/utils/download_sst.py --data_path ./datasets
    python probts/utils/download_sst.py --data_path ./datasets --start_year 2000 --end_year 2020
    python probts/utils/download_sst.py --data_path ./datasets --patch_id 3

Step 2 — Transfer the combined CSV to the VM:
    scp datasets/sst/sst_patch_00.csv user@vm-host:~/ts-forecasting/datasets/sst/

Step 3 — Train on the VM:
    python run.py --config config/default/dyffusion.yaml \\
                  --config config/sst/dyffusion_sst.yaml
"""

import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Default year range (NOAA OISSTv2 starts 1981-09-01)
START_YEAR = 1981
END_YEAR = 2025

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


def extract_patch(nc_path: str, patch_id: int) -> pd.DataFrame:
    """Read NetCDF, extract a single patch, return as DataFrame (kept in memory)."""
    import xarray as xr

    print(f"  Reading {nc_path} ...")
    ds = xr.open_dataset(nc_path)
    sst = ds["sst"]
    lat = ds["lat"].values
    lon = ds["lon"].values
    times = pd.DatetimeIndex(ds["time"].values)

    lat_s, lat_e, lon_s, lon_e = PATCH_DEFS[patch_id]
    lat_idx = np.where((lat >= lat_s) & (lat < lat_e))[0][:60]
    lon_idx = np.where((lon >= lon_s) & (lon < lon_e))[0][:60]

    patch = sst.values[:, lat_idx][:, :, lon_idx]  # (T, 60, 60)
    patch = np.nan_to_num(patch, nan=0.0)
    patch = np.clip(patch, -10, 50)
    flat = patch.reshape(len(times), -1)            # (T, 3600)

    cols = [f"px_{i:04d}" for i in range(flat.shape[1])]
    df = pd.DataFrame(flat, columns=cols)
    df.insert(0, "date", times)

    ds.close()
    return df


def _progress_path(out_dir: str, patch_id: int) -> str:
    """Path to the JSON file that tracks which years have been saved."""
    return os.path.join(out_dir, f"_progress_patch_{patch_id:02d}.json")


def _load_progress(progress_file: str) -> set:
    """Load the set of already-processed years from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            data = json.load(f)
        return set(data.get("done_years", []))
    return set()


def _save_progress(progress_file: str, done_years: set):
    """Persist the set of completed years."""
    with open(progress_file, "w") as f:
        json.dump({"done_years": sorted(done_years)}, f)


def download_all_years(
    data_path: str,
    patch_id: int = 0,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    keep_nc: bool = False,
):
    """
    Progressive download with resume support.

    For each year: download NC -> extract patch -> **append to CSV** -> delete NC.
    A small JSON progress file tracks which years are already saved so the
    script can be interrupted and restarted without re-doing finished years.

    Returns the path to the final combined CSV.
    """
    raw_dir = os.path.join(data_path, "sst", "raw")
    out_dir = os.path.join(data_path, "sst")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"sst_patch_{patch_id:02d}.csv")
    progress_file = _progress_path(out_dir, patch_id)
    done_years = _load_progress(progress_file)

    if done_years:
        print(f"Resuming — {len(done_years)} year(s) already done: "
              f"{min(done_years)}..{max(done_years)}")

    total_years = end_year - start_year + 1

    for year in range(start_year, end_year + 1):
        idx = year - start_year + 1
        print(f"\n=== Year {year} ({idx}/{total_years}) ===")

        if year in done_years:
            print(f"  [skip] already processed")
            continue

        # 1) Download
        try:
            nc_path = download_nc(year, raw_dir)
        except Exception as e:
            print(f"  [WARN] Failed to download {year}: {e}  — skipping")
            continue

        # 2) Extract single patch
        try:
            df = extract_patch(nc_path, patch_id)
            print(f"  Extracted patch {patch_id:02d}: {len(df)} days")
        except Exception as e:
            print(f"  [WARN] Failed to extract {year}: {e}  — skipping")
            # still try to clean up
            if not keep_nc:
                try:
                    os.remove(nc_path)
                except OSError:
                    pass
            continue

        # 3) Append to CSV (write header only if file is new)
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", index=False, header=write_header)
        print(f"  Appended {len(df)} rows to {csv_path}")

        # 4) Record progress
        done_years.add(year)
        _save_progress(progress_file, done_years)

        # 5) Delete the NetCDF to free disk space
        if not keep_nc:
            try:
                os.remove(nc_path)
                print(f"  Deleted {nc_path}")
            except OSError:
                pass

        gc.collect()

    if not done_years:
        raise RuntimeError("No data was extracted — check network / years")

    # Final sort pass — ensure rows are in chronological order
    # (years are processed in order, but just to be safe on resume)
    print(f"\nFinal sort of {csv_path} ...")
    combined = pd.read_csv(csv_path, parse_dates=["date"])
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.to_csv(csv_path, index=False)

    # Clean up progress file — download is complete
    expected = set(range(start_year, end_year + 1))
    if done_years >= expected:
        try:
            os.remove(progress_file)
            print(f"  Removed progress file (all years done)")
        except OSError:
            pass

    print(f"\n=== Saved {csv_path}  ({len(combined)} days, {combined.shape[1]-1} pixels) ===")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NOAA OISSTv2 dataset (all years, single patch)")
    parser.add_argument("--data_path", type=str, default="./datasets",
                        help="Root datasets directory (default: ./datasets)")
    parser.add_argument("--patch_id", type=int, default=0,
                        help="Patch to extract (0-10). Default: 0")
    parser.add_argument("--start_year", type=int, default=START_YEAR,
                        help=f"First year to download (default: {START_YEAR})")
    parser.add_argument("--end_year", type=int, default=END_YEAR,
                        help=f"Last year to download (default: {END_YEAR})")
    parser.add_argument("--keep_nc", action="store_true",
                        help="Keep raw NetCDF files instead of deleting after extraction")
    args = parser.parse_args()

    csv_path = download_all_years(
        data_path=args.data_path,
        patch_id=args.patch_id,
        start_year=args.start_year,
        end_year=args.end_year,
        keep_nc=args.keep_nc,
    )

    print(f"\nDone! Combined CSV: {csv_path}")
    print(f"To train: python run.py --config config/default/dyffusion.yaml --config config/sst/dyffusion_sst.yaml")
