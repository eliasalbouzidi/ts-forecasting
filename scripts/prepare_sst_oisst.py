import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import requests

DEFAULT_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.2025.nc"


def download_file(url: str, dst: Path, chunk_size: int = 1 << 20) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dst


def load_patches(patches_path: Optional[Path]) -> Optional[List[Dict]]:
    if patches_path is None:
        return None
    with open(patches_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Patch file must be a JSON list of patch definitions.")
    return data


def ensure_lon_range(lon: xr.DataArray) -> xr.DataArray:
    # Convert lon to [0, 360) if needed
    if lon.min() < 0:
        lon = (lon + 360) % 360
    return lon


def subset_by_bounds(ds: xr.Dataset, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.Dataset:
    ds = ds.copy()
    ds["lon"] = ensure_lon_range(ds["lon"])
    lon_min = lon_min % 360
    lon_max = lon_max % 360

    if lon_min <= lon_max:
        ds = ds.sel(lon=slice(lon_min, lon_max))
    else:
        # wrap-around selection
        left = ds.sel(lon=slice(lon_min, 360))
        right = ds.sel(lon=slice(0, lon_max))
        ds = xr.concat([left, right], dim="lon")

    if lat_min <= lat_max:
        ds = ds.sel(lat=slice(lat_min, lat_max))
    else:
        ds = ds.sel(lat=slice(lat_max, lat_min))
    return ds


def flatten_patch_to_csv(ds: xr.Dataset, output_path: Path, var_name: str = "sst") -> None:
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset. Available: {list(ds.data_vars)}")

    arr = ds[var_name]
    # Ensure order: time, lat, lon
    arr = arr.transpose("time", "lat", "lon")
    time = pd.to_datetime(arr["time"].values)
    data = arr.values

    num_time = data.shape[0]
    flat = data.reshape(num_time, -1)

    col_count = flat.shape[1]
    cols = [f"f{idx:05d}" for idx in range(col_count)]
    df = pd.DataFrame(flat, columns=cols)
    df.insert(0, "date", time)

    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare OISSTv2 SST data for ProbTS.")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL to OISSTv2 NetCDF file.")
    parser.add_argument("--output-dir", default="./datasets/sst", help="Output directory for CSVs.")
    parser.add_argument("--cache-dir", default="./datasets/raw", help="Cache directory for NetCDF file.")
    parser.add_argument("--var", default="sst", help="Variable name to extract.")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD).")

    parser.add_argument("--patches", default=None, help="Path to JSON list of patches.")
    parser.add_argument("--lat-min", type=float, default=None, help="Single patch lat min.")
    parser.add_argument("--lat-max", type=float, default=None, help="Single patch lat max.")
    parser.add_argument("--lon-min", type=float, default=None, help="Single patch lon min.")
    parser.add_argument("--lon-max", type=float, default=None, help="Single patch lon max.")

    parser.add_argument("--fill-value", type=float, default=0.0, help="Fill value for missing data.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(args.url)
    nc_path = download_file(args.url, cache_dir / filename)

    ds = xr.open_dataset(nc_path)

    if args.start_date or args.end_date:
        ds = ds.sel(time=slice(args.start_date, args.end_date))

    ds = ds.fillna(args.fill_value)

    patches = load_patches(Path(args.patches)) if args.patches else None

    if patches is None:
        if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max):
            raise ValueError("Provide --patches or all of --lat-min/--lat-max/--lon-min/--lon-max for a single patch.")
        patches = [
            {
                "name": "patch00",
                "lat_min": args.lat_min,
                "lat_max": args.lat_max,
                "lon_min": args.lon_min,
                "lon_max": args.lon_max,
            }
        ]

    for patch in patches:
        name = patch.get("name") or "patch"
        lat_min = float(patch["lat_min"])
        lat_max = float(patch["lat_max"])
        lon_min = float(patch["lon_min"])
        lon_max = float(patch["lon_max"])

        patch_ds = subset_by_bounds(ds, lat_min, lat_max, lon_min, lon_max)
        out_path = output_dir / f"{name}.csv"
        flatten_patch_to_csv(patch_ds, out_path, var_name=args.var)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
