#!/usr/bin/env python3
"""W&B run metric extractor.

Creates one CSV row per run tag with selected metric keys.
Missing values are written as NaN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import wandb

NAN_STR = "nan"

DEFAULT_RUNS = [
    "armd_linear_etth1_l1", "armd_linear_etth1_l2",
    "armd_linear_ettm1_l1", "armd_linear_ettm1_l2",
    "armd_linear_solar_l1", "armd_linear_solar_l2",
    "armd_tsmixer_etth1_l1", "armd_tsmixer_etth1_l2",
    "armd_tsmixer_ettm1_l1", "armd_tsmixer_ettm1_l2",
    "armd_tsmixer_solar_l1", "armd_tsmixer_solar_l2",
    "armd_dlinear_etth1_l1_s", "armd_dlinear_etth1_l2_s",
    "armd_dlinear_ettm1_l1_s", "armd_dlinear_ettm1_l2_s",
    "armd_dlinear_solar_l1_s", "armd_dlinear_solar_l2_s",
    "armd_dlinear_etth1_l1_i", "armd_dlinear_etth1_l2_i",
    "armd_dlinear_ettm1_l1_i", "armd_dlinear_ettm1_l2_i",
    "armd_dlinear_solar_l1_i", "armd_dlinear_solar_l2_i",
]

DEFAULT_METRICS = [
    "trainable_parameters",
    "total_parameters",
    "test/96/norm/MAE", "test/96/norm/MSE", "test/96/norm/sMAPE",
    "test/96/norm/CRPS", "test/96/norm/DTW",
]

DATASET_CHANNELS = {
    "etth1": 7,
    "ettm1": 7,
    "solar": 137,
    "solar_nips": 137,
    "electricity": 321,
    "electricity_ltsf": 321,
}


def _to_run_name_list(run_names: str | Sequence[str]) -> list[str]:
    if isinstance(run_names, str):
        return [run_names]
    return list(run_names)


def _safe_dt(value: object) -> datetime:
    if not value:
        return datetime.min
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _best_run_for_name(runs: Iterable[wandb.apis.public.Run], tag: str) -> wandb.apis.public.Run | None:
    same_name = [r for r in runs if (r.name or "") == tag]
    if not same_name:
        return None

    def rank(run: wandb.apis.public.Run) -> tuple[int, datetime]:
        state = (run.state or "").lower()
        finished = 1 if state == "finished" else 0
        updated_at = _safe_dt(getattr(run, "updated_at", None))
        return (finished, updated_at)

    return sorted(same_name, key=rank, reverse=True)[0]


def _pull_metric(run: wandb.apis.public.Run, key: str) -> float | str:
    summary = dict(run.summary)

    value = summary.get(key)
    if value is not None:
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return NAN_STR
            return f
        except Exception:
            return str(value)

    try:
        last_non_null = None
        for row in run.scan_history(keys=[key]):
            if key in row and row[key] is not None:
                last_non_null = row[key]
        if last_non_null is not None:
            try:
                f = float(last_non_null)
                if math.isnan(f) or math.isinf(f):
                    return NAN_STR
                return f
            except Exception:
                return str(last_non_null)
    except Exception:
        pass

    return NAN_STR


def _is_nan_token(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() == NAN_STR


def _format_number(value: object, round_digits: int | None) -> object:
    if round_digits is None or _is_nan_token(value):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return NAN_STR
        return round(float(value), round_digits)
    return value


def _format_params_value(value: object, round_digits: int | None) -> object:
    if _is_nan_token(value):
        return value

    # Keep already-formatted strings (e.g., "9.4 K", "2.6 M") untouched.
    if isinstance(value, str):
        stripped = value.strip()
        if any(u in stripped.upper() for u in (" K", " M", " B")):
            return stripped
        try:
            value = float(stripped)
        except Exception:
            return value

    if not isinstance(value, (int, float)):
        return value

    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return NAN_STR

    digits = 2 if round_digits is None else round_digits
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.{digits}f} M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.{digits}f} K"
    return round(v, digits)


def _infer_backbone_from_tag(tag: str) -> str:
    if "_linear_" in tag:
        return "linear"
    if "_tsmixer_" in tag:
        return "mixed"
    if "_dlinear_" in tag:
        if re.search(r"(_s|_shared)(?:_|$)", tag):
            return "dlinear shared"
        if re.search(r"(_i|_individual)(?:_|$)", tag):
            return "dlinear individual"
        return "dlinear"
    return NAN_STR


def _parse_tag(tag: str) -> tuple[str, str, str | None]:
    m = re.match(r"^armd_(linear|dlinear|tsmixer)_([a-zA-Z0-9_]+?)_(l1|l2)(?:_([si]))?(?:_(short|long))?$", tag)
    if not m:
        return "", "", None
    model, dataset, _, dlin, horizon = m.groups()
    variant = dlin
    if horizon == "short" and not variant:
        variant = "short"
    return model, dataset, variant


def _infer_pred_len(tag: str) -> int:
    return 24 if tag.endswith("_short") else 96


def _theoretical_params(tag: str) -> int | None:
    model, dataset, variant = _parse_tag(tag)
    if not model or not dataset:
        return None

    c = DATASET_CHANNELS.get(dataset)
    if c is None:
        return None

    p = _infer_pred_len(tag)
    t = p

    if model == "linear":
        # LinearBackbone: Linear(p,p) + w[t]
        return (p * p + p) + t

    if model == "dlinear":
        # DLinearBackbone:
        # shared => 2*Linear(p,p) + w[t]
        # individual => 2*C*Linear(p,p) + w[t]
        if variant == "i":
            return 2 * c * (p * p + p) + t
        return 2 * (p * p + p) + t

    if model == "tsmixer":
        # TSMixerBackbone defaults from ARMD:
        # n_blocks=2, hidden_dim=128, temporal_ratio=2.0, channel_ratio=2.0
        n_blocks = 2
        hidden_dim = 128
        ht = max(int(p * 2.0), hidden_dim)
        hc = max(int(c * 2.0), hidden_dim)
        per_block = (2 * p * ht + ht + 3 * p) + (2 * c * hc + hc + 3 * c)
        return n_blocks * per_block + t

    return None


def extract_runs_to_csv(
    run_names: str | Sequence[str],
    metrics: Sequence[str],
    file_name: str,
    entity: str,
    project: str,
    tag_to_backbone: Mapping[str, str] | None = None,
    round_digits: int | None = None,
    format_params_human: bool = False,
    fill_theoretical_params: bool = True,
) -> list[dict[str, object]]:
    """Extract selected metrics for one or multiple W&B run names (tags).

    Args:
        run_names: One tag or a list of tags. Here run name == tag.
        metrics: List of metric keys to pull, e.g. ["train/loss", "val/loss"].
        file_name: Output CSV path.
        entity: W&B entity.
        project: W&B project.
        tag_to_backbone: Optional dict mapping tag -> backbone label.
        round_digits: Number of digits for numeric rounding (e.g., 2).
        format_params_human: If True, fields containing "param" are formatted to K/M.
        fill_theoretical_params: If True, fills missing params with theoretical counts.

    Returns:
        List of row dicts written to CSV.
    """
    tags = _to_run_name_list(run_names)
    api = wandb.Api(overrides={"entity": entity, "project": project})
    all_runs = list(api.runs(f"{entity}/{project}"))

    rows: list[dict[str, object]] = []
    for tag in tags:
        run = _best_run_for_name(all_runs, tag)

        row: dict[str, object] = {
            "tag": tag,
            "backbone": (tag_to_backbone or {}).get(tag, _infer_backbone_from_tag(tag)),
        }

        if run is None:
            for m in metrics:
                row[m] = NAN_STR
        else:
            for m in metrics:
                value = _pull_metric(run, m)
                if format_params_human and "param" in m.lower():
                    value = _format_params_value(value, round_digits)
                else:
                    value = _format_number(value, round_digits)
                row[m] = value

        if fill_theoretical_params:
            theo = _theoretical_params(tag)
            if theo is not None:
                for k in ("trainable_parameters", "total_parameters"):
                    if k in row and _is_nan_token(row[k]):
                        if format_params_human:
                            row[k] = _format_params_value(theo, round_digits)
                        else:
                            row[k] = _format_number(theo, round_digits)

        rows.append(row)

    output_path = Path(file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["tag", "backbone", *metrics]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _load_backbone_map(backbone_map: str | None, backbone_map_file: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if backbone_map:
        mapping.update(json.loads(backbone_map))
    if backbone_map_file:
        file_mapping = json.loads(Path(backbone_map_file).read_text())
        mapping.update(file_mapping)
    return mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected W&B metrics into one-row-per-tag CSV.")
    parser.add_argument("--entity", default="eliasalbouzidics", help="W&B entity")
    parser.add_argument("--project", default="probts", help="W&B project")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run names (tags)")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metric keys to extract")
    parser.add_argument("--file", default="logs/wandb_extract_all_tags.csv", help="Output CSV path")
    parser.add_argument(
        "--round-digits",
        type=int,
        default=2,
        help="Round numeric metrics to this many decimals (e.g., 2).",
    )
    parser.add_argument(
        "--format-params-human",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Format parameter-like fields ("*param*") into K/M units.',
    )
    parser.add_argument(
        "--fill-theoretical-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill missing parameter fields using theoretical formulas from tag.",
    )
    parser.add_argument(
        "--backbone-map",
        default=None,
        help='JSON string mapping tag->backbone, e.g. {"tag1":"linear"}',
    )
    parser.add_argument(
        "--backbone-map-file",
        default=None,
        help="Path to JSON file mapping tag->backbone",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mapping = _load_backbone_map(args.backbone_map, args.backbone_map_file)

    rows = extract_runs_to_csv(
        run_names=args.runs,
        metrics=args.metrics,
        file_name=args.file,
        entity=args.entity,
        project=args.project,
        tag_to_backbone=mapping,
        round_digits=args.round_digits,
        format_params_human=args.format_params_human,
        fill_theoretical_params=args.fill_theoretical_params,
    )
    print(f"Wrote {len(rows)} rows to {args.file}")


if __name__ == "__main__":
    main()
