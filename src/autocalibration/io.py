"""File readers for calibration targets and observations."""

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_mapping(path_or_mapping: str | Path | Mapping[str, Any]) -> dict[str, float]:
    if isinstance(path_or_mapping, Mapping):
        return _float_mapping(path_or_mapping)

    path = Path(path_or_mapping)
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"target profile must be a JSON object: {path}")
    return _float_mapping(payload)


def load_observations(
    path: str | Path,
    parameter_names: Sequence[str],
    metric_columns: Sequence[str] | None = None,
    parameter_paths: Mapping[str, str] | None = None,
    metric_paths: Mapping[str, str] | None = None,
) -> list[dict[str, dict[str, float]]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_observations(path, parameter_names, metric_columns)
    if suffix == ".json":
        return _load_json_observations(path)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5_observations(path, parameter_paths, metric_paths)
    raise ValueError(f"unsupported metrics file type: {path.suffix}")


def _load_csv_observations(
    path: Path,
    parameter_names: Sequence[str],
    metric_columns: Sequence[str] | None,
) -> list[dict[str, dict[str, float]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    fieldnames = set(rows[0])
    missing_parameters = [name for name in parameter_names if name not in fieldnames]
    if missing_parameters:
        raise ValueError(f"CSV file is missing parameter columns: {', '.join(missing_parameters)}")

    if metric_columns is None:
        metric_columns = [name for name in rows[0] if name not in set(parameter_names)]
    missing_metrics = [name for name in metric_columns if name not in fieldnames]
    if missing_metrics:
        raise ValueError(f"CSV file is missing metric columns: {', '.join(missing_metrics)}")

    observations = []
    for row in rows:
        observations.append(
            {
                "parameters": {name: float(row[name]) for name in parameter_names},
                "metrics": {name: float(row[name]) for name in metric_columns},
            }
        )
    return observations


def _load_json_observations(path: Path) -> list[dict[str, dict[str, float]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"observation JSON must be a list: {path}")

    observations = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"observation {index} must be an object")
        parameters = item.get("parameters")
        metrics = item.get("metrics")
        if not isinstance(parameters, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError(f"observation {index} must contain parameter and metric objects")
        observations.append(
            {
                "parameters": _float_mapping(parameters),
                "metrics": _float_mapping(metrics),
            }
        )
    return observations


def _load_hdf5_observations(
    path: Path,
    parameter_paths: Mapping[str, str] | None,
    metric_paths: Mapping[str, str] | None,
) -> list[dict[str, dict[str, float]]]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("HDF5 metrics support requires h5py; install it with `pip install h5py`.") from exc

    if not parameter_paths or not metric_paths:
        raise ValueError("HDF5 input requires parameter_paths and metric_paths")

    with h5py.File(path, "r") as handle:
        parameter_values = {name: _dataset_values(handle[dataset_path]) for name, dataset_path in parameter_paths.items()}
        metric_values = {name: _dataset_values(handle[dataset_path]) for name, dataset_path in metric_paths.items()}

    lengths = {len(values) for values in [*parameter_values.values(), *metric_values.values()]}
    if len(lengths) != 1:
        raise ValueError("all HDF5 parameter and metric datasets must be scalar or same-length arrays")
    count = lengths.pop()

    observations = []
    for index in range(count):
        observations.append(
            {
                "parameters": {name: values[index] for name, values in parameter_values.items()},
                "metrics": {name: values[index] for name, values in metric_values.items()},
            }
        )
    return observations


def _dataset_values(dataset: Any) -> list[float]:
    value = dataset[()]
    if hasattr(value, "shape") and value.shape == ():
        return [float(value)]
    if isinstance(value, bytes):
        return [float(value.decode())]
    try:
        return [float(item) for item in value]
    except TypeError:
        return [float(value)]


def _float_mapping(mapping: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in mapping.items()}
