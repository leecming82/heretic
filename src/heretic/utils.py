# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import csv
import gc
import json
from pathlib import Path
from dataclasses import asdict
from importlib.metadata import version
from typing import TypeVar

import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset
from optuna import Trial
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def load_prompts(specification: DatasetSpecification) -> list[str]:
    dataset_path = Path(specification.dataset)
    if dataset_path.exists():
        return load_local_prompts(dataset_path, specification)

    dataset = load_dataset(specification.dataset, split=specification.split)
    return list(dataset[specification.column])


def load_local_prompts(
    dataset_path: Path,
    specification: DatasetSpecification,
) -> list[str]:
    split_name, slice_spec = parse_split(specification.split)
    records = read_local_split(dataset_path, split_name)
    prompts = [extract_prompt(record, specification.column) for record in records]
    return prompts[slice_spec]


def parse_split(split: str) -> tuple[str, slice]:
    if "[" not in split:
        return split, slice(None)

    base, _, remainder = split.partition("[")
    base = base.strip()
    remainder = remainder.rstrip("]")

    start_str, _, end_str = remainder.partition(":")
    if "%" in start_str or "%" in end_str:
        raise ValueError(
            "Percentage-based splits are not supported for local datasets. "
            f"Invalid split expression: {split}"
        )

    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None
    return base, slice(start, end)


def read_local_split(dataset_path: Path, split_name: str) -> list[dict | str]:
    if dataset_path.is_file():
        return read_local_file(dataset_path)

    if not split_name:
        raise ValueError(
            "Split names must be provided when loading datasets from directories."
        )

    for extension in (".jsonl", ".json", ".csv", ".txt"):
        candidate = dataset_path / f"{split_name}{extension}"
        if candidate.exists():
            return read_local_file(candidate)

    raise FileNotFoundError(
        f"No file found for split '{split_name}' in {dataset_path}. "
        "Expected a file named 'split.jsonl', 'split.json', 'split.csv', or 'split.txt'."
    )


def read_local_file(path: Path) -> list[dict | str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_json_lines(path)
    if suffix == ".json":
        return read_json(path)
    if suffix == ".csv":
        return read_csv_file(path)
    if suffix in (".txt", ".text"):
        return read_text_file(path)

    raise ValueError(
        f"Unsupported local dataset format: {path}. "
        "Supported extensions are .jsonl, .json, .csv, .txt"
    )


def read_json_lines(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json(path: Path) -> list[dict | str]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "records", "prompts"):
            if isinstance(data.get(key), list):
                return data[key]
        # Support records stored as column-oriented dictionaries.
        if all(isinstance(value, list) for value in data.values()):
            length = len(next(iter(data.values()), []))
            return [
                {key: value[index] for key, value in data.items()}
                for index in range(length)
            ]

    raise ValueError(f"Unsupported JSON structure in {path}.")


def read_csv_file(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def read_text_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def extract_prompt(record: dict | str, column: str) -> str:
    if isinstance(record, str):
        if column:
            raise ValueError(
                "Column names cannot be used with plain-text datasets. "
                "Either remove the column specification or convert the dataset "
                "to JSON/CSV with named columns."
            )
        return record

    if column not in record:
        raise KeyError(
            f"Column '{column}' not found in local dataset record: {record.keys()}"
        )

    value = record[column]
    if not isinstance(value, str):
        raise ValueError(
            f"Column '{column}' must contain strings. "
            f"Encountered {type(value)} instead."
        )

    return value


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in asdict(parameters).items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
