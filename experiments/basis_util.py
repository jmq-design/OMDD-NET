import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_train_test_split(dataset, ratio, save_indices_file, split_seed):
    if os.path.exists(save_indices_file):
        print(f"load {ratio}-train_test data from {save_indices_file} ...")
        with open(save_indices_file, "r") as f:
            train_test_indices = f.readlines()
            train_test_indices = [line.strip().split(",") for line in train_test_indices]
            train_test_indices = [[int(num) for num in line if num != ""] for line in train_test_indices]
            train_index = train_test_indices[0]
            test_index = train_test_indices[1]
            print("len(train_index)+len(test_index) :", len(train_index) + len(test_index))
    else:
        data = dataset.iloc[:, 0:-1]
        label = dataset.iloc[:, -1]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=split_seed)
        train_test_indices = sss.split(X=data, y=label)
        print(f"generate {ratio}-train_test data using StratifiedShuffleSplit... {save_indices_file}")
        with open(save_indices_file, "w") as fo:
            for train_index, test_index in train_test_indices:
                fo.write(", ".join(list(map(str, train_index))))
                fo.write("\n")
                fo.write(", ".join(list(map(str, test_index))))
                fo.write("\n")

    return train_index, test_index



def get_k_fold_split(dataset, k_fold, save_indices_file, split_seed):

    fold_indices = []
    if os.path.exists(save_indices_file):
        print(f'load {k_fold}-fold data from {save_indices_file} ...')
        with open(save_indices_file, 'r') as f:
            test_indices_list = f.readlines()
            test_indices_list = [line.strip().split(',') for line in test_indices_list]
            test_indices_list = [[int(num) for num in line] for line in test_indices_list]

            for k in range(k_fold):
                kth_train_indices = []
                for j in range(k_fold):
                    if j!=k:
                        kth_train_indices += test_indices_list[j]

                kth_train_indices = sorted(kth_train_indices)
                kth_test_indices = sorted(test_indices_list[k])

                kth_fold = (kth_train_indices, kth_test_indices)
                fold_indices.append(kth_fold)
    else:
        kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=split_seed)
        data = dataset.iloc[:, 0:-1]
        label =  dataset.iloc[:, -1]

        fold_indices = [
            (np.sort(train_index), np.sort(test_index))
            for train_index, test_index in kf.split(X=data, y=label) # Stratification is done based on the y labels.
        ]

        print(f'generate {k_fold}-fold data using StratifiedKFold... {save_indices_file}')
        with open(save_indices_file, 'w') as fo:
            for _, test_index in fold_indices:
                fo.write(", ".join(list(map(str, test_index))))
                fo.write("\n")

    return fold_indices




def dump_split_csvs(
    dataset: pd.DataFrame,
    train_index: Sequence[int],
    test_index: Sequence[int],
    train_file: str,
    test_file: str,
) -> Tuple[str, str]:
    train_data = dataset.iloc[list(train_index)]
    test_data = dataset.iloc[list(test_index)]

    train_data = train_data.sort_values(by=list(train_data.columns), 
                                            ignore_index=True)

    train_parent = os.path.dirname(train_file)
    test_parent = os.path.dirname(test_file)
    if train_parent:
        ensure_dir(train_parent)
    if test_parent:
        ensure_dir(test_parent)

    if not os.path.exists(train_file):
        print(f"generate train data: {train_file} ...")
        train_data.to_csv(train_file, index=False)
    if not os.path.exists(test_file):
        print(f"generate test data: {test_file} ...")
        test_data.to_csv(test_file, index=False)
    return train_file, test_file


def build_train_test_files(
    dataset_file: str,
    ratio: float,
    split_seed: int,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = pd.read_csv(dataset_file)
    data_filename = os.path.splitext(os.path.basename(dataset_file))[0]
    output_dir = ensure_dir(output_dir or os.path.join(os.path.dirname(dataset_file), f"train-test_{data_filename}"))
    split_data_info = f"train_test-{data_filename}_{ratio}_{split_seed}"
    save_indices_file = os.path.join(output_dir, f"{split_data_info}.indices")
    save_train_file = os.path.join(output_dir, f"{split_data_info}_train.csv")
    save_test_file = os.path.join(output_dir, f"{split_data_info}_test.csv")

    train_index, test_index = get_train_test_split(dataset, ratio, save_indices_file, split_seed)
    dump_split_csvs(dataset, train_index, test_index, save_train_file, save_test_file)
    return {
        "dataset_file": dataset_file,
        # "train_index": train_index,
        # "test_index": test_index,
        "train_file": save_train_file,
        "test_file": save_test_file,
        "indices_file": save_indices_file,
        "output_dir": output_dir,
    }


def build_train_valid_files(
    dataset_file: str,
    valid_ratio: float,
    split_seed: int,
    output_dir: Optional[str] = None,
    prefix: str = "train_valid",
    stratify: bool = True,
) -> Dict[str, Any]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}.")

    dataset = pd.read_csv(dataset_file)
    data_filename = os.path.splitext(os.path.basename(dataset_file))[0]
    output_dir = ensure_dir(output_dir or os.path.join(os.path.dirname(dataset_file), f"{prefix}_{data_filename}"))
    split_data_info = f"{prefix}-{data_filename}_{valid_ratio}_{split_seed}"
    save_indices_file = os.path.join(output_dir, f"{split_data_info}.indices")
    save_train_file = os.path.join(output_dir, f"{split_data_info}_train.csv")
    save_valid_file = os.path.join(output_dir, f"{split_data_info}_valid.csv")

    if os.path.exists(save_indices_file):
        print(f"load {prefix} split from {save_indices_file} ...")
        with open(save_indices_file, "r") as f:
            split_indices = f.readlines()
            split_indices = [line.strip().split(",") for line in split_indices]
            split_indices = [[int(num) for num in line if num != ""] for line in split_indices]
            train_index = split_indices[0]
            valid_index = split_indices[1]
    else:
        data = dataset.iloc[:, 0:-1]
        label = dataset.iloc[:, -1]
        if stratify:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=1.0 - valid_ratio,
                random_state=split_seed,
            )
            split_iter = splitter.split(X=data, y=label)
        else:
            splitter = ShuffleSplit(
                n_splits=1,
                train_size=1.0 - valid_ratio,
                random_state=split_seed,
            )
            split_iter = splitter.split(X=data)

        print(f"generate {prefix} split... {save_indices_file}")
        with open(save_indices_file, "w") as fo:
            for train_index, valid_index in split_iter:
                fo.write(", ".join(list(map(str, train_index))))
                fo.write("\n")
                fo.write(", ".join(list(map(str, valid_index))))
                fo.write("\n")

    dump_split_csvs(dataset, train_index, valid_index, save_train_file, save_valid_file)
    return {
        "dataset_file": dataset_file,
        "train_file": save_train_file,
        "valid_file": save_valid_file,
        "indices_file": save_indices_file,
        "output_dir": output_dir,
    }


def build_k_fold_files(
    dataset_file: str,
    k_fold: int,
    split_seed: int,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = pd.read_csv(dataset_file)
    data_filename = os.path.splitext(os.path.basename(dataset_file))[0]
    output_dir = ensure_dir(output_dir or os.path.join(os.path.dirname(dataset_file), f"{k_fold}-fold_{data_filename}"))
    split_data_info = f"{k_fold}fold_indices-{data_filename}_{split_seed}"
    save_indices_file = os.path.join(output_dir, f"{split_data_info}.indices")
    fold_indices = get_k_fold_split(dataset, k_fold, save_indices_file, split_seed)

    fold_files = []
    for i, (train_index, test_index) in enumerate(fold_indices):
        train_file = os.path.join(output_dir, f"{split_data_info}_train_fold_{i}.csv")
        test_file = os.path.join(output_dir, f"{split_data_info}_test_fold_{i}.csv")
        dump_split_csvs(dataset, train_index, test_index, train_file, test_file)
        fold_files.append(
            {
                "fold": i,
                # "train_index": train_index,
                # "test_index": test_index,
                "train_file": train_file,
                "test_file": test_file,
            }
        )

    return {
        "dataset_file": dataset_file,
        # "fold_indices": fold_indices,
        "fold_files": fold_files,
        "indices_file": save_indices_file,
        "output_dir": output_dir,
    }


def avg_list(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_numeric_lists(metrics: Dict[str, List[float]], prefix: str = "avg_") -> Dict[str, float]:
    return {f"{prefix}{name}": avg_list(values) for name, values in metrics.items() if values}


def write_json(result_file: str, payload: Dict[str, Any]) -> str:
    parent = os.path.dirname(result_file)
    if parent:
        ensure_dir(parent)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return result_file
