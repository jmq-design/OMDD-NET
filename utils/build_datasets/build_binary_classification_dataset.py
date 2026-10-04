

# dataset source:
# - weatherAUS: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
# - BNG_labo: https://www.openml.org/data/download/5579/BNG_labor.arff

from pathlib import Path

import csv
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



# CSV / Orange loading
def detect_delimiter(csv_file):
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except Exception:
        return ","


def strip_orange_prefix(column_name):
    column_name = str(column_name).strip()
    for prefix in ("cD#", "C#", "D#", "i#"):
        if column_name.startswith(prefix):
            return column_name[len(prefix):]
    return column_name


def infer_column_roles(columns, target_columns=None):
    target_columns = set(target_columns or [])
    roles = {}
    for col in columns:
        col = str(col).strip()
        if col in target_columns:
            roles[col] = "target"
        elif col.startswith("i#"):
            roles[col] = "meta"
        elif col.startswith("C#"):
            roles[col] = "continuous"
        elif col.startswith("D#"):
            roles[col] = "discrete"
        else:
            roles[col] = "continuous"
    return roles


def resolve_target_columns(columns):
    """
    Resolve supervised target columns from Orange-style CSV headers.
    """
    columns = [str(col).strip() for col in columns]
    explicit_targets = [col for col in columns if col.startswith("cD#")]
    if explicit_targets:
        if len(explicit_targets) > 1:
            print(f"!! Warning: multiple target columns found by cD# prefix: {explicit_targets}")
        for target_col in explicit_targets:
            if target_col != columns[-1]:
                print(
                    f"! Warning: target column `{target_col}` is not the last source column; "
                    "processed output will move target column(s) to the end."
                )
        return explicit_targets

    if not columns:
        print("!! Warning: no columns found in source CSV.")
        return []

    fallback_target = columns[-1]
    print(
        "!! Warning: no target columns found (expected prefix `cD#`). "
        f"Use the last column `{fallback_target}` as target instead."
    )
    return [fallback_target]


def validate_output_column_name(name, source_col, used_names):
    if name in used_names:
        raise ValueError(
            f"Duplicate processed column name `{name}` after stripping prefixes. "
            f"Current source column: `{source_col}`. Previous source column: `{used_names[name]}`."
        )
    used_names[name] = source_col


def read_source_csv(csv_file):
    delimiter = detect_delimiter(csv_file)
    df = pd.read_csv(
        csv_file,
        sep=delimiter,
        engine="python",
        keep_default_na=True,
        na_values=["NA", "NaN", "nan", ""],
    )
    stripped_columns = [str(col).strip() for col in df.columns]
    if len(stripped_columns) != len(set(stripped_columns)):
        duplicates = sorted({col for col in stripped_columns if stripped_columns.count(col) > 1})
        raise ValueError(f"Duplicate column names after stripping whitespace: {duplicates}")
    df.columns = stripped_columns
    return df


def build_processed_dataframe_by_pandas(csv_file):
    raw_df = read_source_csv(csv_file)
    target_columns = resolve_target_columns(raw_df.columns)
    roles = infer_column_roles(raw_df.columns, target_columns=target_columns)
    if not target_columns:
        raise ValueError(f"No target columns can be resolved from `{csv_file}`.")

    df, dropped_indices = drop_missing_target_rows(raw_df, target_columns=target_columns)
    if dropped_indices:
        print(f"drop rows with missing target in processed data: {len(dropped_indices)}")

    feature_columns = [
        col
        for col in df.columns
        if roles.get(col) not in ("meta", "target")
    ]
    if not feature_columns:
        print(f"!! Warning: no feature columns found in `{csv_file}` after removing meta/target columns.")

    for target_col in target_columns:
        non_missing_values = df[target_col].dropna()
        unique_count = non_missing_values.nunique(dropna=False)
        if unique_count != 2:
            print(
                f"!! Warning: target column `{target_col}` has {unique_count} unique value(s), "
                "but this builder is intended for binary classification."
            )

    output_columns = {}
    used_output_names = {}

    for col in feature_columns:
        role = roles.get(col)
        output_col = strip_orange_prefix(col)
        validate_output_column_name(output_col, col, used_output_names)
        if role == "continuous":
            values = pd.to_numeric(df[col], errors="coerce")
            if values.isna().all():
                print(f"!! Warning: continuous feature `{col}` is all missing; fill with 0.0.")
                imputed = values.fillna(0.0)
            else:
                imputer = SimpleImputer(strategy="median")
                imputed = imputer.fit_transform(values.to_frame()).ravel()
            output_columns[output_col] = imputed
        else:
            values = df[col].astype("object")
            if values.dropna().empty:
                print(f"!! Warning: categorical feature `{col}` is all missing; fill with `missing`.")
                output_columns[output_col] = np.full(len(df), "missing", dtype=object)
            else:
                imputer = SimpleImputer(strategy="most_frequent")
                output_columns[output_col] = imputer.fit_transform(values.to_frame()).ravel()

    for col in target_columns:
        output_col = strip_orange_prefix(col)
        validate_output_column_name(output_col, col, used_output_names)
        output_columns[output_col] = df[col].to_numpy()

    output_df = pd.DataFrame(output_columns, index=df.index).copy()
    return output_df.reset_index(drop=True), [strip_orange_prefix(col) for col in target_columns]




def to_jsonable_value(value):
    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value):
        return None

    return value


def encode_categorical_columns(df):
    encoded_df = df.copy()
    mapping = {}

    for col in encoded_df.columns:
        is_non_numeric = not pd.api.types.is_numeric_dtype(encoded_df[col])
        is_bool = pd.api.types.is_bool_dtype(encoded_df[col])

        if is_non_numeric or is_bool:
            unique_values = pd.Series(encoded_df[col].dropna().unique())

            value_to_id = {
                value: idx
                for idx, value in enumerate(unique_values)
            }

            id_to_value = {
                str(idx): to_jsonable_value(value)
                for idx, value in enumerate(unique_values)
            }

            encoded_df[col] = encoded_df[col].map(value_to_id)
            mapping[col] = id_to_value

    return encoded_df, mapping


def drop_constant_feature_columns(df, target_columns=None):
    if target_columns is None:
        target_columns = [df.columns[-1]] if len(df.columns) > 0 else []
    target_columns = set(target_columns)

    constant_columns = [
        col
        for col in df.columns
        if col not in target_columns and df[col].nunique(dropna=False) <= 1
    ]
    if not constant_columns:
        return df, []
    return df.drop(columns=constant_columns), constant_columns


def drop_missing_target_rows(df, target_columns=None):
    if target_columns is None:
        target_columns = [df.columns[-1]] if len(df.columns) > 0 else []
    target_columns = [col for col in target_columns if col in df.columns]
    if not target_columns:
        return df, []

    missing_mask = df[target_columns].isna().any(axis=1)
    dropped_indices = df.index[missing_mask].tolist()
    if not dropped_indices:
        return df, []
    return df.loc[~missing_mask].copy(), dropped_indices



def _validate_data(df: pd.DataFrame):
    assert np.all(np.isfinite(df.to_numpy())), "expected finite values only"
    duplicate_count = df.duplicated().sum()
    assert duplicate_count == 0, f"found {duplicate_count} duplicate rows"

def save_encoded_src_dataframe(
    df,
    save_file=None,
    mapping_save_file=None,
    drop_duplicates=False,
    target_columns=None,
):
    df, dropped_constant_columns = drop_constant_feature_columns(df, target_columns=target_columns)
    if dropped_constant_columns:
        print(f"drop constant feature columns: {dropped_constant_columns}")

    encoded_df, mapping = encode_categorical_columns(df)

    if drop_duplicates:
        encoded_df = encoded_df.drop_duplicates(keep="first")

    _validate_data(encoded_df)

    if save_file is not None:
        encoded_df.to_csv(save_file, index=False)
        print(f"generating {save_file} ...")

    if mapping_save_file is not None:
        with open(mapping_save_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        print(f"generating {mapping_save_file} ...")

    return encoded_df, mapping




def generate_processed_dataset_file(
    csv_file,
    processed_save_file=None,
    processed_mapping_save_file=None,
):
    """
    Generate {dataset}_processed.csv/json.
    """
    if processed_save_file is None and processed_mapping_save_file is None:
        return None, None

    processed_src_df, target_columns = build_processed_dataframe_by_pandas(csv_file)
    return save_encoded_src_dataframe(
        processed_src_df,
        save_file=processed_save_file,
        mapping_save_file=processed_mapping_save_file,
        drop_duplicates=True,
        target_columns=target_columns,
    )


if __name__ == "__main__":

    large_datasets = [
        "magic04",
        "adult",
        "sec_mushroom",
        "bank_marketing",
        "higgs",
        "weatherAUS",
        "BNG_labor",
        "BNG_credit-g",
    ]

    small_datasets = [
        "christine",
        "musk2",
        "heloc",
        # 
        "tic-tac-toe",
        "splice-1",
        "kr-vs-kp",
        "mushroom",
        # 
        "anneal",
        "hypothyroid",
        "car"
    ]

    datasets = large_datasets + small_datasets

    org_dataset_dir = "./datasets"
    dataset_dir = "./datasets"

    for dataset in datasets:

        org_data_path = (
            f"{org_dataset_dir}/large_datasets"
            if dataset in large_datasets
            else f"{org_dataset_dir}/small_datasets"
        )

        processed_data_path = (
            f"{dataset_dir}/large_datasets/processed"
            if dataset in large_datasets
            else f"{dataset_dir}/small_datasets/processed"
        )

        dataset_file = f"{org_data_path}/{dataset}.csv"

        save_processed_file = f"{processed_data_path}/{dataset}_processed.csv"
        save_processed_mapping_file = f"{processed_data_path}/{dataset}_processed.json"

        need_generate = (
            not os.path.exists(save_processed_file)
            or not os.path.exists(save_processed_mapping_file)
        )

        # if need_generate:
        if True:
            os.makedirs(processed_data_path, exist_ok=True)

            generate_processed_dataset_file(
                dataset_file,
                processed_save_file=save_processed_file,
                processed_mapping_save_file=save_processed_mapping_file,
            )
