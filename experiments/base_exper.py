import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import torch

import sys
sys.path.append(".")

from DDNet import train as ddnet_train
from experiments.basis_util import (
    build_k_fold_files,
    build_train_valid_files,
    build_train_test_files,
    ensure_dir,
    summarize_numeric_lists,
    write_json,
)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _extract_eval_metrics(eval_result: Dict[str, Any], prefix: str) -> Dict[str, float]:
    if eval_result is None:
        return {}
    faithful = eval_result["evaluations"]["faithful"]
    network = eval_result["evaluations"]["network"]
    learned_dd = eval_result["evaluations"]["learned_dd"]
    metrics = {}
    if "score" in faithful:
        metrics[f"{prefix}_acc"] = float(faithful["score"]["acc"])
        metrics[f"{prefix}_f1_macro"] = float(faithful["score"]["f1_macro"])
        metrics[f"{prefix}_network_acc"] = float(network["score"]["acc"])
        metrics[f"{prefix}_network_f1_macro"] = float(network["score"]["f1_macro"])
        if learned_dd.get("available", True) and "score" in learned_dd:
            metrics[f"{prefix}_dd_acc"] = float(learned_dd["score"]["acc"])
            metrics[f"{prefix}_dd_f1_macro"] = float(learned_dd["score"]["f1_macro"])
            if learned_dd.get("size_nonreducedOMDD") is not None:
                metrics[f"{prefix}_size_nonreducedOMDD"] = float(learned_dd["size_nonreducedOMDD"])
            if learned_dd.get("size_reducedOMDD") is not None:
                metrics[f"{prefix}_size_reducedOMDD"] = float(learned_dd["size_reducedOMDD"])
    else:
        metrics[f"{prefix}_rmse"] = float(faithful["rmse"])
        metrics[f"{prefix}_network_rmse"] = float(network["rmse"])
    discretization = eval_result.get("discretization", {})
    for metric_name in (
        "normalized_quantization_error",
        "empty_bin_ratio",
        "min_bin_occupancy_ratio",
        "max_bin_occupancy_ratio",
    ):
        if metric_name in discretization:
            metrics[f"{prefix}_discretization_{metric_name}"] = float(discretization[metric_name])
    consistency = eval_result.get("evaluations", {}).get("consistency_of_preds", {})
    for metric_name, metric_value in consistency.items():
        if metric_value is not None:
            metrics[f"{prefix}_consistency_{metric_name}"] = float(metric_value)
    return metrics


def _extract_learned_dd_artifacts(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    if eval_result is None:
        return {}
    learned_dd = eval_result.get("evaluations", {}).get("learned_dd", {})
    if not learned_dd:
        return {}

    artifacts = {
        "available": learned_dd.get("available", False),
        "size_nonreducedOMDD": learned_dd.get("size_nonreducedOMDD"),
        "size_reducedOMDD": learned_dd.get("size_reducedOMDD"),
        "dot_config": learned_dd.get("dot_config"),
        "branch_labels_by_feature": learned_dd.get("branch_labels_by_feature"),
    }
    if "dot" in learned_dd:
        artifacts["dot"] = learned_dd["dot"]
    if "mdd_case_study" in learned_dd:
        artifacts["mdd_case_study"] = learned_dd["mdd_case_study"]
    if "reason" in learned_dd:
        artifacts["reason"] = learned_dd["reason"]
    return artifacts


def _extract_eval_timing(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    if eval_result is None:
        return {}
    learned_dd = eval_result.get("evaluations", {}).get("learned_dd", {})
    return {
        "total": eval_result.get("timing", {}),
        "network": eval_result.get("evaluations", {}).get("network", {}).get("timing", {}),
        "faithful": eval_result.get("evaluations", {}).get("faithful", {}).get("timing", {}),
        "learned_dd": learned_dd.get("timing", {}),
    }


def _compact_epoch_info(epoch_info: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "epoch": epoch_info["epoch"],
        "global_step_start": epoch_info.get("global_step_start"),
        "global_step": epoch_info.get("global_step"),
        "schedule_progress": epoch_info.get("schedule_progress"),
        "tau": epoch_info["tau"],
        "batch_cnt": epoch_info["batch_cnt"],
        "epoch_task_loss": epoch_info.get("epoch_task_loss"),
        "epoch_loss": epoch_info["epoch_loss"],
        "epoch_reg_loss": epoch_info["epoch_reg_loss"],
        "epoch_reg_loss_details": epoch_info["epoch_reg_loss_details"],
        "elapsed_time": epoch_info["elapsed_time"],
        "timing": epoch_info.get("timing", {}),
        "model_select_source": epoch_info.get("model_select_source"),
        "best_metric_so_far": epoch_info["best_metric_so_far"],
        "best_acc_so_far": epoch_info.get("best_acc_so_far"),
        "best_f1_macro_so_far": epoch_info.get("best_f1_macro_so_far"),
        "best_epoch_so_far": epoch_info["best_epoch_so_far"],
        "best_step_so_far": epoch_info.get("best_step_so_far"),
        "backtracked": epoch_info.get("backtracked", False),
    }
    train_network_eval = epoch_info.get("train_network_eval", {})
    train_faithful_eval = epoch_info.get("train_faithful_eval", {})
    valid_network_eval = epoch_info.get("valid_network_eval", {})
    valid_faithful_eval = epoch_info.get("valid_faithful_eval", {})
    if "score" in train_faithful_eval:
        compact["train_network_acc"] = train_network_eval["score"]["acc"]
        compact["train_network_f1_macro"] = train_network_eval["score"]["f1_macro"]
        compact["train_acc"] = train_faithful_eval["score"]["acc"]
        compact["train_f1_macro"] = train_faithful_eval["score"]["f1_macro"]
        if valid_faithful_eval and "score" in valid_faithful_eval:
            compact["valid_network_acc"] = valid_network_eval["score"]["acc"]
            compact["valid_network_f1_macro"] = valid_network_eval["score"]["f1_macro"]
            compact["valid_acc"] = valid_faithful_eval["score"]["acc"]
            compact["valid_f1_macro"] = valid_faithful_eval["score"]["f1_macro"]
    else:
        compact["train_network_rmse"] = train_network_eval["rmse"]
        compact["train_rmse"] = train_faithful_eval["rmse"]
        if "rmse" in valid_faithful_eval:
            compact["valid_network_rmse"] = valid_network_eval["rmse"]
            compact["valid_rmse"] = valid_faithful_eval["rmse"]
    return compact


def _compress_training_summary(training_summary: Dict[str, Any], save_history: bool = False) -> Dict[str, Any]:
    raw_history = training_summary.get("history", [])
    history_summary = []
    for epoch_info in raw_history:
        if not save_history:
            continue
        history_summary.append(_compact_epoch_info(epoch_info))

    history_preview = {}
    if raw_history:
        history_preview = {
            "first": _compact_epoch_info(raw_history[0]),
            "last": _compact_epoch_info(raw_history[-1]),
        }

    compressed = {
        "elapsed_time": training_summary["elapsed_time"],
        "timing": training_summary.get("timing", {}),
        "stop_reason": training_summary["stop_reason"],
        "stop_mode": training_summary.get("stop_mode"),
        "global_step": training_summary.get("global_step"),
        "step_num": training_summary.get("step_num"),
        "best_epoch": training_summary["best_epoch"],
        "best_step": training_summary.get("best_step"),
        "best_metric": training_summary["best_metric"],
        "best_acc": training_summary.get("best_acc"),
        "best_f1_macro": training_summary.get("best_f1_macro"),
        "model_select_source": training_summary.get("model_select_source"),
        "history_saved": save_history,
        "history_count": len(raw_history),
        "history_preview": history_preview,
    }
    if save_history:
        compressed["history"] = history_summary
    return compressed


def _build_result_paths(args) -> Dict[str, str]:

    dataset_dir = str(Path(args.train_file).parent)
    dataset_name = Path(args.train_file).stem

    _epoch_step_info = f"steps_{args.step_num}" if args.step_num is not None else f"epochs_{args.epoch}"

    config_info = (
        f"result-opt_{args.opt_metric}/"
        f"result-{dataset_name}/"
        f"scale_{args.scale_continuous_features}/"
        f"{args.mode}/"
        f"depth_{args.net_depth}-width_{args.net_width_cap}-intervals_{args.interval_num}-ensembles_{args.ensembles_cnt}-timeout_{args.timeout}-iterations_{args.iterative_iterations}/"
        f"seed_{args.split_seed}-lr_{args.lr}-batch_{args.batch_size}-{_epoch_step_info}/"
        f"val_{args.valid_ratio}-norm_method_{args.normalize_method}-disc_{args.discretizer_type}/"
    )

    result_dir = ensure_dir(args.result_base_path + config_info)

    _backtrack_info = f"dacc_{args.backtrack_dacc}" if args.opt_metric=="acc" else f"dF1_{args.backtrack_dF1}"

    run_info = "".join((
        f"-tau_{args.tau_schedule}_sta{args.tau_start}_end{args.tau_end}_decay_sta{args.tau_decay_start_ratio}_decay_end{args.tau_decay_end_ratio}" if args.normalize_method=="gumbel_softmax" else "",
        f"-{_backtrack_info}",
        f"-a1_{args.a1}-a2_{args.a2}-a3_{args.a3}",
        f"-a4_{args.a4}-dt1_{args.discretize_t1}-dt2_{args.discretize_t2}-dreg_{args.discretize_reg_mode}" if args.discretizer_type=="learned" else "",
        f"-a5_{args.a5}-fsmargin_{args.feat_select_margin_target}" if args.a5!=0 else "",
    ))
    task_info = f"{args.mode}_{dataset_name}"

    return {
        "result_dir": result_dir,
        "summary_file": os.path.join(result_dir, f"res-{run_info}.json"),
        "model_dir": ensure_dir(os.path.join(result_dir, f"model-{run_info}")),
        "split_dir": ensure_dir(os.path.join(dataset_dir, f"{task_info}")),
    }


def run_train_test(args) -> Dict[str, Any]:
    paths = _build_result_paths(args)

    if args.ratio == 1.0 or not args.rest:
        if args.test_file is None:
            raise ValueError("When ratio=1.0 or rest=False, test_file must be provided.")
        train_file = args.train_file
        test_file = args.test_file
        split_info = None
    else:
        split_info = build_train_test_files(
            dataset_file=args.train_file,
            ratio=args.ratio,
            split_seed=args.split_seed,
            output_dir=paths["split_dir"],
        )
        train_file = split_info["train_file"]
        test_file = split_info["test_file"]

    valid_file = None
    valid_split_info = None
    if args.valid_ratio > 0:
        valid_split_info = build_train_valid_files(
            dataset_file=train_file,
            valid_ratio=args.valid_ratio,
            split_seed=args.split_seed,
            output_dir=paths["split_dir"],
            prefix="train_valid",
        )
        train_file = valid_split_info["train_file"]
        valid_file = valid_split_info["valid_file"]

    train_result = ddnet_train(
        train_file=train_file,
        test_file=test_file,
        valid_file=valid_file,
        epoch_num=args.epoch,
        step_num=args.step_num,
        timeout=args.timeout,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_path=os.path.join(paths["model_dir"], "train_test_best.pt"),
        normalize_method=args.normalize_method,
        discretize_t1=args.discretize_t1,
        discretize_t2=args.discretize_t2,
        discretize_reg_mode=args.discretize_reg_mode,
        discretize_cluster_reg_weight=args.discretize_cluster_reg_weight,
        discretize_boundary_reg_weight=args.discretize_boundary_reg_weight,
        discretizer_type=args.discretizer_type,
        discretizer_kwargs={
            "n_bins": args.kbins_n_bins or args.interval_num,
            "encode": args.kbins_encode,
            "strategy": args.kbins_strategy,
            "subsample": args.kbins_subsample,
            "random_state": args.random_seed,
        },
        scale_continuous_features=args.scale_continuous_features,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        tau_decay_start_ratio=args.tau_decay_start_ratio,
        tau_decay_end_ratio=args.tau_decay_end_ratio,
        random_seed=args.random_seed,
        interval_num=args.interval_num,
        dec_level_cnt=args.net_depth,
        ensembles_cnt=args.ensembles_cnt,
        width_cap=args.net_width_cap,
        optimize_metric=args.opt_metric,
        backtrack_dacc=args.backtrack_dacc,
        backtrack_dF1=args.backtrack_dF1,
        coefficients={
            "coef. read_once": args.a1,
            "coef. ent_feat_select": args.a2,
            "coef. ent_dec": args.a3,
            "coef. discretization": args.a4,
            "coef. margin_feat_select": args.a5,
            "margin_feat_select_target": args.feat_select_margin_target,
        },
        verbose=args.verbose,
        print_every=args.print_every,
        learned_dd_dot_edge_label_mode=args.learned_dd_dot_edge_label_mode,
        learned_dd_dot_show_legend=args.learned_dd_dot_show_legend,
        iterative_iterations=args.iterative_iterations,
        iterative_build_final_mdd=args.iterative_build_final_mdd,
        iterative_require_compatible_ordering=args.iterative_require_compatible_ordering,
    )

    summary = {
        "mode": "train-test",
        "train_file": train_file,
        "valid_file": valid_file,
        "test_file": test_file,
        "train_metrics": _extract_eval_metrics(train_result["train_evaluation"], "train"),
        "valid_metrics": _extract_eval_metrics(train_result["valid_evaluation"], "valid"),
        "test_metrics": _extract_eval_metrics(train_result["test_evaluation"], "test"),
        "training_summary": _compress_training_summary(train_result["training_summary"], save_history=args.save_history),
        "evaluation_timing": {
            "train": _extract_eval_timing(train_result["train_evaluation"]),
            "valid": _extract_eval_timing(train_result["valid_evaluation"]),
            "test": _extract_eval_timing(train_result["test_evaluation"]),
        },
        "split_info": {
            "outer_split": split_info,
            "validation_split": valid_split_info,
        },
        "discretized_features": train_result["discretized_features"],
        "hyperparameters": train_result["hyperparameters"],
        "iterative_learning": train_result.get("iterative_learning"),
        "learned_dd_artifacts": {
            "train": _extract_learned_dd_artifacts(train_result["train_evaluation"]),
            # "valid": _extract_learned_dd_artifacts(train_result["valid_evaluation"]),
            # "test": _extract_learned_dd_artifacts(train_result["test_evaluation"]), 
        },
    }
    write_json(paths["summary_file"], summary)
    return summary


def run_k_fold_cross_validation(args, k_fold: int) -> Dict[str, Any]:
    paths = _build_result_paths(args)
    split_info = build_k_fold_files(
        dataset_file=args.train_file,
        k_fold=k_fold,
        split_seed=args.split_seed,
        output_dir=paths["split_dir"],
    )

    fold_results: List[Dict[str, Any]] = []
    aggregated_metrics: Dict[str, List[float]] = {}

    for fold_spec in split_info["fold_files"]:
        fold_id = fold_spec["fold"]
        print(f"\n-------------- {fold_id}th-fold ---------------")
        valid_file = None
        valid_split_info = None
        train_file = fold_spec["train_file"]
        if args.valid_ratio > 0:
            valid_split_info = build_train_valid_files(
                dataset_file=train_file,
                valid_ratio=args.valid_ratio,
                # split_seed=args.split_seed + fold_id,
                split_seed=args.split_seed,
                output_dir=paths["split_dir"],
                prefix=f"fold_{fold_id}_train_valid",
            )
            train_file = valid_split_info["train_file"]
            valid_file = valid_split_info["valid_file"]

        train_result = ddnet_train(
            train_file=train_file,
            test_file=fold_spec["test_file"],
            valid_file=valid_file,
            epoch_num=args.epoch,
            step_num=args.step_num,
            timeout=args.timeout,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            model_path=os.path.join(paths["model_dir"], f"fold_{fold_id}_best.pt"),
            normalize_method=args.normalize_method,
            discretize_t1=args.discretize_t1,
            discretize_t2=args.discretize_t2,
            discretize_reg_mode=args.discretize_reg_mode,
            discretize_cluster_reg_weight=args.discretize_cluster_reg_weight,
            discretize_boundary_reg_weight=args.discretize_boundary_reg_weight,
            discretizer_type=args.discretizer_type,
            discretizer_kwargs={
                "n_bins": args.kbins_n_bins or args.interval_num,
                "encode": args.kbins_encode,
                "strategy": args.kbins_strategy,
                "subsample": args.kbins_subsample,
                "random_state": args.random_seed,
            },
            scale_continuous_features=args.scale_continuous_features,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            tau_decay_start_ratio=args.tau_decay_start_ratio,
            tau_decay_end_ratio=args.tau_decay_end_ratio,
            random_seed=args.random_seed,
            interval_num=args.interval_num,
            dec_level_cnt=args.net_depth,
            ensembles_cnt=args.ensembles_cnt,
            width_cap=args.net_width_cap,
            optimize_metric=args.opt_metric,
            backtrack_dacc=args.backtrack_dacc,
            backtrack_dF1=args.backtrack_dF1,
            coefficients={
                "coef. read_once": args.a1,
                "coef. ent_feat_select": args.a2,
                "coef. ent_dec": args.a3,
                "coef. discretization": args.a4,
                "coef. margin_feat_select": args.a5,
                "margin_feat_select_target": args.feat_select_margin_target,
            },
            verbose=args.verbose,
            print_every=args.print_every,
            learned_dd_dot_edge_label_mode=args.learned_dd_dot_edge_label_mode,
            learned_dd_dot_show_legend=args.learned_dd_dot_show_legend,
            iterative_iterations=args.iterative_iterations,
            iterative_build_final_mdd=args.iterative_build_final_mdd,
            iterative_require_compatible_ordering=args.iterative_require_compatible_ordering,
        )

        fold_summary = {
            "fold": fold_id,
            "train_file": train_file,
            "valid_file": valid_file,
            "test_file": fold_spec["test_file"],
            "training_summary": _compress_training_summary(train_result["training_summary"], save_history=args.save_history),
            "train_metrics": _extract_eval_metrics(train_result["train_evaluation"], "train"),
            "valid_metrics": _extract_eval_metrics(train_result["valid_evaluation"], "valid"),
            "test_metrics": _extract_eval_metrics(train_result["test_evaluation"], "test"),
            "validation_split": valid_split_info,
            "evaluation_timing": {
                "train": _extract_eval_timing(train_result["train_evaluation"]),
                "valid": _extract_eval_timing(train_result["valid_evaluation"]),
                "test": _extract_eval_timing(train_result["test_evaluation"]),
            },
            "iterative_learning": train_result.get("iterative_learning"),
            "learned_dd_artifacts": {
                "train": _extract_learned_dd_artifacts(train_result["train_evaluation"]),
                # "valid": _extract_learned_dd_artifacts(train_result["valid_evaluation"]),
                # "test": _extract_learned_dd_artifacts(train_result["test_evaluation"]),
            },
        }
        fold_results.append(fold_summary)

        for metric_name, metric_value in {
            **fold_summary["train_metrics"],
            **fold_summary["valid_metrics"],
            **fold_summary["test_metrics"],
        }.items():
            aggregated_metrics.setdefault(metric_name, []).append(metric_value)

    summary = {
        "mode": "k-fold",
        "k_fold": k_fold,
        "dataset_file": args.train_file,
        "split_info": {
            "indices_file": split_info["indices_file"],
            "output_dir": split_info["output_dir"],
        },
        "fold_results": fold_results,
        "aggregate_metrics": summarize_numeric_lists(aggregated_metrics),
        "experiment_args": vars(deepcopy(args)),
    }
    write_json(paths["summary_file"], summary)
    return summary


def get_experiment_argument(args_list=None):
    parser = argparse.ArgumentParser(description="DDNet experiment runner")
    parser.add_argument("--mode", choices=["k-fold", "5-fold", "train-test"], default="train-test")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--result_base_path", type=str, default="results/temp_test/")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--rest", type=_str2bool, default=True)

    parser.add_argument("--net_depth", type=int, default=6)
    parser.add_argument("--net_width_cap", type=int, default=None)
    parser.add_argument("--interval_num", type=int, default=3)
    parser.add_argument("--ensembles_cnt", type=int, default=1)

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step_num", "--step", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--normalize_method", choices=["gumbel_softmax", "softmax", "sigmoid"], default="gumbel_softmax")
    parser.add_argument("--discretize_t1", type=float, default=50.0)
    parser.add_argument("--discretize_t2", type=float, default=50.0)
    parser.add_argument("--discretize_reg_mode", choices=["cluster", "spacing", "hybrid"], default="hybrid")
    parser.add_argument("--discretize_cluster_reg_weight", type=float, default=1.0)
    parser.add_argument("--discretize_boundary_reg_weight", type=float, default=0.1)
    parser.add_argument("--discretizer_type", choices=["learned", "kbins"], default="kbins")
    parser.add_argument("--kbins_n_bins", type=int, default=None)
    parser.add_argument("--kbins_encode", choices=["ordinal"], default="ordinal")
    parser.add_argument("--kbins_strategy", choices=["uniform", "quantile", "kmeans"], default="kmeans")
    parser.add_argument("--kbins_subsample", type=int, default=None)
    parser.add_argument("--scale_continuous_features", type=_str2bool, default=True)
    parser.add_argument("--tau_start", type=float, default=10.0)
    parser.add_argument("--tau_end", type=float, default=1.0)
    parser.add_argument("--tau_schedule", choices=["linear", "cosine", "exponential"], default="linear")
    parser.add_argument("--tau_decay_start_ratio", type=float, default=0.0)
    parser.add_argument("--tau_decay_end_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=2024)
    parser.add_argument("--split_seed", type=int, default=2024)
    parser.add_argument("--opt_metric", choices=["acc", "f1_macro"], default="acc")
    parser.add_argument("--backtrack_dacc", type=float, default=1.0)
    parser.add_argument("--backtrack_dF1", type=float, default=1.0)

    parser.add_argument("--a1", type=float, default=5e-3)
    parser.add_argument("--a2", type=float, default=1e-3)
    parser.add_argument("--a3", type=float, default=5e-3)
    parser.add_argument("--a4", type=float, default=5e-4)
    parser.add_argument("--a5", type=float, default=0.0, help="feature-select top1/top2 margin regularization weight")
    parser.add_argument("--feat_select_margin_target", type=float, default=0.2)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_history", type=_str2bool, default=False)
    parser.add_argument("--learned_dd_dot_edge_label_mode", choices=["value", "bin"], default="value")
    parser.add_argument("--learned_dd_dot_show_legend", type=_str2bool, default=True)
    parser.add_argument("--iterative_iterations", "--iterations", type=int, default=1)
    parser.add_argument("--iterative_build_final_mdd", type=_str2bool, default=True)
    parser.add_argument("--iterative_require_compatible_ordering", type=_str2bool, default=False)

    return parser.parse_args(args_list) if args_list is not None else parser.parse_args()


if __name__ == "__main__":
    args = get_experiment_argument()
    if args.net_width_cap is None:
        args.net_width_cap = args.interval_num ** max(args.net_depth - 1, 0) 
    if args.mode == "k-fold":
        result = run_k_fold_cross_validation(args, args.kfold)
    elif args.mode == "5-fold":
        assert args.kfold==5
        result = run_k_fold_cross_validation(args, args.kfold)
    else:
        result = run_train_test(args)

    print(result["mode"])
    if result["mode"] in ["k-fold", "5-fold"]:
        print(result["aggregate_metrics"])
    else:
        print(result["test_metrics"])

