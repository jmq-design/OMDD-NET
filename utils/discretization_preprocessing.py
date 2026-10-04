from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import KBinsDiscretizer


class KBinsDiscretizeLayer(nn.Module):
    """
    Non-differentiable static KBins discretization layer for DDNet.
    """

    def __init__(
        self,
        cont_features_cnt: int,
        interval_num: int,
        featvals_min: Sequence[float],
        featvals_max: Sequence[float],
        # 
        fit_data: Optional[Any] = None,
        n_bins: Optional[int] = None,
        encode: str = "ordinal",
        strategy: str = "kmeans",
        subsample: Optional[int] = None,
        random_state: Optional[int] = 42, 
        device: str = "cpu",
        verbose: bool = False,

        # Allow invalid bins as alignment placeholders
        keep_invalid_bins: bool = True,
        invalid_bin_mode: str = "repeat",  # "repeat" or "jitter"
        clip_to_fitted_range: bool = True,

        **kwargs,
    ):
        super().__init__()

        if n_bins is None:
            n_bins = interval_num
        if isinstance(n_bins, int) and n_bins != interval_num:
            raise ValueError(
                "DDNet topology interval_num and KBins n_bins must match. "
                f"Got interval_num={interval_num}, n_bins={n_bins}."
            )
        if not isinstance(n_bins, int):
            raise ValueError("KBinsDiscretizeLayer currently expects an integer n_bins.")

        if encode != "ordinal":
            raise ValueError(
                "KBinsDiscretizeLayer returns DDNet one-hot output itself; "
                "set KBins encode='ordinal'."
            )

        # 
        if invalid_bin_mode not in {"repeat", "jitter"}:
            raise ValueError(
                "invalid_bin_mode must be either 'repeat' or 'jitter'. "
                f"Got {invalid_bin_mode}."
            )

        self.cont_features_cnt = cont_features_cnt
        self.con_num = cont_features_cnt
        self.interval_num = int(n_bins)
        # self.interval_number = int(n_bins)
        self.featvals_min = list(featvals_min)
        self.featvals_max = list(featvals_max)
        self.device = device
        self.verbose = verbose
        self.eps = 1e-6

        # store the strategy of keeping invalid bins
        self.keep_invalid_bins = keep_invalid_bins
        self.invalid_bin_mode = invalid_bin_mode
        self.clip_to_fitted_range = clip_to_fitted_range

        self.discretizer_config = {
            "n_bins": int(n_bins),
            "encode": encode,
            "strategy": strategy,
            "subsample": subsample,
            "random_state": random_state,
            **kwargs,
        }

        self.discretizer = KBinsDiscretizer(**self.discretizer_config)

        self.register_buffer(
            "i_min_tensor",
            torch.tensor(featvals_min, dtype=torch.float32, device=device).view(cont_features_cnt, 1),
        )
        self.register_buffer(
            "i_max_tensor",
            torch.tensor(featvals_max, dtype=torch.float32, device=device).view(cont_features_cnt, 1),
        )

        self.register_buffer(
            "bin_edges_tensor",
            torch.empty(cont_features_cnt, self.interval_num + 1, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "centers_tensor",
            torch.empty(cont_features_cnt, self.interval_num, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "is_fitted_tensor",
            torch.tensor(False, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "valid_bin_mask_tensor",
            torch.ones(cont_features_cnt, self.interval_num, dtype=torch.bool, device=device),
        )

        self.register_buffer(
            "actual_n_bins_tensor",
            torch.zeros(cont_features_cnt, dtype=torch.long, device=device),
        )

        self.register_buffer(
            "padded_feature_mask_tensor",
            torch.zeros(cont_features_cnt, dtype=torch.bool, device=device),
        )


        self.register_buffer(
            "fitted_low_tensor",
            torch.empty(cont_features_cnt, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "fitted_high_tensor",
            torch.empty(cont_features_cnt, dtype=torch.float32, device=device),
        )

        if fit_data is not None:
            self.fit(fit_data)

    @property
    def binarizeLayer(self):
        return self

    def fit(self, data_x: Any):
        x_np = self._to_numpy(data_x)
        if x_np.ndim != 2 or x_np.shape[1] != self.cont_features_cnt:
            raise ValueError(
                f"Expected fit data shape [n_samples, {self.cont_features_cnt}], got {x_np.shape}."
            )

        self.discretizer.fit(x_np)

        actual_bins = np.asarray(self.discretizer.n_bins_, dtype=np.int64)

        edges_list = []
        centers_list = []
        valid_mask_list = []
        padded_feature_mask = []
        fitted_lows = []
        fitted_highs = []

        for feat_idx in range(self.cont_features_cnt):
            raw_edges = np.asarray(self.discretizer.bin_edges_[feat_idx], dtype=np.float32)
            raw_edges = self._sanitize_raw_edges(raw_edges, x_np[:, feat_idx], feat_idx)

            actual_k = int(actual_bins[feat_idx])
            actual_k_from_edges = len(raw_edges) - 1

            if actual_k_from_edges != actual_k:
                actual_k = actual_k_from_edges
                raise ValueError(f"actual_k_from_edges({actual_k_from_edges}) != actual_k({actual_k})")

            fitted_lows.append(float(raw_edges[0]))
            fitted_highs.append(float(raw_edges[-1]))

            if actual_k == self.interval_num and len(raw_edges) == self.interval_num + 1:
                fixed_edges = raw_edges
                valid_mask = np.ones(self.interval_num, dtype=np.bool_)
                is_padded = False
            else:
                if not self.keep_invalid_bins:
                    raise ValueError(
                        "KBinsDiscretizer produced fewer bins for at least one feature. "
                        f"Expected {self.interval_num} bins for every feature, "
                        f"got {actual_bins.tolist()}."
                    )

                fixed_edges, valid_mask = self._pad_edges_to_fixed_n_bins(
                    raw_edges=raw_edges,
                    feat_idx=feat_idx,
                )
                is_padded = True

            if len(fixed_edges) != self.interval_num + 1:
                raise RuntimeError(
                    f"Internal error: feature {feat_idx} fixed_edges length should be "
                    f"{self.interval_num + 1}, got {len(fixed_edges)}."
                )

            centers = 0.5 * (fixed_edges[:-1] + fixed_edges[1:])

            edges_list.append(fixed_edges.astype(np.float32))
            centers_list.append(centers.astype(np.float32))
            valid_mask_list.append(valid_mask)
            padded_feature_mask.append(is_padded)

        edges = np.stack(edges_list, axis=0)
        centers = np.stack(centers_list, axis=0)
        valid_mask_np = np.stack(valid_mask_list, axis=0)
        padded_feature_mask_np = np.asarray(padded_feature_mask, dtype=np.bool_)

        with torch.no_grad():
            self.bin_edges_tensor.copy_(
                torch.tensor(edges, dtype=torch.float32, device=self.bin_edges_tensor.device)
            )
            self.centers_tensor.copy_(
                torch.tensor(centers, dtype=torch.float32, device=self.centers_tensor.device)
            )
            self.valid_bin_mask_tensor.copy_(
                torch.tensor(valid_mask_np, dtype=torch.bool, device=self.valid_bin_mask_tensor.device)
            )
            self.actual_n_bins_tensor.copy_(
                torch.tensor(actual_bins, dtype=torch.long, device=self.actual_n_bins_tensor.device)
            )
            self.padded_feature_mask_tensor.copy_(
                torch.tensor(
                    padded_feature_mask_np,
                    dtype=torch.bool,
                    device=self.padded_feature_mask_tensor.device,
                )
            )
            self.fitted_low_tensor.copy_(
                torch.tensor(fitted_lows, dtype=torch.float32, device=self.fitted_low_tensor.device)
            )
            self.fitted_high_tensor.copy_(
                torch.tensor(fitted_highs, dtype=torch.float32, device=self.fitted_high_tensor.device)
            )
            self.is_fitted_tensor.fill_(True)

        if self.verbose:
            print("* KBinsDiscretizeLayer fit finished.")
            print(f"\t- expected bins per feature: {self.interval_num}")
            print(f"\t- actual bins from KBins: {actual_bins.tolist()}")
            print(f"\t- padded features: {np.where(padded_feature_mask_np)[0].tolist()}")
            print(f"\t- invalid_bin_mode: {self.invalid_bin_mode}")
            print(f"\t- clip_to_fitted_range: {self.clip_to_fitted_range}")

        return self

    def _sanitize_raw_edges(
        self,
        raw_edges: np.ndarray,
        feature_values: np.ndarray,
        feat_idx: int,
    ) -> np.ndarray:
        raw_edges = np.asarray(raw_edges, dtype=np.float32).reshape(-1)

        if len(raw_edges) >= 2 and np.all(np.isfinite(raw_edges)):
            return raw_edges

        if not np.all(np.isfinite(feature_values)):
            raise ValueError(f"feature {feat_idx} contains infinite values.")

        if np.array_equal(raw_edges, [-np.inf, np.inf]):
            lo = float(np.min(feature_values))
            hi = float(np.max(feature_values))
            if lo==hi:
                return np.asarray([lo, hi], dtype=np.float32)
        
        raise ValueError(f"Unknown error occurs when dealing with feature {feat_idx}.")

    def _pad_edges_to_fixed_n_bins(
        self,
        raw_edges: np.ndarray,
        feat_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        raw_edges = np.asarray(raw_edges, dtype=np.float32).reshape(-1)

        if len(raw_edges) < 2:
            v = float(raw_edges[0]) if len(raw_edges) == 1 else 0.0
            raw_edges = np.asarray([v, v], dtype=np.float32)

        actual_k = len(raw_edges) - 1

        if actual_k > self.interval_num:
            raise ValueError(
                f"Feature {feat_idx} has actual_k={actual_k}, "
                f"which is larger than interval_num={self.interval_num}."
            )

        missing = self.interval_num - actual_k

        valid_mask = np.asarray(
            [True] * actual_k + [False] * missing,
            dtype=np.bool_,
        )

        if missing == 0:
            return raw_edges.astype(np.float32), valid_mask

        if self.invalid_bin_mode == "repeat":
            pad_edges = np.full(missing, raw_edges[-1], dtype=np.float32)
            fixed_edges = np.concatenate([raw_edges, pad_edges], axis=0)

        elif self.invalid_bin_mode == "jitter":
            fixed_edges_list = []

            last = None
            for v in raw_edges:
                v = np.float32(v)

                if last is not None and not (v > last):
                    v = self._next_float32(last)

                fixed_edges_list.append(v)
                last = v

            for _ in range(missing):
                last = self._next_float32(last)
                fixed_edges_list.append(last)

            fixed_edges = np.asarray(fixed_edges_list, dtype=np.float32)

        else:
            raise ValueError(f"Unknown invalid_bin_mode={self.invalid_bin_mode}.")

        if len(fixed_edges) != self.interval_num + 1:
            raise RuntimeError(
                f"Feature {feat_idx}: expected {self.interval_num + 1} fixed edges, "
                f"got {len(fixed_edges)}."
            )

        return fixed_edges.astype(np.float32), valid_mask

    @staticmethod
    def _next_float32(x: float) -> np.float32:
        x = np.float32(x)
        y = np.float32(np.nextafter(x, np.float32(np.inf)))

        if not np.isfinite(y):
            raise ValueError(f"Cannot create next float32 after {x}.")

        return y

    def reset_verbose(self, verbose):
        self.verbose = verbose

    def _to_numpy(self, data_x: Any) -> np.ndarray:
        if isinstance(data_x, torch.Tensor):
            data_x = data_x.detach().cpu().numpy()
        return np.asarray(data_x, dtype=np.float32)

    def _get_centers(self):
        self._require_fitted()
        return self.centers_tensor

    def _get_valid_bin_mask(self):
        self._require_fitted()
        return self.valid_bin_mask_tensor

    def _get_padded_feature_mask(self):
        self._require_fitted()
        return self.padded_feature_mask_tensor

    def _get_bin_ids(self, data_x: torch.Tensor):
        self._require_fitted()

        values = data_x.contiguous()

        if self.clip_to_fitted_range:
            lo = self.fitted_low_tensor.to(device=data_x.device, dtype=data_x.dtype).view(1, -1)
            hi = self.fitted_high_tensor.to(device=data_x.device, dtype=data_x.dtype).view(1, -1)
            values = torch.minimum(torch.maximum(values, lo), hi)

        internal_edges = self.bin_edges_tensor[:, 1:-1].to(
            device=data_x.device,
            dtype=data_x.dtype,
        )

        return (values.unsqueeze(-1) > internal_edges.unsqueeze(0)).sum(dim=-1).long()

    def _require_fitted(self):
        if not bool(self.is_fitted_tensor.item()):
            raise RuntimeError("KBinsDiscretizeLayer must be fitted before forward/evaluation.")

    def forward(self, data_x, tau=None, test_flag=False):
        bin_ids = self._get_bin_ids(data_x)

        one_hot = F.one_hot(bin_ids, num_classes=self.interval_num).type_as(data_x)
        out = one_hot.reshape(data_x.shape[0], -1)

        if self.verbose:
            with torch.no_grad():
                print("* KBinsDiscretizeLayer forward.")
                print("\t- bin_ids: ", bin_ids)
                print("\t- discretized_data_x: ", out)
                print("\t- valid_bin_mask: ", self.valid_bin_mask_tensor)

        return out

    def regularization(self, coefficients: Dict[str, float]):
        return torch.tensor(0.0, device=self.bin_edges_tensor.device), {}

    def get_discretized_feats_name(
        self,
        feats_name: List[str],
        mean=None,
        std=None,
        return_centers: bool = True,
    ):
        self._require_fitted()
        bound_name = []
        centers_info = {}

        def _to_original_scale(v, fi_name):
            if mean is not None and std is not None:
                return float(v) * float(std[fi_name]) + float(mean[fi_name])
            return float(v)

        edges = self.bin_edges_tensor.detach().cpu()
        centers = self.centers_tensor.detach().cpu()

        valid_mask = self.valid_bin_mask_tensor.detach().cpu()
        actual_n_bins = self.actual_n_bins_tensor.detach().cpu()
        padded_feature_mask = self.padded_feature_mask_tensor.detach().cpu()

        for feat_idx in range(self.cont_features_cnt):
            fi_name = feats_name[feat_idx]

            feat_edges = edges[feat_idx].tolist()
            feat_centers = centers[feat_idx].tolist()

            edges_scaled = [_to_original_scale(v, fi_name) for v in feat_edges]
            centers_scaled = [_to_original_scale(v, fi_name) for v in feat_centers]

            valid_bins_list = [bool(x) for x in valid_mask[feat_idx].tolist()]

            centers_info[fi_name] = {
                "centers": centers_scaled,
                "boundaries": edges_scaled[1:-1],
                "intervals": [
                    float(edges_scaled[j + 1] - edges_scaled[j])
                    for j in range(self.interval_num)
                ],
                "bin_edges": edges_scaled,
                "method": "KBinsDiscretizer",
                "strategy": self.discretizer_config["strategy"],

                "actual_n_bins": int(actual_n_bins[feat_idx].item()),
                "expected_n_bins": int(self.interval_num),
                "valid_bins": valid_bins_list,
                "has_padded_invalid_bins": bool(padded_feature_mask[feat_idx].item()),
                "invalid_bin_mode": self.invalid_bin_mode,
                "clip_to_fitted_range": self.clip_to_fitted_range,
            }

            for bin_idx in range(self.interval_num):

                if not bool(valid_mask[feat_idx, bin_idx].item()):
                    bound_name.append(
                        "{}: invalid padded bin {} (alignment only)".format(
                            fi_name,
                            bin_idx,
                        )
                    )
                    continue

                if self.interval_num == 1:
                    bound_name.append("{}: all values".format(fi_name))
                elif bin_idx == 0:
                    bound_name.append("{} <= {:.3f}".format(fi_name, edges_scaled[1]))
                elif bin_idx == self.interval_num - 1:
                    bound_name.append("{:.3f} < {}".format(edges_scaled[-2], fi_name))
                else:
                    bound_name.append(
                        "{:.3f} < {} <= {:.3f}".format(
                            edges_scaled[bin_idx],
                            fi_name,
                            edges_scaled[bin_idx + 1],
                        )
                    )

        if return_centers:
            return bound_name, centers_info
        return bound_name
