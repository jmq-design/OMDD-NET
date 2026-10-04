from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


EPSILON1 = 1e-3
EPSILON2 = 1e-3
SUPPORTED_REG_MODES = {"cluster", "spacing", "hybrid"}

class Enforced(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > EPSILON2, X, torch.full_like(X, EPSILON2))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    

class SimpBinarizeLayer(nn.Module):
    """
    Simplified AutoInt-style BinarizeLayer.
    """

    def __init__(
        self,
        input_dim: List[int], # [disc_num, con_num]
        i_min: List[float],
        i_max: List[float],
        interval: torch.Tensor, # shape(con_num, interval_num)
        t1: float = 50.0,
        t2: float = 10.0,
        use_not: bool = False,
        hard: bool = True,
        eps: float = EPSILON2,
        device: Optional[str] = None,
        regularization_mode: str = "hybrid",
        cluster_reg_weight: float = 1.0,
        boundary_reg_weight: float = 0.1,
        # 
        # the following parameters are only for aligning with the original API
        interval2=None,
        use_kmeans_loss=None
    ):
        assert isinstance(input_dim, list) and (len(input_dim)==2) 
        disc_num, con_num = input_dim[0], input_dim[1]
        assert con_num==len(i_min)==len(i_max)
        
        if not isinstance(interval, torch.Tensor):
            interval = torch.tensor(interval, dtype=torch.float32, device=device)
        assert (interval.shape)[0] == con_num, f"Expected shape ({con_num}, interval_num), got {interval.shape}"
        assert torch.all(interval >= 0), f"All interval values must be non-negative"

        super(SimpBinarizeLayer, self).__init__()

        self.input_dim = input_dim
        self.disc_num = disc_num
        self.con_num = con_num
        self.i_min = i_min
        self.i_max = i_max

        self.interval_number = len(interval[0])
        self.hard = hard
        self.use_not = use_not
        self.t1 = t1 if t1 is not None else 50.0
        self.t2 = t2
        self.eps = eps
        self.minimum_gap_ratio = 0.01
        self.spacing_reg_weight = 0.1
        self.latest_kmeans_loss = None
        self.regularization_mode = regularization_mode
        self.cluster_reg_weight = cluster_reg_weight
        self.boundary_reg_weight = boundary_reg_weight
        if self.regularization_mode not in SUPPORTED_REG_MODES:
            raise ValueError(
                f"regularization_mode must be one of {sorted(SUPPORTED_REG_MODES)}, "
                f"got {self.regularization_mode!r}"
            )

        if self.use_not:
            self.disc_num *= 2

        self.output_dim = self.disc_num + self.interval_number * self.con_num
        self.layer_type = "binarization"
        self.dim2id = {i: i for i in range(self.output_dim)}

        device = device if device is None else interval.device

        i_min_tensor = torch.tensor(i_min, dtype=torch.float32, device=device).view(self.con_num, 1)
        i_max_tensor = torch.tensor(i_max, dtype=torch.float32, device=device).view(self.con_num, 1)
        self.register_buffer("i_min_tensor", i_min_tensor)
        self.register_buffer("i_max_tensor", i_max_tensor)


        ''' define trainable parameters '''
        self.params_interval = Parameter(interval)


    def freeze_parameters(self, reset_uniform_boundaries:bool=False):
        with torch.no_grad():
            if reset_uniform_boundaries:    
                assert self.interval_number>=2

                _interval_unit = [(val_max-val_min)/(self.interval_number-1) for val_max, val_min in zip(self.i_max, self.i_min)] # shape(cont_features_cnt)
                _interval = [ [v/2 for v in _interval_unit] ] \
                            +  [ _interval_unit for _ in range(self.interval_number-2) ] \
                            +  [ [v/2 for v in _interval_unit] ] # shape(interval_num, cont_features_cnt)
                self.params_interval = Parameter(torch.tensor(_interval).T) # shape(cont_features_cnt, interval_num)


            self.params_interval.requires_grad = False

    @staticmethod
    def _inverse_softplus(x):
        return torch.log(torch.expm1(x).clamp_min(1e-12))

    def _get_positive_intervals(self):

        interval_pos = Enforced.apply(self.params_interval)
        return interval_pos

    def _get_centers(self):
        interval_pos = self._get_positive_intervals()
        centers = self.i_min_tensor + torch.cumsum(interval_pos, dim=1)

        return centers

    def _resolve_temperature(self, tau=None):
        return max(float(self.t2), EPSILON1)

    def regularization(self):
        centers = self._get_centers()
        value_range = (self.i_max_tensor - self.i_min_tensor).clamp_min(self.eps)

        if self.latest_kmeans_loss is None:
            kmeans_loss = centers.new_zeros(())
        else:
            kmeans_loss = self.latest_kmeans_loss

        first_center = centers[:, :1]
        lower_violation = (torch.relu(self.i_min_tensor - first_center) / value_range).pow(2)
        last_center = centers[:, -1:]
        upper_violation = (torch.relu(last_center - self.i_max_tensor) / value_range).pow(2)
        boundary_loss = lower_violation.mean() + upper_violation.mean()

        if self.interval_number > 1:
            center_gap = centers[:, 1:] - centers[:, :-1]
            minimum_gap = self.minimum_gap_ratio * value_range
            spacing_loss = (
                torch.relu(minimum_gap - center_gap) / value_range
            ).pow(2).mean()
        else:
            spacing_loss = centers.new_zeros(())

        if self.regularization_mode == "cluster":
            return self.cluster_reg_weight * kmeans_loss + self.spacing_reg_weight * spacing_loss
        if self.regularization_mode == "spacing":
            return spacing_loss + boundary_loss
        return (
            self.cluster_reg_weight * kmeans_loss
            + self.spacing_reg_weight * spacing_loss
            + self.boundary_reg_weight * boundary_loss
        )

    def forward(self, x, tau=None, test_flag=False):

        if self.con_num <= 0:
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            return x

        x_disc = x[:, 0:self.input_dim[0]]
        x_cont = x[:, self.input_dim[0]:]

        if self.use_not:
            x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)

        temperature = max(float(self.t2), EPSILON1)

        x_cont_expanded = x_cont.unsqueeze(-1) # shape(batch_size, con_num, 1)

        centers = self._get_centers()
        centers_expanded = centers.unsqueeze(0) # shape(1, con_num, interval_number)

        distance = (x_cont_expanded - centers_expanded) ** 2 # shape(batch_size, con_num, interval_number)

        value_range = (self.i_max_tensor - self.i_min_tensor).clamp_min(self.eps)
        cluster_distance = distance / value_range.unsqueeze(0).pow(2)   
        cluster_temperature = max(float(self.t1), EPSILON1)
        cluster_distance_min = cluster_distance.min(dim=-1, keepdim=True).values
        cluster_logits = -cluster_temperature * (cluster_distance - cluster_distance_min)
        cluster_probs = F.softmax(cluster_logits, dim=-1)
        self.latest_kmeans_loss = (cluster_distance * cluster_probs.detach()).sum(dim=-1).mean()

        distance_min = distance.min(dim=-1, keepdim=True).values
        logits = -temperature * (distance - distance_min)
        probs = F.softmax(logits, dim=-1)

        use_hard = self.hard or test_flag
        if use_hard:
            bin_ids = torch.argmax(probs, dim=-1)
            hard_onehot = F.one_hot(
                bin_ids,
                num_classes=self.interval_number
            ).type_as(probs)
            out = hard_onehot.detach() + probs - probs.detach()
        else:
            out = probs

        out = out.reshape(x.shape[0], -1)   # shape(batch_size, con_num * interval_number)

        return torch.cat((x_disc, out), dim=1), self.latest_kmeans_loss

    def binarized_forward(self, x):
        with torch.no_grad():
            return self.forward(x)

    def get_bound_name(self, feature_name, mean=None, std=None, return_centers=True):
        """
        Return learned discretization names according to the real forward boundary.
        """

        bound_name = []
        centers_info = {}

        # discrete features
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])

        if self.use_not:
            for i in range(self.input_dim[0]):
                bound_name.append("~" + feature_name[i])

        if self.con_num <= 0:
            if return_centers:
                return bound_name, centers_info
            return bound_name

        def _to_original_scale(v, fi_name):
            if mean is not None and std is not None:
                return float(v) * float(std[fi_name]) + float(mean[fi_name])
            return float(v)

        with torch.no_grad():
            centers = self._get_centers().detach().cpu()
            intervals = self._get_positive_intervals().detach().cpu()

            for feat_idx in range(self.con_num):
                fi_name = feature_name[self.input_dim[0] + feat_idx]

                feat_centers = centers[feat_idx].tolist()
                feat_intervals = intervals[feat_idx].tolist()

                # boundaries between adjacent centers
                feat_boundaries = []
                for j in range(self.interval_number - 1):
                    b = 0.5 * (feat_centers[j] + feat_centers[j + 1])
                    feat_boundaries.append(b)

                # convert to original scale if needed
                centers_scaled = [
                    _to_original_scale(c, fi_name)
                    for c in feat_centers
                ]

                boundaries_scaled = [
                    _to_original_scale(b, fi_name)
                    for b in feat_boundaries
                ]

                intervals_scaled = [
                    float(v) * float(std[fi_name]) if mean is not None and std is not None else float(v)
                    for v in feat_intervals
                ]

                centers_info[fi_name] = {
                    "centers": centers_scaled,
                    "boundaries": boundaries_scaled,
                    "intervals": intervals_scaled,
                }

                # Build interval names according to real argmin/argmax behavior.
                for j in range(self.interval_number):
                    if self.interval_number == 1:
                        bound_name.append("{}: all values".format(fi_name))

                    elif j == 0:
                        # x <= b0
                        cr = boundaries_scaled[0]
                        bound_name.append("{} <= {:.3f}".format(fi_name, cr))

                    elif j == self.interval_number - 1:
                        # b_{last} < x
                        cl = boundaries_scaled[-1]
                        bound_name.append("{:.3f} < {}".format(cl, fi_name))

                    else:
                        # b_{j-1} < x <= b_j
                        cl = boundaries_scaled[j - 1]
                        cr = boundaries_scaled[j]
                        bound_name.append(
                            "{:.3f} < {} <= {:.3f}".format(cl, fi_name, cr)
                        )

        if return_centers:
            return bound_name, centers_info

        return bound_name

