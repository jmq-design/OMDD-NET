import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")

from torch.nn.parameter import Parameter
import time
import random
import tempfile
import numpy as np
import pandas as pd
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader

from utils.data import ClassificationDataset, RegressionDataset
from utils.discretization_preprocessing import KBinsDiscretizeLayer
from utils.layers import DiscretizeLayer, CategFeatureSelectLayer, RegressionLayer, DecisionLayer
from utils.netTopology import OrderedDDNetTopology
from utils.metrics import accuracy, classification_score, root_mean_squared_error
from utils.MDD import MDD
from utils.MDD_mdd_bridge import MDD_to_mdd, mdd_predict, mdd_to_MDD


# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.metrics import mean_squared_error

from typing import Any, Dict, List, Literal, Optional, Tuple, Set
from torch import Tensor


class ProbCrossEntropyLoss(nn.Module):
    """
    Cross-entropy for probability inputs.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob: Tensor, target_prob: Tensor) -> Tensor:
        pred_prob = pred_prob.clamp_min(self.eps)
        return -(target_prob * torch.log(pred_prob)).sum(dim=1).mean()


def _build_adamw_optimizer(model: nn.Module, lr: float) -> torch.optim.AdamW:
    """Keep learned discretization intervals free from AdamW shrinkage."""
    interval_params = []
    regular_params = []
    for name, param in model.named_parameters():
        if name.endswith("params_interval"):
            interval_params.append(param)
        else:
            regular_params.append(param)

    parameter_groups = []
    if regular_params:
        parameter_groups.append({"params": regular_params})
    if interval_params:
        parameter_groups.append({"params": interval_params, "weight_decay": 0.0})
    return torch.optim.AdamW(parameter_groups, lr=lr)


class DDNet(nn.Module):
    '''
        OrderedDDNet (OMDDNet).

    '''
    def __init__(self, 
                net_topology : OrderedDDNetTopology,  
                # features_cnt : int,
                # cont_features_cnt : int,
                # interval_num_for_discretize : int,
                cont_featvals_min : List[float],
                cont_featvals_max : List[float],
                coefficients : Dict[str, float],
                discretize_t2: float = 10.0,
                discretize_t1: float = 50.0,
                discretize_reg_mode: str = "hybrid",
                discretize_cluster_reg_weight: float = 1.0,
                discretize_boundary_reg_weight: float = 0.1,
                discretizer_type: Literal["learned", "kbins"] = "learned",
                discretizer_kwargs: Optional[Dict[str, Any]] = None,
                discretizer_fit_data: Optional[Tensor] = None,
                preprocessing_feature_scaling: Optional[Dict[str, Any]] = None,
                normalize_method : Literal["gumbel_softmax", "softmax"]="gumbel_softmax",
                tau_start: float=1.0,
                device : str='cuda:0' if torch.cuda.is_available() else 'cpu',
                verbose=False):
        '''
            :param - net_topology:   the number of the vertices of DD at level l
            :param - normalize_method:    normalization method for training. When testing, it shifts to "softmax" by default. 

        '''
        super(DDNet, self).__init__()
        self.net_topology = net_topology
        # self.features_cnt = features_cnt
        # self.cont_features_cnt = cont_features_cnt
        # self.interval_num_for_discretize = interval_num_for_discretize
        self.cont_featvals_min = cont_featvals_min
        self.cont_featvals_max = cont_featvals_max
        self.coefficients = coefficients
        self.normalize_method = normalize_method
        self.is_regression = net_topology.is_regression
        self.discretizer_type = discretizer_type
        self.discretizer_kwargs = dict(discretizer_kwargs or {})
        self.preprocessing_feature_scaling = dict(preprocessing_feature_scaling or {})

        self.verbose = verbose
        self.device = device

        self.tau=Parameter(torch.tensor(tau_start, dtype=torch.float32, device=device), requires_grad=False)

        features_cnt, _, interval_num_for_discretize = net_topology.get_DiscretizeLayer_shape()
        discretize_min = cont_featvals_min
        discretize_max = cont_featvals_max
        if discretizer_type == "learned":
            self.discretizeLayer = DiscretizeLayer(
                                                    cont_features_cnt=features_cnt,
                                                    interval_num=interval_num_for_discretize,
                                                    featvals_min=discretize_min,
                                                    featvals_max=discretize_max,
                                                    device=device,
                                                    t1=discretize_t1,
                                                    t2=discretize_t2,
                                                    discretize_reg_mode=discretize_reg_mode,
                                                    discretize_cluster_reg_weight=discretize_cluster_reg_weight,
                                                    discretize_boundary_reg_weight=discretize_boundary_reg_weight,
                                                    verbose=verbose)
        elif discretizer_type == "kbins":
            training_fit_data = None
            if discretizer_fit_data is not None:
                training_fit_data = discretizer_fit_data.to(device)
            self.discretizeLayer = KBinsDiscretizeLayer(
                                                    cont_features_cnt=features_cnt,
                                                    interval_num=interval_num_for_discretize,
                                                    featvals_min=discretize_min,
                                                    featvals_max=discretize_max,
                                                    fit_data=training_fit_data,
                                                    device=device,
                                                    verbose=verbose,
                                                    **self.discretizer_kwargs)
        else:
            raise ValueError("discretizer_type must be one of ['learned', 'kbins']")

        discretized_features_cnt, dec_levels_cnt, features_domain_size = net_topology.get_CategFeatureSelectLayer_shape()
        self.featSelectLayer = CategFeatureSelectLayer(
                                                        features_cnt=discretized_features_cnt,
                                                        k_valued=features_domain_size,
                                                        # net_topology=net_topology, 
                                                        dec_levels_cnt=dec_levels_cnt, 
                                                        device=device,
                                                        normalize_method=normalize_method,
                                                        verbose=verbose)


        in_height, cur_out_height, inout_width = self.net_topology.get_EnsLayer_shape()
        self.ensembleLayer = DecisionLayer(
                                            in_height=in_height,
                                            cur_out_height=cur_out_height,
                                            inout_width=inout_width,
                                            skip_out_height=None,
                                            device=device,
                                            normalize_method="softmax",     
                                            verbose=verbose)    
        
        self.decisionLayers = nn.ModuleList()
        for ith in range(self.net_topology.get_DecisionLayers_cnt()):
            in_height, cur_out_height, inout_width, skip_out_height = self.net_topology.get_ith_DecLayer_shape(ith)
            decisionLayer = DecisionLayer(
                                        in_height=in_height,
                                        cur_out_height=cur_out_height,
                                        inout_width=inout_width,
                                        skip_out_height=skip_out_height,
                                        device=device,
                                        normalize_method=normalize_method,
                                        verbose=verbose)
            self.decisionLayers.append(decisionLayer)

        self.decisionLayers_cnt = len(self.decisionLayers)
        self.init_reach_prob4in = torch.tensor([[[1.0]]], requires_grad=False).to(device)   # shape(1[To_Expand_to__batch_size], 1, in_height=1)
        self.init_branch_prob4in = torch.tensor([[[1.0]]], requires_grad=False).to(device)  # shape(1[To_Expand_to__batch_size], in_height=1, inout_width=1)

        if self.is_regression:
            features_cnt, hidden_dim_regress, output_dim = net_topology.get_RegressLayer_shape()
            self.regressionLayer = RegressionLayer(
                                                in_dim=features_cnt, 
                                                hidden_dim=hidden_dim_regress, 
                                                out_dim=output_dim, 
                                                activation_func_method=None,
                                                device=device,
                                                verbose=verbose)

    def reset_verbose(self, verbose):
        self.verbose = verbose
        self.discretizeLayer.reset_verbose(verbose)
        self.ensembleLayer.reset_verbose(verbose)
        self.featSelectLayer.reset_verbose(verbose)
        for decisionLayer in self.decisionLayers:
            decisionLayer.reset_verbose(verbose)
        if self.is_regression:
            self.regressionLayer.reset_verbose(verbose)

    def reset_tau(self, tau: float):
        '''
            the temperature parameter used in gumbel_softmax when the `normalize_method` is set to "gumbel_softmax"
        '''
        self.tau = tau

    def forward(self, data_x, test_flag=False, interpret_flag=False):
        '''
            Forward process of the DDNet.

            :param - test_flag:   if True, apply the softmax function as the normalize_method to get an encoding
            :param - interpret_flag:   if True, the forward will perform at the interpreted faithful encoding; otherwise the original (non-faithful) encoding.  Note that test_flag is restricted to True when interpret_flag=True to ensure determinism.

        '''
        assert not (test_flag==False and interpret_flag==True), f"Note that test_flag is restricted to True when interpret_flag=True to ensure determinism. But (test_flag={test_flag}, interpret_flag={interpret_flag}) got."
        tau = self.tau

        if self.verbose:
            print("** DDNet Forward.")

        discretized_data_x = self.discretizeLayer(data_x, tau=tau, test_flag=test_flag)    

        branch_prob4in_per_layer = self.featSelectLayer(discretized_data_x, tau=tau, test_flag=test_flag, interpret_flag=interpret_flag)

        reach_prob4skip_out_2_dec_levels = dict() # reach_prob4skip_out linked to a decision layer
        reach_prob4cur_out, reach_prob4skip_out = self.ensembleLayer(reach_prob4in=self.init_reach_prob4in,
                                                branch_prob4in=self.init_branch_prob4in,
                                                tau=tau, test_flag=test_flag)


        if self.verbose:
            with torch.no_grad():
                # print("** DDNet Forward.")
                # print("* featSelectLayer forward.")
                # print("\t- branch_prob4in_per_layer: ", branch_prob4in_per_layer)
                print("* ensembleLayer forward.")
                print("\t- reach_prob4in: ", self.init_reach_prob4in)
                print("\t- branch_prob4in: ", self.init_branch_prob4in)
                print("\t- reach_prob4cur_out: ", reach_prob4cur_out)
                print("\t- reach_prob4skip_out: ", reach_prob4skip_out)
           

        for ith, decisionLayer in enumerate(self.decisionLayers):
            reach_prob4out = reach_prob4cur_out if (ith not in reach_prob4skip_out_2_dec_levels.keys()) \
                            else (reach_prob4cur_out+sum(reach_prob4skip_out_2_dec_levels[ith]))    
            reach_prob4cur_out, reach_prob4skip_out = decisionLayer(reach_prob4in=reach_prob4out,
                                                                branch_prob4in=branch_prob4in_per_layer[ith],
                                                                tau=tau, test_flag=test_flag, interpret_flag=interpret_flag)

            if (reach_prob4skip_out is not None) and (self.net_topology.get_ith_DecLayer_residual_link(ith) is not None):
                layer_tolink= self.net_topology.get_ith_DecLayer_residual_link(ith)
                if layer_tolink in reach_prob4skip_out_2_dec_levels.keys():
                    reach_prob4skip_out_2_dec_levels[layer_tolink].append(reach_prob4skip_out)
                else:
                    reach_prob4skip_out_2_dec_levels[layer_tolink] = [reach_prob4skip_out]

            if self.verbose:
                with torch.no_grad():
                    print(f"* decisionLayer {ith} forward.")
                    print("\t- reach_prob4in: ", reach_prob4out)
                    print("\t- branch_prob4in: ", branch_prob4in_per_layer[ith])
                    print("\t- reach_prob4cur_out: ", reach_prob4cur_out)
                    print("\t- reach_prob4skip_out: ", reach_prob4skip_out)

        
        terminal_dec_level_id = self.net_topology.get_DecLevel_cnt()
        classification_prediction = reach_prob4cur_out if (terminal_dec_level_id not in reach_prob4skip_out_2_dec_levels.keys()) \
                            else (reach_prob4cur_out+sum(reach_prob4skip_out_2_dec_levels[terminal_dec_level_id]))   #  shape(batch_size, 1, out_height)
        class_pred_prob = classification_prediction[:,0,:] #  shape(batch_size, out_height)

        if self.is_regression:
            regress_x = self.regressionLayer(data_x)        # prepare to compute regress_pred

            regress_pred = regress_x * class_pred_prob   

            regress_pred = regress_pred.sum(dim=-1).squeeze()  #  shape(batch_size)
        
        return regress_pred if self.is_regression else class_pred_prob

    def regularization(self):
        coefficients = self.coefficients
        
        reg_loss = torch.tensor(0.0, device=self.device)

        discretization_loss, _disc_loss_details = self.discretizeLayer.regularization(coefficients)
        feat_select_loss, _feat_select_loss_details = self.featSelectLayer.regularization(coefficients)

        _reg_loss = torch.tensor(0.0, device=self.device)
        decisionLayers_cnt = len(self.decisionLayers)
        for decisionLayer in self.decisionLayers:
            _reg_loss += decisionLayer.regularization(coefficients)
        ent_dec_loss = _reg_loss / decisionLayers_cnt 
        
        reg_loss += discretization_loss + feat_select_loss + ent_dec_loss 

        _ent_dec_loss = float(ent_dec_loss.detach().cpu().item())

        reg_loss_details = {
        # "disc_loss": _discretization_loss,
        **_disc_loss_details,
        **_feat_select_loss_details,
        "ent_dec_loss": _ent_dec_loss,
        }

        return reg_loss, reg_loss_details

    def freeze_discretizeLayer(self):
        self.discretizeLayer.freeze_parameters(reset_uniform_boundaries=True)

    def fit_discretizer(self, data_x: Tensor):
        if not hasattr(self.discretizeLayer, "fit"):
            return
        self.discretizeLayer.fit(data_x.to(self.device))

    def get_discretized_feats_name(self, feats_name : List[str], return_centers: bool = True):
        mean = None
        std = None
        if self.preprocessing_feature_scaling.get("enabled", False):
            mean = dict(zip(feats_name, self.preprocessing_feature_scaling["min"]))
            std = dict(zip(feats_name, self.preprocessing_feature_scaling["range"]))
        return self.discretizeLayer.get_discretized_feats_name(
            feats_name,
            mean=mean,
            std=std,
            return_centers=return_centers,
        )

    def _interpret_faithful_encoding(self, validate: bool = False):
        with torch.no_grad():
            self.featSelectLayer._interpret_faithful_encoding()
            for decisionLayer in self.decisionLayers:
                decisionLayer._interpret_faithful_encoding()

            if validate:
                feats_selected_per_level = [
                    torch.nonzero(row == 1, as_tuple=True)[0].tolist()
                    for row in self.featSelectLayer.faithfulenc_prob_feat_select
                ]

                assert len(feats_selected_per_level)==self.net_topology.DecLevel_cnt and all(len(feats_specific_level)==1 for feats_specific_level in feats_selected_per_level), "Error: faithfulness violated: it is expected that a single feature to be selected per level"



    def decode(self, feature_names: Optional[List[str]] = None):
        '''
            The decode function of DDNet. Decode the faithful OMDD encoding to get a OMDD representation

        '''
        assert self.net_topology.residual_link==None, "decoding for DDNet that uses residual_link is not supported currently"
        assert self.net_topology.ensembles_cnt==1, "decoding for DDNet that has more than one ensemble is not supported currently"
        assert not self.is_regression, "decoding DDNet for regression is not supported currently"
        assert self.net_topology.get_output_dim()==2, "decoding via mdd.build currently supports binary classification only"

        with torch.no_grad():
            # print("decode....")
            self._interpret_faithful_encoding(validate=True)

            node_naming = lambda lev, no: f"{lev}_{no}"

            # MDD representation
            root: str = None
            reachable_nodes: List[str] = []
            node_labels: Dict[str, int|str] = {}
            edges: Dict[int, Dict[str, str]] = {}

            root_level, root_node_idx = 0, 0    
            root = node_naming(root_level, root_node_idx)   # step-1
            reachable_by_level: Dict[int, Set[int]] = {root_level: {root_node_idx}}

            for level, decisionLayer in enumerate(self.decisionLayers):
                faithful_in2out = decisionLayer.faitfulenc_in2out.detach().clone().float()
                reachable_src = reachable_by_level.get(level, set())
                next_reachable: Set[int] = set()

                for src_idx in sorted(reachable_src):
                    if src_idx >= faithful_in2out.shape[1]:
                        continue
                    for branch_value in range(faithful_in2out.shape[0]):
                        dst_candidates = torch.nonzero(
                            faithful_in2out[branch_value, src_idx] == 1,
                            as_tuple=True,
                        )[0]
                        if len(dst_candidates) != 1:
                            raise ValueError(
                                "faithfulness violated: expected exactly one destination "
                                f"for level={level}, src={src_idx}, branch={branch_value}, "
                                f"but got {len(dst_candidates)}"
                            )
                        dst_idx = int(dst_candidates.detach().cpu().item())       # step-2
                        if level < len(self.decisionLayers) - 1:
                            next_reachable.add(dst_idx)
                            edges.setdefault(branch_value, {})[node_naming(level,src_idx)] = node_naming(level+1,dst_idx) # step-4
                        else:
                            next_reachable.add(dst_idx)
                            leaf = f"{node_naming(level+1,dst_idx)}_class"
                            edges.setdefault(branch_value, {})[node_naming(level,src_idx)] = leaf

                reachable_by_level[level + 1] = next_reachable
            
            feature_ids = torch.nonzero(self.featSelectLayer.faithfulenc_prob_feat_select == 1, as_tuple=True)[1].detach().cpu().tolist()

            for node_level, node_idx_s in reachable_by_level.items():
                for node_idx in sorted(node_idx_s):         # step-3
                    if node_level < len(self.decisionLayers):    # internal level
                        node = node_naming(node_level, node_idx)
                        node_labels[node] = feature_ids[node_level]   
                    else:                                   # terminal level
                        node = f"{node_naming(node_level,node_idx)}_class"
                        node_labels[node] = "F" if node_idx == 0 else "T"
                    reachable_nodes.append(node)

            
            mdd = MDD(feature_names=feature_names)
            mdd.build(
                mdd_nodes=reachable_nodes,
                node_labels=node_labels,
                edges=edges,
                root=root,
                edge_format="by_branch"
            )

            return mdd


    def predict(self, dataset: Dataset, interpret_flag=True):
        '''
            The prediction function of DDNet.

            :param - interpret_flag:    if True, interpretation of faithful encoding will apply to the trained network (i.e., original (non-faithful) encoding), 
                                        and then the forward process will perform at the interpreted faithful encoding; 
                                        otherwise at the original (non-faithful) encoding.

        '''
        with torch.no_grad():
            if interpret_flag:
                self._interpret_faithful_encoding()

            dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
            y_pred_original = []     
            # y_true = []
            for data_x, data_y in dataloader:
                data_x = data_x.to(self.device, non_blocking=True)
                pred_batch = self.forward(data_x, test_flag=True, interpret_flag=interpret_flag) 
        
                y_pred_original += pred_batch.cpu().numpy().tolist()     

            y_true = dataset.get_data_y().cpu().numpy().tolist()   

            if self.is_regression:
                y_pred = y_pred_original
            else:
                y_pred__one_hot_indices = np.argmax(y_pred_original, axis=1) 
                y_pred = dataset.interpret_y_pred_from_one_hot_indices(y_pred__one_hot_indices) 

        return y_true, y_pred       


    def print_paramsAencs(self):
        with torch.no_grad():
            print(f'\n--------Parameters---------')
            for name, param in self.named_parameters():
                print(f'Parameter {name}: Value = {param} \n Gradient = {param.grad}')

            print(f'\n--------Encodings---------')
            self.featSelectLayer.build_FeatureSelectLayer_encoding(test_flag=True)
            print("Encoding featSelectLayer.enc_prob_feat_select: ", self.featSelectLayer.enc_prob_feat_select)
            self.ensembleLayer.build_DecisionLayer_encoding(test_flag=True)
            print("Encoding ensembleLayer.enc_in2out: ", self.ensembleLayer.enc_in2out)
            for idx, decisionLayer in enumerate(self.decisionLayers):
                decisionLayer.build_DecisionLayer_encoding(test_flag=True)
                print(f"Encoding decisionLayer.{idx}.enc_in2out: ", decisionLayer.enc_in2out)

            print(f'\n--------Faithful Encodings---------')
            self._interpret_faithful_encoding()
            print("Encoding featSelectLayer.faithfulenc_prob_feat_select: ", self.featSelectLayer.faithfulenc_prob_feat_select)
            for idx, decisionLayer in enumerate(self.decisionLayers):
                print(f"Faithful encoding decisionLayer.{idx}.faithfulenc_in2out: ", decisionLayer.faitfulenc_in2out)
            

# ================================


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _build_dataloader_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _compute_tau(
    epoch: int,
    epoch_num: int,
    tau_start: float,
    tau_end: float,
    schedule: str = "linear",
    decay_start_ratio: float = 0.0,
    decay_end_ratio: float = 0.8,
    progress: Optional[float] = None,
) -> float:
    if progress is None and epoch_num <= 0:
        return tau_end
    if tau_start <= 0 or tau_end <= 0:
        return tau_end
    if progress is None:
        progress = epoch / epoch_num
    progress = min(max(float(progress), 0.0), 1.0)
    decay_start_ratio = min(max(decay_start_ratio, 0.0), 1.0)
    decay_end_ratio = min(max(decay_end_ratio, 0.0), 1.0)

    if decay_end_ratio <= decay_start_ratio:
        decay_end_ratio = decay_start_ratio

    if progress <= decay_start_ratio:
        return float(tau_start)
    if progress >= decay_end_ratio:
        return float(tau_end)

    decay_progress = (progress - decay_start_ratio) / (decay_end_ratio - decay_start_ratio)

    if schedule == "linear":
        tau = tau_start - ((tau_start - tau_end) * decay_progress)
    elif schedule == "exponential":
        tau = tau_start * ((tau_end / tau_start) ** decay_progress)
    elif schedule == "cosine":
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
        tau = tau_end + (tau_start - tau_end) * cosine_factor
    else:
        raise ValueError(f"`{schedule}` is not the supported schedule method of tau")

    return float(tau)


def _schedule_coefficients(
    base_coefficients: Dict[str, float],
    epoch: int,
    epoch_num: int,
    start_ratio: float = 0.3,
) -> Dict[str, float]:
    
    ramp = 1.0
    coefficients = dict(base_coefficients)
    for key in ("coef. read_once", "coef. ent_dec"):
        if key in coefficients:
            coefficients[key] = base_coefficients[key] * ramp
    return coefficients


def _classification_model_is_better(
    current_acc: float,
    current_f1_macro: float,
    best_acc: float,
    best_f1_macro: float,
    optimize_metric: str,
) -> bool:
    if optimize_metric == "acc":
        return current_acc > best_acc or (
            current_acc == best_acc and current_f1_macro > best_f1_macro
        )
    if optimize_metric == "f1_macro":
        return current_f1_macro > best_f1_macro or (
            current_f1_macro == best_f1_macro and current_acc > best_acc
        )
    raise ValueError(f"Unsupported classification optimize_metric: {optimize_metric}")


def _classification_should_backtrack(
    current_acc: float,
    current_f1_macro: float,
    best_acc: float,
    best_f1_macro: float,
    optimize_metric: str,
    backtrack_dacc: float,
    backtrack_dF1: float,
) -> bool:
    if optimize_metric == "acc":
        degradation = best_acc - current_acc
        return degradation > backtrack_dacc and not np.isclose(degradation, backtrack_dacc)
    if optimize_metric == "f1_macro":
        degradation = best_f1_macro - current_f1_macro
        return degradation > backtrack_dF1 and not np.isclose(degradation, backtrack_dF1)
    raise ValueError(f"Unsupported classification optimize_metric: {optimize_metric}")


def _build_default_topology(
    feats_cnt: int,
    output_dim: int,
    interval_num: int = 2,
    dec_level_cnt: int = 6,
    ensembles_cnt: int = 1,
    width_cap: Optional[int] = None,
    residual_link: Optional[Dict[int, int]] = None,
    is_regression: bool = False,
    hidden_dim_regress: int = 16,
    verbose: bool = False,
) -> OrderedDDNetTopology:
    if width_cap is None:
        width_cap = interval_num ** max(dec_level_cnt - 1, 0)      

    dec_level_shape_s = []
    for ith in range(dec_level_cnt):
        in_height = ensembles_cnt if ith == 0 else min(interval_num ** ith, width_cap) 
        dec_level_shape_s.append((in_height, interval_num))

    return OrderedDDNetTopology(
        input_dim=feats_cnt,
        output_dim=output_dim,
        interval_num_for_discretize=interval_num,
        DecLevel_cnt=dec_level_cnt,
        DecLevel_shape_s=dec_level_shape_s,
        residual_link=residual_link,
        ensembles_cnt=ensembles_cnt,
        is_regression=is_regression,
        hidden_dim_regress=hidden_dim_regress,
        verbose=verbose,
    )


def _evaluate_model(model: DDNet, dataset: Dataset, interpret_flag: bool, return_predictions:bool=False) -> Dict[str, Any]:
    timing: Dict[str, float] = {}
    total_start = _timer_start(model.device)
    with torch.no_grad():
        model.eval()
        predict_start = _timer_start(model.device)
        y_true, y_pred = model.predict(dataset, interpret_flag=interpret_flag)
        timing["predict"] = _timer_elapsed(predict_start, model.device)

        result = {
            "interpret_flag": interpret_flag,
        }

        if return_predictions:
            result["y_pred"] = y_pred
            result["y_true"] = y_true

        score_start = _timer_start(model.device)
        if model.is_regression:
            result["rmse"] = root_mean_squared_error(y_true, y_pred)
        else:
            # result["acc"] = accuracy(y_true, y_pred)
            result["score"] = classification_score(y_true, y_pred)
        timing["score"] = _timer_elapsed(score_start, model.device)
        timing["total"] = _timer_elapsed(total_start, model.device)
        result["timing"] = timing
        
        model.train()

    return result


def _sync_device(device: Optional[str] = None) -> None:
    if torch.cuda.is_available() and device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _timer_start(device: Optional[str] = None) -> float:
    _sync_device(device)
    return time.time()


def _timer_elapsed(start_time: float, device: Optional[str] = None) -> float:
    _sync_device(device)
    return time.time() - start_time


def consistency_of_preds(left: Optional[List[Any]], right: Optional[List[Any]]) -> Optional[float]:
    if left is None or right is None:
        return None
    if len(left) != len(right):
        raise ValueError(f"Prediction lengths differ: {len(left)} vs {len(right)}")
    if len(left) == 0:
        return None
    same = sum(1 for a, b in zip(left, right) if a == b)
    return same / len(left)


def _get_discretized_bin_ids(model: DDNet, dataset: Dataset, batch_size: int = 2048) -> List[List[int]]:
    bin_id_batches = []
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for data_x, _ in dataloader:
            data_x = data_x.to(model.device, non_blocking=True)

            if hasattr(model.discretizeLayer, "_get_bin_ids"):
                bin_ids = model.discretizeLayer._get_bin_ids(data_x)
            elif hasattr(model.discretizeLayer, "binarizeLayer") and hasattr(model.discretizeLayer.binarizeLayer, "_get_bin_ids"):
                bin_ids = model.discretizeLayer.binarizeLayer._get_bin_ids(data_x)
            else:
                discretized_data_x = model.discretizeLayer(data_x, tau=model.tau, test_flag=True)
                features_cnt, _, domain_size = model.net_topology.get_DiscretizeLayer_shape()
                bin_ids = discretized_data_x.reshape(data_x.shape[0], features_cnt, domain_size).argmax(dim=-1)

            bin_id_batches.extend(bin_ids.detach().cpu().long().tolist())
    return bin_id_batches


def _get_mdd_branch_labels_by_feature(model: DDNet, dataset: Dataset) -> Dict[int, List[str]]:
    feats_name = dataset.get_feats_name()
    interval_num = model.net_topology.get_interval_num_for_discretize()
    discretized_feats_name = model.get_discretized_feats_name(feats_name, return_centers=False)

    branch_labels_by_feature: Dict[int, List[str]] = {}
    for feature_idx in range(model.net_topology.get_input_dim()):
        start = feature_idx * interval_num
        end = start + interval_num
        branch_labels_by_feature[feature_idx] = discretized_feats_name[start:end]
    return branch_labels_by_feature


def _evaluate_interpreted_mdd(
    model: DDNet,
    dataset: Dataset,
    return_predictions: bool = False,
    dot_edge_label_mode: Literal["value", "bin"] = "value",
    dot_show_legend: bool = True,
) -> Dict[str, Any]:
    timing: Dict[str, float] = {}
    total_start = _timer_start(model.device)
    with torch.no_grad():
        model.eval()
        result: Dict[str, Any] = {
            "available": False,
            "timing": timing,
        }

        try:
            if model.is_regression:
                result["reason"] = "learned_dd evaluation is only implemented for classification"
                return result
            if model.net_topology.get_output_dim() != 2:
                result["reason"] = "dd.mdd bridge currently supports Boolean/binary outputs only"
                return result

            step_start = _timer_start(model.device)
            branch_labels_by_feature = _get_mdd_branch_labels_by_feature(model, dataset)
            timing["branch_labels"] = _timer_elapsed(step_start, model.device)

            step_start = _timer_start(model.device)
            decoded_mdd = model.decode(feature_names=dataset.get_feats_name())
            timing["decode_to_MDD"] = _timer_elapsed(step_start, model.device)
            size_nonreduced_omdd = decoded_mdd.actual_size()

            step_start = _timer_start(model.device)
            dd_mdd, root, feature_to_var = MDD_to_mdd(decoded_mdd)
            timing["MDD_to_dd_mdd"] = _timer_elapsed(step_start, model.device)

            step_start = _timer_start(model.device)
            discretized_bin_ids = _get_discretized_bin_ids(model, dataset)
            timing["get_discretized_bin_ids"] = _timer_elapsed(step_start, model.device)

            step_start = _timer_start(model.device)
            y_pred_indices = mdd_predict(
                dd_mdd,
                root,
                discretized_bin_ids,
                feature_to_var=feature_to_var,
            )
            timing["dd_mdd_predict"] = _timer_elapsed(step_start, model.device)
            y_pred = dataset.interpret_y_pred_from_one_hot_indices(y_pred_indices)
            y_true = dataset.get_data_y().cpu().numpy().tolist()

            step_start = _timer_start(model.device)
            reduced_mdd = mdd_to_MDD(
                dd_mdd,
                root,
                feature_names=decoded_mdd.feature_names,
            )
            timing["dd_mdd_to_MDD_recover"] = _timer_elapsed(step_start, model.device)
            size_reduced_omdd = reduced_mdd.actual_size()

            step_start = _timer_start(model.device)
            score = classification_score(y_true, y_pred)
            timing["score"] = _timer_elapsed(step_start, model.device)

            step_start = _timer_start(model.device)
            nonreduced_case_study = _mdd_case_study_payload(
                decoded_mdd,
                branch_labels_by_feature=branch_labels_by_feature,
                dot_edge_label_mode=dot_edge_label_mode,
                dot_show_legend=dot_show_legend,
            )
            timing["case_study_nonreduced_omdd"] = _timer_elapsed(step_start, model.device)

            step_start = _timer_start(model.device)
            reduced_case_study = _mdd_case_study_payload(
                reduced_mdd,
                branch_labels_by_feature=branch_labels_by_feature,
                dot_edge_label_mode=dot_edge_label_mode,
                dot_show_legend=dot_show_legend,
            )
            timing["case_study_reduced_omdd"] = _timer_elapsed(step_start, model.device)

            result = {
                "available": True,
                "size_nonreducedOMDD": size_nonreduced_omdd,
                "size_reducedOMDD": size_reduced_omdd,
                "score": score,
                "timing": timing,
                "feature_to_var": feature_to_var,
                "dot_config": {
                    "edge_label_mode": dot_edge_label_mode,
                    "show_legend": dot_show_legend,
                },
                "branch_labels_by_feature": branch_labels_by_feature,
                # "dot": {
                #     "nonreduced_omdd": str(dot_nonreduced_omdd),
                #     "reduced_omdd": str(dot_reduced_omdd),
                # },
                "mdd_case_study": {
                    "nonreduced_omdd": nonreduced_case_study,
                    "reduced_omdd": reduced_case_study,
                },
            }
            if return_predictions:
                result["y_pred"] = y_pred
                result["y_true"] = y_true

            return result
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
            # raise  
            return result
        finally:
            timing["total"] = _timer_elapsed(total_start, model.device)
            model.train()


def _evaluate_discretization(model: DDNet, dataset: Dataset, batch_size: int = 2048) -> Dict[str, Any]:
    """
    Measure distribution coverage independently from downstream predictions.
    """
    layer = model.discretizeLayer
    features_cnt, _, interval_num = model.net_topology.get_DiscretizeLayer_shape()
    occupancy = torch.zeros(features_cnt, interval_num, dtype=torch.float64)
    total_normalized_error = 0.0
    example_count = 0

    with torch.no_grad():
        if hasattr(layer, "_get_centers"):
            centers = layer._get_centers()
        elif hasattr(layer, "params_bin_centers"):
            centers = torch.sort(
                torch.clamp(
                    layer.params_bin_centers,
                    min=layer.featvals_min_tensor,
                    max=layer.featvals_max_tensor,
                ),
                dim=-1,
            ).values
        elif hasattr(layer, "binarizeLayer") and hasattr(layer.binarizeLayer, "_get_centers"):
            centers = layer.binarizeLayer._get_centers()
        else:
            centers = None

        if centers is not None:
            centers = centers.to(model.device)
            value_range = (centers.max(dim=1).values - centers.min(dim=1).values).clamp_min(1e-8)
        else:
            value_range = None

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for data_x, _ in dataloader:
            data_x = data_x.to(model.device)
            if hasattr(layer, "_get_bin_ids"):
                bin_ids = layer._get_bin_ids(data_x)
            elif hasattr(layer, "binarizeLayer") and hasattr(layer.binarizeLayer, "_get_bin_ids"):
                bin_ids = layer.binarizeLayer._get_bin_ids(data_x)
            else:
                discretized_data_x = layer(data_x, tau=model.tau, test_flag=True)
                bin_ids = discretized_data_x.reshape(data_x.shape[0], features_cnt, interval_num).argmax(dim=-1)

            if centers is not None and value_range is not None:
                normalized_distance = (
                    (data_x.unsqueeze(-1) - centers.unsqueeze(0)) / value_range.unsqueeze(0).unsqueeze(-1)
                ).pow(2)
                nearest_distance = normalized_distance.min(dim=-1).values
                total_normalized_error += nearest_distance.sum().item()
            example_count += data_x.shape[0]
            for bin_idx in range(interval_num):
                occupancy[:, bin_idx] += (bin_ids == bin_idx).sum(dim=0).cpu()

    occupancy_ratio = occupancy / max(example_count, 1)
    return {
        "normalized_quantization_error": total_normalized_error / max(example_count * features_cnt, 1),
        "empty_bin_ratio": float((occupancy == 0).double().mean().item()),
        "min_bin_occupancy_ratio": float(occupancy_ratio.min().item()),
        "max_bin_occupancy_ratio": float(occupancy_ratio.max().item()),
        "feature_bin_occupancy_ratio": occupancy_ratio.tolist(),
    }


def test(
    model: DDNet,
    test_file: Optional[str] = None,
    test_dataset: Optional[Dataset] = None,
    device: Optional[str] = None,
    return_predictions: bool = False,
    learned_dd_dot_edge_label_mode: Literal["value", "bin"] = "value",
    learned_dd_dot_show_legend: bool = True,
) -> Dict[str, Any]:
    total_start = _timer_start(device or model.device)
    if test_dataset is None:
        if test_file is None:
            raise ValueError("Either test_file or test_dataset must be provided.")
        dataset_load_start = _timer_start(device or model.device)
        dataset_cls = RegressionDataset if model.is_regression else ClassificationDataset
        preprocessing_scaling = getattr(model, "preprocessing_feature_scaling", {})
        test_dataset = dataset_cls(
            test_file,
            scale_continuous_features=preprocessing_scaling.get("enabled", False),
            feature_scaling=preprocessing_scaling if preprocessing_scaling.get("enabled", False) else None,
        )
        dataset_load_time = _timer_elapsed(dataset_load_start, device or model.device)
    else:
        dataset_load_time = 0.0

    dataset_size, feats_cnt, target_dim, target_distribution, feats_name, featvals_min, featvals_max = \
        test_dataset.get_dataset_statistic()
    
    if test_file is not None:
        print(f"\n\nThe statistic of dataset {test_file}: \ndataset_size: {dataset_size} \n"
            + f"feats_cnt: {feats_cnt} \n"
            + f"target_dim: {target_dim} \n"
            + f"target_distribution: {target_distribution} \n"
            )
    
    result = {
        "dataset_statistic": {
            "dataset_size": dataset_size,
            "feats_cnt": feats_cnt,
            "target_dim": target_dim,
            "target_distribution": target_distribution,
            "feats_name": feats_name,
            "featvals_min": featvals_min,
            "featvals_max": featvals_max,
        },
        "evaluations": {},
        "timing": {
            "dataset_load": dataset_load_time,
        },
    }

    with torch.no_grad():
        model.eval()

        step_start = _timer_start(model.device)
        network_eval = _evaluate_model(model, test_dataset, interpret_flag=False, return_predictions=True)
        result["timing"]["network_eval"] = _timer_elapsed(step_start, model.device)
        result["evaluations"]["network"] = network_eval

        step_start = _timer_start(model.device)
        faithful_eval = _evaluate_model(model, test_dataset, interpret_flag=True, return_predictions=True)
        result["timing"]["faithful_eval"] = _timer_elapsed(step_start, model.device)
        result["evaluations"]["faithful"] = faithful_eval  

        step_start = _timer_start(model.device)
        learned_dd_eval = _evaluate_interpreted_mdd(
            model,
            test_dataset,
            return_predictions=True,
            dot_edge_label_mode=learned_dd_dot_edge_label_mode,
            dot_show_legend=learned_dd_dot_show_legend,
        )
        result["timing"]["learned_dd_eval"] = _timer_elapsed(step_start, model.device)
        result["evaluations"]["learned_dd"] = learned_dd_eval

        learned_dd_pred = learned_dd_eval.get("y_pred") if learned_dd_eval.get("available", False) else None
        result["evaluations"]["consistency_of_preds"] = {
            "network_vs_faithful": consistency_of_preds(network_eval.get("y_pred"), faithful_eval.get("y_pred")),
            "network_vs_learned_dd": consistency_of_preds(network_eval.get("y_pred"), learned_dd_pred),
            "faithful_vs_learned_dd": consistency_of_preds(faithful_eval.get("y_pred"), learned_dd_pred),
        }

        step_start = _timer_start(model.device)
        result["discretization"] = _evaluate_discretization(model, test_dataset)
        result["timing"]["discretization_eval"] = _timer_elapsed(step_start, model.device)
        result["timing"]["total"] = _timer_elapsed(total_start, model.device)

        if not return_predictions:
            for eval_name in ("network", "faithful", "learned_dd"):
                result["evaluations"][eval_name].pop("y_pred", None)
                result["evaluations"][eval_name].pop("y_true", None)


    return result


def _classification_labels_as_indices(dataset: ClassificationDataset) -> List[int]:
    y_old = dataset.get_data_y().cpu().numpy().tolist()
    return [int(dataset.labels_mapping_old2new[label]) for label in y_old]


def _classification_indices_to_labels(dataset: ClassificationDataset, indices: List[int]) -> List[Any]:
    return dataset.interpret_y_pred_from_one_hot_indices([int(x) for x in indices])


def _json_ready(value: Any) -> Any:
    """Convert numpy/torch/container values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(_json_ready(key)): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _mdd_case_study_payload(
    mdd: MDD,
    branch_labels_by_feature: Optional[Dict[int, List[str]]],
    dot_edge_label_mode: Literal["value", "bin"],
    dot_show_legend: bool,
    max_dot_chars: int = 200000,
) -> Dict[str, Any]:
    """Persist enough MDD information for later dot/case-study analysis."""
    dot = mdd.get_dot_description(
        branch_labels_by_feature=branch_labels_by_feature,
        edge_label_mode=dot_edge_label_mode,
        show_legend=dot_show_legend,
    )
    payload = {
        "mdd_size": mdd.actual_size(),
        "mdd_feature_ordering": _json_ready(mdd.get_feature_ordering()[0]),
        "mdd_dict": _json_ready(mdd.to_dict()),
        "mdd_dot_chars": len(dot),
        "mdd_dot_truncated": len(dot) > max_dot_chars,
    }
    if len(dot) <= max_dot_chars:
        payload["mdd_dot"] = dot
    else:
        payload["mdd_dot_preview"] = dot[:max_dot_chars]
    return payload


def train(
    train_file: str,
    test_file: Optional[str] = None,
    valid_file: Optional[str] = None,
    model: Optional[DDNet] = None,
    net_topology: Optional[OrderedDDNetTopology] = None,
    coefficients: Optional[Dict[str, float]] = None,
    epoch_num: int = 1000,
    step_num: Optional[int] = None,
    timeout: int = 900,
    batch_size: int = 512,
    lr: float = 0.02,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    normalize_method: Literal["gumbel_softmax", "softmax"] = "gumbel_softmax",
    discretize_t1: float = 50.0,
    discretize_t2: float = 10.0,
    discretize_reg_mode: str = "hybrid",
    discretize_cluster_reg_weight: float = 1.0,
    discretize_boundary_reg_weight: float = 0.1,
    discretizer_type: Literal["learned", "kbins"] = "learned",
    discretizer_kwargs: Optional[Dict[str, Any]] = None,
    scale_continuous_features: bool = True,
    tau_start: float = 10.0,
    tau_end: float = 1.0,
    tau_schedule: Literal["linear", "cosine", "exponential"] = "cosine",
    tau_decay_start_ratio: float = 0.3,
    tau_decay_end_ratio: float = 2.0 / 3.0,
    random_seed: int = 2024,
    optimize_metric: Literal["acc", "f1_macro", "rmse"] = None,
    backtrack_dacc: float = 1.0,
    backtrack_dF1: float = 1.0,
    # 
    # shuffle: bool = True,
    verbose: bool = False,
    print_every: int = 20,
    # 
    # used to build net_topology if not provided before
    interval_num: int = 2,
    dec_level_cnt: int = 6,
    ensembles_cnt: int = 1,
    width_cap: Optional[int] = None,
    residual_link: Optional[Dict[int, int]] = None,
    is_regression: bool = False,
    hidden_dim_regress: int = 16,
    # 
    load_best_state_at_end: bool = True,
    learned_dd_dot_edge_label_mode: Literal["value", "bin"] = "value",
    learned_dd_dot_show_legend: bool = True,
    iterative_iterations: int = 1,
    iterative_build_final_mdd: bool = True,
    iterative_require_compatible_ordering: bool = False,
    return_model: bool = False,
    feature_scaling_override: Optional[Dict[str, Any]] = None,
    discretizer_fit_data_override: Optional[Tensor] = None,
    classification_label_values_override: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    
    train_total_start = _timer_start(device)
    training_timing: Dict[str, float] = {}
    _set_random_seed(random_seed)

    if coefficients is None:
        coefficients = {
            "coef. read_once": 0.01,
            "coef. ent_feat_select": 1e-6,
            "coef. ent_dec": 1e-4,
            "coef. discretization": 1e-2,
        }
    base_coefficients = dict(coefficients)

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    train_total_start = _timer_start(device)

    if iterative_iterations > 1:
        pass
        raise ValueError("iterative learning is not supported currently")
        
    if model_path is not None:
        model_parent = os.path.dirname(model_path)
        if model_parent:
            os.makedirs(model_parent, exist_ok=True)

    dataset_cls = RegressionDataset if is_regression else ClassificationDataset
    train_dataset_kwargs: Dict[str, Any] = {}
    if not is_regression and classification_label_values_override is not None:
        train_dataset_kwargs["label_values"] = classification_label_values_override
    dataset_prepare_start = _timer_start(device)
    train_dataset = dataset_cls(
        train_file,
        scale_continuous_features=scale_continuous_features,
        feature_scaling=feature_scaling_override if scale_continuous_features else None,
        **train_dataset_kwargs,
    )
    feature_scaling = train_dataset.get_feature_scaling()
    valid_dataset = (
        dataset_cls(
            valid_file,
            scale_continuous_features=scale_continuous_features,
                feature_scaling=feature_scaling if scale_continuous_features else None,
                **train_dataset_kwargs,
            )
        if valid_file is not None
        else None
    )
    dataset_size, feats_cnt, target_dim, target_distribution, feats_name, featvals_min, featvals_max = \
        train_dataset.get_dataset_statistic()
    training_timing["dataset_prepare"] = _timer_elapsed(dataset_prepare_start, device)
    
    print(f"\n\nThe statistic of dataset {train_file}: \ndataset_size: {dataset_size} \n"
        + f"feats_cnt: {feats_cnt} \n"
        + f"target_dim: {target_dim} \n"
        + f"target_distribution: {target_distribution} \n"
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,         # NOTE: 
        generator=_build_dataloader_generator(random_seed, device=device),
        worker_init_fn=_seed_dataloader_worker,
        pin_memory=torch.cuda.is_available() and str(device).startswith("cuda"),
    )

    if net_topology is None:
        net_topology = _build_default_topology(
            feats_cnt=feats_cnt,
            output_dim=target_dim,
            interval_num=interval_num,
            dec_level_cnt=dec_level_cnt,
            ensembles_cnt=ensembles_cnt,
            width_cap=width_cap,
            residual_link=residual_link,
            is_regression=is_regression,
            hidden_dim_regress=hidden_dim_regress,
            verbose=True,
        )

    model_prepare_start = _timer_start(device)
    if model is None:
        model = DDNet(
            net_topology=net_topology,
            cont_featvals_min=featvals_min,
            cont_featvals_max=featvals_max,
            coefficients=coefficients,
            discretize_t1=discretize_t1,
            discretize_t2=discretize_t2,
            discretize_reg_mode=discretize_reg_mode,
            discretize_cluster_reg_weight=discretize_cluster_reg_weight,
            discretize_boundary_reg_weight=discretize_boundary_reg_weight,
            discretizer_type=discretizer_type,
            discretizer_kwargs=discretizer_kwargs,
            discretizer_fit_data=discretizer_fit_data_override if discretizer_fit_data_override is not None else train_dataset.get_data_x(),
            preprocessing_feature_scaling=feature_scaling if scale_continuous_features else None,
            normalize_method=normalize_method,
            tau_start=tau_start,
            device=device,
            verbose=verbose,
        )
    elif discretizer_type == "kbins" and hasattr(model.discretizeLayer, "fit"):
        model.fit_discretizer(train_dataset.get_data_x())  

    model.to(device)
    optimizer = _build_adamw_optimizer(model, lr=lr)
    if is_regression:
        criterion = nn.MSELoss() 
    else:
        if "softmax" in normalize_method:
            criterion = ProbCrossEntropyLoss()      
        else:
            criterion = nn.CrossEntropyLoss()   
    training_timing["model_prepare"] = _timer_elapsed(model_prepare_start, device)


    optimize_metric = optimize_metric or ("rmse" if is_regression else "acc")
    if backtrack_dacc < 0 or backtrack_dF1 < 0:
        raise ValueError("backtrack_dacc and backtrack_dF1 must be non-negative")
    if is_regression:
        assert optimize_metric in ["rmse"]
    else:
        assert optimize_metric in ["acc", "f1_macro"]
    if step_num is not None and step_num <= 0:
        raise ValueError("step_num must be positive when provided")
    best_metric = float("inf") if is_regression else -float("inf")
    best_acc = -float("inf")
    best_f1_macro = -float("inf")
    best_epoch = -1
    best_step = -1
    best_state_dict = None
    stop_mode = "step" if step_num is not None else "epoch"
    stop_reason = "step_limit" if step_num is not None else "epoch_limit"
    history = []
    start_time = time.time()
    train_loop_start = _timer_start(device)
    global_step = 0
    epoch = 0

    while True:
        if step_num is None and epoch > epoch_num:
            break
        if step_num is not None and global_step >= step_num:
            break

        if step_num is None and epoch == epoch_num and verbose:
            print(f"\n ====== {epoch_num}-th epoch verbose ====== \n")
            model.reset_verbose(verbose=True)
        else:
            model.reset_verbose(verbose=False)

        model.train()
        schedule_progress = (global_step / step_num) if step_num is not None else None
        tau = _compute_tau(
            epoch,
            epoch_num,
            tau_start,
            tau_end,
            schedule=tau_schedule,
            decay_start_ratio=tau_decay_start_ratio,
            decay_end_ratio=tau_decay_end_ratio,
            progress=schedule_progress,
        )
        model.tau.data.fill_(tau)
        model.coefficients = _schedule_coefficients(
            base_coefficients,
            epoch,
            epoch_num,
            start_ratio=tau_decay_start_ratio,
        )

        epoch_task_loss = 0.0
        epoch_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_reg_loss_details = None
        batch_cnt = 0
        epoch_start_step = global_step
        epoch_train_start = _timer_start(device)

        for data_x, data_y in train_dataloader:
            if step_num is not None and global_step >= step_num:
                stop_reason = "step_limit"
                break

            data_x = data_x.to(device, non_blocking=True)
            data_y = data_y.to(device, non_blocking=True)
            prediction = model(data_x, test_flag=False)
            task_loss = criterion(prediction, data_y)
            reg_loss, reg_loss_details = model.regularization()
            total_loss = task_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += float(task_loss.detach().cpu().item())
            epoch_loss += float(total_loss.detach().cpu().item())
            epoch_reg_loss += float(reg_loss.detach().cpu().item()) if torch.is_tensor(reg_loss) else float(reg_loss)
            epoch_reg_loss_details = reg_loss_details if epoch_reg_loss_details is None else {k: v + reg_loss_details[k] for k, v in epoch_reg_loss_details.items()}
            batch_cnt += 1
            global_step += 1

            if time.time() - start_time > timeout:
                stop_reason = "timeout"
                break

        epoch_train_time = _timer_elapsed(epoch_train_start, device)

        model.reset_verbose(verbose=False)

        if batch_cnt > 0:
            epoch_task_loss /= batch_cnt
            epoch_loss /= batch_cnt
            epoch_reg_loss /= batch_cnt

            epoch_reg_loss_details = {k:(v/batch_cnt) for k, v in epoch_reg_loss_details.items()}

        eval_start = _timer_start(device)
        train_network_eval = _evaluate_model(model, train_dataset, interpret_flag=False)
        train_network_eval_time = _timer_elapsed(eval_start, device)

        eval_start = _timer_start(device)
        train_faithful_eval = _evaluate_model(model, train_dataset, interpret_flag=True)
        train_faithful_eval_time = _timer_elapsed(eval_start, device)
        valid_network_eval = None
        valid_faithful_eval = None
        valid_network_eval_time = 0.0
        valid_faithful_eval_time = 0.0
        model_select_source = "train"

        if valid_dataset is not None:
            eval_start = _timer_start(device)
            valid_network_eval = _evaluate_model(model, valid_dataset, interpret_flag=False)
            valid_network_eval_time = _timer_elapsed(eval_start, device)
            eval_start = _timer_start(device)
            valid_faithful_eval = _evaluate_model(model, valid_dataset, interpret_flag=True)
            valid_faithful_eval_time = _timer_elapsed(eval_start, device)
            model_select_source = "valid"

        selected_eval = valid_faithful_eval if valid_faithful_eval is not None else train_faithful_eval

        backtracked = False
        if is_regression:
            metric_value = float(selected_eval[optimize_metric])
            improved = metric_value < best_metric
        else:
            current_acc = float(selected_eval["score"]["acc"])
            current_f1_macro = float(selected_eval["score"]["f1_macro"])
            metric_value = current_acc if optimize_metric == "acc" else current_f1_macro
            improved = _classification_model_is_better(
                current_acc=current_acc,
                current_f1_macro=current_f1_macro,
                best_acc=best_acc,
                best_f1_macro=best_f1_macro,
                optimize_metric=optimize_metric,
            )

        if improved:
            best_metric = metric_value
            if not is_regression:
                best_acc = current_acc
                best_f1_macro = current_f1_macro
            best_epoch = epoch
            best_step = global_step
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if model_path is not None:
                torch.save(best_state_dict, model_path)
        elif not is_regression and best_state_dict is not None:
            backtracked = _classification_should_backtrack(
                current_acc=current_acc,
                current_f1_macro=current_f1_macro,
                best_acc=best_acc,
                best_f1_macro=best_f1_macro,
                optimize_metric=optimize_metric,
                backtrack_dacc=backtrack_dacc,
                backtrack_dF1=backtrack_dF1,
            )
            if backtracked:
                model.load_state_dict(best_state_dict)

        epoch_record = {
            "epoch": epoch,
            "global_step_start": epoch_start_step,
            "global_step": global_step,
            "schedule_progress": min(max(schedule_progress if schedule_progress is not None else (epoch / epoch_num if epoch_num > 0 else 1.0), 0.0), 1.0),
            "tau": tau,
            "epoch_loss": epoch_loss,
            "epoch_task_loss": epoch_task_loss,
            "epoch_reg_loss": epoch_reg_loss,
            "epoch_reg_loss_details": epoch_reg_loss_details,
            "batch_cnt": batch_cnt,
            "elapsed_time": time.time() - start_time,
            "timing": {
                "train_batches": epoch_train_time,
                "train_network_eval": train_network_eval_time,
                "train_faithful_eval": train_faithful_eval_time,
                "valid_network_eval": valid_network_eval_time,
                "valid_faithful_eval": valid_faithful_eval_time,
                "epoch_total": epoch_train_time + train_network_eval_time + train_faithful_eval_time + valid_network_eval_time + valid_faithful_eval_time,
            },
            "train_network_eval": train_network_eval,
            "train_faithful_eval": train_faithful_eval,
            "valid_network_eval": valid_network_eval,
            "valid_faithful_eval": valid_faithful_eval,
            "model_select_source": model_select_source,
            "best_metric_so_far": best_metric,
            "best_acc_so_far": best_acc if not is_regression else None,
            "best_f1_macro_so_far": best_f1_macro if not is_regression else None,
            "best_epoch_so_far": best_epoch,
            "best_step_so_far": best_step,
            "backtracked": backtracked,
        }
        history.append(epoch_record)

        if print_every > 0 and epoch % print_every == 0:
            if is_regression:
                print(
                    f"epoch: {epoch} | train_time: {epoch_record['elapsed_time']:.2f}s | "
                    f"step: {global_step} | tau: {tau:.4f} | epoch_loss: {epoch_loss:.4f} | epoch_reg_loss: {epoch_reg_loss:.4f} | "
                    f"rmse_network: {train_network_eval['rmse']:.4f} | rmse: {train_faithful_eval['rmse']:.4f}"
                )
            else:
                print(
                    f"epoch: {epoch} | train_time: {epoch_record['elapsed_time']:.2f}s | "
                    f"step: {global_step} | tau: {tau:.4f} | epoch_loss: {epoch_loss:.4f} | epoch_reg_loss: {epoch_reg_loss:.4f} | "
                    f"accuracy_network: {train_network_eval['score']['acc'] * 100:.2f}% | "
                    f"f1_macro_network: {train_network_eval['score']['f1_macro'] * 100:.2f}% | "
                    f"accuracy: {train_faithful_eval['score']['acc'] * 100:.2f}% | "
                    f"f1_macro: {train_faithful_eval['score']['f1_macro'] * 100:.2f}%"
                )
                print( "\t " + " | ".join([f"{reg_loss_item}: {val:.6f}" for reg_loss_item, val in epoch_reg_loss_details.items()]) )
                if valid_faithful_eval is not None:
                    print(
                        f"\t valid_accuracy_network: {valid_network_eval['score']['acc'] * 100:.2f}% | "
                        f"valid_f1_macro_network: {valid_network_eval['score']['f1_macro'] * 100:.2f}% | "
                        f"valid_accuracy: {valid_faithful_eval['score']['acc'] * 100:.2f}% | "
                        f"valid_f1_macro: {valid_faithful_eval['score']['f1_macro'] * 100:.2f}% | "
                        f"select_on: {model_select_source}"
                    )

        if stop_reason == "timeout":
            print(f"Timeout! Terminate training. cur_epoch:{epoch}, cur_batch:{batch_cnt}")
            break
        if stop_reason == "step_limit" and step_num is not None and global_step >= step_num:
            print(f"Step limit reached. Terminate training. cur_epoch:{epoch}, global_step:{global_step}")
            break
        epoch += 1

    training_timing["training_loop"] = _timer_elapsed(train_loop_start, device)

    if load_best_state_at_end and best_state_dict is not None:
        load_best_start = _timer_start(device)
        model.load_state_dict(best_state_dict)
        training_timing["load_best_state"] = _timer_elapsed(load_best_start, device)
    elif load_best_state_at_end and best_state_dict is None and model_path is not None:
        load_best_start = _timer_start(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        training_timing["load_best_state"] = _timer_elapsed(load_best_start, device)

    with torch.no_grad():
        model.eval()
        final_eval_start = _timer_start(device)
        feature_name_start = _timer_start(device)
        discretized_feats_name, centers = model.get_discretized_feats_name(feats_name)
        training_timing["final_discretized_feature_names"] = _timer_elapsed(feature_name_start, device)
        eval_start = _timer_start(device)
        train_evaluation = test(
            model=model,
            test_dataset=train_dataset,
            return_predictions=True,
            learned_dd_dot_edge_label_mode=learned_dd_dot_edge_label_mode,
            learned_dd_dot_show_legend=learned_dd_dot_show_legend,
        )
        training_timing["final_train_evaluation"] = _timer_elapsed(eval_start, device)
        valid_evaluation = None
        if valid_dataset is not None:
            eval_start = _timer_start(device)
            valid_evaluation = test(
                model=model,
                test_dataset=valid_dataset,
                return_predictions=True,
                learned_dd_dot_edge_label_mode=learned_dd_dot_edge_label_mode,
                learned_dd_dot_show_legend=learned_dd_dot_show_legend,
            )
            training_timing["final_valid_evaluation"] = _timer_elapsed(eval_start, device)
        test_evaluation = None
        if test_file is not None:
            eval_start = _timer_start(device)
            test_evaluation = test(
                model=model,
                test_file=test_file,
                device=device,
                return_predictions=True,
                learned_dd_dot_edge_label_mode=learned_dd_dot_edge_label_mode,
                learned_dd_dot_show_legend=learned_dd_dot_show_legend,
            )
            training_timing["final_test_evaluation"] = _timer_elapsed(eval_start, device)
        training_timing["final_evaluation"] = _timer_elapsed(final_eval_start, device)

        if verbose:
            model.print_paramsAencs()

    training_timing["total"] = _timer_elapsed(train_total_start, device)

    result = {
        "model_path": model_path,
        "net_topology": net_topology,
        "train_file": train_file,
        "test_file": test_file,
        "valid_file": valid_file,
        "device": device,
        "is_regression": is_regression,
        "hyperparameters": {
            "coefficients": base_coefficients,
            "epoch_num": epoch_num,
            "step_num": step_num,
            "timeout": timeout,
            "batch_size": batch_size,
            "lr": lr,
            "normalize_method": normalize_method,
            "discretize_t2": discretize_t2,
            "discretize_t1": discretize_t1,
            "discretize_reg_mode": discretize_reg_mode,
            "discretize_cluster_reg_weight": discretize_cluster_reg_weight,
            "discretize_boundary_reg_weight": discretize_boundary_reg_weight,
            "discretizer_type": discretizer_type,
            "discretizer_kwargs": dict(discretizer_kwargs or {}),
            "scale_continuous_features": scale_continuous_features,
            "tau_start": tau_start,
            "tau_end": tau_end,
            "tau_schedule": tau_schedule,
            "tau_decay_start_ratio": tau_decay_start_ratio,
            "tau_decay_end_ratio": tau_decay_end_ratio,
            "random_seed": random_seed,
            "optimize_metric": optimize_metric,
            "backtrack_dacc": backtrack_dacc,
            "backtrack_dF1": backtrack_dF1,
            "interval_num": interval_num,
            "dec_level_cnt": dec_level_cnt,
            "ensembles_cnt": ensembles_cnt,
            "width_cap": width_cap,
            "residual_link": residual_link,
            "hidden_dim_regress": hidden_dim_regress,
            "learned_dd_dot_edge_label_mode": learned_dd_dot_edge_label_mode,
            "learned_dd_dot_show_legend": learned_dd_dot_show_legend,
            "iterative_iterations": iterative_iterations,
            "iterative_build_final_mdd": iterative_build_final_mdd,
            "iterative_require_compatible_ordering": iterative_require_compatible_ordering,
        },
        "train_dataset_statistic": {
            "dataset_size": dataset_size,
            "feats_cnt": feats_cnt,
            "target_dim": target_dim,
            "target_distribution": target_distribution,
            "feats_name": feats_name,
            "featvals_min": featvals_min,
            "featvals_max": featvals_max,
            "feature_scaling": {
                **feature_scaling,
                "original_min": train_dataset.get_original_featvals_min(),
                "original_max": train_dataset.get_original_featvals_max(),
            },
        },
        "training_summary": {
            "elapsed_time": time.time() - start_time,
            "timing": training_timing,
            "stop_reason": stop_reason,
            "stop_mode": stop_mode,
            "global_step": global_step,
            "step_num": step_num,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_metric": best_metric,
            "best_acc": best_acc if not is_regression else None,
            "best_f1_macro": best_f1_macro if not is_regression else None,
            "model_select_source": "valid" if valid_dataset is not None else "train",
            "history": history,
        },
        "train_evaluation": train_evaluation,
        "valid_evaluation": valid_evaluation,
        "test_evaluation": test_evaluation,
        "discretized_features": {
            "names": discretized_feats_name,
            "centers": centers,
        },
    }
    if return_model:
        result["model"] = model
    return result


