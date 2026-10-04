import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.simplified_binarize_layer import SimpBinarizeLayer

from typing import Dict, List, Tuple, Set
from torch import Tensor

TAU_SWITCH_TO_SOFTMAX = 2.0
EPS = 1e-8


def _build_encoding(params: Tensor, dim: int, normalize_method: str, tau=None, test_flag: bool=False):
    assert dim==-1 or dim==len(params.size())-1

    if test_flag or normalize_method == "softmax":
        return torch.softmax(params, dim=dim)
    if normalize_method == "gumbel_softmax":
        # 先不要基于阈值的调度
        # if tau is None or float(tau) > TAU_SWITCH_TO_SOFTMAX:
        #     return F.gumbel_softmax(params, tau=tau, hard=False, dim=dim)
        # return torch.softmax(params, dim=dim)
        # 
        return F.gumbel_softmax(params, tau=tau, hard=False, dim=dim)
    elif normalize_method == "sigmoid":     # use for ablation study 
        return F.sigmoid(params)  
    elif normalize_method == "argmax":      # just use for debugging
        with torch.no_grad():
            max_ids = torch.argmax(params, dim=dim)
            one_hot = F.one_hot(max_ids, num_classes=params.size(dim)).to(torch.float32) # compute the one-hot representation of params
        soft = torch.softmax(params, dim=dim)
        return one_hot + soft - soft.detach() 
    raise ValueError(f"the normalization method {normalize_method} is not supported.")



def _normalize_linearly(params, dim):

    pos_params = torch.sigmoid(params)
    total = torch.sum(pos_params, dim=dim, keepdim=True)
    
    epsilon = torch.finfo(torch.float32).eps  
    total = torch.clamp(total, min=epsilon) 

    normalized_params = pos_params / total
    
    return normalized_params



def argmax_obeys_readonce_up2bottom(martrix : Tensor):      # martrix : 2D-tensor with row_cnt<=col_cnt
    assert len(martrix.size())==2
    assert martrix.size(0) <= martrix.size(1)

    row_cnt = martrix.size(0)
    excluded_mask = torch.zeros(martrix.size(1), dtype=torch.bool, device=martrix.device)
    max_val_idx_s = []
    max_val_s = []
    for row in range(row_cnt):
        allowed_indices = torch.nonzero(~excluded_mask, as_tuple=True)[0]
        allowed_values = martrix[row].index_select(0, allowed_indices)
        max_val_tensor, local_max_idx_tensor = torch.max(allowed_values, dim=0)
        max_val = max_val_tensor.detach().cpu().item()
        max_val_idx = int(allowed_indices[local_max_idx_tensor].detach().cpu().item())
        max_val_idx_s.append(max_val_idx)
        max_val_s.append(max_val)
        excluded_mask[max_val_idx] = True
    return max_val_idx_s, max_val_s

class ClippingLayer(torch.nn.Module):
    '''
        Class ClippingLayer. apply the minmax operation (a variant of relu) to a tensor
    '''
    def __init__(self, **kwargs):
        super(ClippingLayer, self).__init__(**kwargs)
    
    def forward(self, x):
        x[x<0] = x[x<0]*0.01        # for an element<0  
        x[x>1] = x[x>1]*0.01 + 0.99 # for an element>1 
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

clippingLayer = ClippingLayer()



class DiscretizeLayer(nn.Module):
    '''
        DiscretizeLayer.
    '''
    def __init__(self,
                cont_features_cnt : int,
                interval_num : int,
                featvals_min : List[float],
                featvals_max : List[float],
                t1=50, t2=1000,
                discretize_reg_mode: str = "hybrid",
                discretize_cluster_reg_weight: float = 1.0,
                discretize_boundary_reg_weight: float = 0.1,
                device : str='cuda:0' if torch.cuda.is_available() else 'cpu',
                verbose=False):
    
        assert cont_features_cnt==len(featvals_min)==len(featvals_max)

        super(DiscretizeLayer, self).__init__()

        self.cont_features_cnt = cont_features_cnt
        self.interval_num = interval_num
        self.featvals_min = featvals_min
        self.featvals_max = featvals_max

        self.verbose = verbose
        self.device = device

        equal_interval = [(val_max-val_min)/interval_num for val_max, val_min in zip(featvals_max, featvals_min)]
        interval = torch.tensor(equal_interval).unsqueeze(1).repeat(1,interval_num) # shape(cont_features_cnt, interval_num) 
        # interval.to(device)

        ''' adpot simplified BinarizeLayer of AutoInt as the differentiable binarization method'''
        self.binarizeLayer = SimpBinarizeLayer(
                                            input_dim=[0, cont_features_cnt], 
                                            i_min=featvals_min, 
                                            i_max=featvals_max, 
                                            t1=t1, t2=t2, 
                                            interval=interval, 
                                            regularization_mode=discretize_reg_mode,
                                            cluster_reg_weight=discretize_cluster_reg_weight,
                                            boundary_reg_weight=discretize_boundary_reg_weight,
                                            interval2=None, 
                                            use_not=False,
                                            use_kmeans_loss=False,
                                            )

    def reset_parameters(self):
        pass
        
    def freeze_parameters(self, reset_uniform_boundaries:bool=False):
        self.binarizeLayer.freeze_parameters(reset_uniform_boundaries)


    def reset_verbose(self, verbose):
        self.verbose = verbose

    def forward(self, data_x, tau=None, test_flag=False): 
        discretized_data_x, self.kmeans_loss = self.binarizeLayer.forward(data_x, tau=tau, test_flag=test_flag)     # shape(batch_size, features_cnt*k-valued)
        if self.verbose:
            with torch.no_grad():
                print("* discretizeLayer forward.")
                print("\t- discretized_data_x: ", discretized_data_x)
        return discretized_data_x
     
    def regularization(self, coefficients : Dict[str, float]):
        coef = "coef. discretization"
        assert coef in coefficients.keys()

        reg_loss_details = {}

        reg_loss = torch.tensor(0.0, device=self.device)
        if coefficients[coef] != 0:
            reg_loss = coefficients[coef] * self.binarizeLayer.regularization()

            reg_loss_details["disc_loss"] = float(reg_loss.detach().cpu().item())

        return reg_loss, reg_loss_details


    def get_discretized_feats_name(
        self,
        feats_name: List[str],
        mean=None,
        std=None,
        return_centers: bool = True,
    ):
        return self.binarizeLayer.get_bound_name(
            feats_name,
            mean=mean,
            std=std,
            return_centers=return_centers,
        )



class CategFeatureSelectLayer(nn.Module):
    '''
        CategFeatureSelectLayer.
    '''
    def __init__(self,
                features_cnt : int,
                k_valued : int,
                # net_topology : NetTopology,  
                dec_levels_cnt : int,
                normalize_method : str="gumbel_softmax",
                device : str='cuda:0' if torch.cuda.is_available() else 'cpu',
                verbose=False):
        super(CategFeatureSelectLayer, self).__init__()

        self.dec_levels_cnt = dec_levels_cnt
        self.features_cnt = features_cnt
        self.k_valued = k_valued
        self.normalize_method = normalize_method

        self.verbose = verbose
        self.device = device


        ''' define trainable parameters'''
        self.params_prob_feat_select = Parameter(torch.randn(self.dec_levels_cnt, features_cnt)) 

        self.reset_parameters()
        

    def reset_parameters(self):
        nn.init.orthogonal_(self.params_prob_feat_select)
    
    def reset_verbose(self, verbose):
        self.verbose = verbose

    def _build_FeatureSelectLayer_encoding(self, tau=None, test_flag=False):
        '''
            :param - tau:   the temperature parameter used in Gumbel Softmax function
            :param - test_flag:   if True, apply the softmax function as the normalize_method to get an encoding

        '''
        return _build_encoding(
            self.params_prob_feat_select,
            dim=1,
            normalize_method=self.normalize_method,
            tau=tau,
            test_flag=test_flag,
        )
    
    def build_FeatureSelectLayer_encoding(self, tau=None, test_flag=False):
        self.enc_prob_feat_select = self._build_FeatureSelectLayer_encoding(tau=tau, test_flag=test_flag)


    def _interpret_faithful_encoding(self):
        '''interpret faithful encoding from the trained parameters. '''
        with torch.no_grad():
            self.build_FeatureSelectLayer_encoding(test_flag=True)
            max_ids, _ = argmax_obeys_readonce_up2bottom(self.enc_prob_feat_select)
            max_ids = torch.Tensor(max_ids).long().to(self.device)
            self.faithfulenc_prob_feat_select = F.one_hot(max_ids, num_classes=self.enc_prob_feat_select.size(1))  # compute the one-hot representation of enc_prob_feat_select 


    def _forward(self, data_x, interpret_flag=False):
        if not interpret_flag:
            prob_feat_select = self.enc_prob_feat_select
        else:
            prob_feat_select = self.faithfulenc_prob_feat_select.float()

        batch_size = len(data_x)

        reshaped_data_x = data_x.reshape(batch_size, self.features_cnt, self.k_valued)   # shape(batch_size, features_cnt, k_valued)
        _branch_prob = prob_feat_select.matmul(reshaped_data_x) # shape(batch_size, dec_levels_cnt, inout_width[is-exactly-k_valued])
        branch_prob = _branch_prob.permute(1, 0, 2) # shape(dec_levels_cnt, batch_size, inout_width)
        branch_prob4in_per_layer = branch_prob.unsqueeze(2)    # shape(dec_levels_cnt, batch_size, 1[To_Expand_to__in_height], inout_width)

        branch_prob4in_per_layer = clippingLayer( branch_prob4in_per_layer )
        return branch_prob4in_per_layer



    def forward(self, data_x, tau=None, test_flag=False, interpret_flag=False):  

        assert data_x.size()[-1]==self.features_cnt*self.k_valued # shape(batch_size, features_cnt*k_valued), with each row in the form of [k_values_of_feature_0, k_values_of_feature_1, ...] 
        # batch_size = len(data_x)

        if not interpret_flag:
            self.build_FeatureSelectLayer_encoding(tau=tau, test_flag=test_flag)

        branch_prob4in_per_layer = self._forward(data_x, interpret_flag=interpret_flag)

        if self.verbose:
            with torch.no_grad():
                # print("** DDNet Forward.")
                print("* featSelectLayer forward.")
                print("\t- enc_prob_feat_select: ", self.enc_prob_feat_select)
                if interpret_flag:
                    print("\t- faithfulenc_prob_feat_select: ", self.faithfulenc_prob_feat_select)

                # print("\t- branch_prob4in_per_layer: ", branch_prob4in_per_layer)

        return branch_prob4in_per_layer


    def regularization(self, coefficients : Dict[str, float]):
        coef_read_once = "coef. read_once"
        coef_ent_feat_select = "coef. ent_feat_select"
        # 
        coef_margin_feat_select = "coef. margin_feat_select"
        margin_target_key = "margin_feat_select_target"
        # 
        reg_loss_details = {}

        assert (coef_read_once in coefficients.keys()) and (coef_ent_feat_select in coefficients.keys())

        # 
        enc_prob_feat_select = self._build_FeatureSelectLayer_encoding(test_flag=True) 
        dec_levels_cnt, features_cnt = enc_prob_feat_select.size()

        reg_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = reg_loss.new_zeros(())
        read_once_loss = reg_loss.new_zeros(())
        # 
        margin_loss = reg_loss.new_zeros(())
        margin_mean = reg_loss.new_zeros(())
        margin_min = reg_loss.new_zeros(())

        if coefficients[coef_read_once]!=0:
            read_once_loss_per_column = torch.relu(torch.sum(enc_prob_feat_select, dim=0) - 1) / dec_levels_cnt # shape(features_cnt)
            read_once_loss = read_once_loss_per_column.mean()
            reg_loss += coefficients[coef_read_once] * read_once_loss

            reg_loss_details["read_once_loss"] = float(coefficients[coef_read_once] * read_once_loss.detach().cpu().item())

        if coefficients[coef_ent_feat_select]!=0:
            entropy_per_row = - (enc_prob_feat_select.mul(torch.log(enc_prob_feat_select + EPS))).sum(dim=1)   # shape(dec_levels_cnt)
            entropy_norm = entropy_per_row / math.log(features_cnt) 
            entropy_loss = entropy_norm.mean() 
            reg_loss += coefficients[coef_ent_feat_select] * entropy_loss

            reg_loss_details["ent_feat_select_loss"] = float(coefficients[coef_ent_feat_select] * entropy_loss.detach().cpu().item())

        ''' feature-selection margin regularization '''
        margin_coef = coefficients.get(coef_margin_feat_select, 0.0)
        if margin_coef != 0:
            margin_target = float(coefficients.get(margin_target_key, 0.2))
            top2 = torch.topk(enc_prob_feat_select, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
            margin_mean = margins.mean()
            margin_min = margins.min()
            margin_loss = torch.relu(margin_target - margins).pow(2).mean()
            reg_loss += margin_coef * margin_loss

            reg_loss_details["margin_feat_select_loss"] = float(margin_coef * margin_loss.detach().cpu().item())
            reg_loss_details["feat_select_margin_mean"] = float(margin_mean.detach().cpu().item())
            reg_loss_details["feat_select_margin_min"] = float(margin_min.detach().cpu().item())


        return reg_loss, reg_loss_details



class DecisionLayer(nn.Module):
    '''
        DecisionLayer.

    '''
    def __init__(self, 
                in_height, 
                cur_out_height, 
                inout_width, 
                skip_out_height=None, 
                device : str='cuda:0' if torch.cuda.is_available() else 'cpu',
                normalize_method : str="gumbel_softmax",
                verbose=False):

        super(DecisionLayer, self).__init__()
        self.in_height = in_height
        self.cur_out_height = cur_out_height
        self.skip_out_height = skip_out_height
        self.inout_width = inout_width
        self.normalize_method = normalize_method

        self.verbose = verbose
        self.device = device

        ''' define trainable parameters'''
        self.params_in2cur_out = Parameter(torch.randn(inout_width, in_height, cur_out_height)) # 3-D tensor: the probability to consecutive layers connections
        self.params_in2skip_out = None if self.skip_out_height is None \
                        else Parameter(torch.randn(inout_width, in_height, skip_out_height))    # 3-D tensor: the probability to consecutive layers connections
                        
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.inout_width):
            nn.init.orthogonal_(self.params_in2cur_out[k])

        if self.skip_out_height is None:
            pass
        else:
            for k in range(self.inout_width):
                nn.init.orthogonal_(self.params_in2skip_out[k])
        
        if (self.in_height==self.cur_out_height==1) and (self.skip_out_height is None):     # for the case of EnsembleLayer with ensembles_cnt=1
            self.params_in2cur_out.requires_grad = False
            self.params_in2cur_out.data.fill_(1)

    def reset_verbose(self, verbose):
        self.verbose = verbose


    def _build_DecisionLayer_encoding(self, tau=None, test_flag=False):
        '''
            :param - tau:   the temperature parameter used in Gumbel Softmax function
            :param - test_flag:   if True, apply the softmax function as the normalize_method to get an encoding

        '''
        if self.skip_out_height is None:
            in2out = self.params_in2cur_out
        else:
            in2out = torch.cat((self.params_in2cur_out, self.params_in2skip_out), dim=2)

        if self.normalize_method=="normalize_linearly":
            return _normalize_linearly(in2out, dim=2)
        else:
            return _build_encoding(
                in2out,
                dim=2,
                normalize_method=self.normalize_method,
                tau=tau,
                test_flag=test_flag,
            )
    

    def build_DecisionLayer_encoding(self, tau=None, test_flag=False):
        self.enc_in2out = self._build_DecisionLayer_encoding(tau=tau, test_flag=test_flag)


    def _interpret_faithful_encoding(self):
        '''interpret faithful encoding from the trained parameters. '''
        with torch.no_grad():
            self.build_DecisionLayer_encoding(test_flag=True)
            max_ids = torch.argmax(self.enc_in2out, dim=2)
            self.faitfulenc_in2out = F.one_hot(max_ids, num_classes=self.enc_in2out.size(2))  # compute the one-hot representation of enc_in2out 

    
    def _forward(self, reach_prob4in, branch_prob4in, interpret_flag=False):

        if interpret_flag:
            trans_in2out = self.faitfulenc_in2out
        else:
            trans_in2out = self.enc_in2out

        temp_in2out = trans_in2out.permute(2, 1, 0)  
        _temp_in2out = temp_in2out.unsqueeze(0) # shape(1[To_Expand_to__batch_size], cur_out_height+skip_out_height, in_height, inout_width) 
        _branch_prob4in = branch_prob4in.unsqueeze(1) # shape(batch_size, 1[To_Expand_to__out_height], 1[To_Expand_to__in_height], inout_width)
        in2out_flat = torch.sum(_branch_prob4in.mul(_temp_in2out), dim=-1).permute(0, 2, 1)  #  shape(batch_size, in_height, cur_out_height+skip_out_height)  
        in2out_flat = clippingLayer(in2out_flat)  
        reach_prob4out = reach_prob4in.matmul(in2out_flat)  #  shape(batch_size, 1, cur_out_height+skip_out_height)
        
        reach_prob4cur_out = reach_prob4out[:,:,:self.cur_out_height]   # shape(batch_size, 1, cur_out_height)
        reach_prob4skip_out = None if self.skip_out_height is None else reach_prob4out[:,:,self.cur_out_height:]  # shape(batch_size, 1, skip_out_height)
        # 
        reach_prob4cur_out = clippingLayer( reach_prob4cur_out )   # shape(batch_size, 1, cur_out_height)
        reach_prob4skip_out = None if reach_prob4skip_out is None else clippingLayer( reach_prob4skip_out )  # shape(batch_size, 1, skip_out_height)


        return reach_prob4cur_out, reach_prob4skip_out

    
    def forward(self, reach_prob4in, branch_prob4in, tau=None, test_flag=False, interpret_flag=False):

        if not interpret_flag:
            self.build_DecisionLayer_encoding(tau=tau, test_flag=test_flag)

        reach_prob4cur_out, reach_prob4skip_out = self._forward(reach_prob4in, branch_prob4in, interpret_flag=interpret_flag)

        if self.verbose:
            print("params_in2cur_out:", self.params_in2cur_out)
            print("enc_in2out:", self.enc_in2out)
            if interpret_flag:
                print("faitfulenc_in2out:", self.faitfulenc_in2out)

        return reach_prob4cur_out, reach_prob4skip_out
    

    def regularization(self, coefficients : Dict[str, float]):
        coef_ent_dec = "coef. ent_dec"
        assert coef_ent_dec in coefficients.keys()

        reg_loss = torch.tensor(0.0, device=self.device)

        enc_in2out = self._build_DecisionLayer_encoding(test_flag=True) 
        inout_width, in_height, out_height = enc_in2out.size()

        if coefficients[coef_ent_dec]!=0:
        
            entropy_per_row = - (enc_in2out.mul(torch.log(enc_in2out + EPS))).sum(dim=2)   # shape(inout_width, in_height)
            entropy_norm = entropy_per_row / math.log(out_height) 
            entropy_loss = entropy_norm.mean() 
            reg_loss += coefficients[coef_ent_dec] * entropy_loss
        return reg_loss


class RegressionLayer(nn.Module):
    '''
        RegressionLayer.

    '''
    def __init__(self, 
                in_dim, 
                hidden_dim, 
                out_dim, 
                activation_func_method : str=None,
                device : str='cuda:0' if torch.cuda.is_available() else 'cpu',
                verbose=False):

        super(RegressionLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation_func_method = activation_func_method

        self.verbose = verbose
        self.device = device

        ''' define the network'''
        if activation_func_method is None:
            activation_func = None
        elif activation_func_method=="ReLU":
            activation_func = nn.ReLU()
        elif activation_func_method=="sigmoid":
            activation_func = nn.Sigmoid()
        elif activation_func_method=="softmax":
            activation_func = nn.Softmax()
        else:
            raise ValueError()

        if activation_func is None:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                activation_func,
                nn.Linear(hidden_dim, out_dim),
                activation_func
            )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def reset_verbose(self, verbose):
        self.verbose = verbose

    def forward(self, x):
        return self.layers(x)


