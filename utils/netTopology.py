from typing import Dict, List, Tuple, Set, Union
from torch import Tensor



class DDNetTopology(object):
    '''
        DDNetTopology.

    '''
    def __init__(self, 
                input_dim : int,
                output_dim : int,
                DecLevel_cnt : int,
                DecLevel_shape_s : List[Tuple[int]],
                residual_link : Dict[int,int]=None,
                ensembles_cnt : int=1,
                is_regression : bool=False,
                hidden_dim_regress : int=16,
                verbose=False):
        '''
            Let DecLevel_cnt = n.
            Topology of Network :                                         INPUT
                                                     ---------------------- ↓ ------------------
                                            TOImpletment[ FeatureSplitLayer ]                   |
                                                    ↓                                           ↓
                                    [1*EnsembleLayer → (n)*DecisionLayer]         OPtional[ 1*RegressionLayer ]
                                                    ↓                                           ↓
                                                    -------------------- OUTPUT -----------------
                                    
            NOTE: EnsembleLayer is a special DecisionLayer that the in_height is set 1 with no associated feature
            NOTE: DecLevel refers to a single level of a DecisionLayer ("in" level). Consecutive DecLayers have not necessary the same inout_width.

            :param - input_dim:  number of the (initial) input features
            :param - output_dim:  number of the (final) ouput 
            :param - DecLevel_cnt:  number of the DecisionLayers
            :param - DecLevel_shape_s:   list of 2-tuples with each indicating the (in-)shape (in_height, inout_width) of a decision layers
            :param - residual_link:  residual links among the DecisionLayers
            :param - ensembles_cnt:  number of the ensembles (i.e., the out_height of EnsembleLayer)
            :param - is_regression:  whether the learning task is regression. If True, ODD-Net acts as a regressor, otherwise as a classifier
            :param - hidden_dim_regress:  dimension of the hidden layer in RegressionLayer
        '''
        assert DecLevel_cnt>0 
        assert DecLevel_cnt==len(DecLevel_shape_s)
        for DecLevel_shape in DecLevel_shape_s:
            assert len(DecLevel_shape)==2
        if residual_link:
            for key, value in residual_link.items():
                assert 0 <= key <= DecLevel_cnt-1, f"the source {key} of a residual link much be a valid Decision Level"
                assert 0 <= value <= DecLevel_cnt-1, f"the end {value} of a residual link much be a valid Decision Level"

        self.input_dim = input_dim
        self.output_dim = output_dim


        self.DecLevel_cnt = DecLevel_cnt
        self.DecLevel_shape_s = DecLevel_shape_s
        self.residual_link = residual_link

        self.ensembles_cnt = ensembles_cnt
        self.EnsLayer_inshape = (1, 1)

        self.is_regression = is_regression
        self.hidden_dim_regress = hidden_dim_regress

        self.verbose = verbose


    def get_input_dim(self):
        return self.input_dim
    
    def get_output_dim(self):
        return self.output_dim

    def get_hidden_dim_regress(self):
        return self.hidden_dim_regress


    def get_DecLevel_cnt(self):     # the same as DecisionLayers_cnt
        return self.DecLevel_cnt 

    def get_DecisionLayers_cnt(self):
        return self.DecLevel_cnt

    def get_DecVertices_cnt(self):
        DecVertices_cnt = 0
        for ith in range(self.DecLevel_cnt):
            DecVertices_cnt += self.DecLevel_shape_s[ith][0]
        return DecVertices_cnt

    def get_ensembles_cnt(self):
        return self.ensembles_cnt



    def get_ith_DecLayer_inshape(self, ith):      
        assert ith<self.DecLevel_cnt
        return self.DecLevel_shape_s[ith]

    def get_ith_DecLayer_inheight(self, ith):      
        # assert ith<self.DecLevel_cnt
        return self.get_ith_DecLayer_inshape(ith)[0]

    def get_ith_DecLayer_inoutwidth(self, ith):      
        # assert ith<self.DecLevel_cnt
        return self.get_ith_DecLayer_inshape(ith)[1]

    def get_ith_DecLayer_residual_link(self, ith):      
        assert ith<self.DecLevel_cnt
        if (self.residual_link is not None) and  (ith in self.residual_link.keys()):
            return self.residual_link[ith]
        else:
            return None


    def get_EnsLayer_inshape(self):
        return self.EnsLayer_inshape

    def get_EnsLayer_inheight(self):
        return self.get_EnsLayer_inshape()[0]
        
    def get_EnsLayer_inoutwidth(self):
        return self.get_EnsLayer_inshape()[1]

    # ---

    # NOTE: the shape of a DecLevel is of form: "(in_height, inout_width)", corresponding to the in-shape of a DecisionLayer(short for DecLayer);
    #       while the shape of a DecLayer is of form: "(in_height, cur_out_height, inout_width, skip_out_height)";
    def get_ith_DecLayer_shape(self, ith):
        assert ith<self.DecLevel_cnt

        residual_link = self.get_ith_DecLayer_residual_link(ith)
        if residual_link:
            skip_out_height = self.get_ith_DecLayer_inheight(residual_link)
        else:
            skip_out_height = None

        cur_out_height = self.get_output_dim() if ith==(self.DecLevel_cnt-1) else self.get_ith_DecLayer_inheight(ith+1)
        return self.get_ith_DecLayer_inheight(ith), cur_out_height, self.get_ith_DecLayer_inoutwidth(ith), skip_out_height

    def get_EnsLayer_shape(self):
        return self.get_EnsLayer_inheight(), self.get_ith_DecLayer_inheight(0), self.get_EnsLayer_inoutwidth()

    def get_RegressLayer_shape(self):
        return self.input_dim, self.hidden_dim_regress, self.output_dim

    def print_NetTopology(self):
        print("** NetTopology.")
        print("* FORMAT of a layer: TypeofLayer | FormofShape (in_dim, Optional[ hidden_dim ], out_dim, Optional[ inout_width ]) | DetailofShape")
        print("\t- EnsembleLayer: \t | (in_height, out_height, inout_width) \t | ", tuple(self.get_EnsLayer_shape()))
        print("\t- DecisionLayer: \t | (in_height, out_height, inout_width) ")
        for ith in range(self.DecLevel_cnt):
            print(f"\t\t-- DecisionLayer {ith}: ", tuple(self.get_ith_DecLayer_shape(ith)[:-1]))
        if self.is_regression:
            print("\t- RegressionLayer: \t | (in_dim, hidden_dim, out_dim) \t | ", tuple(self.get_RegressLayer_shape()))

        print("* FORMAT of a residual link : DecLevel i -> DecLevel j")
        if self.residual_link is None:
            print("\t- None")
        else:
            for link_s,link_d in self.residual_link.items():
                 print(f"\t- {link_s} -> {link_d} ")
        print("\n")


class OrderedDDNetTopology(DDNetTopology):
    '''
        OrderedDDNetTopology.

    '''
    def __init__(self, 
                input_dim : int,
                output_dim : int,
                interval_num_for_discretize : int,     # new arg
                DecLevel_cnt : int,
                DecLevel_shape_s : List[Tuple[int]],
                residual_link : Dict[int,int]=None,
                ensembles_cnt : int=1,
                is_regression : bool=False,
                hidden_dim_regress : int=16,
                verbose=False):
        '''
            Let DecLevel_cnt = n.
            Topology of Network :                                         INPUT
                                                     ---------------------- ↓ ------------------
                                    [1*DiscretizeLayer → 1*CategFeatureSelectLayer]             |
                                                    ↓                                           ↓
                                    [1*EnsembleLayer → (n)*DecisionLayer]         OPTIONAL[ 1*RegressionLayer ]
                                                    ↓                                           ↓
                                                    -------------------- OUTPUT -----------------
                                    
            :param - interval_num_for_discretize:  pre-defined number of the interval for features discretization in DiscretizeLayer

        '''
        for DecLevel_shape in DecLevel_shape_s:
            assert len(DecLevel_shape)==2
            assert DecLevel_shape[-1]==interval_num_for_discretize, f"the inout_width of each DecisionLayer mush be {interval_num_for_discretize}"

        super().__init__(
                input_dim=input_dim,
                output_dim=output_dim,
                DecLevel_cnt=DecLevel_cnt,
                DecLevel_shape_s=DecLevel_shape_s,
                residual_link=residual_link,
                ensembles_cnt=ensembles_cnt,
                is_regression=is_regression,
                hidden_dim_regress=hidden_dim_regress,
                verbose=verbose)

        # new attributes
        self.interval_num_for_discretize = interval_num_for_discretize

        if self.verbose:
            self.print_NetTopology()
    
    def get_interval_num_for_discretize(self):
        return self.interval_num_for_discretize

    def get_DiscretizeLayer_shape(self):
        return self.input_dim, self.input_dim, self.interval_num_for_discretize  # (in_dim, out_dim, out_width), meaning (cont_features_cnt, discretized_features_cnt, interval_num_for_discretize)
    
    def get_CategFeatureSelectLayer_shape(self):
        return self.input_dim, self.get_DecisionLayers_cnt(), self.interval_num_for_discretize  # (in_dim, out_dim, inout_width), meaning (discretized_features_cnt, dec_levels_cnt, features_domain_size)


    def print_NetTopology(self):
        print("** OrderedDDNetTopology.") 
        print("* FORMAT of a layer: TypeofLayer | FormofShape (in_dim, Optional[ hidden_dim ], out_dim, Optional[ inout_width ]) | DetailofShape")
        print("\t- DiscretizeLayer: \t | (in_dim, out_dim, out_width) \t | ", tuple(self.get_DiscretizeLayer_shape()))
        print("\t- CategFeatureSelectLayer: \t | (in_dim, out_dim, inout_width) \t | ", tuple(self.get_CategFeatureSelectLayer_shape()))
        print("\t- EnsembleLayer: \t | (in_height, out_height, inout_width) \t | ", tuple(self.get_EnsLayer_shape()))
        print("\t- DecisionLayer: \t | (in_height, out_height, inout_width) ")
        for ith in range(self.DecLevel_cnt):
            print(f"\t\t-- DecisionLayer {ith}: ", tuple(self.get_ith_DecLayer_shape(ith)[:-1]))
        if self.is_regression:
            print("\t- RegressionLayer: \t | (in_dim, hidden_dim, out_dim) \t | ", tuple(self.get_RegressLayer_shape()))

        print("* FORMAT of a residual link : DecLevel i -> DecLevel j")
        if self.residual_link is None:
            print("\t- None")
        else:
            for link_s,link_d in self.residual_link.items():
                 print(f"\t- {link_s} -> {link_d} ")
        print("\n")


