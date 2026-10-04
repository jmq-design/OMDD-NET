import torch
import math
from torch.utils.data import Dataset, DataLoader
import os
# import json
import pandas as pd
from typing import Dict, Optional, Sequence


class myDataset(Dataset):
    '''
        myDataset.

        The data is in a .csv file where 
        - the header indicates the corresponding feature names;
        - the last column is the label (class for classification / goal for regression).
    '''
    def __init__(
        self,
        dataset_dir,
        device=None,
        scale_continuous_features: bool = False,
        feature_scaling: Optional[Dict] = None,
    ):

        self.feats_name = pd.read_csv(dataset_dir, nrows=0).columns.values.tolist()[:-1] # get the header (features_name)
        raw_dataset = pd.read_csv(dataset_dir, header=0).values  # start from line 0
        raw_x = torch.tensor(raw_dataset[:, :-1], dtype=torch.float32)
        self.original_featvals_min = raw_x.min(dim=0).values.tolist()
        self.original_featvals_max = raw_x.max(dim=0).values.tolist()
        self.scale_continuous_features = scale_continuous_features

        if scale_continuous_features:
            if feature_scaling is None:
                feature_min = raw_x.min(dim=0).values
                feature_max = raw_x.max(dim=0).values
                feature_range = feature_max - feature_min
                feature_scale = torch.where(
                    feature_range > 0,
                    feature_range,
                    torch.ones_like(feature_range),     # avoid using 0 as the divisor
                )
                self.feature_scaling = {
                    "enabled": True,
                    "method": "train_min_max",
                    "min": feature_min.tolist(),
                    "range": feature_range.tolist(),
                    "scale": feature_scale.tolist(),
                }
            else:
                self.feature_scaling = dict(feature_scaling)
                feature_min = torch.tensor(self.feature_scaling["min"], dtype=torch.float32)
                if "scale" in self.feature_scaling:
                    feature_scale = torch.tensor(self.feature_scaling["scale"], dtype=torch.float32)
                else:
                    feature_range = torch.tensor(self.feature_scaling["range"], dtype=torch.float32)
                    feature_scale = torch.where(
                        feature_range > 0,
                        feature_range,
                        torch.ones_like(feature_range),
                    )
                    self.feature_scaling["scale"] = feature_scale.tolist()
            data_x = (raw_x - feature_min) / feature_scale
        else:
            self.feature_scaling = {
                "enabled": False,
                "method": None,
                "min": self.original_featvals_min,
                "range": [
                    float(maximum - minimum)
                    for minimum, maximum in zip(self.original_featvals_min, self.original_featvals_max)
                ],
                "scale": [
                    float(maximum - minimum) if maximum > minimum else 1.0
                    for minimum, maximum in zip(self.original_featvals_min, self.original_featvals_max)
                ],
            }
            data_x = raw_x

        _data_y = pd.DataFrame({'data_y': raw_dataset[:, -1]})
        _data_y['data_y'] = _data_y['data_y'].astype('category')
        self.label_distribution = _data_y['data_y'].value_counts(normalize=True).to_dict()

        self.data_x = data_x
        self.data_y = torch.tensor(raw_dataset[:, -1], dtype=torch.float32)
        self.featvals_min = self.data_x.min(dim=0).values.tolist()
        self.featvals_max = self.data_x.max(dim=0).values.tolist()

        self.device = device

        assert self.get_feats_cnt()==len(self.data_x[0])
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        # to implement in its sub-class
        pass

    def get_feats_name(self):
        return self.feats_name

    def get_feats_cnt(self):
        return len(self.feats_name)

    def get_examples_cnt(self):
        return len(self.data_x)
        
    def get_data_x(self):
        return self.data_x

    def get_data_y(self):
        return self.data_y

    def get_featvals_min(self):
        return self.featvals_min

    def get_featvals_max(self):
        return self.featvals_max

    def get_original_featvals_min(self):
        return self.original_featvals_min

    def get_original_featvals_max(self):
        return self.original_featvals_max

    def get_feature_scaling(self):
        return dict(self.feature_scaling)

    def size(self):
        return len(self.data_x), self.get_feats_cnt()


class ClassificationDataset(myDataset):
    '''
        ClassificationDataset.

    '''
    def __init__(
        self,
        dataset_dir,
        device=None,
        scale_continuous_features: bool = False,
        feature_scaling: Optional[Dict] = None,
        label_values: Optional[Sequence] = None,
    ):
        myDataset.__init__(self, dataset_dir, device, scale_continuous_features, feature_scaling)

        labels = list(label_values) if label_values is not None else list(self.label_distribution.keys())
        labels = sorted(labels)
        self.classes_cnt = len(labels)
        # mapping the old labels to the new ones {0, 1, ..., classes_cnt-1} (according to their indices in the sorted list)
        self.labels_mapping_old2new = {old_label: ith for ith, old_label in enumerate(labels)} 
        self.labels_mapping_new2old = {ith : old_label for ith, old_label in enumerate(labels)} 
        data_y_renamed = [self.labels_mapping_old2new[elem] for elem in self.data_y.tolist()]
        data_y_renamed = torch.tensor(data_y_renamed, dtype=torch.long)

        _one_hot_data_y = torch.nn.functional.one_hot(data_y_renamed, num_classes=self.classes_cnt)
        self.one_hot_data_y = _one_hot_data_y.float()  

        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.one_hot_data_y[idx]
    
    def get_data_y_onehot(self):
        return self.one_hot_data_y
    
    def interpret_y_pred_from_one_hot_indices(self, y_pred__one_hot_indices):
        assert max(y_pred__one_hot_indices)<self.classes_cnt and min(y_pred__one_hot_indices)>=0
        y_pred = []
        for elem in y_pred__one_hot_indices:
            y_pred.append(self.labels_mapping_new2old[elem])
        return y_pred

    def get_classes_cnt(self):
        return self.classes_cnt 
    
    def get_class_distribution(self):
        return self.label_distribution

    def get_dataset_statistic(self):
        return self.get_examples_cnt(), \
            self.get_feats_cnt(), \
            self.get_classes_cnt(), \
            self.get_class_distribution(), \
            self.get_feats_name(), \
            self.get_featvals_min(), \
            self.get_featvals_max()


class RegressionDataset(myDataset):
    '''
        RegressionDataset.

    '''
    def __init__(
        self,
        dataset_dir,
        device=None,
        scale_continuous_features: bool = False,
        feature_scaling: Optional[Dict] = None,
    ):
        myDataset.__init__(self, dataset_dir, device, scale_continuous_features, feature_scaling)

        self.goal_scope = (self.data_y.min().item(), self.data_y.max().item())
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
    def get_goal_scope(self):
        return self.goal_scope

    def get_goal_distribution(self):
        return self.label_distribution

    def get_dataset_statistic(self):
        return self.get_examples_cnt(), \
            self.get_feats_cnt(), \
            self.get_goal_scope(), \
            self.get_goal_distribution(), \
            self.get_feats_name(), \
            self.get_featvals_min(), \
            self.get_featvals_max()



def _test_classification_dataset():
    dataset_dir = "../datasets/xxx.csv"
    device = 'cpu'
    dataset = ClassificationDataset(dataset_dir, device)

    dataset_size, feats_cnt, classes_cnt, class_distribution,\
                        feats_name, featvals_min, featvals_max = dataset.get_dataset_statistic()

    print(f"\n\nThe statistic of dataset {dataset_dir}: \ndataset_size: {dataset_size} \n"
            + f"feats_cnt: {feats_cnt} \n"
            # + f"examples_cnt: {dataset.get_examples_cnt()} \n"
            + f"classes_cnt: {classes_cnt} \n"
            + f"class_distribution: {class_distribution} \n"
            + f"feats_name: {feats_name} \n"
            + f"featvals_min: {featvals_min} \n"
            + f"featvals_max: {featvals_max} \n"
            )



if __name__ == '__main__':

    _test_classification_dataset()
    




    
