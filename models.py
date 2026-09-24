# models

# imports:
import torch
from torch import nn
import torch.nn.functional as F
from GeM_Classifier import multiple_featuremaps_GEM_classification, t1_t2_GEM_head
from feature_extractor import  nnUnet_feature_maps_multiple

class GEM_multiple_classification(nn.Module):
    def __init__(self, nnunet_pathway, frozen_decoder, in_channels,label_num):
        super(GEM_multiple_classification, self).__init__()
        self.nnunet_feature_maps = nnUnet_feature_maps_multiple(nnunet_pathway, frozen_decoder)
        self.GEM_head = multiple_featuremaps_GEM_classification(in_channels, label_num)

    def forward(self, img, seg):
        feature_maps1, feature_maps2, feature_maps3, feature_maps4, feature_maps5, feature_maps6 = self.nnunet_feature_maps(img)
        classification = self.GEM_head(feature_maps1, feature_maps2, feature_maps3,feature_maps4,feature_maps5, seg)
        return classification, feature_maps1

class GEM_multiple_T1_T2_classification(nn.Module):
    def __init__(self, nnunet_pathway_t2, nnunet_pathway_t1, frozen_decoder, in_channels,label_num, modality_type):
        super(GEM_multiple_T1_T2_classification, self).__init__()
        self.nnunet_feature_maps1 = nnUnet_feature_maps_multiple(nnunet_pathway_t1, frozen_decoder)
        self.nnunet_feature_maps2 = nnUnet_feature_maps_multiple(nnunet_pathway_t2, frozen_decoder)
        self.GEM_head = t1_t2_GEM_head(in_channels, label_num, modality_type)

    def forward(self, t1, t2, seg):
        t1_1, t1_2, t1_3, t1_4, t1_5, t1_6 = self.nnunet_feature_maps1(t1)
        t2_1,t2_2, t2_3, t2_4, t2_5, t2_6 = self.nnunet_feature_maps2(t2)
        classification = self.GEM_head(seg, t1_1, t1_2, t1_3, t1_4, t1_5, t2_1,t2_2, t2_3, t2_4, t2_5)
        return classification, t1_1

