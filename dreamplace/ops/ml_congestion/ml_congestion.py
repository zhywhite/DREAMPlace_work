##
# @file   ml_congestion.py
# @author Yibo Lin
# @date   Oct 2022
#

import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb
from scipy import ndimage
import numpy as np

import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pinrudy.pinrudy as pinrudy
############## Your code block begins here ##############
# import your ML model 
from models.gpdl import GPDL
############## Your code block ends here ################

class MLCongestion(nn.Module):
    """
    @brief compute congestion map based on a neural network model 
    @param fixed_node_map_op an operator to compute fixed macro map given node positions 
    @param rudy_utilization_map_op an operator to compute RUDY map given node positions
    @param pinrudy_utilization_map_op an operator to compute pin RUDY map given node positions 
    @param pin_pos_op an operator to compute pin positions given node positions 
    @param xl left boundary 
    @param yl bottom boundary 
    @param xh right boundary 
    @param yh top boundary 
    @param num_bins_x #bins in horizontal direction, assume to be the same as horizontal routing grids 
    @param num_bins_y #bins in vertical direction, assume to be the same as vertical routing grids 
    @param unit_horizontal_capacity amount of routing resources in horizontal direction in unit distance
    @param unit_vertical_capacity amount of routing resources in vertical direction in unit distance
    @param pretrained_ml_congestion_weight_file file path for pretrained weights of the machine learning model 
    """
    def __init__(self,
                 fixed_node_map_op,
                 rudy_utilization_map_op, 
                 pinrudy_utilization_map_op, 
                 pin_pos_op, 
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x,
                 num_bins_y,
                 unit_horizontal_capacity,
                 unit_vertical_capacity,
                 pretrained_ml_congestion_weight_file):
        super(MLCongestion, self).__init__()
        ############## Your code block begins here ##############
        self.fixed_node_map_op = fixed_node_map_op
        self.rudy_utilization_map_op = rudy_utilization_map_op
        self.pinrudy_utilization_map_op = pinrudy_utilization_map_op
        self.pin_pos_op = pin_pos_op
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.unit_horizontal_capacity = unit_horizontal_capacity
        self.unit_vertical_capacity = unit_vertical_capacity
        self.pretrained_ml_congestion_weight_file = pretrained_ml_congestion_weight_file
        ############## Your code block ends here ################

    def __call__(self, pos):
        return self.forward(pos)

    # for data process
    def resize(self, input):
        dimension = input.shape
        result = ndimage.zoom(input.cpu().numpy(), (256 / dimension[0], 256 / dimension[1]), order=3)
        return result

    def std(self, input):
        if input.max() == 0:
            return input
        else:
            result = (input-input.min()) / (input.max()-input.min())
            return result

    def forward(self, pos):
        ############## Your code block begins here ##############
        print("I am doing ML")
        # input process
        fixed_node_map = self.fixed_node_map_op(pos)
        rudy_utilization_map = self.rudy_utilization_map_op(pos)
        pin_rudy_utilization_map = self.pinrudy_utilization_map_op(pos)

        # print("map_finished")
        feature_list = []

        # normalize and process
        feature_list.append(self.std(self.resize(fixed_node_map)))
        feature_list.append(self.std(self.resize(rudy_utilization_map)))
        feature_list.append(self.std(self.resize(pin_rudy_utilization_map)))
        feature = np.array(feature_list)
        input = torch.from_numpy(feature.astype(np.float32).reshape(1,3,256,256))
        input = input.cuda()
        # print(input.shape)
        # print(type(input))
        # model define & load
        model = GPDL(in_channels=3,out_channels=1)
        model.load_state_dict(torch.load(self.pretrained_ml_congestion_weight_file)['state_dict'])
        model.eval()
        model = model.cuda()

        # print("model define finished")
        # congestion prediction
        congestion_map = model(input)

        print("congestion_map compute finished")
        return congestion_map
        ############## Your code block ends here ################
