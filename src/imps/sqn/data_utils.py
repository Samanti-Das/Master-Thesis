import torch
from torch_geometric.nn.pool import knn
import numpy as np

from .interpolation import keops_knn


def prepare_input(xyz, k, num_layers, encoder_dims, sub_sampling_ratio, device, keops=True):
    batch_pc = xyz

    input_points = []
    input_neighbors = []
    input_pools = []

    # FIXME: Mini-batching is not supported!!
    assert xyz.shape[0] == 1, "Batch size must be 1"


    for i in range(num_layers):
        if not keops:
            assign_idx = knn(batch_pc.squeeze(0), batch_pc.squeeze(0), k)[1]
        else:
            assign_idx = keops_knn(batch_pc, batch_pc, k)
            
        neighbour_idx = assign_idx.reshape(1, batch_pc.shape[1], k)


        sub_points = batch_pc[:, :batch_pc.shape[1] // sub_sampling_ratio, :]

        pool_i = neighbour_idx[:, :batch_pc.shape[1] // sub_sampling_ratio, :]

        input_points.append(batch_pc.to(device))
        input_neighbors.append(neighbour_idx.to(device))
        input_pools.append(pool_i.to(device))
        batch_pc = sub_points


    input_points.append(batch_pc.to(device))
    feat_shape = 2*(np.sum(encoder_dims))
                    
    return input_points, input_neighbors, input_pools, feat_shape


    