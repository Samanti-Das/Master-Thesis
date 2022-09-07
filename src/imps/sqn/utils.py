import torch
import torch.nn.functional as F
from torch_geometric.nn import knn
from torch_scatter import scatter_add


# BATCHING IS NOT SUPPORTED!!!! DEPRECATED
def knn_feat_interpolate(x, pos_x, pos_y, type='inv', batch_x=None, batch_y=None, k=3, num_workers=1):
    assert type in ('inv', 'exp')

    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y,
                           num_workers=num_workers)
        y_idx, x_idx = assign_index
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        mean_distance = torch.mean(squared_distance)

        if type=='inv':
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        elif type=='exp':
            weights = torch.exp(-(squared_distance/mean_distance)**2)
        else:
            raise Exception()

    y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

    return y

