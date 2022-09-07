import torch
from torch_geometric.nn import knn_interpolate
from pykeops.torch import LazyTensor


# HUGE TODO: Does not support batching
def torch_knn_interpolate(e, p, xyz_query, k):
    n_batch_query = xyz_query.shape[0]
    n_points_query = xyz_query.shape[1]
        
    _, n_point, f_dim = e.shape
    e_expand = e.expand(n_batch_query, n_point, f_dim)
    p_expand = p.expand(n_batch_query, n_point, 3)

    batch_x = torch.cat([torch.ones((1, n_point), dtype=torch.int64)*i for i in range(n_batch_query)],
                        dim=0).reshape(-1).to(xyz_query.device)
    batch_y = torch.cat([torch.ones((1, n_points_query), dtype=torch.int64)*i for i in range(n_batch_query)], 
                        dim=0).reshape(-1).to(xyz_query.device)
    
    interp = knn_interpolate(e_expand.reshape(-1, f_dim), p_expand.reshape(-1, 3), xyz_query.reshape(-1, 3),
                             k=k, batch_x=batch_x, batch_y=batch_y)
    interp = interp.reshape(n_batch_query, n_points_query, -1)
    
    return interp


def keops_knn(pos_x, pos_y, k):
    B, N, D = pos_x.shape
    B, M, D = pos_y.shape
    
    x_i = LazyTensor(pos_x.view(B, N, 1, D))
    y_j = LazyTensor(pos_y.view(B, 1, M, D))
    
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (B, N, M)
    indices = D_ij.argKmin(k, dim=2)
    
    return indices

# HUGE TODO: Does not support batching
def keops_knn_interpolate(x, pos_x, pos_y, k):
    idxs = keops_knn(pos_y, pos_x, k)
        
    n_query_pts = pos_y.shape[1]
    feat_dim = x.shape[-1]

    neighbor_points = pos_x[0, idxs[0].view(-1), :].reshape(n_query_pts, k, 3)
    neighbor_feats = x[0, idxs[0].view(-1), :].reshape(n_query_pts, k, feat_dim)

    dist = torch.square(neighbor_points - pos_y[0][:, None, :])
    dist = dist.sum(dim=-1)
    weight = 1.0 / torch.clamp(dist, min=1e-16)
    norm_weight = weight / weight.sum(dim=-1, keepdim=True)

    interp = (neighbor_feats * norm_weight.unsqueeze(-1)).sum(dim=1)
    
    return interp.unsqueeze(0)
