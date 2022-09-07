# Main implementation is taken from: https://github.com/qiqihaer/RandLA-Net-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


from .interpolation import torch_knn_interpolate, keops_knn_interpolate


class Randla(nn.Module):

    def __init__(self, d_feature, d_in, encoder_dims, interpolator, device='cuda', create_decoder=True, num_class=-1):
        """
        d_features: Feature dimensionality of the points
        d_in: Output dimensionality of the intial shared-MLP layer
        k: number of kNN neighbors
        sub_sampling_ratio: Sub-sampling rate at each layer
        encoder_dims: Hidden layer sizes of the encoder
        decoder_dims: Hidden layer sizes of the decoder
        activation: Output activation (all intermediate layers are ReLU)
        """
        super().__init__()

        assert interpolator in ('torch', 'keops')

        if interpolator == 'torch':
            self.inp_fn = torch_knn_interpolate
        else:
            self.inp_fn = keops_knn_interpolate
            
        self.n_layers = len(encoder_dims)

        self.fc0 = Conv1d(d_feature, d_in, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        self.latent_size = 0
        self.encoder_dims = encoder_dims

        for d_out in encoder_dims:
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
            self.latent_size += d_in

        self.decoder_modules = None
        d_out = d_in

        if create_decoder:
            assert num_class > 0, "Provide number of classes"

            self.decoder_blocks = nn.ModuleList()
            decoder0 = Conv1d(d_in, d_out, bn=True, activation=nn.ReLU(inplace=True))
            self.decoder_blocks.append(decoder0)
            #print("length of decoder block {}".format(len(self.decoder_blocks)))

            for i in range(len(encoder_dims)-1, 0, -1):
                #print(i)
                #print("enocder dims by layer i".format(encoder_dims[i]))
                #print("enocder dims by layer i-1".format(encoder_dims[i-1]))
                self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))

            #print("encoder dims by layer 0 {}".format(encoder_dims[0]))
            self.decoder_blocks.append(Conv1d(2*encoder_dims[0], encoder_dims[0], bn=True, activation=nn.ReLU(inplace=True)))
            self.decoder_blocks.append(Conv1d(encoder_dims[0], num_class, bn=False, activation=None))

        self.n_neighbor = 1

        self = self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return 'cuda' in str(self.device)

    def encoder(self, features, xyz_points, xyz_neighbors, xyz_sub_samples):
        """
        features.shape = batch x npoints x dims
        xyz_points is a list of xyz[i].shape = batch x npoints[i] x 3
        xyz_neighbors is a list of xyz[i].shape = batch x npoints[i] x K
        xyz_sub_samples is a list of xyz[i].shape = batch x npoints[i+1] x K
        xyz_query is a tensor fo shape batch x n_points_query x 3 (for labeled query points)
        """
        assert features.shape[0] == 1, "This model only supports scene input of batch size 1"

        features = features.permute(0,2,1) # Batch*npoints*3 -> batch*3*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1
        f_encoder_list = []

        for i in range(len(self.dilated_res_blocks)):
            xyz = xyz_points[i]
            neigh_idx = xyz_neighbors[i]
            sub_idx = xyz_sub_samples[i]
            f_encoder_i = self.dilated_res_blocks[i](features, xyz, neigh_idx)
            features = self.random_sample(f_encoder_i, sub_idx)

            f_encoder_list.append(features)

        return f_encoder_list

    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        assert self.decoder is not None, "Create decoder first"
        
        N = len(self.encoder_dims)
        #print("length of N is {}".format(N))
        #print("length of decoder_blocks is {}".format(len(self.decoder_blocks)))
        #print("first decoder block is {}".format(self.decoder_blocks[0]))
        #print("shape of feature encoder list last item {}".format(f_encoder_list[-1].shape))
        #print("shape of feature encoder list last item squeezed {}".format(f_encoder_list[-1].squeeze(-1).shape))
        #print("feature encoder list squeezed value {}".format(f_encoder_list[-1].squeeze(-1)))
        d = self.decoder_blocks[0](f_encoder_list[-1].squeeze(-1))
        
        #print("value of d is {}".format(d))
        #print("shaoe of d is {}".format(d.shape))
              

        for i in range(N):
            #print("enetering loop; value of N is {}".format(i))
            #print("encoder layer number {}".format(N-i+1))
            l = f_encoder_list[N-i-1].squeeze(-1)
            #print("encoder features by layer squeezed {}".format(l))
            #print("shape of encoder features by layer squeezed {}".format(l.shape))
            feats = (d + l).permute(0,2,1)
            #print("summing of decoder and encoder features {}".format(feats))
            #print("summing of decoder and encoder features by shape {}".format(feats.shape))
            interp = self.inp_fn(feats, xyz_points[N-i], xyz_points[N-i-1], k=1).permute(0,2,1) #working with only the second layer, so what should be done?
            #print("interp shape {}".format(interp.shape))
            #print("decoder block layer i {}".format(self.decoder_blocks[1:][i]))
            d = self.decoder_blocks[1:][i](interp) #understand
            #print("the new decoder features {}".format(d))
            #print("shape of new decoder features {}".format(d.shape))

        logits = self.decoder_blocks[-1](d).permute(0,2,1)
        #print("logits shape {}".format(logits.shape))

        return logits

    def forward(self, features, xyz_points, xyz_neighbors, xyz_sub_samples, **kwargs):
        f_encoder_list = self.encoder(features, xyz_points, xyz_neighbors, xyz_sub_samples)
        y = self.decoder(f_encoder_list, xyz_points, **kwargs)
        return y

    def get_features(self, xyz_query, f_encoder_list, xyz_points):

        query_featues = []

        # intermediate features corresponds to sub_sample points. Therefore sub_samples
        # are the reference points for the queries. That corresponds to xyz_points[1:]

        n_batch_query = xyz_query.shape[0]
        n_points_query = xyz_query.shape[1]

        batch_y = torch.cat([torch.ones((1, n_points_query), dtype=torch.int64)*i for i in range(n_batch_query)], dim=0).reshape(-1).to(self.device)
    
        for e, p in zip(f_encoder_list, xyz_points[1:]):
            e = e.squeeze(-1).permute(0,2,1)

            interp = self.inp_fn(e, p, xyz_query, k=self.n_neighbor)
            interp = interp.reshape(n_batch_query, n_points_query, -1)
            query_featues.append(interp)

        return query_featues

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features


class SQN(Randla):
    def __init__(self, d_feature, d_in, encoder_dims, interpolator, decoder_dims=None, interpolation_type='inv', device='cuda', activation=None, skip_connections=True, second_head=None):
        super().__init__(d_feature, d_in, encoder_dims, interpolator, 'cpu', create_decoder=False)
        """
        We create a custom decoder. Don't create Randla-net's decoder. We want to send everyting to the GPU at the same time so put initial 
        layers to cpu first.

        interpolation: Can be one of 'position' or 'feature'. If 'position' PointNet++ interpolation is used. If 'feature' bilateral feature
        interpolation is used.
        """

        self.n_neighbor = 3
        self.interpolation_type = interpolation_type
        self.skip_connections = skip_connections

        self.decoder_blocks = nn.ModuleList()

        d_decoder = self.latent_size

        if self.skip_connections:
            if decoder_dims is not None:
                print("Warning: skip_connections is True and decoder_dims is given. Will not take decoder_dims into account")
            self.decoder_blocks.append(Conv1d(d_decoder, 2*encoder_dims[-1], bn=True, activation=nn.ReLU(inplace=True)))

            for i in range(len(encoder_dims)-1, 0, -1):
                self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))
            
            self.decoder_blocks.append(Conv1d(2*encoder_dims[0], 1, bn=False, activation=activation))

        else:
            for d_dim in decoder_dims[:-1]:
                self.decoder_blocks.append(Conv1d(d_decoder, d_dim, bn=True, activation=nn.ReLU(inplace=True)))
                d_decoder = d_dim
            self.decoder_blocks.append(Conv1d(d_decoder, decoder_dims[-1], bn=False, activation=activation))

        if second_head is not None:
            in_size = self.decoder_blocks[-1].conv.in_channels
            self.decoder_second_head = Conv1d(in_size, second_head, bn=False, activation=None)
        else:
            self.decoder_second_head = None

        self.to(device)


    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        xyz_query = kwargs.get('xyz_query', None)
        assert xyz_query is not None, "Musy provide xyz_query"

        query_features = self.get_features(xyz_query, f_encoder_list, xyz_points)
        latent_features = torch.cat(query_features, dim=-1)

        x = latent_features.permute(0,2,1)
        y = None

        if self.skip_connections:
            N = len(self.encoder_dims)
            x = self.decoder_blocks[0](x)

            for i in range(N-1):
                l = query_features[N-i-1].permute(0,2,1)
                x = x + l
                x = self.decoder_blocks[i+1](x)
            
            x = x + query_features[0].permute(0,2,1)
            occ_out = self.decoder_blocks[-1](x)
        else:
            for dec_layer in self.decoder_blocks[:-1]:
                x = dec_layer(x)
            occ_out = self.decoder_blocks[0](x)

        occ_out = occ_out.permute(0,2,1)

        if self.decoder_second_head is not None:
            y = self.decoder_second_head(x)
            y = y.permute(0,2,1)

            return occ_out, y
            
        return occ_out



class SQNv2(Randla):
    def __init__(self, d_feature, d_in, encoder_dims, embedding_dim, interpolator, device='cuda', activation=None, second_head=None):
        super().__init__(d_feature, d_in, encoder_dims, interpolator, 'cpu', create_decoder=False)
        """
        We create a custom decoder. Don't create Randla-net's decoder. We want to send everyting to the GPU at the same time so put initial 
        layers to cpu first.

        interpolation: Can be one of 'position' or 'feature'. If 'position' PointNet++ interpolation is used. If 'feature' bilateral feature
        interpolation is used.
        """

        self.n_neighbor = 3
        self.decoder_blocks = nn.ModuleList()


        decoder_input_dim = 2*encoder_dims[-1]  + embedding_dim
        self.decoder_blocks.append(Conv1d(decoder_input_dim, 2*encoder_dims[-2], bn=True, activation=nn.ReLU(inplace=True)))

        for i in range(len(encoder_dims)-2, 0, -1):
            self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))
            
        self.decoder_blocks.append(Conv1d(2*encoder_dims[0], 1, bn=False, activation=activation))


        if second_head is not None:
            in_size = self.decoder_blocks[-1].conv.in_channels
            self.decoder_second_head = Conv1d(in_size, second_head, bn=False, activation=None)
        else:
            self.decoder_second_head = None

        self.to(device)


    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        xyz_query = kwargs.get('xyz_query', None)
        assert xyz_query is not None, "Musy provide xyz_query"
        query_embedding = kwargs.get('query_embedding', None)
        assert query_embedding is not None, "Must provide query_embedding"

        x = query_embedding.permute(0,2,1)
        y = None

        query_features = self.get_features(xyz_query, f_encoder_list, xyz_points)

        encoded = query_features[-1].permute(0,2,1)
        x = torch.cat([x, encoded], dim=1)

        N = len(self.encoder_dims)
        x = self.decoder_blocks[0](x)

        for i in range(1, N-1):
            l = query_features[N-i-1].permute(0,2,1)
            x = x + l
            x = self.decoder_blocks[i](x)
            
        x = x + query_features[0].permute(0,2,1)
        occ_out = self.decoder_blocks[-1](x)

        occ_out = occ_out.permute(0,2,1)

        if self.decoder_second_head is not None:
            y = self.decoder_second_head(x)
            y = y.permute(0,2,1)
            
        return occ_out, y



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


import torch.nn as nn
from typing import List, Tuple


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm
                )
            )


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
# Main implementation is taken from: https://github.com/qiqihaer/RandLA-Net-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


from .interpolation import torch_knn_interpolate, keops_knn_interpolate


class Randla(nn.Module):

    def __init__(self, d_feature, d_in, encoder_dims, interpolator, device='cuda', create_decoder=True, num_class=-1):
        """
        d_features: Feature dimensionality of the points
        d_in: Output dimensionality of the intial shared-MLP layer
        k: number of kNN neighbors
        sub_sampling_ratio: Sub-sampling rate at each layer
        encoder_dims: Hidden layer sizes of the encoder
        decoder_dims: Hidden layer sizes of the decoder
        activation: Output activation (all intermediate layers are ReLU)
        """
        super().__init__()

        assert interpolator in ('torch', 'keops')

        if interpolator == 'torch':
            self.inp_fn = torch_knn_interpolate
        else:
            self.inp_fn = keops_knn_interpolate
            
        self.n_layers = len(encoder_dims)

        self.fc0 = Conv1d(d_feature, d_in, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        self.latent_size = 0
        self.encoder_dims = encoder_dims

        for d_out in encoder_dims:
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
            self.latent_size += d_in

        self.decoder_modules = None
        d_out = d_in

        if create_decoder:
            assert num_class > 0, "Provide number of classes"

            self.decoder_blocks = nn.ModuleList()
            decoder0 = Conv1d(d_in, d_out, bn=True, activation=nn.ReLU(inplace=True))
            self.decoder_blocks.append(decoder0)
            #print("length of decoder block {}".format(len(self.decoder_blocks)))

            for i in range(len(encoder_dims)-1, 0, -1):
                #print(i)
                #print("enocder dims by layer i".format(encoder_dims[i]))
                #print("enocder dims by layer i-1".format(encoder_dims[i-1]))
                self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))

            #print("encoder dims by layer 0 {}".format(encoder_dims[0]))
            self.decoder_blocks.append(Conv1d(2*encoder_dims[0], encoder_dims[0], bn=True, activation=nn.ReLU(inplace=True)))
            self.decoder_blocks.append(Conv1d(encoder_dims[0], num_class, bn=False, activation=None))

        self.n_neighbor = 1

        self = self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return 'cuda' in str(self.device)

    def encoder(self, features, xyz_points, xyz_neighbors, xyz_sub_samples):
        """
        features.shape = batch x npoints x dims
        xyz_points is a list of xyz[i].shape = batch x npoints[i] x 3
        xyz_neighbors is a list of xyz[i].shape = batch x npoints[i] x K
        xyz_sub_samples is a list of xyz[i].shape = batch x npoints[i+1] x K
        xyz_query is a tensor fo shape batch x n_points_query x 3 (for labeled query points)
        """
        assert features.shape[0] == 1, "This model only supports scene input of batch size 1"

        features = features.permute(0,2,1) # Batch*npoints*3 -> batch*3*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1
        f_encoder_list = []

        for i in range(len(self.dilated_res_blocks)):
            xyz = xyz_points[i]
            neigh_idx = xyz_neighbors[i]
            sub_idx = xyz_sub_samples[i]
            f_encoder_i = self.dilated_res_blocks[i](features, xyz, neigh_idx)
            features = self.random_sample(f_encoder_i, sub_idx)

            f_encoder_list.append(features)

        return f_encoder_list

    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        assert self.decoder is not None, "Create decoder first"
        
        N = len(self.encoder_dims)
        #print("length of N is {}".format(N))
        #print("length of decoder_blocks is {}".format(len(self.decoder_blocks)))
        #print("first decoder block is {}".format(self.decoder_blocks[0]))
        #print("shape of feature encoder list last item {}".format(f_encoder_list[-1].shape))
        #print("shape of feature encoder list last item squeezed {}".format(f_encoder_list[-1].squeeze(-1).shape))
        #print("feature encoder list squeezed value {}".format(f_encoder_list[-1].squeeze(-1)))
        d = self.decoder_blocks[0](f_encoder_list[-1].squeeze(-1))
        
        #print("value of d is {}".format(d))
        #print("shaoe of d is {}".format(d.shape))
              

        for i in range(N):
            #print("enetering loop; value of N is {}".format(i))
            #print("encoder layer number {}".format(N-i+1))
            l = f_encoder_list[N-i-1].squeeze(-1)
            #print("encoder features by layer squeezed {}".format(l))
            #print("shape of encoder features by layer squeezed {}".format(l.shape))
            feats = (d + l).permute(0,2,1)
            #print("summing of decoder and encoder features {}".format(feats))
            #print("summing of decoder and encoder features by shape {}".format(feats.shape))
            interp = self.inp_fn(feats, xyz_points[N-i], xyz_points[N-i-1], k=1).permute(0,2,1) #working with only the second layer, so what should be done?
            #print("interp shape {}".format(interp.shape))
            #print("decoder block layer i {}".format(self.decoder_blocks[1:][i]))
            d = self.decoder_blocks[1:][i](interp) #understand
            #print("the new decoder features {}".format(d))
            #print("shape of new decoder features {}".format(d.shape))

        logits = self.decoder_blocks[-1](d).permute(0,2,1)
        #print("logits shape {}".format(logits.shape))

        return logits

    def forward(self, features, xyz_points, xyz_neighbors, xyz_sub_samples, **kwargs):
        f_encoder_list = self.encoder(features, xyz_points, xyz_neighbors, xyz_sub_samples)
        y = self.decoder(f_encoder_list, xyz_points, **kwargs)
        return y

    def get_features(self, xyz_query, f_encoder_list, xyz_points):

        query_featues = []

        # intermediate features corresponds to sub_sample points. Therefore sub_samples
        # are the reference points for the queries. That corresponds to xyz_points[1:]

        n_batch_query = xyz_query.shape[0]
        n_points_query = xyz_query.shape[1]

        batch_y = torch.cat([torch.ones((1, n_points_query), dtype=torch.int64)*i for i in range(n_batch_query)], dim=0).reshape(-1).to(self.device)
    
        for e, p in zip(f_encoder_list, xyz_points[1:]):
            e = e.squeeze(-1).permute(0,2,1)

            interp = self.inp_fn(e, p, xyz_query, k=self.n_neighbor)
            interp = interp.reshape(n_batch_query, n_points_query, -1)
            query_featues.append(interp)

        return query_featues

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features


class SQN(Randla):
    def __init__(self, d_feature, d_in, encoder_dims, interpolator, decoder_dims=None, interpolation_type='inv', device='cuda', activation=None, skip_connections=True, second_head=None):
        super().__init__(d_feature, d_in, encoder_dims, interpolator, 'cpu', create_decoder=False)
        """
        We create a custom decoder. Don't create Randla-net's decoder. We want to send everyting to the GPU at the same time so put initial 
        layers to cpu first.

        interpolation: Can be one of 'position' or 'feature'. If 'position' PointNet++ interpolation is used. If 'feature' bilateral feature
        interpolation is used.
        """

        self.n_neighbor = 3
        self.interpolation_type = interpolation_type
        self.skip_connections = skip_connections

        self.decoder_blocks = nn.ModuleList()

        d_decoder = self.latent_size

        if self.skip_connections:
            if decoder_dims is not None:
                print("Warning: skip_connections is True and decoder_dims is given. Will not take decoder_dims into account")
            self.decoder_blocks.append(Conv1d(d_decoder, 2*encoder_dims[-1], bn=True, activation=nn.ReLU(inplace=True)))

            for i in range(len(encoder_dims)-1, 0, -1):
                self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))
            
            self.decoder_blocks.append(Conv1d(2*encoder_dims[0], 1, bn=False, activation=activation))

        else:
            for d_dim in decoder_dims[:-1]:
                self.decoder_blocks.append(Conv1d(d_decoder, d_dim, bn=True, activation=nn.ReLU(inplace=True)))
                d_decoder = d_dim
            self.decoder_blocks.append(Conv1d(d_decoder, decoder_dims[-1], bn=False, activation=activation))

        if second_head is not None:
            in_size = self.decoder_blocks[-1].conv.in_channels
            self.decoder_second_head = Conv1d(in_size, second_head, bn=False, activation=None)
        else:
            self.decoder_second_head = None

        self.to(device)


    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        xyz_query = kwargs.get('xyz_query', None)
        assert xyz_query is not None, "Musy provide xyz_query"

        query_features = self.get_features(xyz_query, f_encoder_list, xyz_points)
        latent_features = torch.cat(query_features, dim=-1)

        x = latent_features.permute(0,2,1)
        y = None

        if self.skip_connections:
            N = len(self.encoder_dims)
            x = self.decoder_blocks[0](x)

            for i in range(N-1):
                l = query_features[N-i-1].permute(0,2,1)
                x = x + l
                x = self.decoder_blocks[i+1](x)
            
            x = x + query_features[0].permute(0,2,1)
            occ_out = self.decoder_blocks[-1](x)
        else:
            for dec_layer in self.decoder_blocks[:-1]:
                x = dec_layer(x)
            occ_out = self.decoder_blocks[0](x)

        occ_out = occ_out.permute(0,2,1)

        if self.decoder_second_head is not None:
            y = self.decoder_second_head(x)
            y = y.permute(0,2,1)

            return occ_out, y
            
        return occ_out



class SQNv2(Randla):
    def __init__(self, d_feature, d_in, encoder_dims, embedding_dim, interpolator, device='cuda', activation=None, second_head=None):
        super().__init__(d_feature, d_in, encoder_dims, interpolator, 'cpu', create_decoder=False)
        """
        We create a custom decoder. Don't create Randla-net's decoder. We want to send everyting to the GPU at the same time so put initial 
        layers to cpu first.

        interpolation: Can be one of 'position' or 'feature'. If 'position' PointNet++ interpolation is used. If 'feature' bilateral feature
        interpolation is used.
        """

        self.n_neighbor = 3
        self.decoder_blocks = nn.ModuleList()


        decoder_input_dim = 2*encoder_dims[-1]  + embedding_dim
        self.decoder_blocks.append(Conv1d(decoder_input_dim, 2*encoder_dims[-2], bn=True, activation=nn.ReLU(inplace=True)))

        for i in range(len(encoder_dims)-2, 0, -1):
            self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))
            
        self.decoder_blocks.append(Conv1d(2*encoder_dims[0], 1, bn=False, activation=activation))


        if second_head is not None:
            in_size = self.decoder_blocks[-1].conv.in_channels
            self.decoder_second_head = Conv1d(in_size, second_head, bn=False, activation=None)
        else:
            self.decoder_second_head = None

        self.to(device)


    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        xyz_query = kwargs.get('xyz_query', None)
        assert xyz_query is not None, "Musy provide xyz_query"
        query_embedding = kwargs.get('query_embedding', None)
        assert query_embedding is not None, "Must provide query_embedding"

        x = query_embedding.permute(0,2,1)
        y = None

        query_features = self.get_features(xyz_query, f_encoder_list, xyz_points)

        encoded = query_features[-1].permute(0,2,1)
        x = torch.cat([x, encoded], dim=1)

        N = len(self.encoder_dims)
        x = self.decoder_blocks[0](x)

        for i in range(1, N-1):
            l = query_features[N-i-1].permute(0,2,1)
            x = x + l
            x = self.decoder_blocks[i](x)
            
        x = x + query_features[0].permute(0,2,1)
        occ_out = self.decoder_blocks[-1](x)

        occ_out = occ_out.permute(0,2,1)

        if self.decoder_second_head is not None:
            y = self.decoder_second_head(x)
            y = y.permute(0,2,1)
            
        return occ_out, y



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


import torch.nn as nn
from typing import List, Tuple


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm
                )
            )


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
# Main implementation is taken from: https://github.com/qiqihaer/RandLA-Net-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


from .interpolation import torch_knn_interpolate, keops_knn_interpolate


class Randla(nn.Module):

    def __init__(self, d_feature, d_in, encoder_dims, interpolator, device='cuda', create_decoder=True, num_class=-1):
        """
        d_features: Feature dimensionality of the points
        d_in: Output dimensionality of the intial shared-MLP layer
        k: number of kNN neighbors
        sub_sampling_ratio: Sub-sampling rate at each layer
        encoder_dims: Hidden layer sizes of the encoder
        decoder_dims: Hidden layer sizes of the decoder
        activation: Output activation (all intermediate layers are ReLU)
        """
        super().__init__()

        assert interpolator in ('torch', 'keops')

        if interpolator == 'torch':
            self.inp_fn = torch_knn_interpolate
        else:
            self.inp_fn = keops_knn_interpolate
            
        self.n_layers = len(encoder_dims)

        self.fc0 = Conv1d(d_feature, d_in, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        self.latent_size = 0
        self.encoder_dims = encoder_dims

        for d_out in encoder_dims:
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
            self.latent_size += d_in

        self.decoder_modules = None
        d_out = d_in

        if create_decoder:
            assert num_class > 0, "Provide number of classes"

            self.decoder_blocks = nn.ModuleList()
            decoder0 = Conv1d(d_in, d_out, bn=True, activation=nn.ReLU(inplace=True))
            self.decoder_blocks.append(decoder0)
            #print("length of decoder block {}".format(len(self.decoder_blocks)))

            for i in range(len(encoder_dims)-1, 0, -1):
                #print(i)
                #print("enocder dims by layer i".format(encoder_dims[i]))
                #print("enocder dims by layer i-1".format(encoder_dims[i-1]))
                self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))

            #print("encoder dims by layer 0 {}".format(encoder_dims[0]))
            self.decoder_blocks.append(Conv1d(2*encoder_dims[0], encoder_dims[0], bn=True, activation=nn.ReLU(inplace=True)))
            self.decoder_blocks.append(Conv1d(encoder_dims[0], num_class, bn=False, activation=None))

        self.n_neighbor = 1

        self = self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return 'cuda' in str(self.device)

    def encoder(self, features, xyz_points, xyz_neighbors, xyz_sub_samples):
        """
        features.shape = batch x npoints x dims
        xyz_points is a list of xyz[i].shape = batch x npoints[i] x 3
        xyz_neighbors is a list of xyz[i].shape = batch x npoints[i] x K
        xyz_sub_samples is a list of xyz[i].shape = batch x npoints[i+1] x K
        xyz_query is a tensor fo shape batch x n_points_query x 3 (for labeled query points)
        """
        assert features.shape[0] == 1, "This model only supports scene input of batch size 1"

        features = features.permute(0,2,1) # Batch*npoints*3 -> batch*3*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1
        f_encoder_list = []

        for i in range(len(self.dilated_res_blocks)):
            xyz = xyz_points[i]
            neigh_idx = xyz_neighbors[i]
            sub_idx = xyz_sub_samples[i]
            f_encoder_i = self.dilated_res_blocks[i](features, xyz, neigh_idx)
            features = self.random_sample(f_encoder_i, sub_idx)

            f_encoder_list.append(features)

        return f_encoder_list

    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        assert self.decoder is not None, "Create decoder first"
        
        N = len(self.encoder_dims)
        #print("length of N is {}".format(N))
        #print("length of decoder_blocks is {}".format(len(self.decoder_blocks)))
        #print("first decoder block is {}".format(self.decoder_blocks[0]))
        #print("shape of feature encoder list last item {}".format(f_encoder_list[-1].shape))
        #print("shape of feature encoder list last item squeezed {}".format(f_encoder_list[-1].squeeze(-1).shape))
        #print("feature encoder list squeezed value {}".format(f_encoder_list[-1].squeeze(-1)))
        d = self.decoder_blocks[0](f_encoder_list[-1].squeeze(-1))
        
        #print("value of d is {}".format(d))
        #print("shaoe of d is {}".format(d.shape))
              

        for i in range(N):
            #print("enetering loop; value of N is {}".format(i))
            #print("encoder layer number {}".format(N-i+1))
            l = f_encoder_list[N-i-1].squeeze(-1)
            #print("encoder features by layer squeezed {}".format(l))
            #print("shape of encoder features by layer squeezed {}".format(l.shape))
            feats = (d + l).permute(0,2,1)
            #print("summing of decoder and encoder features {}".format(feats))
            #print("summing of decoder and encoder features by shape {}".format(feats.shape))
            interp = self.inp_fn(feats, xyz_points[N-i], xyz_points[N-i-1], k=1).permute(0,2,1) #working with only the second layer, so what should be done?
            #print("interp shape {}".format(interp.shape))
            #print("decoder block layer i {}".format(self.decoder_blocks[1:][i]))
            d = self.decoder_blocks[1:][i](interp) #understand
            #print("the new decoder features {}".format(d))
            #print("shape of new decoder features {}".format(d.shape))

        logits = self.decoder_blocks[-1](d).permute(0,2,1)
        #print("logits shape {}".format(logits.shape))

        return logits

    def forward(self, features, xyz_points, xyz_neighbors, xyz_sub_samples, **kwargs):
        f_encoder_list = self.encoder(features, xyz_points, xyz_neighbors, xyz_sub_samples)
        y = self.decoder(f_encoder_list, xyz_points, **kwargs)
        return y

    def get_features(self, xyz_query, f_encoder_list, xyz_points):

        query_featues = []

        # intermediate features corresponds to sub_sample points. Therefore sub_samples
        # are the reference points for the queries. That corresponds to xyz_points[1:]

        n_batch_query = xyz_query.shape[0]
        n_points_query = xyz_query.shape[1]

        batch_y = torch.cat([torch.ones((1, n_points_query), dtype=torch.int64)*i for i in range(n_batch_query)], dim=0).reshape(-1).to(self.device)
    
        for e, p in zip(f_encoder_list, xyz_points[1:]):
            e = e.squeeze(-1).permute(0,2,1)

            interp = self.inp_fn(e, p, xyz_query, k=self.n_neighbor)
            interp = interp.reshape(n_batch_query, n_points_query, -1)
            query_featues.append(interp)

        return query_featues

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features


class SQN(Randla):
    def __init__(self, d_feature, d_in, encoder_dims, interpolator, decoder_dims=None, interpolation_type='inv', device='cuda', activation=None, skip_connections=True, second_head=None):
        super().__init__(d_feature, d_in, encoder_dims, interpolator, 'cpu', create_decoder=False)
        """
        We create a custom decoder. Don't create Randla-net's decoder. We want to send everyting to the GPU at the same time so put initial 
        layers to cpu first.

        interpolation: Can be one of 'position' or 'feature'. If 'position' PointNet++ interpolation is used. If 'feature' bilateral feature
        interpolation is used.
        """

        self.n_neighbor = 3
        self.interpolation_type = interpolation_type
        self.skip_connections = skip_connections

        self.decoder_blocks = nn.ModuleList()

        d_decoder = self.latent_size

        if self.skip_connections:
            if decoder_dims is not None:
                print("Warning: skip_connections is True and decoder_dims is given. Will not take decoder_dims into account")
            self.decoder_blocks.append(Conv1d(d_decoder, 2*encoder_dims[-1], bn=True, activation=nn.ReLU(inplace=True)))

            for i in range(len(encoder_dims)-1, 0, -1):
                self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))
            
            self.decoder_blocks.append(Conv1d(2*encoder_dims[0], 1, bn=False, activation=activation))

        else:
            for d_dim in decoder_dims[:-1]:
                self.decoder_blocks.append(Conv1d(d_decoder, d_dim, bn=True, activation=nn.ReLU(inplace=True)))
                d_decoder = d_dim
            self.decoder_blocks.append(Conv1d(d_decoder, decoder_dims[-1], bn=False, activation=activation))

        if second_head is not None:
            in_size = self.decoder_blocks[-1].conv.in_channels
            self.decoder_second_head = Conv1d(in_size, second_head, bn=False, activation=None)
        else:
            self.decoder_second_head = None

        self.to(device)


    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        xyz_query = kwargs.get('xyz_query', None)
        assert xyz_query is not None, "Musy provide xyz_query"

        query_features = self.get_features(xyz_query, f_encoder_list, xyz_points)
        latent_features = torch.cat(query_features, dim=-1)

        x = latent_features.permute(0,2,1)
        y = None

        if self.skip_connections:
            N = len(self.encoder_dims)
            x = self.decoder_blocks[0](x)

            for i in range(N-1):
                l = query_features[N-i-1].permute(0,2,1)
                x = x + l
                x = self.decoder_blocks[i+1](x)
            
            x = x + query_features[0].permute(0,2,1)
            occ_out = self.decoder_blocks[-1](x)
        else:
            for dec_layer in self.decoder_blocks[:-1]:
                x = dec_layer(x)
            occ_out = self.decoder_blocks[0](x)

        occ_out = occ_out.permute(0,2,1)

        if self.decoder_second_head is not None:
            y = self.decoder_second_head(x)
            y = y.permute(0,2,1)

            return occ_out, y
            
        return occ_out



class SQNv2(Randla):
    def __init__(self, d_feature, d_in, encoder_dims, embedding_dim, interpolator, device='cuda', activation=None, second_head=None):
        super().__init__(d_feature, d_in, encoder_dims, interpolator, 'cpu', create_decoder=False)
        """
        We create a custom decoder. Don't create Randla-net's decoder. We want to send everyting to the GPU at the same time so put initial 
        layers to cpu first.

        interpolation: Can be one of 'position' or 'feature'. If 'position' PointNet++ interpolation is used. If 'feature' bilateral feature
        interpolation is used.
        """

        self.n_neighbor = 3
        self.decoder_blocks = nn.ModuleList()


        decoder_input_dim = 2*encoder_dims[-1]  + embedding_dim
        self.decoder_blocks.append(Conv1d(decoder_input_dim, 2*encoder_dims[-2], bn=True, activation=nn.ReLU(inplace=True)))

        for i in range(len(encoder_dims)-2, 0, -1):
            self.decoder_blocks.append(Conv1d(2*encoder_dims[i], 2*encoder_dims[i-1], bn=True, activation=nn.ReLU(inplace=True)))
            
        self.decoder_blocks.append(Conv1d(2*encoder_dims[0], 1, bn=False, activation=activation))


        if second_head is not None:
            in_size = self.decoder_blocks[-1].conv.in_channels
            self.decoder_second_head = Conv1d(in_size, second_head, bn=False, activation=None)
        else:
            self.decoder_second_head = None

        self.to(device)


    def decoder(self, f_encoder_list, xyz_points, **kwargs):
        xyz_query = kwargs.get('xyz_query', None)
        assert xyz_query is not None, "Musy provide xyz_query"
        query_embedding = kwargs.get('query_embedding', None)
        assert query_embedding is not None, "Must provide query_embedding"

        x = query_embedding.permute(0,2,1)
        y = None

        query_features = self.get_features(xyz_query, f_encoder_list, xyz_points)

        encoded = query_features[-1].permute(0,2,1)
        x = torch.cat([x, encoded], dim=1)

        N = len(self.encoder_dims)
        x = self.decoder_blocks[0](x)

        for i in range(1, N-1):
            l = query_features[N-i-1].permute(0,2,1)
            x = x + l
            x = self.decoder_blocks[i](x)
            
        x = x + query_features[0].permute(0,2,1)
        occ_out = self.decoder_blocks[-1](x)

        occ_out = occ_out.permute(0,2,1)

        if self.decoder_second_head is not None:
            y = self.decoder_second_head(x)
            y = y.permute(0,2,1)
            
        return occ_out, y



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


import torch.nn as nn
from typing import List, Tuple


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm
                )
            )


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
