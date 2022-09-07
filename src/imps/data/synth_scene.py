import os
from collections import OrderedDict

import torch
import numpy as np

from imps.sqn.data_utils import prepare_input


N_CLASS = 5
CLASS_NAMES = OrderedDict({
    -1: "ground-plane",
    0: "cabinet",
    1: "chair",
    2: "lamp",
    3: "sofa",
    4: "table",
    6: "free-space"
})

def pos_embed(pos, L):
    embs = []
    
    for l in range(L):
        sin_emb = np.sin((2^l)*np.pi*pos)
        cos_emb = np.cos((2^l)*np.pi*pos)
        embs += [sin_emb, cos_emb]
    
    return np.concatenate(embs, axis=-1).astype(np.float)


class SynthSceneDataset(object):
    def __init__(self, base_dir, device, L=10):
        self.base_dir = base_dir
        self.device = device
        self.L = L

        train_lst = os.path.join(base_dir, 'train.lst')
        test_lst = os.path.join(base_dir, 'train.lst')
        val_lst = os.path.join(base_dir, 'train.lst')

        with open(train_lst, 'r') as f:
            train_scenes = f.read().splitlines()
        self.train_dirs = list(map(lambda x: os.path.join(base_dir, x), train_scenes))
            
        with open(test_lst, 'r') as f:
            test_scenes = f.read().splitlines() 
        self.test_dirs = list(map(lambda x: os.path.join(base_dir, x), test_scenes))
            
        with open(val_lst, 'r') as f:
            val_scenes = f.read().splitlines()
        self.val_dirs = list(map(lambda x: os.path.join(base_dir, x), val_scenes))

    def read_scene_data(self, scene_dir, num, seed):
        scene = np.load(os.path.join(scene_dir, 'pointcloud', f'pointcloud_{num}.npz'))
        scene_points = scene['points']
        
        if (self.L > 0) and (self.L is not None):
            features = torch.FloatTensor(pos_embed(scene_points, self.L)).unsqueeze(0).to(self.device)
        else:
            features = torch.FloatTensor(scene_points).unsqueeze(0).to(self.device)
            
        xyz = torch.FloatTensor(scene_points).unsqueeze(0)
        
        if (seed is not None) and (seed != -1):
            torch.manual_seed(seed)
        elif seed == -1:
            torch.manual_seed(31)
        
        point_perm = torch.randperm(xyz.size()[1])
        xyz = xyz[:, point_perm]
        features = features[:, point_perm]
        
        return features, xyz

    def read_query_data(self, scene_dir, num):
        query_iou = np.load(os.path.join(scene_dir, 'points_iou', f'points_iou_{num}.npz'))
        query_points = query_iou['points']
        query_occ = np.unpackbits(query_iou['occupancies'])
        query_semantics = query_iou['semantics']
        
        pos_w = np.sum(query_occ==0) / np.sum(query_occ==1)
        
        query = torch.FloatTensor(query_points).unsqueeze(0).to(self.device)
        query_occ = torch.FloatTensor(query_occ).unsqueeze(0).to(self.device)
        pos_w = torch.FloatTensor([pos_w]).to(self.device).unsqueeze(0)
        query_semantics = torch.LongTensor(query_semantics).unsqueeze(0).to(self.device)
        
        return query, query_occ, pos_w, query_semantics

    def get_scene_data(self, scene_dir, scene_num, seed=None, iou_nums=None):
        """
        iou_nums: denotes which iou files to read from scene_id/points_iou. If None reads eveyrthing into query's batch.
        This works since all queries have same number of points.
        During training, for a given scene, average out the query batches for the gradient update.
        Scenes can't be batched since each has different number of points.
        
        scene_num: denotes which scene to read from scene_id/pointcloud. This is the random sampled points from the
        surface of the same mesh but with different sampling. This does not have batching or a correspondance with
        iou_nums.
        """
        features, xyz = self.read_scene_data(scene_dir, scene_num, seed)

        input_points, input_neighbors, input_pools = prepare_input(xyz, k=8, num_layers=3, sub_sampling_ratio=4, device=self.device)
        
        if iou_nums is None:
            iou_nums = ['%02d' % i for i in range(10)]
        
        query_batch, query_occ_batch, pos_w_batch, query_semantics_batch = [], [], [], []
        
        for num in iou_nums:
            query, query_occ, pos_w, query_semantics = self.read_query_data(scene_dir, num)
            query_batch.append(query)
            query_occ_batch.append(query_occ)
            pos_w_batch.append(pos_w)
            query_semantics_batch.append(query_semantics)
        
        query_semantics = torch.cat(query_semantics_batch, dim=0)
        class_counts = torch.zeros(N_CLASS)

        for c in range(N_CLASS):
            class_counts[c] = torch.sum(query_semantics == c)
        class_weights = class_counts / class_counts.sum()
        
        data = {
            'features': features,
            'input_points': input_points,
            'input_neighbors': input_neighbors,
            'input_pools': input_pools,
            'query': torch.cat(query_batch, dim=0),
            'query_occ': torch.cat(query_occ_batch, dim=0),
            'query_semantics': query_semantics,
            'pos_w':torch.cat(pos_w_batch, dim=0),
            'class_weights': class_weights.to(self.device)
        }
        
        return data