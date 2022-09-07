import os
import json
import time
import trimesh
#import igl
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree as KDTree
from matplotlib import cm
from collections import OrderedDict

# TODO: Generate free space points going inwards the room

UNANNOTATED_LABEL = 'unannotated'

CLASS_NAMES = OrderedDict([
                 ('unannotated', (0, 0, 0)),
                 ('wall', (174, 199, 232)),
                 ('floor', (152, 223, 138)),
                 ('cabinet', (31, 119, 180)),
                 ('bed', (255, 187, 120)),
                 ('chair', (188, 189, 34)),
                 ('sofa', (140, 86, 75)),
                 ('table', (255, 152, 150)),
                 ('door', (214, 39, 40)),
                 ('window', (197, 176, 213)),
                 ('bookshelf', (148, 103, 189)),
                 ('picture', (196, 156, 148)),
                 ('counter', (23, 190, 207)),
                 ('desk', (247, 182, 210)),
                 ('curtain', (219, 219, 141)),
                 ('refrigerator', (255, 127, 14)),
                 ('showercurtain', (158, 218, 229)),
                 ('toilet', (44, 160, 44)),
                 ('sink', (112, 128, 144)),
                 ('bathtub', (227, 119, 194)),
                 ('otherfurniture', (82, 84, 163)),
         ])



MOVABLE_INSTANCE_NAMES = OrderedDict([
                 ('cabinet', (31, 119, 180)),
                 ('bed', (255, 187, 120)),
                 ('chair', (188, 189, 34)),
                 ('sofa', (140, 86, 75)),
                 ('table', (255, 152, 150)),
                 ('bookshelf', (148, 103, 189)),
                 ('counter', (23, 190, 207)),
                 ('desk', (247, 182, 210)),
                 ('refrigerator', (255, 127, 14)),
                 ('toilet', (44, 160, 44)),
                 ('sink', (112, 128, 144)),
                 ('bathtub', (227, 119, 194)),
                 ('otherfurniture', (82, 84, 163)),
         ])

#CLASS_NAMES = OrderedDict([
#            ('unannotated', (0, 0, 0)),
#            ('wall', (174, 199, 232)),
#            ('floor', (152, 223, 138)),
#            ('cabinet', (31, 119, 180)),
#            ('bed', (255, 187, 120)),
#            ('chair', (188, 189, 34)),
#            ('sofa', (140, 86, 75)),
#            ('table', (255, 152, 150)),
#            ('door', (214, 39, 40)),
#            ('window', (197, 176, 213)),
#            ('desk', (247, 182, 210)),
#            ('otherfurniture', (82, 84, 163)),
#    ])

LABEL_COLORS = list(CLASS_NAMES.values())

CLASS_IDXS = {c: i for i,c in enumerate(CLASS_NAMES.keys())}


def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open('../../configs/scannet-labels.combined.tsv','r', encoding='utf-8')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(CLASS_NAMES)
        elements = lines[i].split('\t')
        raw_name = elements[0]
        nyu40_name = elements[6]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = UNANNOTATED_LABEL
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

RAW2SCANNET = get_raw2scannet_label_map()


class ScanNetScene(object):
    def __init__(self, scene_dir, scale=None, normalize=True):
        self.scene_name = scene_dir.split('/')[-1]
        self.mesh_dir = os.path.join(scene_dir, f"{self.scene_name}_vh_clean_2.ply")
        self.mesh_seg_filename = os.path.join(scene_dir, f'{self.scene_name}_vh_clean_2.0.010000.segs.json')
        self.annotation_filename = os.path.join(scene_dir, f'{self.scene_name}.aggregation.json')

        self.mesh = trimesh.load(self.mesh_dir, process=False)
        if scale is None:
            self.scale = (self.mesh.bounds[1] - self.mesh.bounds[0]).max()
        else:
            self.scale = scale
            
        self.centers = (self.mesh.bounds[1] + self.mesh.bounds[0]) / 2

        self.is_normalized = False

        if normalize:
            self.normalize()

        with open(self.mesh_seg_filename) as jsondata:
            d = json.load(jsondata)
            seg = np.array(d['segIndices'])

        seg2label = {}
        object_id_to_segs = {}
        seg2idx = {c: i for i,c in enumerate(CLASS_NAMES)}

        with open(self.annotation_filename) as jsondata:
            d = json.load(jsondata)
            for x in d['segGroups']:
                object_id = x['objectId'] 
                segs = x['segments']
                object_id_to_segs[object_id] = segs
                for s in x['segments']:
                    if x['label'] not in RAW2SCANNET:
                        label = UNANNOTATED_LABEL
                    else:
                        label = RAW2SCANNET[x['label']]
                    seg2label[s] = label
        self.seg_labels = np.vectorize(lambda x: seg2label.get(x, UNANNOTATED_LABEL))(seg)
        self.seg_idxs = np.vectorize(lambda x: CLASS_IDXS.get(x))(self.seg_labels)
        
        # instance array creation
        seg_to_verts = {}
        for i in range(len(seg)):
            seg_id = seg[i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
                
        instance_ids = np.zeros(shape=(len(seg)), dtype=np.uint32)  # 0: unannotated

        for object_id, segs in object_id_to_segs.items():
            
        #print("object id is {}".format(object_id))
            for seg in segs:
                #print(seg)
                verts = seg_to_verts[seg]
                #print(verts)
                instance_ids[verts] = object_id
                
        self.instance_ids = instance_ids

    @property
    def bounds(self):
        return self.mesh.bounds

    def normalize(self):
        self.translate(-self.centers)
        self.scale_down(self.scale)

        self.centers = (self.mesh.bounds[1] + self.mesh.bounds[0]) / 2
        self.is_normalized = True
    
    def translate(self, x):
        self.mesh.apply_translation(x)

    def scale_down(self, x):
        self.mesh.apply_scale(1/x)

    def voxelize(self, res, n_points, o3d_format=False):
        # voxel_min, voxel_max = self.bounds.min(), self.bounds.max()
        # DANGER! the mesh has to be normalized for this to work
        # We want the whole scene scaled to 1x1x1 meters
        assert self.is_normalized
        voxel_min, voxel_max = -0.5, +0.5
        voxel_size = (voxel_max - voxel_min) / res

        dim_bins = np.linspace(voxel_min, voxel_max-voxel_size, res)
        x_, y_, z_ = np.meshgrid(dim_bins, dim_bins, dim_bins, indexing='ij')
        x_ = x_.reshape((np.prod(x_.shape),))
        y_ = y_.reshape((np.prod(y_.shape),))
        z_ = z_.reshape((np.prod(z_.shape),))

        voxel_samples = int(min(n_points*5, 5e6))
        points_voxel, sample_idxs = self.mesh.sample(voxel_samples, return_index=True)
        sample_colors = self.mesh.visual.face_colors[sample_idxs, :-1] / 255
        grid_points = np.column_stack((x_, y_, z_))

        kd_tree = KDTree(grid_points)
        _, idx = kd_tree.query(points_voxel)

        occupancies = np.zeros(len(grid_points))
        colors = np.ones((len(grid_points), 3)) * -1
        # The mapping between idx and color is one to many since we are sort of downsampling. Just select the most recent one.
        # A potential improvement can be averaging
        colors[idx] = sample_colors 
        occupancies[idx] = 1
        occupancies = occupancies.reshape((len(dim_bins), )*3)

        if o3d_format:
            #occupied_colors = np.ones_like(grid_points)*0.8
            #occupied_colors[occupancies.reshape(-1) == 1] = colors[occupancies.reshape(-1) == 1]
            #occupied_points = grid_points
            occupied_points = grid_points[occupancies.reshape(-1) == 1]
            occupied_colors = colors[occupancies.reshape(-1) == 1]
            voxel_pcd = o3d.geometry.PointCloud()
            voxel_pcd.points = o3d.utility.Vector3dVector(occupied_points)
            voxel_pcd.colors = o3d.utility.Vector3dVector(occupied_colors)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, voxel_size=voxel_size)

            return voxel_grid

        return occupancies
    
    def create_points_colors_labels_from_pc(self, n_points):
        start_func_time = time.time()
        surface_points, surface_idxs = self.mesh.sample(n_points, return_index=True)
        surface_colors = self.mesh.visual.face_colors[surface_idxs, :-1] / 255
        fx = self.mesh.faces[surface_idxs][:, 0]
        fy = self.mesh.faces[surface_idxs][:, 1]
        fz = self.mesh.faces[surface_idxs][:, 2]

        start_time = time.time()
        fx_label = get_one_hot(self.seg_idxs[fx], len(CLASS_NAMES))[:, :, None]
        fy_label = get_one_hot(self.seg_idxs[fy], len(CLASS_NAMES))[:, :, None]
        fz_label = get_one_hot(self.seg_idxs[fz], len(CLASS_NAMES))[:, :, None]
        end_time = time.time()
        #print("time is {}".format(end_time-start_time))
        
        start_time = time.time()
        fx_instance = get_one_hot(self.instance_ids[fx], np.max(self.instance_ids[fx], 0)+1)[:, :, None]
        fy_instance = get_one_hot(self.instance_ids[fy], np.max(self.instance_ids[fy], 0)+1)[:, :, None]
        fz_instance = get_one_hot(self.instance_ids[fz], np.max(self.instance_ids[fz], 0)+1)[:, :, None]
        end_time = time.time()
        #print("time is {}".format(end_time-start_time))

        # Majority voting
        start_time = time.time()
        face_labels = np.concatenate([fx_label, fy_label, fz_label], axis=2)
        label_votes = face_labels.sum(axis=-1)
        point_labels = np.argmax(label_votes, axis=-1)
        end_time = time.time()
        #print("time is {}".format(end_time-start_time))
        
        start_time = time.time()
        face_instances = np.concatenate([fx_instance, fy_instance, fz_instance], axis=2)
        instance_votes = face_instances.sum(axis=-1)
        point_instances = np.argmax(instance_votes, axis=-1)
        end_time = time.time()
        #print("time is {}".format(end_time-start_time))
        
        instance_number = np.max(self.instance_ids[fz])
        end_func_time = time.time()
        #print("the function takes {} s to execute".format(end_func_time - start_func_time))
        return surface_points, surface_colors, point_labels, point_instances, instance_number


        

    def colorize_labels(self, point_labels):
        point_label_colors = np.vectorize(lambda x: LABEL_COLORS[x])(point_labels)
        point_label_colors = np.vstack(point_label_colors).T / 255
        return point_label_colors

    def create_if_data(self, res, n_points, sigmas=None, o3d_format=False, vicinity_ratio=None, scale_sigmas=True):
        # Creates data compatible with IF-Net model: https://virtualhumans.mpi-inf.mpg.de/ifnets/
        # Sigmas are in cm. They are scaled down

        surface_points, surface_idxs = self.mesh.sample(n_points, return_index=True)

        surface_colors = self.mesh.visual.face_colors[surface_idxs, :-1] / 255
        vicinity_points = []
        distances = []
    
        voxel_grid = self.voxelize(res, n_points, o3d_format=o3d_format)

        if sigmas is not None:
            if scale_sigmas:
                sigmas = sigmas/self.scale

            if vicinity_ratio is not None:
                N_sampled = int(vicinity_ratio*n_points)

            for i, s in enumerate(sigmas):
                pts = surface_points + s*np.random.randn(len(surface_points), 3)

                vicinity_points += [pts]
                distances += [np.abs(igl.signed_distance(pts, self.mesh.vertices, self.mesh.faces)[0])]

            distances = np.concatenate(distances, axis=0)
            vicinity_points = np.concatenate(vicinity_points, axis=0)       

        fx = self.mesh.faces[surface_idxs][:, 0]
        fy = self.mesh.faces[surface_idxs][:, 1]
        fz = self.mesh.faces[surface_idxs][:, 2]

        fx_label = get_one_hot(self.seg_idxs[fx], len(CLASS_NAMES))[:, :, None]
        fy_label = get_one_hot(self.seg_idxs[fy], len(CLASS_NAMES))[:, :, None]
        fz_label = get_one_hot(self.seg_idxs[fz], len(CLASS_NAMES))[:, :, None]

        # Majority voting
        face_labels = np.concatenate([fx_label, fy_label, fz_label], axis=2)
        label_votes = face_labels.sum(axis=-1)
        point_labels = np.argmax(label_votes, axis=-1)

        if o3d_format:
            seg_pcd = o3d.geometry.PointCloud()
            seg_pcd.points = o3d.utility.Vector3dVector(surface_points)
            seg_pcd.colors = o3d.utility.Vector3dVector(self.colorize_labels(point_labels))
            point_labels = seg_pcd
            
            if sigmas is None:
                vicinity_points = None
            else:
                viridis = cm.get_cmap('Reds')
                dist_norm = (distances-distances.min()) / (distances.max()-distances.min())
                
                vicinity_pcd = o3d.geometry.PointCloud()
                vicinity_pcd.points = o3d.utility.Vector3dVector(vicinity_points)
                vicinity_pcd.colors = o3d.utility.Vector3dVector(viridis(distances)[..., :-1])
                vicinity_points = vicinity_pcd

            surface_pcd = o3d.geometry.PointCloud()
            surface_pcd.points = o3d.utility.Vector3dVector(surface_points)
            surface_pcd.colors = o3d.utility.Vector3dVector(surface_colors)
            surface_points = surface_pcd

        if not o3d_format:
            return voxel_grid, surface_points, surface_colors, vicinity_points, distances, point_labels
        else:
            return voxel_grid, surface_points, vicinity_points, point_labels