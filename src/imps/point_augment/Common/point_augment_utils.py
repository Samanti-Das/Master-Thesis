from collections import defaultdict, OrderedDict
import numpy as np
import torch
import scipy
import scipy.interpolate
#import albumentations as A
import volumentations as V


def get_instance_indices_dict(seg2idx, movable_instances, surface_points, point_labels, point_instances, lable_to_instance_dict=None):
    
    label_index_dict = {} 
    for index, name in enumerate(list(seg2idx.keys())):
        res_list = np.where(point_labels == index) # extracting indices corresponding to a label
        label_index_dict[name] = res_list
        
    point_instances_dict = {}
    for key, value in movable_instances.items():
        res_list = np.where(point_labels == value)
        instance_indices = point_instances[res_list]
        point_instances_dict[key] = instance_indices
     
    point_instances_dict = dict( [(k,v) for k,v in point_instances_dict.items() if len(v)>0])
    object_id_instances_list = {}
    instance_indices_list = {}
    
    for key, value in point_instances_dict.items():
        positions = defaultdict(set)
        for index, value in enumerate(point_instances_dict[key]):
            positions[value].add(index)
         
        if lable_to_instance_dict is not None:
        
            instance_number_for_label = len(lable_to_instance_dict[key])
            instances = [None]*instance_number_for_label
            instance_indices = [None]*instance_number_for_label
            
            for k, v in dict(positions).items():
                
                if k in lable_to_instance_dict[key]:
                    
                    index = lable_to_instance_dict[key].index(k)

                    instances[index] = k
                    instance_indices[index] = list(v)             

                object_id_instances_list[key] = instances
                instance_indices_list[key] = instance_indices
                                
        else:
            instances = []
            instance_indices = []
            for k, v in dict(positions).items():
                instances.append(k)
                instance_indices.append(list(v))
                object_id_instances_list[key] = instances
                instance_indices_list[key] = instance_indices


    original_instance_indices= {}

    for key in instance_indices_list.keys():    
        instance_indices_list_ = []
        for instance_indices in instance_indices_list[key]:
            
            original_indices = np.array(label_index_dict[key]).squeeze(0)[instance_indices]
            instance_indices_list_.append(original_indices)
            original_instance_indices[key] = instance_indices_list_
    
    instance_coordinates = {}
    surface_points_arr = np.array(surface_points)
    
    #print(original_instance_indices)

    for key, value in original_instance_indices.items():
        instance_coord_list = []
        for coord in value:
            instance_coordinates_ = surface_points_arr[coord]
            instance_coord_list.append(instance_coordinates_)
        instance_coordinates[key]= instance_coord_list
    
          
    return object_id_instances_list, original_instance_indices, instance_coordinates


def add_aug_instance_to_pc(aug_instance, surface_points, index_list):
    aug_instance = aug_instance.squeeze(0).transpose(1,0)
    aug_instance = aug_instance.cpu().detach()
    surface_points[index_list] = aug_instance
    return surface_points

def collect_label_and_instance_pools(labels, instances, input_pools, N_CLASS, instance_number):
    label_pools = []
    instance_pools = []
    prev_label_pools = labels
    prev_instance_pools = instances
    
    for ip in input_pools:
        ip = ip.squeeze().cpu().numpy()
        n_pts, k = ip.shape
        ip = ip.reshape(-1)
        
        pool_pt_labels = prev_label_pools[ip]
        pool_pt_instances = prev_instance_pools[ip]
        
        oh_labels = np.eye(N_CLASS)[pool_pt_labels]
        oh_instances = np.eye(instance_number+1)[pool_pt_instances]
        
        pooled_labels = oh_labels.reshape(n_pts, k, N_CLASS)
        pooled_instances = oh_instances.reshape(n_pts, k, instance_number+1)
        
        pooled_votes_label = pooled_labels.sum(axis=1)
        pooled_votes_instance = pooled_instances.sum(axis=1)

        prev_label_pools = pooled_votes_label.argmax(axis=-1)
        prev_instance_pools = pooled_votes_instance.argmax(axis=-1)
        
        label_pools.append(prev_label_pools)
        instance_pools.append(prev_instance_pools)
        
    return label_pools, instance_pools



def averaging_labels_from_randla_decoder(label_index_dict, logits, seg2idx):
    
    output_label_dict={} # initializing an empty dict
    for label, value in seg2idx.items(): #len_instances = len(CLASS_NAMES) = 21 , each class representing an instance
        array_for_label = logits.squeeze()[label_index_dict[label]]
        label_array = torch.mean(array_for_label, dim = 0) # average out the value for all the indices
        output_label_dict[value] = label_array # get a 1x21 array for the entire instance; thereby getting the class_label for a particular instance from the decoder
                
    return output_label_dict


def collect_instance_global_features(label_pools, instance_pools, features_encoder_list, seg2idx, movable_instances, surface_points, feat_shape, object_id_instances_list):
    
    pooled_instance_indices_list = [] 
    pooled_object_id_instance_list = []  
    for index, label in enumerate(label_pools):
        pooled_object_id_instances_list, pooled_instance_indices, pooled_instance_coordinates = get_instance_indices_dict(seg2idx, movable_instances, surface_points, label, instance_pools[index], lable_to_instance_dict = object_id_instances_list)
        pooled_instance_indices_list.append(pooled_instance_indices)
        pooled_object_id_instance_list.append(pooled_object_id_instances_list)
    total_number_of_instances = 0    
    #print("pooled_instance_indices_list {}".format(pooled_object_id_instance_list[1]))
    instance_number_dict = {}
    for key,value in pooled_object_id_instance_list[1].items():
        number_of_instances = len(pooled_object_id_instance_list[1][key])
        instance_number_dict[key] = number_of_instances
        total_number_of_instances += number_of_instances
        
    global_features_by_layer = []
    for index, pool_layer in enumerate(pooled_instance_indices_list):
        feature = features_encoder_list[index].squeeze(0).squeeze(-1).permute(1,0)
        global_instance_features_dict = {}
        
        for key, value in pool_layer.items():
            #print(key)
            global_features_list = []
            for instance_number, indices in enumerate(value):
                instance_features = feature[indices]
                global_instance_features, indices =torch.max(instance_features, 0)
                #print(global_instance_features.shape)
                global_features_list.append(global_instance_features)
                #print(global_features_list)
            global_instance_features_dict[f'{key}']= global_features_list
        #print("global_instance_features_dict {}".format(global_instance_features_dict))
        global_features_by_layer.append(global_instance_features_dict)
    
    #print(global_features_by_layer[1])
    #for key, value in global_features_by_layer[1].items():
    #    for index, features in enumerate(value):
    #        if len(features)!=64:
    #            value.pop(index)
                
    
    #listKeys = list(global_features_by_layer[1].keys())
    #print(listKeys)
    #global_features_unclean = {}
    #for k in listKeys:
    #    listValues = [] 
        #print(k)
        #for d in global_features_by_layer:
    #    try:
            #print("entering try loop")
    #        listValues.append(global_features_by_layer[1][k])
            #print("listValues are {}".format(listValues))
    #    except:
            #print("entering except loop")
    #        pass
    #    if listValues:
            #print("entering if condition")
    #        global_features_unclean[k] = torch.cat(tuple(listValues), dim=0)

    #print("global_features_unclean is {}".format(global_features_unclean))
    pooled_object_id_instance_number_list = {}
    for key,value in pooled_object_id_instance_list[1].items():

        instance_list = []
        for val in value:
            if val != None :
                instance_list.append(val)
        pooled_object_id_instance_number_list[key] = instance_list

    global_features_layer_2 = global_features_by_layer[1] #{k: v for k, v in global_features_unclean.items() if len(v) == 64}
    #pooled_instance_list = pooled_object_id_instance_number_list
    return global_features_layer_2, pooled_object_id_instance_number_list, total_number_of_instances, instance_number_dict


        
def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.
    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
    pointcloud = pointcloud.cpu().detach().numpy()
    pointcloud = pointcloud.squeeze(0)
    pointcloud = pointcloud.transpose(1,0)
    #print(pointcloud.shape)
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    pointcloud = pointcloud.transpose(1,0)
    pointcloud = torch.FloatTensor(pointcloud).to('cuda')
    pointcloud = pointcloud.unsqueeze(0)
    #print(pointcloud.shape)
    #pointcloud = torch.FloatTensor(pointcloud).to('cuda')
    return pointcloud


def scaling_point_cloud(surface_points):
    center = surface_points.mean(axis=0, keepdims=True)
    surface_points -= center
    pts_min = surface_points.min(axis=0, keepdims=True)
    pts_max = surface_points.max(axis=0, keepdims=True)
    surface_points = (surface_points - pts_min) / (pts_max - pts_min)
    surface_points -= 0.5   
    return surface_points, center, pts_min, pts_max

def descaling_point_cloud(surface_points_new, center, pts_min, pts_max):
    surface_points_new += 0.5
    surface_points_new = surface_points_new * (pts_max - pts_min) + pts_min
    surface_points_new += center 
    return surface_points_new

def get_logits_dict(movable_instances, object_id_instances_list, point_labels, logits):
    logits_dict = {}
    for key,value in movable_instances.items():
        if key in object_id_instances_list.keys():
            indices = torch.where(point_labels == value)
            #print(indices[0].shape)
            logits=logits.squeeze(0)
            logits_object = logits[indices[0]]
            logits_dict[f'{key}'] = logits_object
    return logits_dict

def flip_in_center(coordinates):
    # moving coordinates to center
    #print(coordinates.shape)
    coordinates = coordinates.squeeze(0)
    coordinates = coordinates.transpose(1,0)
    #print(coordinates.shape)
    coordinates = coordinates.cpu().detach().numpy()
    #print(coordinates.shape)
    
    coordinates -= coordinates.mean(0)
    #print(coordinates.shape)
    aug = V.Compose(
        [
            V.Flip(axis=(0, 1, 0), always_apply=True),
            V.Flip(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    #print(first_crop.shape)
    first_crop &= coordinates[:, 1] > 0
    #print(first_crop.shape)
    # x -y
    #print(first_crop)
    second_crop = coordinates[:, 0] > 0
    #print(second_crop.shape)
    second_crop &= coordinates[:, 1] < 0
    #print(second_crop.shape)
    # -x y
    third_crop = coordinates[:, 0] < 0
    #print(third_crop.shape)
    third_crop &= coordinates[:, 1] > 0
    #print(third_crop.shape)
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    #print(fourth_crop.shape)
    fourth_crop &= coordinates[:, 1] < 0
    #print(fourth_crop.shape)
    #print("end crop")
    
    if first_crop.size > 1:
        if coordinates[first_crop].size > 1:
        #print(first_crop.size)
          coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        #print(second_crop.size)
        #print(coordinates[second_crop])
        if coordinates[second_crop].size > 1 :
            minimum = coordinates[second_crop].min(0)
            minimum[2] = 0
            minimum[0] = 0
            coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
            coordinates[second_crop] += minimum
    if third_crop.size > 1:
        if coordinates[third_crop].size > 1 :
            minimum = coordinates[third_crop].min(0)
            minimum[2] = 0
            minimum[1] = 0
            coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
            coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        if coordinates[fourth_crop].size > 1 :
            minimum = coordinates[fourth_crop].min(0)
            minimum[2] = 0
            coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
            coordinates[fourth_crop] += minimum
            
            
    coordinates = coordinates.transpose(1,0)
    coordinates = torch.FloatTensor(coordinates).to('cuda')
    coordinates = coordinates.unsqueeze(0)
    #print(coordinates.shape)

    return coordinates