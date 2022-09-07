import re
import os
import sys
import time
import statistics
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict, OrderedDict

from imps.data.scannet import ScanNetScene, CLASS_NAMES, MOVABLE_INSTANCE_NAMES
from imps.sqn.model import Randla
from imps.sqn.data_utils import prepare_input

from imps.point_augment.Common import loss_utils, point_augment_utils
from imps.point_augment.Augmentor.augmentor import Augmentor
from imps.point_augment.Classifier.classifier import RandlaClassifier


SCENE_DIR = '/app/mnt/scans/'

N_POINTS = int(1.5e5)
DEVICE = 'cuda'

IGNORED_LABELS = [0]

with open('../../data/train_dataset.txt') as f:
    train_data = f.read().splitlines()

seg2idx = {c: i for i,c in enumerate(CLASS_NAMES)}
movable_instances = dict((k, seg2idx[k]) for k in list(set(MOVABLE_INSTANCE_NAMES)))


randla = Randla(d_feature=3, d_in=8, encoder_dims=[8, 32, 64], device=DEVICE, num_class=len(CLASS_NAMES), interpolator='keops')

# initialize dimension, augmentor, optimizers

# memory used: 22MiB

dim = 3
augmentor = Augmentor().cuda()
optimizer_r = optim.Adam(randla.parameters(), lr=1e-3) # optimizer for randla 
optimizer_a = torch.optim.Adam(                        # optimzer for PA
            augmentor.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
        )



mse_fn = torch.nn.MSELoss(reduction = 'mean')

#Training Loop

randla.train()
augmentor=augmentor.train()

Augmentor_loss_list = []
Augmentor_loss_list_ = []
Randla_loss_list = []


for epoch in range(0, 10):  
    
    start_step_begin = time.time()
    
    for index, value in enumerate(train_data):
        #print(value)
        print(index + 1)
        
        # Getting the scene from the Scannet dataset and doing required scaling operation:
        time1 = time.time()
        scene = ScanNetScene(SCENE_DIR + value)
        surface_points, surface_colors, point_labels_, point_instances, instance_number = scene.create_points_colors_labels_from_pc(N_POINTS)
        surface_points_, center, pts_min, pts_max = point_augment_utils.scaling_point_cloud(surface_points)   
        time2 = time.time()
        # Calculation of class weights:
        class_counts = []
        for c in range(len(CLASS_NAMES.keys())):
            class_counts.append(np.sum(point_labels_ == c))
        class_counts = np.array(class_counts)
        for ign in IGNORED_LABELS:
            class_counts[ign] = 0
        class_weights = class_counts / class_counts.sum() 
        time3 = time.time()

        # Generating instance info in a scene:
        object_id_instances_list, original_instance_indices, instance_coordinates = point_augment_utils.get_instance_indices_dict(seg2idx, movable_instances, surface_points_, point_labels_, point_instances)   
        
        time4 = time.time()
        # Transfering features to GPU:
        features = torch.FloatTensor(surface_colors).unsqueeze(0).to(DEVICE)
        xyz = torch.FloatTensor(surface_points_).unsqueeze(0)
        point_labels = torch.LongTensor(point_labels_).unsqueeze(0).to(DEVICE)
        time5 = time.time()
        # Generating pooled points:
        input_points, input_neighbors, input_pools, feat_shape = prepare_input(xyz, k=16, num_layers=3, encoder_dims = [8,32,64], sub_sampling_ratio=4, 
                                                               device=DEVICE)
        time6 = time.time()
        # Generating features from RandLA Net Encoder:
        features_encoder_list = randla.encoder(features, input_points, input_neighbors, input_pools)
        #print(features_encoder_list[1].shape)
        time7 = time.time()
        
        # For now only considering augmentation for scenes with movable objects: 
        if len(object_id_instances_list) != 0:
            #print("Instances in the scene: {}".format(object_id_instances_list))
            time8 = time.time()
            # Gathering logits for unaugmented point cloud:
            label_pools, instance_pools = point_augment_utils.collect_label_and_instance_pools(point_labels.squeeze().cpu().numpy(), point_instances, input_pools, N_CLASS=len(CLASS_NAMES), instance_number=instance_number) 
            time9 = time.time()
            global_features, pooled_instance_list, total_number_of_instances, instance_number_dict = point_augment_utils.collect_instance_global_features(label_pools, instance_pools, features_encoder_list, seg2idx, movable_instances, surface_points_, feat_shape, object_id_instances_list)
            #print("global features shape {}".format(global_features))
            #print("Instances in the scene after pooling: {}".format(pooled_instance_list))
            #print("number of instances {}".format(total_number_of_instances))
            #print("dictionary {}".format(instance_number_dict))
            #print("Instance_coordinates after pooling: {}".format(pooled_instance_coordinates))
            time10 = time.time()
            logits = randla.decoder(features_encoder_list, input_points)
            time11 = time.time()
            logits_dict = point_augment_utils.get_logits_dict(movable_instances, object_id_instances_list, point_labels, logits)
            time12 = time.time()


            optimizer_r.zero_grad()
            optimizer_a.zero_grad()

            # Performing Augmentation:
            time13 = time.time()
            for key,value in instance_coordinates.items():
                if key in pooled_instance_list.keys():
                #print("entering performing augmentation group key is {}".format(key))
                    global_features_list = global_features[key]
                    for index, coord in enumerate(value):
                        coord, center_, pts_min_, pts_max_  = point_augment_utils.scaling_point_cloud(coord)
                        #print("Instance Augmentation:")
                        #print("Coordinate shape for instance {}".format(coord.shape))
                        #print("Min pts for scaling for instance: {}".format(pts_min_))
                        #print("Max pts for scaling for instance: {}".format(pts_max_))
                        #print("denominator for instance {}".format(pts_max_ - pts_min_))
                        if coord.shape[0] > 10:
                            coord = torch.FloatTensor(coord).unsqueeze(0).to(DEVICE)
                            coord=coord.transpose(2,1).contiguous()            
                            noise = 0.02 * torch.randn(1, 64).cuda()
                            aug_instance = augmentor(coord, global_features_list[index], noise)
                            aug_instance = aug_instance.squeeze(0).cpu().detach().numpy()
                            aug_instance = point_augment_utils.descaling_point_cloud(aug_instance.transpose(1,0), center_, pts_min_, pts_max_)
                            aug_instance = torch.FloatTensor(aug_instance).unsqueeze(0).to(DEVICE)
                            augmented_scene = point_augment_utils.add_aug_instance_to_pc(aug_instance.transpose(2,1), surface_points_, original_instance_indices[key][index])

            augmented_scene = torch.FloatTensor(augmented_scene).unsqueeze(0)
            time14 = time.time()


            # Gathering logits for augmented point cloud:
            input_points_aug, input_neighbors_aug, input_pools_aug, feat_shape = prepare_input(augmented_scene, k=16, num_layers=3, encoder_dims = [8,32,64], sub_sampling_ratio=4, 
                                                               device=DEVICE)  
            time15 = time.time()
            features_encoder_list_aug = randla.encoder(features, input_points_aug, input_neighbors_aug, input_pools_aug)
            time16 = time.time()
            label_pools_aug, instance_pools_aug = point_augment_utils.collect_label_and_instance_pools(point_labels.squeeze().cpu().numpy(), point_instances, input_pools_aug, N_CLASS=len(CLASS_NAMES), instance_number=instance_number)
            time17 = time.time()
            #global_features_aug = point_augment_utils.collect_instance_global_features(label_pools_aug, instance_pools_aug, features_encoder_list_aug, seg2idx, movable_instances, augmented_scene.squeeze(0), feat_shape, object_id_instances_list)
            #print("aug global features shape {}".format(global_features_aug))
            time18 = time.time()
            aug_logits = randla.decoder(features_encoder_list_aug, input_points_aug)
            time19 = time.time()
            aug_logits_dict = point_augment_utils.get_logits_dict(movable_instances, object_id_instances_list, point_labels, aug_logits)
            #print(aug_logits_dict.keys())
            #print(aug_logits_dict)
            time20 = time.time()

            # Calculating Augmentor Loss:
            time21 = time.time()
            aug_loss_cumulated = 0  
            aug_loss_cumulated_ = 0 
            aug_diff_cumulated = 0
            multiplicative_factor_list = []
            len_key = len(pooled_instance_list.keys())
            #print(len_key)
            for key, value in aug_logits_dict.items():
                if key in pooled_instance_list.keys():
                    #print("augmentor loss loop key {}".format(key))
                    if total_number_of_instances == 1 or len_key == 1:
                          multiplicative_factor = 1
                    else:
                          multiplicative_factor = (1 - (instance_number_dict[key] / total_number_of_instances))
                            
                    multiplicative_factor_list.append(multiplicative_factor)
                    #print("multiplicating factor for key {}".format(multiplicative_factor))
                    #print("multiplicating factor list for key {}".format(multiplicative_factor_list))
                
            weight_multiplied = sum(multiplicative_factor_list)
            #print("weight multilied is {}".format(weight_multiplied))
            for key, value in aug_logits_dict.items():
                if key in pooled_instance_list.keys():
                    #print("augmentor loss loop key {}".format(key))
                    target = torch.LongTensor([seg2idx[key]]).to("cuda")
                    aug_loss, aug_loss_  = loss_utils.aug_loss_(logits_dict[key], aug_logits_dict[key], target)
                    if total_number_of_instances == 1 or len_key == 1:
                          multiplicative_factor = 1
                    else:
                          multiplicative_factor = (1 - (instance_number_dict[key] / total_number_of_instances))

                    multiplicative_factor_list.append(multiplicative_factor)
                    aug_loss_updated = aug_loss * multiplicative_factor / weight_multiplied
                    #print("aug_loss is {}".format(aug_loss))
                    #print("aug_loss updated is {}".format(aug_loss_updated))
                    aug_loss_cumulated += aug_loss_updated
                    #print("aug_loss cumulated is {}".format(aug_loss_cumulated))
                    aug_loss_cumulated_ += aug_loss_
        
            augLoss = aug_loss_cumulated/total_number_of_instances
            #print(augLoss)
            augLoss_ = aug_loss_cumulated_/total_number_of_instances
            
            time22 = time.time()
            augLoss.backward(retain_graph=True)
            
            # Calculating loss from RandLA Net:
            time23 = time.time()
            features_encoder_list_squeezed_layer_0 = features_encoder_list[0].squeeze(0)
            features_encoder_list_squeezed_layer_0 = features_encoder_list_squeezed_layer_0.squeeze(2)
            features_encoder_list_aug_squeezed_layer_0 = features_encoder_list_aug[0].squeeze(0)
            features_encoder_list_aug_squeezed_layer_0 = features_encoder_list_aug_squeezed_layer_0.squeeze(2)
            feat_diff_layer_0 = 10.0*mse_fn(features_encoder_list_squeezed_layer_0,features_encoder_list_aug_squeezed_layer_0)

            features_encoder_list_squeezed_layer_1 = features_encoder_list[1].squeeze(0)
            features_encoder_list_squeezed_layer_1 = features_encoder_list_squeezed_layer_1.squeeze(2)
            features_encoder_list_aug_squeezed_layer_1 = features_encoder_list_aug[1].squeeze(0)
            features_encoder_list_aug_squeezed_layer_1 = features_encoder_list_aug_squeezed_layer_1.squeeze(2)
            feat_diff_layer_1 = 10.0*mse_fn(features_encoder_list_squeezed_layer_1,features_encoder_list_aug_squeezed_layer_1)

            features_encoder_list_squeezed_layer_2 = features_encoder_list[2].squeeze(0)
            features_encoder_list_squeezed_layer_2 = features_encoder_list_squeezed_layer_2.squeeze(2)
            features_encoder_list_aug_squeezed_layer_2 = features_encoder_list_aug[2].squeeze(0)
            features_encoder_list_aug_squeezed_layer_2 = features_encoder_list_aug_squeezed_layer_2.squeeze(2)
            feat_diff_layer_2 = 10.0*mse_fn(features_encoder_list_squeezed_layer_2,features_encoder_list_aug_squeezed_layer_2)

            feat_diff = feat_diff_layer_0 + feat_diff_layer_1 + feat_diff_layer_2
            #print(feat_diff)
            randlaLoss = loss_utils.get_randla_loss(logits, point_labels, class_weights)
            randlaLoss_aug = loss_utils.get_randla_loss(aug_logits, point_labels, class_weights)
            #print(randlaLoss_aug)
            semseg_loss = randlaLoss + randlaLoss_aug + feat_diff
            semseg_loss.backward(retain_graph=True)
            time24 = time.time()
            
            optimizer_r.step()
            optimizer_a.step()

        else:
            optimizer_r.zero_grad()
            logits = randla.decoder(features_encoder_list, input_points)
            semseg_loss = loss_utils.get_randla_loss(logits, point_labels, class_weights)
            semseg_loss.backward(retain_graph=True)
            optimizer_r.step()
            

        
    
    start_step_end = time.time()
    
    print(f"Epoch {epoch+1}, Augmentor loss:{round(augLoss.item(), 4)}, RandlaNet loss:{round(semseg_loss.item(), 4)}, Epoch_duration: {round((start_step_end-start_step_begin)/60,2)} mins")
    Augmentor_loss_list.append(round(augLoss.item(), 4))
    Augmentor_loss_list_.append(round(augLoss_.item(), 4)) 
    Randla_loss_list.append(round(semseg_loss.item(), 4))
    if ((epoch + 1) % 2) == 0:
                model_state = {
                      'epoch': epoch + 1,
                      'state_dict_randla': randla.state_dict(),
                      'optimizer_randla': optimizer_r.state_dict(),  
                      'loss': semseg_loss,
                  }
                torch.save(model_state, f'../../processed/periodic_models/epoch_{epoch+1}_state_model_PA_extension_1_disp_fac_whole_dataset')


torch.save(randla.state_dict(), '../../processed/saved_models/6_epochs_200_scenes_PA_extension_1_with_disp_fac')