import argparse
import sys
import os
import glob
from datetime import datetime

sys.path.insert(1, '../src')
sys.path.insert(1, '/home/cem/Documents/implicit_scene/LLFF')

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

from imps.nerf.data_utils import get_poses_bounds, generate_data
from imps.nerf.model import get_embedder, get_single_nerf
from imps.nerf.ray_utils import sample_points_on_ray
from imps.nerf.render import render_hierarchical, render_single_model
from imps.nerf.args import parser

'''
This implementation of NeRF and its variants assumes big scenes are being modelled not single objects.
'''

embedding_include_input = True

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.no_log:
        logdir = f"../logs/{args.exp_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        train_logdir = logdir + '/train'
        eval_logdir = logdir + '/eval'

        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        eval_summary_writer = tf.summary.create_file_writer(eval_logdir)

    poses_bounds = get_poses_bounds(args.data_dir, args.colmap, args.near, args.far, args.img_format, args.img_prefix)
    
    # Now load the image data and train the nerf model :)

    tf.keras.backend.clear_session()
    lr = args.initial_lr
    if args.lr_decay > 0:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=args.lr_decay * 1000, decay_rate=0.1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    add_input = int(embedding_include_input) * 3

    pos_embedder = get_embedder(args.Lp, include_input=embedding_include_input)
    dir_embedder = get_embedder(args.Ld, include_input=embedding_include_input)

    coarse_nerf = get_single_nerf(args.n_layer, args.n_hidden, args.skip_layer, args.N_coarse, args.Lp*6+add_input, args.Ld*6+add_input, name='coarse')
    fine_nerf = get_single_nerf(args.n_layer, args.n_hidden, args.skip_layer, args.N_coarse+args.N_fine, args.Lp*6+add_input, args.Ld*6+add_input, name='fine')

    image_files = sorted(glob.glob(os.path.join(args.data_dir, f'images/*{args.img_format}')))
    assert len(image_files) > 0, f"No images under /images folder with extension {args.img_format}"

    trainable_vars = coarse_nerf.trainable_variables + fine_nerf.trainable_variables

    step = 0

    origins, viewdirs, imgs, bounds = generate_data(image_files, poses_bounds, down_scale=args.down_scale)
    
    # Use the data from the first view-point for evaluation
    eval_image = imgs[0]
    origins_eval = origins[0].reshape(-1, 3)
    viewdirs_eval = viewdirs[0].reshape(-1, 3)
    pixels_eval = imgs[0].reshape(-1, 3)
    bounds_eval = bounds[0].reshape(-1, 2)

    # Use the remaining as training data
    origins = origins[1:].reshape(-1, 3)
    viewdirs = viewdirs[1:].reshape(-1, 3)
    imgs = imgs[1:].reshape(-1, 3)
    bounds = bounds[1:].reshape(-1, 2)

    ray_idxs = np.arange(imgs.shape[0])

    for e in range(args.n_epochs):
        tf.keras.models.save_model(coarse_nerf, logdir+'/coarse')
        tf.keras.models.save_model(fine_nerf, logdir+'/fine')

        np.random.shuffle(ray_idxs)

        for b in tqdm(range(0, len(ray_idxs), args.batch_size), desc=f'Epoch {e+1}'):
            batch_idxs = ray_idxs[b:b+args.batch_size]

            origins_batch = tf.cast(origins[batch_idxs], tf.float32)
            directions_batch = tf.cast(viewdirs[batch_idxs], tf.float32)
            pixels_batch = tf.cast(imgs[batch_idxs], tf.float32)
            bounds_batch = tf.cast(bounds[batch_idxs], tf.float32)
            
            # Sample and perturb ray locations
            coarse_locs = sample_points_on_ray(bounds_batch, args.N_coarse, perturb=True)
            
            with tf.GradientTape() as tape:
                coarse_rgb, fine_rgb = render_hierarchical(coarse_locs, coarse_nerf, fine_nerf, args.N_fine, pos_embedder, 
                                                            dir_embedder, origins_batch, directions_batch, only_rgb=True)
                loss_fine = tf.reduce_mean(tf.square(fine_rgb - pixels_batch))
                loss_coarse = tf.reduce_mean(tf.square(coarse_rgb - pixels_batch))
                loss = loss_fine + loss_coarse

                grads = tape.gradient(loss, trainable_vars)

            optimizer.apply_gradients(zip(grads, trainable_vars))

            if not args.no_log:
                # Compute PSNR from the fine_rgb prediction
                psnr_fine = tf.image.psnr(tf.expand_dims(pixels_batch, 1), 
                                        tf.expand_dims(fine_rgb, 1), max_val=1.0)
                
                with train_summary_writer.as_default():
                    tf.summary.scalar('psnr', psnr_fine, step=step)
                    tf.summary.scalar('loss', loss, step=step)

            if ((step % args.eval_step) == 0) and (not args.no_log):
                pred_image = np.zeros_like(pixels_eval)
                for b in range(0, len(bounds_eval), args.batch_size):
                    origins_batch = tf.cast(origins_eval[b:b+args.batch_size], tf.float32)
                    directions_batch = tf.cast(viewdirs_eval[b:b+args.batch_size], tf.float32)
                    pixels_batch = tf.cast(pixels_eval[b:b+args.batch_size], tf.float32)
                    bounds_batch = tf.cast(bounds_eval[b:b+args.batch_size], tf.float32)
                    
                    # Sample and perturb ray locations
                    coarse_locs = sample_points_on_ray(bounds_batch, args.N_coarse, perturb=False)
                    
                    coarse_rgb, fine_rgb = render_hierarchical(coarse_locs, coarse_nerf, fine_nerf, args.N_fine, pos_embedder, 
                                                     dir_embedder, origins_batch, directions_batch, only_rgb=True)
                    pred_image[b:b+args.batch_size] = fine_rgb

                pred_image = pred_image.reshape(*eval_image.shape)
                psnr_fine = tf.image.psnr(eval_image, pred_image, max_val=1.0)

                with eval_summary_writer.as_default():
                    side2side = tf.stack([eval_image, pred_image])
                    tf.summary.scalar('psnr', psnr_fine, step=step)
                    tf.summary.image('image', side2side, step=step)
                
            step += 1