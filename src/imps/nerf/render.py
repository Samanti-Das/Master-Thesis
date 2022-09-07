import tensorflow as tf

from .model import inverse_transform_sampling


def render_single_model(locs, nerf, pos_embedder, dir_embedder, origins, directions):
    """
    locs: (batch_size, N_rays) locations of the rays
    nerf: the nerf model
    pos_embedder: positional embedding function
    dir_embedder: directional embedding function
    origins: (batch_size, 3) origin of the rays
    directions: (batch_size, 3) viewdirections of the rays
    """
    locs = tf.cast(tf.expand_dims(locs, -1), tf.float32)
    directions = tf.cast(tf.expand_dims(directions, 1), tf.float32)
    origins = tf.cast(tf.expand_dims(origins, 1), tf.float32)

    positions = origins + locs*directions
    viewdir = tf.broadcast_to(directions, positions.shape)

    # embed
    pos_emb = pos_embedder(positions)
    dir_emb = dir_embedder(viewdir)
    # predict
    rgb, density = nerf.predict_step([pos_emb, dir_emb])
    weights = get_weights_from_density(density, locs, directions)
    pixel_color = tf.reduce_sum(weights*rgb, axis=1)

    return pixel_color, density, weights


def render_hierarchical(coarse_locs, coarse_nerf, fine_nerf, N_fine, pos_embedder, dir_embedder, origins, directions, only_rgb=False):
    """
    coarse_locs: (bath_size, N_rays)
    """
    coarse_pixel, coarse_density, coarse_weights = render_single_model(coarse_locs, coarse_nerf, pos_embedder, dir_embedder, origins, directions)
    # Fine position sampling
    norm_weights = coarse_weights / tf.reduce_sum(coarse_weights+1e-8, axis=1, keepdims=True)
    bins = 0.5*(coarse_locs[:,1:] + coarse_locs[:,:-1])
    fine_locs = inverse_transform_sampling(norm_weights, bins, N_fine)
    # Put together the coarse and fine samples
    all_locs = tf.sort(tf.concat([coarse_locs, fine_locs], axis=1), axis=1)
    # Get fine results
    all_pixel, all_density, all_weights = render_single_model(all_locs, fine_nerf, pos_embedder, dir_embedder, origins, directions)

    if only_rgb:
        return coarse_pixel, all_pixel
    return coarse_pixel, coarse_density, coarse_weights, all_pixel, all_density, all_weights, all_locs


def get_weights_from_density(ray_density, ray_locations, ray_direction):
    '''
    density: (N_ray, N_sample, 1) predicted density values of sampled points on rays.
    ray_locations: (N_ray, N_sample, 1) locations of sampled points on rays. Not position. Position is calculated as p = origin + ray_locaiton*ray_direction
    '''
    dists = ray_locations[:,1:]-ray_locations[:,:-1]
    dists = tf.concat([dists, tf.broadcast_to([1e10], dists[:, :1, :].shape)], axis=1)
    dists = dists * tf.linalg.norm(ray_direction, axis=-1, keepdims=True)
    alpha = 1 - tf.exp(-ray_density*dists)
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=1, exclusive=True)
    
    return weights