import tensorflow as tf
import tensorflow.keras.layers as tfkl

import numpy as np


def get_single_nerf(n_layer, n_hidden, skip_layer, n_sample, n_dim_pos, n_dim_dir, name=None):
    scope = f"{name}/" if name else ''
    pos_inp = tfkl.Input(shape=(n_sample, n_dim_pos), name=scope+'input_pos')
    dir_inp = tfkl.Input(shape=(n_sample, n_dim_dir), name=scope+'input_dir')
    
    z = tfkl.Dense(n_hidden, activation='relu', name=scope+'dense0')(pos_inp)
    for i in range(1, skip_layer):
        z = tfkl.Dense(n_hidden, activation='relu', name=scope+f'dense{i}')(z)
        
    z = tf.concat([z, pos_inp], name=scope+f'skip_pos', axis=-1)
    for i in range(skip_layer, n_layer-1):
        z = tfkl.Dense(n_hidden, activation='relu', name=scope+f'dense{i}')(z)
    
    # Density is independent of the viewing direction
    density = tfkl.Dense(1, activation='relu', name=scope+'output_density')(z)
    
    z = tf.concat([z, dir_inp], name=scope+f'skip_dir', axis=-1)
    z = tfkl.Dense(n_hidden//2, activation='relu', name=scope+f'dense{n_layer-1}')(z)
    rgb = tfkl.Dense(3, activation='sigmoid', name=scope+'output_rgb')(z)
    
    return tf.keras.Model(inputs=[pos_inp, dir_inp], outputs=[rgb, density])

# TODO: The multiplicaiton with pi is removed here. Put it back but before scale down the poses to the room boundaries.
def get_embedder(L, include_input=False, fns=[tf.cos, tf.sin]):
    mults = 2**tf.range(L, dtype=tf.float32)
    functions = []
    
    if include_input:
        functions.append(lambda x: x)
        
    for m in mults:
        for f in fns:
            functions.append(lambda x: f(m*x))
    
    def embed(inp):
        return tf.concat([f(inp) for f in functions], axis=-1)
    
    return embed
    

def inverse_transform_sampling(pdf, bins, N_samples):
    '''
    pdf: normalized set of weights of shape: (batch, N_bins). The dimensions with shape 1 are squeezed.
    bins: Values of the bins: (batch, N_bins-1). The dimensions with shape 1 are squeezed.
    N_samples: Number of samples obtained from the pdf
    
    Taken from the official NeRF repo
    '''
    batch_size = pdf.shape[0]
    cdf = tf.squeeze(tf.cumsum(pdf, axis=1))
    bins = tf.squeeze(bins)
    uniform_sampled = tf.random.uniform(shape=(batch_size, N_samples))
        
    inds = tf.searchsorted(cdf, uniform_sampled, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (uniform_sampled-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    tf.stop_gradient(samples)

    
    return samples