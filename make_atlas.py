#!/usr/bin/env python3

import os
import time
import argparse


# Arguments.
f = argparse.ArgumentDefaultsHelpFormatter
p = argparse.ArgumentParser(formatter_class=f)
p.add_argument('images', nargs='+', help='input images')
p.add_argument('-t', dest='trans', help='output transform name')
p.add_argument('-o', dest='moved', default='moved_{:04d}.mgz',help='output image name')
p.add_argument('-a', dest='atlas', default='atlas_{:03d}.mgz', help='output atlas name')
p.add_argument('-T', dest='target', help='initial registration target')
p.add_argument('-w', dest='weights', default='hyp_mse_uni_256_lm10_mid.h5', help='weights')
p.add_argument('-s', dest='smooth', type=float, default=0.5, help='warp smoothness')
p.add_argument('-i', dest='iter', type=int, default=4, help='iterations')
p.add_argument('-e', dest='extent', type=int, default=(192,) * 3, help='atlas shape')
p.add_argument('-r', dest='res', type=float, default=1, help='atlas resolution')
p.add_argument('-f', dest='average', default='median', help='initial averaging function')
p.add_argument('-g', dest='gpu', action='store_true', help='use the GPU')
arg = p.parse_args()


def load_image(f, res=arg.res, shape=arg.extent):
    f = sf.load_volume(f).reorient('LIA').resize(res).reshape(shape)
    f = tf.cast(f.framed_data, tf.float32)
    f -= tf.reduce_min(f)
    f /= tf.reduce_max(f)
    return f[None]


def save_image(path, im, res=arg.res):
    geom = sf.ImageGeometry(shape=im.shape[1:-1], voxsize=res)
    sf.Volume(data=np.squeeze(im), geometry=geom).save(path)


# Environment.
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu if arg.gpu else ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Third-party imports.
import numpy as np
import surfa as sf
import tensorflow as tf
import voxelmorph as vxm


# Setup.
images = list(map(load_image, arg.images))
prop = dict(mid_space=True, in_shape=arg.extent)
model = vxm.networks.HyperVxmJoint(**prop)
model.load_weights(arg.weights)
hyper = tf.cast([arg.smooth], tf.float32)


# Initialization.
if arg.target:
    atlas = load_image(arg.target)
else:
    atlas = getattr(np, arg.average)(images, axis=0)
save_image(arg.atlas.format(0), atlas)


for i in range(arg.iter):
    start = time.time()

    # Transforms.
    trans = [model((hyper, f, atlas)) for f in images]
    mean = tf.reduce_mean(trans, axis=0)
    trans = [t - mean for t in trans]

    # Images.
    layer = vxm.layers.SpatialTransformer
    moved = [layer(fill_value=0)((f, t)) for f, t in zip(images, trans)]

    # Atlas.
    atlas = tf.reduce_mean(moved, axis=0)
    save_image(arg.atlas.format(i + 1), atlas)

    sec = time.time() - start
    print(f'Finished iteration {i + 1} ({sec:.1f} sec)', flush=True)


# Output images.
if arg.moved:
    for i, m in enumerate(moved):
        save_image(arg.moved.format(i), m)


# Output transforms.
if arg.trans:
    for i, t in enumerate(trans):
        save_image(arg.trans.format(i), t)
