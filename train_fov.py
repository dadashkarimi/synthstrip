import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import tensorflow as tf
from tensorflow.keras.models import Model
from neurite.tf import models  # Assuming the module's location
import voxelmorph.tf.losses as vtml
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import neurite as ne
import sys
import nibabel as nib
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full

import tensorflow.keras.layers as KL
import voxelmorph as vxm
from utils import *
import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pathlib
import surfa as sf
import re
import json
from keras import backend as K

from tensorflow.keras.layers import Input, BatchNormalization

# from tensorflow.keras.layers import Input, InstanceNormalization
from tensorflow_addons.layers import InstanceNormalization


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.0001, help="learning rate")
parser.add_argument('-zb','--zero_background',type=float, default=0.2, help="zero background")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-sc','--scale',type=float,default=0.2,help="scale")
parser.add_argument('-body_scale','--body_scale',type=float,default=2,help="body scale")
parser.add_argument('-wm','--warp_max',type=float,default=0.2,help="scale")
parser.add_argument('-bsh','--brain_shift',type=int,default=5,help="brain shift")
parser.add_argument('-body_shift','--body_shift',type=int,default=0,help="brain shift")

parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-m','--num_dims',default=256,type=int,help="number of dims")
parser.add_argument('-e', '--encoder_layers', nargs='+', type=int, help="A list of dimensions for the encoder")
parser.add_argument('-d', '--decoder_layers', nargs='+', type=int, help="A list of dimensions for the decoder")
# parser.add_argument('-mgh', '--mgh_label_maps', choices=['brain', 'body', 'both'], default='both',
#                     help="Specify the type of label maps (choices: brain, body, both)")

parser.add_argument('-ring', '--ring', action='store_true', default=False, help="A list of dimensions for the decoder")
parser.add_argument('-shapes', '--shapes', action='store_true', default=False, help="A list of dimensions for the decoder")
parser.add_argument('-feta', '--feta', action='store_true', default=False, help="feta")
parser.add_argument('-synth', '--synth', action='store_true', default=False, help="feta")
parser.add_argument('-conc', '--conc', action='store_true', default=False, help="concentricity")

# parser.add_argument('-synth', '--synth', action='store_true', default=False, help="feta")

parser.add_argument('-body', '--body', action='store_true', default=False, help="feta")
parser.add_argument('-brain', '--brain', action='store_true', default=False, help="feta")
parser.add_argument('-gmm', '--gmm', action='store_true', default=False, help="feta")
parser.add_argument('-norm', '--norm', action='store_true', default=False, help="feta")
parser.add_argument('-det', '--detection', action='store_true', default=False, help="detection")
parser.add_argument('-cm', '--center_of_mass', action='store_true', default=False, help="detection")
parser.add_argument('-np', '--ninty_percentile', action='store_true', default=False, help="detection")
parser.add_argument('-bn', '--batch_norm', action='store_true', default=False, help="detection")

args = parser.parse_args()
# dimx=192
# dimy=192
# dimz=192
dimx=args.num_dims
dimy=args.num_dims
dimz=args.num_dims
mgh= None
# nb_features=64
if args.encoder_layers:
    nb_features = '_'.join(map(str, args.encoder_layers))
# mgh = pathlib.Path('fetus_brain_label')

# feta = pathlib.Path('/autofs/space/bal_004/users/jd1677/synthstrip/feta_3d')
# feta_files = list(feta.glob('sub-???/anat/sub-???_rec-mial_dseg.nii.gz'))

ngpus =len(os.environ["CUDA_VISIBLE_DEVICES"])
print(f'using {ngpus} gpus')
if ngpus > 1:
    model_device = '/gpu:0'
    synth_device = '/gpu:1'
    synth_gpu = 1
    dev_str = ", ".join(map(str, range(ngpus)))
    print("dev_str:",dev_str)
else:
    model_device = '/gpu:0'
    synth_device = model_device
    synth_gpu = 0
    dev_str = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = dev_str
print(f'model_device {model_device}, synth_device {synth_device}, dev_str {dev_str}')
print(f'physical GPU # is {os.getenv("SLURM_STEP_GPUS")}')

if args.num_dims==256:
    print("loading 256 fetus dataset.")
    feta = pathlib.Path('feta_resized_256')
    with open("params.json", "r") as json_file:
        config = json.load(json_file)
elif args.num_dims==192:
    print("loading 192 fetus dataset.")
    feta = pathlib.Path('feta_resized_192')
    with open("params_192.json", "r") as json_file:
        config = json.load(json_file)

feta_files = list(feta.glob('*.nii.gz'))
feta_label_maps = [np.uint8(f.dataobj) for f in map(nib.load, feta_files)]


# if args.synth:
log_dir = 'logs/logs_synth'
models_dir = 'models_synth'
# else:
#     log_dir = 'logs'
#     models_dir = 'models'

if args.conc:
    log_dir += '_less_conc'
    models_dir += '_less_conc'

if args.detection:
    log_dir += '_detection'
    models_dir += '_detection'

if args.batch_norm:
    log_dir += '_bn'
    models_dir += '_bn'


if args.center_of_mass:
    log_dir += '_cm'
    models_dir += '_cm'

if args.ninty_percentile:
    log_dir += '_90'
    models_dir += '_90'


if args.norm:
    log_dir += '_norm'
    models_dir += '_norm'
    
if args.gmm:
    log_dir += '_gmm'
    models_dir += '_gmm'   
    
if args.ring:
    log_dir += '_ring'
    models_dir += '_ring'
if args.shapes:
    log_dir += '_shapes'
    models_dir += '_shapes'
if args.feta:
    log_dir += '_feta'
    models_dir += '_feta'
    
if args.brain:
    log_dir += '_mgh_brain'
    models_dir += '_mgh_brain'



if args.body:
    log_dir += '_mgh_body'
    models_dir += '_mgh_body'

log_dir += '_'+str(args.num_dims)
models_dir += '_'+str(args.num_dims)

# en = args.encoder_layers
# de = args.decoder_layers

# if args.synth:
#     en = [16, 16, 32 ,32 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64, 256 ,256]
#     de = [256 ,256 ,64 ,64, 64 ,64 ,64, 64, 64, 64 ,64, 32 ,32 ,16 ,16 ,2]
mgh_files = []
mgh_label_maps = []
if args.brain or args.body:
    mgh = pathlib.Path('fetus_label_map')
    mgh_files = list(mgh.glob('*.nii.gz'))
    label_maps = [np.uint8(sf.load_volume(str(file_path)).reshape([dimx,dimy,dimz]).data) for file_path in mgh_files]
    # labels_in = np.unique(label_maps) # change this for feta
    in_shape = label_maps[0].shape
    print("brain or body!")

else:
    label_maps = feta_label_maps
    labels = np.unique(label_maps)
    num_labels=9
    in_shape = label_maps[0].shape
    print("in_shape",in_shape)
    # labels_in = range(max(labels) + num_labels + 1)



num_shapes = 8

import os, shutil, glob

latest_weight = max(glob.glob(os.path.join(models_dir, 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

if latest_weight:
    shutil.move(latest_weight, os.path.join(models_dir, 'weights_epoch_0.h5'))


# initial_epoch=args.initial_epoch




if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

checkpoint_path=models_dir+'/weights_epoch_0.h5'

if not os.listdir(log_dir):
    initial_epoch = 0
    print("initial epoch is 0")
elif latest_weight:
    match = re.search(r'(\d+)', latest_weight)
    initial_epoch = int(match.group())
else:
    initial_epoch= args.initial_epoch




batch_size=args.batch_size
warp_max=2   
warp_max=2.5
warp_min=.5
warp_blur_min=np.array([2, 4, 8])
warp_blur_max=warp_blur_min*2
bias_blur_min=np.array([2, 4, 8])
bias_blur_max=bias_blur_min*2
initial_lr=args.learning_rate
nb_conv_per_level=args.nb_conv_per_level
# lr = args.learning_rate
nb_levels=5
conv_size=3
num_epochs=50000
# num_bg_labels=16
warp_fwhm_min=10
warp_fwhm_max=20
warp_min_shapes=10
warp_max_shapes=50
# in_shape=(dimx,dimy,dimz)
bg_brain = True

warp_max=2
warp_min=1
image_fwhm_min=20
image_fwhm_max=40
aff_shift=30
aff_rotate=180
aff_shear=0.1
blur_max=2.4
slice_prob=1
crop_prob=1
bias_min=0.01
bias_max=0.2
zero_background=args.zero_background
aff_scale=args.scale
up_scale=False

# labels_in = range(max(labels) + num_labels + 1)


# def create_model(model_config):
#     return ne.models.labels_to_image_new(**model_config)




# Access the configuration
model1_config = config["brain"]
model2_config = config["body"]
model_feta_config = config["feta"]

model_shapes_config = config["shapes"]

model3_config = config["labels_to_image_model"]

# if args.conc:
#     model4_config = config["labels_to_image_model_with_shapes_24"]
# else:
# model4_config = config["labels_to_image_model_with_shapes"]

# Convert labels_out keys to integers for all models
# model1_config["labels_out"] = {int(key): value for key, value in model1_config["labels_out"].items()}
# model2_config["labels_out"] = {int(key): value for key, value in model2_config["labels_out"].items()}


model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}

# model4_config["labels_out"] = {int(key): value for key, value in model4_config["labels_out"].items()}
# Now you have the modified configuration
# Brain
# model1 = create_model(model1_config)
# Body
# model2 = create_model(model2_config)
# FeTA brain
model_feta = create_model(model_feta_config)

#shapes
model_shapes = create_model(model_shapes_config)


# Model
labels_to_image_model = create_model(model3_config)

# labels_to_image_model_with_shapes = create_model(model4_config)

# gen_arg = {
#     'in_shape': in_shape,
#     'labels_in': labels_in,
#     'labels_out': {f: 1 if f in (1, 2, 3, 4, 5, 6, 7) else 0 for f in labels_in},
#     'warp_min': 0.01,  # Adjust this value for a small warping change
#     'warp_max': 2,  # Adjust this value for a small warping change
#     'blur_max': 1,
#     'noise_max': 0.1,
#     'one_hot': True,
#     'aff_scale': 0.5,
#     'axes_flip': True,
#     'zero_background': zero_background,
#     'mean_min': [0.2 if f in (1, 2, 3, 4, 5, 6, 7) else 0.0 for f in labels_in],
#     'mean_max': [1.0 if f in (1, 2, 3, 4, 5, 6, 7) else 0.8 for f in labels_in]

# }



from scipy.ndimage import binary_erosion

from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation

# #brain
# model1 = ne.models.labels_to_image_new(
#     in_shape=in_shape,
#     labels_in=labels_in,
#     labels_out={i: i if i in (1, 2, 3, 4, 5, 6, 7) else 0 for i in labels_in},
#     warp_min=0.1,  # Adjust this value for a small warping change
#     warp_max=args.warp_max,  # Adjust this value for a small warping change
#     one_hot=False,
#     aff_rotate=20,
#     aff_shift=args.brain_shift,
#     up_scale=False,
#     aff_scale=args.scale

# )

# #body
# model2 = ne.models.labels_to_image_new(
#     in_shape=in_shape,
#     labels_in=labels_in,
#     labels_out={i: 0 if i in (1, 2, 3, 4, 5, 6, 7) else i for i in labels_in},
#     aff_rotate=5,
#     aff_shear=0.0,
#     blur_max=1,
#     warp_min=0.01,  # Adjust this value for a small warping change
#     warp_max=2,  # Adjust this value for a small warping change
#     slice_prob=1,
#     one_hot=False,
#     crop_prob=1,
#     aff_shift=args.body_shift,
#     aff_scale=args.body_scale

# )

import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_erosion



    



reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, verbose=1, min_lr=1e-7)

weights_saver = PeriodicWeightsSaver(filepath=models_dir, save_freq=20)  # Save weights every 100 epochs

# TB_callback = CustomTensorBoard(
#     base_log_dir=log_dir,
#     histogram_freq=100,
    
#     write_graph=True,
#     write_images=False,
#     write_steps_per_second=False,
#     update_freq='epoch',
#     profile_batch=0,
#     embeddings_freq=0,
#     embeddings_metadata=None
# )

TB_callback = CustomTensorBoard(
    base_log_dir=log_dir,
    histogram_freq=1000,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None
)






if __name__ == "__main__":
    en = args.encoder_layers
    de = args.decoder_layers
    random.seed(3000)
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )

    unet_model = vxm.networks.Unet(inshape=(*in_shape, 1), nb_features=(en, de), 
                                   nb_conv_per_level=nb_conv_per_level, batch_norm=args.batch_norm,
                                   final_activation_function='softmax')
    input_img = Input(shape=(*in_shape,1))
    
    generated_img, y = labels_to_image_model(input_img)
    
    generated_img_norm = min_max_norm(generated_img)
    
    segmentation = unet_model(generated_img)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.add_loss(soft_dice(y, segmentation))
    
    combined_model.compile(optimizer=Adam(learning_rate=initial_lr))

    
    if args.ring:
        brain_maps = add_ring(input_img)

    if args.detection:
        generated_img, y = labels_to_image_model(input_img)

        segmentation = unet_model(generated_img)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        box = fit_bounding_rect_box(y,margin=5)
        if args.center_of_mass:
            print("center of mass")
            combined_model.add_loss(center_of_mass_mse(box, segmentation))
        else:
            combined_model.add_loss(soft_dice(box, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=initial_lr))

        mgh = pathlib.Path('fetus_label_map')
        mgh_files = list(mgh.glob('*.nii.gz'))
        label_maps = [np.uint8(sf.load_volume(str(file_path)).reshape([dimx,dimy,dimz]).data) for file_path in mgh_files]
    
        brain_maps = feta_label_maps

        fov_maps = get_fov(label_maps)
        shapes = [draw_shapes(in_shape, num_labels) for _ in range(num_shapes)]
        shapes = map(np.squeeze, shapes)
        shapes = map(np.uint8, shapes)
        shapes = [f + 7 + 1 for f in shapes]
        shapes = np.concatenate((shapes, fov_maps), axis=0)

        gen = generator(brain_maps, shapes)
        
    elif args.conc:
        print("concentricity shapes!")

        unet_model = vxm.networks.Unet(inshape=(*in_shape, 1), nb_features=(en, de), batch_norm=args.batch_norm,
                               nb_conv_per_level=nb_conv_per_level,
                               final_activation_function='softmax')
        input_img = Input(shape=(*in_shape,1))

        _, fg = model_feta(input_img)
        shapes = draw_shapes_easy(shape = (192,)*3)

        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.uint8)

        # shapes = deform(shapes)
        _, bg = model_shapes(shapes[None,...,None])
        bg = bg + 8
        
        fg_inner = fg[0, ..., 0]
        bg_inner = tf.reshape(bg[0, ..., 0], fg_inner.shape)
        mask = tf.cast(tf.equal(fg_inner, 0), fg_inner.dtype)
        
        result = fg_inner + bg_inner * mask
        
        generated_img , y = labels_to_image_model(result[None,...,None])
        
        if args.ninty_percentile:
            generated_img_norm = percentile_norm_tf(generated_img)
        elif args.norm:
            generated_img_norm = min_max_norm(generated_img)
        else:
            generated_img_norm = generated_img
                
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=initial_lr))
        
        brain_maps = feta_label_maps
        brain_maps = [tf.cast(brain, tf.uint8) for brain in brain_maps]

        gen = generator_brain(brain_maps)

    elif args.shapes:
        brain_maps = get_brain(label_maps)
        shapes = [draw_shapes(in_shape, num_labels) for _ in range(num_shapes)]
        print("shapes:",shapes[0].shape)
        shapes = map(np.squeeze, shapes)
        shapes = map(np.uint8, shapes)
        print("brain_maps:",brain_maps[0])#,shapes[0].shape)
        shapes = [f + 7 + 1 for f in shapes]
        gen = generator(brain_maps, shapes)
    elif args.feta and args.body:
        unet_model = vxm.networks.Unet(inshape=(*in_shape, 1), nb_features=(en, de), 
                       nb_conv_per_level=nb_conv_per_level,
                       final_activation_function='softmax')
        input_img = Input(shape=(*in_shape,1))
        input_brain = get_brain_tf(input_img)
        _, y_brain = model_feta(input_brain)

        input_fov = get_fov_tf(input_img)
        
        _, y_fov = model2(input_fov)
        # input_final = y_brain + y_fov * (y_brain == 0)
        input_final = y_brain + tf.cast(y_fov, dtype=tf.int32) * tf.cast(y_brain == 0, dtype=tf.int32)

        generated_img, y = labels_to_image_model(input_final)
        segmentation = unet_model(generated_img)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=initial_lr))

        fov_maps = get_fov(label_maps)
        gen = gen_brain_feta_fov(feta_label_maps, fov_maps)
    elif args.feta and args.brain and args.body:
        unet_model = vxm.networks.Unet(inshape=(*in_shape, 1), nb_features=(en, de), 
                       nb_conv_per_level=nb_conv_per_level,
                       final_activation_function='softmax')
        input_img = Input(shape=(*in_shape,1))
        input_brain = get_brain_tf(input_img)
        _, y_brain = model1(input_brain)

        input_fov = get_fov_tf(input_img)
        
        _, y_fov = model2(input_fov)
        # input_final = y_brain + y_fov * (y_brain == 0)
        input_final = y_brain + tf.cast(y_fov, dtype=tf.float32) * tf.cast(y_brain == 0, dtype=tf.float32)

        generated_img, y = labels_to_image_model_with_shapes(input_final)
        segmentation = unet_model(generated_img)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=initial_lr))

        brain_maps = get_brain(label_maps)
        fov_maps = get_fov(label_maps)
        brain_maps = feta_label_maps + brain_maps
        gen = gen_brain_feta_fov(brain_maps, fov_maps)
    elif args.brain or args.body:
        unet_model = vxm.networks.Unet(inshape=(*in_shape, 1), nb_features=(en, de), 
               nb_conv_per_level=nb_conv_per_level,
               final_activation_function='softmax')
        
        input_img = Input(shape=(*in_shape,1))
        input_brain = get_brain_tf(input_img)
        _, y_brain = model1(input_brain)

        input_fov = get_fov_tf(input_img)
        
        _, y_fov = model2(input_fov)
        # input_final = y_brain + y_fov * (y_brain == 0)
        input_final = y_brain + tf.cast(y_fov, dtype=tf.int32) * tf.cast(y_brain == 0, dtype=tf.int32)

        generated_img, y = labels_to_image_model(input_final)
        segmentation = unet_model(generated_img)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=initial_lr))

        brain_maps = get_brain(label_maps)
        fov_maps = get_fov(label_maps)
        gen = gen_brain_feta_fov(brain_maps, fov_maps)
    else:
        print("wrong arguments specified!")
    

    callbacks_list = [TB_callback, weights_saver]

    
    # output_label = gen_brain_fov(output_brain, output_fov)

    ################################
    if os.path.exists(checkpoint_path):
        print(checkpoint_path)
        combined_model.load_weights(checkpoint_path)
        print("Loaded weights from the checkpoint and continued training.")

    else:
        print("Checkpoint file not found.")
 
    hist = combined_model.fit(
        gen,
        epochs=num_epochs,  # Set the total number of epochs including previous ones
        initial_epoch=initial_epoch,  # Specify the initial epoch
        verbose=0,
        steps_per_epoch=100,
        callbacks=callbacks_list
    )
