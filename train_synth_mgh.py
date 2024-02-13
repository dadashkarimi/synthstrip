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
import os
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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.0001, help="learning rate")
parser.add_argument('-zb','--zero_background',type=float, default=0.1, help="zero background")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-sc','--scale',type=float,default=0.2,help="scale")
parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-m','--num_dims',default=256,type=int,help="number of dims")
parser.add_argument('-e', '--encoder_layers', nargs='+', type=int, help="A list of dimensions for the encoder")
parser.add_argument('-d', '--decoder_layers', nargs='+', type=int, help="A list of dimensions for the decoder")
parser.add_argument('-mgh', '--mgh_label_maps', action='store_true', default=False, help="A list of dimensions for the decoder")


args = parser.parse_args()
# dimx=192
# dimy=192
# dimz=192
dimx=args.num_dims
dimy=args.num_dims
dimz=args.num_dims

# nb_features=64
if args.encoder_layers:
    nb_features = '_'.join(map(str, args.encoder_layers))

log_dir='logs_synth_FOV_brain_params_dim_'+str(dimx)+'_'+str(nb_features)+'_nc_'+str(args.nb_conv_per_level)+'_bs_'+str(args.batch_size)+'_sc_'+str(args.scale)+'_zb_'+str(args.zero_background)+'_mgh_'+str(args.mgh_label_maps)
models_dir='models_synth_FOV_brain_params_dim_'+str(dimx)+'_'+str(nb_features)+'_nc_'+str(args.nb_conv_per_level)+'_bs_'+str(args.batch_size)+'_sc_'+str(args.scale)+'_zb_'+str(args.zero_background)+'_mgh_'+str(args.mgh_label_maps)
if not args.mgh_label_maps:
    print("using Feta label maps")
    feta = pathlib.Path('/autofs/space/bal_004/users/jd1677/synthstrip/feta_3d')
    files = list(feta.glob('sub-???/anat/sub-???_rec-mial_dseg.nii.gz'))
else:
    print("using MGH label maps")
    mgh = pathlib.Path('fetus_label_map')
    files = list(mgh.glob('*.nii.gz'))

label_maps = [np.uint8(f.dataobj) for f in map(nib.load, files)]


initial_epoch=args.initial_epoch
checkpoint_path=models_dir+'/weights_epoch_'+str(initial_epoch)+'.h5'
# label_maps = [np.uint8(sf.load_volume(str(file_path)).reshape((dimx, dimy, dimz)).data) for file_path in files]
# label_maps = [np.uint8(sf.load_volume(str(file_path)).data) for file_path in mgh_files]

# label_maps = crop_img(label_maps,dimx,dimy,dimz)
labels = np.unique(label_maps)
in_shape = label_maps[0].shape

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    

    
num_labels = 8
num_shapes = 80

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
num_bg_labels=16
warp_fwhm_min=10
warp_fwhm_max=20
warp_min_shapes=10
warp_max_shapes=50
in_shape=(dimx,dimy,dimz)
bg_brain = True

warp_max=2
warp_min=1
image_fwhm_min=20
image_fwhm_max=40
aff_shift=5
aff_rotate=0
aff_shear=0.0
blur_max=2.4
slice_prob=1
crop_prob=1
bias_min=0.01
bias_max=0.2
zero_background=args.zero_background
aff_scale=args.scale
up_scale=False

labels_in = range(max(labels) + num_labels + 1)

gen_arg = {
    'in_shape': in_shape,
    'labels_in': labels_in,
    'labels_out': {f: 1 if f in (1, 2, 3, 4, 5, 6, 7) else 0 for f in labels_in},
    'one_hot':True,
    'aff_shift':aff_shift,
    'aff_scale':aff_scale,
    'up_scale':False,
    'aff_rotate':aff_rotate,
    'aff_shear':aff_shear,
    'warp_min':warp_min,
    'warp_max':warp_max,
    'bias_min':bias_min,
    'bias_max':bias_max,
    'zero_background':zero_background
}

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, verbose=1, min_lr=1e-7)

weights_saver = PeriodicWeightsSaver(filepath=models_dir, save_freq=10)  # Save weights every 100 epochs

TB_callback = CustomTensorBoard(
    base_log_dir=log_dir,
    histogram_freq=0,
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
    de= args.decoder_layers
    random.seed(3000)
    
    unet_model = vxm.networks.Unet(inshape=(*in_shape, 1), nb_features=(en, de), 
                                   nb_conv_per_level=nb_conv_per_level,
                                   final_activation_function='softmax')
    
    labels_to_image_model = ne.models.labels_to_image_new(**gen_arg)
    
    input_img = Input(shape=(*in_shape,1))
    generated_img, y = labels_to_image_model(input_img)
    # print(generated_img.shape,y.shape)
    
    segmentation = unet_model(generated_img)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.add_loss(soft_dice(y, segmentation))
    
    combined_model.compile(optimizer=Adam(learning_rate=initial_lr))
    # data_generator = synth_generator(y_train, batch_size=batch_size, same_subj=False, flip=False,**kwargs_shapes)
    ################################
    shapes = [draw_shapes(in_shape, num_labels) for _ in range(num_shapes)]
    shapes = map(np.squeeze, shapes)
    shapes = map(np.uint8, shapes)
    shapes = [f + max(labels) + 1 for f in shapes]
    gen = generator3D(label_maps, shapes, zero_background)
    # gen = generator3D_noshape(label_maps)
    ################################
    if os.path.exists(checkpoint_path):
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
        callbacks=[weights_saver, TB_callback])
