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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.0001, help="learning rate")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-e', '--encoder_layers', nargs='+', type=int, help="A list of dimensions for the encoder")
parser.add_argument('-d', '--decoder_layers', nargs='+', type=int, help="A list of dimensions for the decoder")

args = parser.parse_args()

# nb_features=64
if args.encoder_layers:
    nb_features = '_'.join(map(str, args.encoder_layers))

log_dir='logs_feta_mom_brain_params_'+str(nb_features)+'_nc_'+str(args.nb_conv_per_level)+'_bs_'+str(args.batch_size)
models_dir='models_feta_mom_brain_params_'+str(nb_features)+'_nc_'+str(args.nb_conv_per_level)+'_bs_'+str(args.batch_size)
data_dir = 'feta_2d/'
initial_epoch=args.initial_epoch
checkpoint_path=models_dir+'/weights_epoch_'+str(initial_epoch)+'.h5'

feta = pathlib.Path('/autofs/space/bal_004/users/jd1677/synthstrip/feta_2d')
files = list(feta.glob('sub-???/sub-???_dseg.nii.gz'))
label_maps = [np.uint8(f.dataobj) for f in map(nib.load, files)]
labels = np.unique(label_maps)
in_shape = label_maps[0].shape

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
num_labels = 8
num_shapes = 80

dimx=256
dimy=256



batch_size=args.batch_size
  

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
warp_fwhm_min=40
warp_fwhm_max=80

in_shape=(dimx,dimy)

warp_max=2
warp_min=1
image_fwhm_min=20
image_fwhm_max=40
aff_shift=30
aff_rotate=180
aff_shear=0.1
blur_max=3.4
slice_prob=1
crop_prob=1
bias_min=0.01
bias_max=0.2
zero_background=0.1
aff_scale=0.8
up_scale=False

# aff_shear=0.1

kwargs_shapes = {
'num_label': num_bg_labels,
'nb_labels':num_labels,
'warp_min': warp_min,
'warp_max': warp_max,
'image_fwhm_max':image_fwhm_max,
'image_fwhm_min':image_fwhm_min,
'zero_background':zero_background
}

labels_in = range(max(labels) + num_labels + 1)

gen_arg = {
    'in_shape': in_shape,
    'labels_in': labels_in,
    'labels_out': {f: 1 if f in (1, 2, 3, 4, 5, 6, 7) else 0 for f in labels_in},
    'one_hot':True,
    'axes_flip':True,
    'axes_swap':True,
    'aff_shift':aff_shift,
    'aff_scale':aff_scale,
    'up_scale':False,
    'aff_rotate':aff_rotate,
    'aff_shear':aff_shear,
    'zero_background':zero_background
}
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=30, verbose=1, min_lr=1e-7)

weights_saver = PeriodicWeightsSaver(filepath=models_dir, save_freq=40)  # Save weights every 100 epochs

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

    
# def load_data(data_dir):
    
#     subject_dirs = [d for d in os.listdir(data_dir) if d.startswith('sub-')]
#     seg_image_filenames = []
#     real_image_filenames = []
#     fetal_data = []  # List to store fetal images
#     fetal_segmentation_masks = []  # List to store fetal segmentation masks
    
#     for subject_dir in subject_dirs:
#         subject_prefix = subject_dir
    
#         # Search for the T2-weighted image and segmented brain image with flexible naming patterns
#         t2w_path = None
#         dseg_path = None
#         # print(os.path.join(data_dir, subject_prefix))
#         for filename in os.listdir(os.path.join(data_dir, subject_prefix)):
#             if "_T2w.nii.gz" in filename:
#                 t2w_path = os.path.join(data_dir, subject_prefix, filename)
#             elif "_dseg.nii.gz" in filename:
#                 dseg_path = os.path.join(data_dir, subject_prefix, filename)
    
#         if t2w_path is None or dseg_path is None:
#             print(f"Data not found for subject {subject_prefix}. Skipping...")
#             continue
    
#         # Append the filenames to the respective lists
#         real_image_filenames.append(t2w_path)
#         seg_image_filenames.append(dseg_path)
        
#     for i in range(len(real_image_filenames)):
#         # Load the 2D image
#         img_path = real_image_filenames[i]
#         real_img = nib.load(img_path).get_fdata()

#         seg_img_path = seg_image_filenames[i]
#         seg_img = nib.load(seg_img_path).get_fdata()

#         min_value = np.min(real_img)
#         max_value = np.max(real_img)
#         real_img = (real_img - min_value) / (max_value - min_value)

#         real_img = tf.expand_dims(real_img, axis=0)  # Shape becomes (1, 160, 192, 1)
#         fetal_data.append(real_img)
#         fetal_segmentation_masks.append(seg_img)
#     return fetal_data, fetal_segmentation_masks



# fetal_data, fetal_segmentation_masks = load_data(data_dir)
# X_train, X_test, y_train, y_test = train_test_split(fetal_data, fetal_segmentation_masks, test_size=0.2, random_state=42)






if __name__ == "__main__":
    en = args.encoder_layers
    de= args.decoder_layers
    random.seed(3000)
    # input_img = Input(shape=(dimx, dimy,1))
    # labels_to_image_model = ne.models.labels_to_image_new(**gen_arg)#,input_model=labels_to_labels_model)
    
    
    unet_model = vxm.networks.Unet(inshape=(dimx, dimy, 1), nb_features=[en, de], 
                                   nb_conv_per_level=nb_conv_per_level,
                                   final_activation_function='softmax')
    
    labels_to_image_model = ne.models.labels_to_image_new(**gen_arg)
    
    input_img = Input(shape=(dimx, dimy,1))
    generated_img, y = labels_to_image_model(input_img)    
    segmentation = unet_model(generated_img)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.add_loss(soft_dice(y, segmentation))
    combined_model.compile(optimizer=Adam(learning_rate=initial_lr))
    ################################
    num_labels = 8
    num_shapes = 80
    shapes = [draw_shapes(in_shape, num_labels) for _ in range(num_shapes)]
    shapes = map(np.squeeze, shapes)
    shapes = map(np.uint8, shapes)
    shapes = [f + max(labels) + 1 for f in shapes]
    gen = generator(label_maps, shapes)
    ################################

    # data_generator = my_generator(y_train, batch_size=batch_size, same_subj=False, flip=False,**kwargs_shapes)

    if os.path.exists(checkpoint_path):
        combined_model.load_weights(checkpoint_path)
        print("Loaded weights from the checkpoint and continued training.")
    else:
        print("Checkpoint file not found.")
    # combined_model.sum
    # print("Metrics being used:", combined_model.summary)

    hist = combined_model.fit(
        gen,
        epochs=num_epochs,  # Set the total number of epochs including previous ones
        initial_epoch=initial_epoch,  # Specify the initial epoch
        verbose=0,
        steps_per_epoch=100,
        callbacks=[weights_saver, TB_callback]#,reduce_lr]
    )
