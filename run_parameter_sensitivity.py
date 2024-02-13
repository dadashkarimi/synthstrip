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
# from neurite.tf.utils.augment import labels_to_labels

# import ne.keras as nke
import tensorflow.keras.layers as KL
import voxelmorph as vxm

tf.keras.utils.disable_interactive_logging()

tf.get_logger().setLevel('ERROR')
log_dir='logs_feta_mom_brain_params2'
models_dir='models_feta_mom_brain_params2'
data_dir = 'feta_2d/'
initial_epoch=36100
checkpoint_path=models_dir+'/weights_epoch_'+str(initial_epoch)+'.h5'
print(checkpoint_path)
nb_labels=8
dimx=256
dimy=256


nb_features=64
batch_size=2
warp_max=2.5
warp_min=.5
warp_min_shapes=0.5
warp_max_shapes=3
warp_blur_min=np.array([2, 4, 8])
warp_blur_max=warp_blur_min*2
bias_blur_min=np.array([2, 4, 8])
bias_blur_max=bias_blur_min*2
bias_min=0.01
bias_max=0.3
aff_scale=0.1
aff_shear=0.1
aff_shift=32

initial_lr=1e-4
lr = 1e-4
lr_lin = 1e-4
nb_levels=5
conv_size=3
num_epochs=50000
num_bg_labels=16
warp_fwhm_min=10
warp_fwhm_max=20
in_shape=(dimx,dimy)
bg_brain = True
labels_in=[i for i in range(num_bg_labels+nb_labels)]
labels_in_unique = np.unique(labels_in).astype(int)


kwargs_shapes = {
'num_label': num_bg_labels,
'warp_min': warp_min_shapes,
'warp_max': warp_max_shapes,
'image_fwhm_max':image_fwhm_max,
'image_fwhm_min':image_fwhm_min
}

gen_arg = {
    'in_shape': in_shape,
    'labels_in': labels_in,
    'labels_out': {i: 1 if i > 0 and i < nb_labels else 0 for i in range(num_bg_labels+nb_labels)},  
    'warp_min': warp_min,
    'one_hot':True,
    'axes_flip':True,
    'axes_swap':True,
    'warp_max': warp_max,
    'aff_shift':aff_shift,
    'aff_rotate':180,
    'aff_scale':aff_scale,
    'aff_shear':aff_shear,
    'bias_min':bias_min,
    'bias_max':bias_max,
    'bias_func':tf.exp,
    'zero_background': 0.1
}


def draw_shapes(
    shape, num_label=16, warp_min=1, warp_max=20, dtype=None, seed=None,
    image_fwhm_min=20, image_fwhm_max=40, warp_fwhm_min=40, warp_fwhm_max=80,
):
    # Data types.
    # shape=(dimx,dimy)
    type_fp = tf.float16
    type_int = tf.int32
    if dtype is None:
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dtype = tf.dtypes.as_dtype(dtype)

    # Randomization.
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(np.int32).max, dtype=np.int32)
    prop = lambda: dict(isotropic=False, batched=False, featured=True, seed=seed(), dtype=type_fp, reduce=tf.math.reduce_max)
    # print(shape)
    # Images and transforms.
    v = draw_perlin_full(
        shape=(*shape, 1),
        fwhm_min=image_fwhm_min, fwhm_max=image_fwhm_max, **prop(),
    )
    t = draw_perlin_full(
        shape=(*shape, len(shape)), noise_min=warp_min, noise_max=warp_max,
        fwhm_min=warp_fwhm_min, fwhm_max=warp_fwhm_max, **prop(),
    )

    # Application and background.
    v = ne.utils.minmax_norm(v)
    v = vxm.utils.transform(v, t, fill_value=-1)
    v = tf.floor(v * (num_label - 1))
    fg = tf.greater_equal(v, 0)
    out = (v + 1) * tf.cast(fg, v.dtype)

    return tf.cast(out, dtype) if out.dtype != dtype else out

def nothing_to_shapes(out_shape, **kwargs):
    nothing = Input(shape=[])
    generate = lambda _: draw_shapes(out_shape, **kwargs)
    return tf.keras.Model(
        inputs=nothing,
        outputs=Lambda(lambda x: tf.map_fn(generate, x))(nothing),
    )
    
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, base_log_dir, **kwargs):
        super(CustomTensorBoard, self).__init__(**kwargs)
        self.base_log_dir = base_log_dir

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 200 == 0:  # Check if it's the start of a new set of 50 epochs
            self.log_dir = f"{self.base_log_dir}/epoch_{epoch}"
            super().set_model(self.model)




class PeriodicWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=200, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        # Save the weights every `save_freq` epochs
        if (epoch + 1) % self.save_freq == 0:
            # sys.stdout.close()
            # sys.stdout = old_stdout
            
            weights_path = os.path.join(self.filepath, f"weights_epoch_{epoch + 1}.h5")
            self.model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")
        else:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            # sys.stdout.close()
            sys.stdout = self.old_stdout


weights_saver = PeriodicWeightsSaver(filepath=models_dir, save_freq=50)  # Save weights every 5 epochs

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


def dice_loss(y_true, y_pred):
    # y_pred = tf.argmax(y_pred, axis=-1)
    # y_true = tf.squeeze(y_true,-1)
    print("dice",y_true.shape,y_pred.shape)
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))
    # print(y_true.shape,y_pred.shape)
    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

    div_no_nan = tf.math.divide_no_nan if hasattr(
        tf.math, 'divide_no_nan') else tf.div_no_nan  # pylint: disable=no-member
    dice = tf.reduce_mean(div_no_nan(top, bottom))
    return -dice
    
def dice_coefficient(y_true, y_pred):
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims + 1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

    div_no_nan = tf.math.divide_no_nan if hasattr(
    tf.math, 'divide_no_nan') else tf.div_no_nan  # pylint: disable=no-member
    dice = tf.reduce_mean(div_no_nan(top, bottom))
    return dice
    
def load_data(data_dir):
    
    subject_dirs = [d for d in os.listdir(data_dir) if d.startswith('sub-')]
    seg_image_filenames = []
    real_image_filenames = []
    fetal_data = []  # List to store fetal images
    fetal_segmentation_masks = []  # List to store fetal segmentation masks
    
    for subject_dir in subject_dirs:
        subject_prefix = subject_dir
    
        # Search for the T2-weighted image and segmented brain image with flexible naming patterns
        t2w_path = None
        dseg_path = None
        # print(os.path.join(data_dir, subject_prefix))
        for filename in os.listdir(os.path.join(data_dir, subject_prefix)):
            if "_T2w.nii.gz" in filename:
                t2w_path = os.path.join(data_dir, subject_prefix, filename)
            elif "_dseg.nii.gz" in filename:
                dseg_path = os.path.join(data_dir, subject_prefix, filename)
    
        if t2w_path is None or dseg_path is None:
            print(f"Data not found for subject {subject_prefix}. Skipping...")
            continue
    
        # Append the filenames to the respective lists
        real_image_filenames.append(t2w_path)
        seg_image_filenames.append(dseg_path)
        
    for i in range(len(real_image_filenames)):
        # Load the 2D image
        img_path = real_image_filenames[i]
        real_img = nib.load(img_path).get_fdata()

        seg_img_path = seg_image_filenames[i]
        seg_img = nib.load(seg_img_path).get_fdata()

        min_value = np.min(real_img)
        max_value = np.max(real_img)
        real_img = (real_img - min_value) / (max_value - min_value)

        real_img = tf.expand_dims(real_img, axis=0)  # Shape becomes (1, 160, 192, 1)
        fetal_data.append(real_img)
        fetal_segmentation_masks.append(seg_img)
    return fetal_data, fetal_segmentation_masks

fetal_data, fetal_segmentation_masks = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(fetal_data, fetal_segmentation_masks, test_size=0.2, random_state=42)

input_img = Input(shape=(dimx, dimy,1))

labels_to_image_model = ne.models.labels_to_image_new(**gen_arg)#,input_model=labels_to_labels_model)
# max_label=20

temp_im = []
temp_lm = []

def my_generator(label_maps, batch_size=batch_size, same_subj=False, flip=False):
    in_shape = label_maps[0].shape
    
    num_dim = len(in_shape)
    void = np.zeros((batch_size, *in_shape), dtype='float32')
    rand = np.random.default_rng()
    prop = dict(replace=False, shuffle=False)
    num_batches = len(label_maps) // batch_size

    while True:
        gen_model = nothing_to_shapes((dimx, dimy),**kwargs_shapes)
        shapes = gen_model.predict(np.ones((1,)))
        ind = rand.integers(len(label_maps), size=2 * batch_size)
        x = [label_maps[i] for i in ind]
        if same_subj:
            x = x[:batch_size] * 2
        x = np.stack(x)[..., None]

        if flip:
            axes = rand.choice(num_dim, size=rand.integers(num_dim + 1), **prop)
            x = np.flip(x, axis=axes + 1)

        offset = nb_labels
        shapes += offset
        
        fg = x > 0
        zero_background = gen_arg['zero_background']
        bg_rand = np.random.uniform()
        bg_zero_background = bg_rand < zero_background
        bg_zero_background_number = np.int64(bg_zero_background)
        bg_shapes = shapes * (1 - fg)* (1 - bg_zero_background_number)
        combined = x * fg + bg_shapes
        x = combined[:batch_size, ...,0]

        y = np.array(void)
        yield x, y

