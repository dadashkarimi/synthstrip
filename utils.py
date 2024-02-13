import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import neurite as ne
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full
import voxelmorph as vxm
import os
import glob
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation


def get_brain(a):
    a_copy = np.copy(a)
    for i in range(len(a)):
        a_copy[i][a[i] >7 ] = 0
    return a_copy

def get_fov(a, max_shift=90):
    a_copy = np.copy(a)
    
    for i in range(len(a)):
        b = a_copy[i]
        m = (b >= 1) & (b < 8)
        m = binary_dilation((m > 0), structure=np.ones((2, 2, 2)))

        max_sum = 0
        best_shift = 0
        best_direction = 0

        for d in range(3):
            for shift in range(-max_shift, max_shift + 1):
                shifted_m = np.roll(m, shift, axis=d)
                current_sum = np.sum(b[shifted_m])

                if current_sum > max_sum:
                    max_sum = current_sum
                    best_shift = shift
                    best_direction = d

        shifted_m = np.roll(m, best_shift, axis=best_direction)
        b[m] = 0
        b[m] = a_copy[i][shifted_m]
        
        a_copy[i] = b

    return a_copy
    


def add_ring(label_maps):
    for i in range(len(label_maps)):
        label_maps[i] = segment_brain_with_ring(label_maps[i])
    return label_maps

def segment_brain_with_ring(image):
    brain_mask = (image > 0).astype(np.uint8)
    expanded_ring = binary_dilation(brain_mask.astype(bool),structure=np.ones((5,5, 5)))
    ring_mask = expanded_ring & ~(brain_mask.astype(bool))
    image[ring_mask] = 8  # You can use any non-zero label for the ring
    image[(brain_mask == 1) & (image == 0)] = 1
    return image
    
def draw_shapes(
    shape,
    num_label=16,
    warp_min=1,
    warp_max=20,
    dtype=None,
    seed=None,
    image_fwhm_min=20,
    image_fwhm_max=40,
    warp_fwhm_min=40,
    warp_fwhm_max=80,
):
    # Data types.
    type_fp = tf.float16
    type_int = tf.int32
    if dtype is None:
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dtype = tf.dtypes.as_dtype(dtype)
    
    # Randomization.
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(np.int32).max, dtype=np.int32)
    prop = lambda: dict(isotropic=False, batched=False, featured=True, seed=seed(), dtype=type_fp, reduce=tf.math.reduce_max)
    
    # Images and transforms.
    v = ne.utils.augment.draw_perlin_full(
        shape=(*shape, 1),
        fwhm_min=image_fwhm_min, fwhm_max=image_fwhm_max, **prop(),
    )
    
    t = ne.utils.augment.draw_perlin_full(
        shape=(*shape, len(shape)), noise_min=warp_min, noise_max=warp_max,
        fwhm_min=warp_fwhm_min, fwhm_max=warp_fwhm_max, **prop(),
    )
    
    # Application and background.
    v = ne.utils.minmax_norm(v)
    v = vxm.utils.transform(v, t, fill_value=0)
    v = tf.math.ceil(v * (num_label - 1))

    return tf.cast(v, dtype) if v.dtype != dtype else v
    
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
        if epoch % 100 == 0:  # Check if it's the start of a new set of 50 epochs
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
            weights_path = os.path.join(self.filepath, f"weights_epoch_{epoch + 1}.h5")
            self.model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")

def crop_img(label_maps,dimx,dimy,dimz):
    cropped_images = []
    crop_size = (dimx,dimy,dimz)
    for img in label_maps:
        img_shape = img.shape
        start_indices = [(img_shape[i] - crop_size[i]) // 2 for i in range(3)]
        end_indices = [start_indices[i] + crop_size[i] for i in range(3)]
        cropped_img = img[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
        cropped_images.append(cropped_img)
    return cropped_images
    
def load_npz_files(folder_path,dimx,dimy,dimz):
    npz_files = glob.glob(folder_path + '/*.npz')
    data = []

    for file in npz_files:
        loaded_data = np.load(file)
        for key in loaded_data.files:
            image = loaded_data[key]
            
            orig_shape = image.shape
            if orig_shape[0] > dimx or orig_shape[1] > dimy or orig_shape[2] > dimz:
                crop_x = min(orig_shape[0], dimx)
                crop_y = min(orig_shape[1], dimy)
                crop_z = min(orig_shape[2], dimz)
                image = image[:crop_x, :crop_y, :crop_z]

            new_shape = (dimx, dimy, dimz)
            padded_image = np.zeros(new_shape, dtype=image.dtype)

            pad_x = max(0, (dimx - image.shape[0]) // 2)
            pad_y = max(0, (dimy - image.shape[1]) // 2)
            pad_z = max(0, (dimz - image.shape[2]) // 2)
            
            end_x = min(pad_x + image.shape[0], dimx)
            end_y = min(pad_y + image.shape[1], dimy)
            end_z = min(pad_z + image.shape[2], dimz)
            
            padded_image[pad_x:end_x, pad_y:end_y, pad_z:end_z] = image[:end_x - pad_x, :end_y - pad_y, :end_z - pad_z]
            
            data.append(padded_image)

    return data
    
def soft_dice(a, b):
    dim = len(a.shape) - 2
    space = list(range(1, dim + 1))

    top = 2 * tf.reduce_sum(a * b, axis=space)
    bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)
    
    out = tf.divide(top, bot + 1e-6)
    return -tf.reduce_mean(out)

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

def synth_generator(label_maps, batch_size=1, same_subj=False, flip=False,**kwargs_shapes):
    in_shape = label_maps[0].shape
    
    num_dim = len(in_shape)
    dimx=in_shape[0]
    dimy=in_shape[1]
    dimz=in_shape[2]
    print(dimx,dimy,dimz)
    void = np.zeros((batch_size, *in_shape), dtype='float32')
    rand = np.random.default_rng()
    prop = dict(replace=False, shuffle=False)
    num_batches = len(label_maps) // batch_size

    # offset = kwargs_shapes.get('nb_labels',None)
    nb_bg_labels = kwargs_shapes.get('num_label',None)
    # print("ddd",offset)
    while True:
        # gen_model = nothing_to_shapes((dimx, dimy),**kwargs_shapes)
        # shapes = gen_model.predict(np.ones((1,)))
        ind = rand.integers(len(label_maps), size=2 * batch_size)
        x = [label_maps[i] for i in ind]
        if same_subj:
            x = x[:batch_size] * 2
        x = np.stack(x)[..., None]

        if flip:
            axes = rand.choice(num_dim, size=rand.integers(num_dim + 1), **prop)
            x = np.flip(x, axis=axes + 1)

        gg = x[:batch_size, ...,0]
        y = np.array(void)
        yield gg, y
        
def generator(label_maps, shapes):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    shapes = np.asarray(shapes)
    
    while True:
        fg = rand.choice(label_maps)
        bg = rand.choice(shapes)
        out = fg + bg * (fg == 0)
        yield out[None, ..., None]

def generator3D_noshape(label_maps):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)
        yield fg[None, ..., None]

def generatorFOV(brain_maps, fov_maps):
    rand = np.random.default_rng()
    brain_maps = np.asarray(brain_maps)
    fov_maps = np.asarray(fov_maps)
    
    while True:
        idx = rand.choice(len(brain_maps))
        brain_label = brain_maps[idx]
        body_label = fov_maps[idx]
        yield brain_label[None, ..., None], body_label[None, ..., None]

def generator_feta_FOV(brain_maps, fov_maps):
    rand = np.random.default_rng()
    brain_maps_ = brain_maps.copy()
    fov_maps_ = fov_maps.copy()
    
    brain_maps_ = np.asarray(brain_maps_)
    fov_maps = np.asarray(fov_maps_)
    
    while True:
        # idx1 = rand.choice(len(brain_maps_))
        # idx2 = rand.choice(len(fov_maps_))
        # brain_label = brain_maps_[idx1]
        # body_label = fov_maps_[idx2]
        # yield brain_label[None, ..., None], body_label[None, ..., None]
        #         idx1 = rand.choice(len(brain_maps_))
        # idx2 = rand.choice(len(fov_maps_))
        brain_label = rand.choice(brain_maps_)
        body_label = rand.choice(fov_maps)
        yield brain_label[None, ..., None], body_label[None, ..., None]
        
def generator3D(label_maps, shapes, zero_background):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    shapes = np.asarray(shapes)
    
    while True:
        fg = rand.choice(label_maps)
        
        # Determine whether to add background shapes
        if rand.random() > zero_background:
            bg = rand.choice(shapes)
            fg = fg + bg * (fg == 0)
        
        yield fg[None, ..., None]

# def gen_brain_fov_non_flip(brain_maps, fov_maps, model1, model2):

#     gen1 = generatorFOV(brain_maps, fov_maps)
    
#     input_brain, input_fov = next(gen1)
    
#     _ , output_brain = model1(input_brain)
#     _ , output_fov = model2(input_fov)
    
#     output_brain = np.squeeze(output_brain, axis=(0, -1))
#     output_fov = np.squeeze(output_fov, axis=(0, -1))


#     output_brain = output_brain.copy()
#     output_fov = output_fov.copy()
#     # shrunken_boundaries = binary_erosion((output_brain > 0), structure=np.ones((2, 2, 2)))
#     # difference_region1 = np.logical_xor(output_brain > 0, shrunken_boundaries)
    

#     r2 = np.random.randint(2, 4) 
#     expanded_ring = binary_dilation((output_brain > 0), structure=np.ones((r2, r2, r2)))

#     difference_region2 = np.logical_xor(output_brain > 0, expanded_ring)
    
#     # output_fov[output_brain > 0] = 1
#     # output_label[difference_region1]=7
#     output_fov[difference_region2] = 8

#     output_label = output_brain + output_fov * (output_brain == 0) 

#     # output_label = output_brain + output_fov * (output_brain == 0) 

#     output_label = np.float32(output_label + output_fov * (output_label == 0))
#     output_label = np.expand_dims(output_label, axis=(0, -1))
    
#     return output_label
def create_model(model_config):
    model_config_ = model_config.copy()
    return ne.models.labels_to_image_new(**model_config_)

def gen_brain_feta_fov(brain_maps, fov_maps, model1, model2):

    rand = np.random.default_rng()
    brain_maps_ = np.asarray(brain_maps)
    fov_maps_ = np.asarray(fov_maps)
    
    while True:
        brain_label = rand.choice(brain_maps_)
        body_label = rand.choice(fov_maps_)
        input_brain = brain_label[None, ..., None]
        input_fov = body_label[None, ..., None]
        _ , output_brain = model1(input_brain)
        _ , output_fov = model2(input_fov)
        output_brain = np.array(output_brain)
        output_fov = np.array(output_fov)
        output_label = output_brain + output_fov * (output_brain == 0) 
        yield output_label[None, ..., None]


def gen_brain_feta_fov2(brain_maps, fov_maps, model1, model2):

    gen1 = generator_feta_FOV(brain_maps, fov_maps)
    
    input_brain, input_fov = next(gen1)
    
    _ , output_brain = model1(input_brain)
    _ , output_fov = model2(input_fov)
    # print("shapes:",output_brain.shape,output_fov.shape)



    output_brain = output_brain.copy()
    output_fov = output_fov.copy()

    output_label = output_brain + output_fov * (output_brain == 0) 
    return output_label[None, ..., None]
    
# def gen_brain_fov_with_flip(brain_maps, fov_maps, model1, model2):

#     gen1 = generatorFOV(brain_maps, fov_maps)
    
#     input_brain, input_fov = next(gen1)
    
#     _ , output_brain = model1(input_brain)
#     _ , output_fov = model2(input_fov)
    
#     output_brain = np.squeeze(output_brain, axis=(0, -1))
#     output_fov = np.squeeze(output_fov, axis=(0, -1))


#     output_brain = output_brain.copy()
#     output_fov = output_fov.copy()

#     flip_axes = [0, 1, 2]

#     intersection_original = np.sum((output_brain > 0) & (output_fov > 0), axis=(0, 1, 2))
#     intersections_flipped = []
    
#     for axis in flip_axes:
#         output_brain_flipped = np.flip(output_brain, axis=axis)
#         intersection_flipped = np.sum((output_brain_flipped > 0) & (output_fov > 0), axis=(0, 1, 2))
#         intersections_flipped.append(intersection_flipped)
    
#     max_intersection_axis = flip_axes[np.argmax(intersections_flipped)]

#     output_brain = np.flip(output_brain, axis=max_intersection_axis)
    
#     r1 = np.random.randint(1, 3) 
#     shrunken_boundaries = binary_erosion((output_brain > 0), structure=np.ones((r1, r1, r1)))
#     difference_region1 = np.logical_xor(output_brain > 0, shrunken_boundaries)

#     r2 = np.random.randint(1, 4) 
#     expanded_ring = binary_dilation((output_brain > 0), structure=np.ones((r2, r2, r2)))

#     difference_region2 = np.logical_xor(output_brain > 0, expanded_ring)
    
#     # output_brain[difference_region1]=7
#     output_fov[difference_region2] = 8
    
#     output_label = output_brain + output_fov * (output_brain == 0)

#     output_label = np.expand_dims(output_label, axis=(0, -1))
    
#     return output_label
    
def resize(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of 
        length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    If you find this function useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims

    # Avoid unnecessary work.
    if all(z == 1 for z in zoom_factor):
        return vol

    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [tf.linspace(0., vol_shape[d] - 1., new_shape[d]) for d in range(ndims)]
    grid = ne.utils.ndgrid(*lin)

    return ne.utils.interpn(vol, grid, interp_method=interp_method)



def my_generator(label_maps, batch_size=1, same_subj=False, flip=False,**kwargs_shapes):
    in_shape = label_maps[0].shape
    
    num_dim = len(in_shape)
    dimx=in_shape[0]
    dimy=in_shape[1]
    print(dimx,dimy)
    void = np.zeros((batch_size, *in_shape), dtype='float32')
    rand = np.random.default_rng()
    prop = dict(replace=False, shuffle=False)
    num_batches = len(label_maps) // batch_size

    offset = kwargs_shapes.get('nb_labels',None)
    nb_bg_labels = kwargs_shapes.get('num_label',None)
    # print("ddd",offset)
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
        # vv=np.unique(shapes).astype(int)
        # print(shapes.shape,vv,offset,nb_bg_labels)
        
        shapes = tf.add(shapes,offset)
        shapes=tf.cast(shapes,dtype=tf.int64)
        # print(offset)
        # print(np.unique(shapes))
        fg = tf.cast(x > 0, x.dtype)
        zero_background = kwargs_shapes.get('zero_background', None)
        bg_rand = tf.random.uniform(())
        bg_zero_background = tf.cast(bg_rand < zero_background, x.dtype)
        bg_zero_background_number = tf.cast(bg_zero_background, x.dtype)
        bg_shapes = tf.cast(shapes, x.dtype) * (1 - fg) * (1 - bg_zero_background_number)  # Cast shapes to match x.dtype

        # bg_shapes = shapes * (1 - fg) * (1 - bg_zero_background_number)
        combined = x * fg + bg_shapes
        # fg = x > 0
        # zero_background = kwargs_shapes.get('zero_background', None)
        # bg_rand = np.random.uniform()
        # bg_zero_background = bg_rand < zero_background
        # bg_zero_background_number = np.int64(bg_zero_background)
        # bg_shapes = shapes * (1 - fg)* (1 - bg_zero_background_number)
        # combined = x * fg + bg_shapes
        gg = combined[:batch_size, ...,0]
        y = np.array(void)
        yield gg, y
