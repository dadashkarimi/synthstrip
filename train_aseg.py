
import socket, os
import numpy as np
from tensorflow.keras import layers as KL
from tqdm import tqdm
import glob, copy
import scipy
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K

from freesurfer import deeplearn as fsd
import freesurfer as fs

import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import surfa as sf

import layer_dict as ld
import pdb as gdb

vscale = 2
vscale = 1

dofit = False
dofit = True

which_loss = 'both'
which_loss = 'mse'
which_loss = 'cce'
which_loss = 'dice'

same_contrast=False
same_contrast=True

oshapes = False
oshapes = True

fit_lin = True
fit_lin = False

save_model = False
save_model = True

doaff = False
doaff = True

test_adni = False


model_dir = 'models'
gpuid = -1
host = socket.gethostname()
from neurite_sandbox.tf.utils.utils import plot_fit_callback as pfc


print(f'host name {socket.gethostname()}')

ngpus = 1 if os.getenv('NGPUS') is None else int(os.getenv('NGPUS'))

print(f'using {ngpus} gpus')
if ngpus > 1:
    model_device = '/gpu:0'
    synth_device = '/gpu:1'
    synth_gpu = 1
    dev_str = "0, 1"
else:
    model_device = '/gpu:0'
    synth_device = model_device
    synth_gpu = 0
    dev_str = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = dev_str

print(f'model_device {model_device}, synth_device {synth_device}, dev_str {dev_str}')

print(f'physical GPU # is {os.getenv("SLURM_STEP_GPUS")}')
ret = ne.utils.setup_device(dev_str)
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')


print(f'dofit {dofit}, doaff {doaff}, fit_lin {fit_lin}, oshapes {oshapes}, save_model {save_model}')


batch_size = 1

adir = '/autofs/cluster/freesurfer/subjects/atlases/aseg_atlas'
mname = 'seg_edited.mgz'
vname = 'norm.mgz'
sfile = os.path.join(adir, 'scripts', 'subjects.txt')
with open(sfile, 'r') as f:
    man_subjects = f.read().split('\n')[0:-1]


odir = '/autofs/vast/braindev/braindev/OASIS/OASIS1/synth-high-res/recon_subject'
subjects = [f for f in Path(odir).iterdir() if 'OASIS_OAS1_0' in str(f)]
seg_files = [f/f'mri/aparc+aseg.mgz' for f in tqdm(subjects)]
seg_files = [f/f'mri/aseg.mgz' for f in tqdm(subjects)]

if dofit:
    print(f'TRAINING model with loss {which_loss}')
else:
    print(f'loading model trained with {which_loss} loss')

target_shape = (192,)*3
inshape = target_shape

def minmax_norm(mri, axis=None):
    """
    Min-max normalize an mri struct using a safe division.

    Arguments:
        x: np.array to be normalized
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized tensor.
    """
    x = mri.data
    x_min = x.min()
    x_max = x.max()
    mri.data = (x - x_min) / (x_max - x_min) 
    return mri


ntest = 25
crop = -1 if dofit else ntest

lut = fs.lookups.default()
# lut = fs.lookups.LookupTable.read(os.path.join(odir, 'seg35_labels.txt'))
# lut = fs.lookups.LookupTable.read(os.path.join(odir, f'seg{nb_labels}_labels.txt'))

# csf = lut.search('CSF')[0]
lesion_label_orig = lut.search('Left-Lesion')
if len(lesion_label_orig) > 0:
    lesion_label_orig = lesion_label_orig[0]
else:   # not in the lut - add a new one
    lesion_label_orig = 77
    lut.add(lesion_label_orig, 'Lesion', color=[240,240,240])

if 'inited' not in locals() and 'inited' not in globals():
    inited = False

if not inited:
    mri_segs_orig = [fs.Volume.read(str(fname)) for fname in tqdm(seg_files[:crop])]
    # mri_segs = [sf.load_volume(str(fname)) for fname in tqdm(seg_files[:crop])]
    if vscale > 1:
        print(f'downsampling by {vscale}')
        mri_segs = [mri.reslice(vscale) for mri in tqdm(mri_segs_orig)]
    else:
        print(f'cropping to {target_shape}')
        mri_segs = [mri.fit_to_shape(target_shape, center='bbox') for mri in tqdm(mri_segs_orig)]
        # mri_segs = [mri.reshape(target_shape, center='bbox') for mri in tqdm(mri_segs)]

    np_segs_orig = [mri.data for mri in mri_segs]
    print(f'finding unique labels in {len(mri_segs_orig)} datasets...')
    labels_orig = np.unique(np.array(np_segs_orig))

    # mapping = fs.lookups.tissue_type_recoder_no_skull(include_lesions=use_lesions)
    mapping = fs.lookups.nonlateral_aseg_recoder()
    target_lut = mapping.target_lut
    lut_name = 'nonlat.txt'
    mri_segs_recoded = [fs.label.recode(mri, mapping) for mri in mri_segs]  
    lesion_label = target_lut.search('Left-Lesion')[0]
    np_segs = [vol.data for vol in mri_segs_recoded]
    labels_in = np.unique(np.array(np_segs)).astype(int)
    if lesion_label not in labels_in:
        l = list(labels_in)
        l.append(lesion_label)
        labels_in = np.array(l)
    nlabels_small = len(labels_in)
    label_map = {}
    keys = mapping.mapping.keys()
    lab_to_ind = np.zeros((labels_orig.max()+1,), dtype=int)
    for label in labels_orig:
        if label not in keys:
            output_label = 0
        else:
            output_label = mapping.mapping[label]
        label_map[label] = output_label
        lab_to_ind[label] = output_label

    mri_man_segs = []  # manual segs
    mri_norms = []  # mri vols
    mri_norms_orig = []
    mri_man_segs_orig = []
    for s in tqdm(man_subjects):
        mri_seg_orig = fs.Volume.read(os.path.join(adir, s, 'mri', mname))
        mri_man_segs_orig.append(mri_seg_orig)
        mri_seg = mri_seg_orig.fit_to_shape(target_shape, center='bbox')
        mri_man_segs.append(mri_seg)
        mri_norm_orig = fs.Volume.read(os.path.join(adir, s, 'mri', vname))
        mri_norm = mri_norm_orig.resample_like(mri_seg)
        mri_norms.append(mri_norm)
        mri_norms_orig.append(mri_norm_orig)

    mri_man_segs_recoded = [fs.label.recode(mri, mapping) for mri in tqdm(mri_man_segs)]

    mri_seg_atlas = fs.Volume.read("aseg_atlas.mgz")
    hard_seg = np.argmax(mri_seg_atlas.data, axis=-1)
    mri_hard_seg = mri_seg_atlas.copy(hard_seg).fit_to_shape(target_shape, center='bbox')
    mri_norm_atlas = fs.Volume.read("norm_atlas.mgz").resample_like(mri_hard_seg)
    mri_seg_atlas = mri_seg_atlas.resample_like(mri_hard_seg)
    norm_atlas = (mri_norm_atlas.data / mri_norm_atlas.data.max())[np.newaxis, ..., np.newaxis]

    import generators as gens
    inited = True

# these control how much the labels are warped to create a new "subject"
warp_max=2   
warp_max=2.5
warp_min=.5
warp_blur_min=np.array([2, 4, 8])
warp_blur_max=warp_blur_min*2
bias_blur_min=np.array([2, 4, 8])
bias_blur_max=bias_blur_min*2

print(f'using warp max = {warp_max} and nlabels {nlabels_small}')
inshape=np_segs[0].shape
gen_args = dict(
    warp_min=warp_min,
    warp_max=warp_max,
    blur_max=2,  # was .5, then 1
    bias_max=.25,  # was 2 then .5
    bias_blur_min=bias_blur_min,
    bias_blur_max=bias_blur_max,
    gamma=0,
    # warp_zero_mean=True,
    zero_background=.75,
    noise_max=.2,   
    noise_min=.1
)

# generate a  synth image
# inputs are synth images
# outputs are label maps
def synth_gen(label_vols, gen_model, lab_to_ind, labels_in, batch_size=8, use_rand=True, gpuid=1, 
              seg_resize=1, num_outside_shapes_to_add=8, use_log=False, debug=False, 
              add_outside=True):

    inshape = label_vols[0].shape
    nlabels = gen_model.outputs[-1].get_shape().as_list()[-1]  # number of compressed labels

    if add_outside:
        l2l = nes.models.labels_to_labels(
            labels_in,
            shapes_num=num_outside_shapes_to_add,
            in_shape=label_vols[0].shape,
            shapes_add=True
        )
        li = np.concatenate([labels_in, np.arange(labels_in.max()+1, 
                                                  labels_in.max()+1+num_outside_shapes_to_add)])
        l2i = nes.models.labels_to_image(
            labels_in=li,
            labels_out=None,
            in_shape=label_vols[0].shape,
            zero_background=.5,  #  was 1
            noise_max=.2,
            noise_min=.1,
            warp_max=0
        )

    # outputs [6] and [7] are the t2 labels without (6) and with (7) atrophy
    batch_input_labels = np.zeros((batch_size, *inshape, 1))
    label_shape = tuple(np.array(inshape) // seg_resize)
    batch_onehots = np.zeros((batch_size, *label_shape, nlabels))
    batch_images = np.zeros((batch_size, *inshape, 1))

    if debug:
        batch_orig_images1 = np.zeros((batch_size, *inshape, 1))
        batch_orig_images2 = np.zeros((batch_size, *inshape, 1))
        batch_orig_labels1 = np.zeros((batch_size, *label_shape, 1))
        batch_orig_labels2 = np.zeros((batch_size, *label_shape, 1))

    if gpuid >= 0:
        device = '/gpu:' + str(gpuid)
    else:
        device = '/physical_device:CPU:0'
        device = '/cpu:0'

    ind = -1
    while (True):
        for bind in range(batch_size):
            if use_rand:
                ind = np.random.randint(0, len(label_vols))
            else:
                ind = np.mod(ind+1, len(label_vols))
            batch_input_labels[bind,...] = label_vols[ind].data[...,np.newaxis]

        with tf.device(device):
            pred = gen_model.predict_on_batch(batch_input_labels)

        for bind in range(batch_size):
            im = pred[0][bind,...]
            onehot = pred[1][bind,...]
            if add_outside:
                 im = nes.utils.augment.add_outside_shapes(im[..., 0], np.argmax(onehot, axis=-1), labels_in, l2l=l2l, l2i=l2i)[..., np.newaxis]

            if use_log:
                onehot[onehot == 0] = -10
                onehot[onehot == 1] = 10

            batch_images[bind, ...] = im
            batch_onehots[bind, ...] = onehot

        inputs = [batch_images]
        outputs = [batch_onehots]

        yield inputs, outputs
                
    return 0

# build lists for unet architecture
nfeats = 64
nfeats = 60
nsmall = 0
nb_levels = int(np.log2(inshape[0]))-(1+nsmall)   # 4,4,4 is lowest level
nb_conv_per_level = 2
unet_scale = 1
nfeats_small = int(nfeats // 3)
    
unet_nf = []
fscale = 1
fscale = 1.1
fscale = 1.2


for level in range(nb_levels-nsmall):
    filters_in_this_level = []
    for layer in range(nb_conv_per_level):
        filters_in_this_level.append(int(fscale**level*nfeats))
        
    unet_nf.append(filters_in_this_level)

unet_nf = [[nfeats_small] * nb_conv_per_level] * nsmall + unet_nf

tf.compat.v1.enable_eager_execution()

lr = 1e-5
lr = 1e-4
lr_lin = 1e-4
name = f'aseg.outside.unet_nf.{nfeats}.{nfeats_small}.{nsmall}.warp_max.{warp_max}.oshapes.{oshapes}'
if nb_levels > 5:
    name = f'aseg.outside.unet_nf.{nfeats}.{nfeats_small}.{nsmall}.levels.{nb_levels}.warp_max.{warp_max}.oshapes.{oshapes}'

label_weights = np.ones((1,nlabels_small,))
# label_weights[0,-1] = .01  # downweight lesion class
lfunc = ne.losses.Dice(nb_labels=nlabels_small, weights=None, check_input_limits=False).mean_loss
thresh = -.2*2 

cooldown = 25
patience = 600

mc_cb_lin = tf.keras.callbacks.ModelCheckpoint(name+'.checkpoint.lin.tf', save_best_only=True)
write_cb_lin = nes.callbacks.WriteHist(name+'.lin.txt')

unet_device = model_device if (fit_lin or dofit) else synth_device
with tf.device(unet_device): 
    model_lin = ne.models.unet(unet_nf, inshape+(1,), None, 3, nlabels_small, feat_mult=None, final_pred_activation='linear')
    print(f'unet model created with nf {unet_nf}')
    softmax_out = KL.Softmax(name='seg')(model_lin.outputs[0])
    model = tf.keras.Model(model_lin.inputs, [softmax_out])


if fit_lin:
    thresh_lin = .5
    thresh = -.5
    print(f'thresh {thresh}, thresh_lin {thresh_lin}')
    lr_cb_lin = nes.tf.callbacks.ReduceLRWithModelCheckpointAndRecovery(name+'.lin.h5', monitor='loss',
                                                                        verbose=1, cooldown=cooldown, 
                                                                        recovery_decrease_factor=.8,
                                                                        factor=.8, patience=patience, 
                                                                        thresh_increase_factor=2,
                                                                        thresh=thresh_lin,
                                                                        save_weights_only=True, 
                                                                        burn_in=10,
                                                                        min_lr=1e-7,
                                                                        warm_restart_epoch=None,
                                                                        nloss=5,
                                                                        warm_restart_lr=None)

    callbacks_lin = [lr_cb_lin, write_cb_lin, ne.callbacks.LRLog(), mc_cb_lin]
    lfunc_lin = ne.losses.MeanSquaredErrorProb().loss
    with tf.device(synth_device):
        gen_lin = synth_gen(mri_segs_recoded, gen_model, None, labels_in, batch_size=batch_size, 
                            use_rand=True, gpuid=synth_gpu,use_log=True, add_outside=oshapes)
        gen = synth_gen(mri_segs_recoded, gen_model, None, labels_in, batch_size=batch_size, 
                        use_rand=True, gpuid=synth_gpu, debug=False, add_outside=oshapes)
        vgen = gens.real_gen(mri_man_segs_recoded, mri_norms, vxm_model, norm_atlas, 
                             None, labels_in, 
                             batch_size=batch_size, use_rand=True, 
                             gpuid=synth_gpu, debug=False, add_outside=oshapes)

    loss_funcs_lin = [lfunc_lin]
    print(f'fitting linear model and saving hist to {write_cb_lin.fname} and model to {lr_cb_lin.fname}')
    with tf.device(model_device):
        nes.utils.check_and_compile(unet_model_lin, gen_lin, optimizer=keras.optimizers.Adam(learning_rate=lr_lin), 
                                    loss=loss_funcs_lin, loss_weights=loss_weights, check_layers=False, run_eagerly=True)
        fhist_lin = unet_model_lin.fit(gen_lin, epochs=int(10), steps_per_epoch=50, initial_epoch=0, callbacks=callbacks_lin)
        unet_model_lin.save_weights(lr_cb_lin.fname)
else:
    fname = name + '.lin.h5'
    if os.path.exists(fname) and dofit:
        print(f'loading linear weights from {fname}')
        model_lin.load_weights(fname)

vxm_model = gens.read_vxm_model(inshape)
vxm_smooth_wt = np.zeros((1, 1))
vxm_smooth_wt[0,0] = .3   # warp regularization hyper parameter

gen_model = gens.create_gen_model(np_segs, oshapes, synth_device, nlabels_small, labels_in, inshape, warp_max)

with tf.device(synth_device):
    val_size = 20
    gen = gens.synth_gen(mri_segs_recoded, gen_model, vxm_model, norm_atlas, 
                         None, labels_in, batch_size=batch_size, 
                         use_rand=True, gpuid=synth_gpu, debug=False, add_outside=oshapes)
    vgen = gens.real_gen(mri_man_segs_recoded, mri_norms, vxm_model, norm_atlas, 
                         None, labels_in, 
                         batch_size=batch_size, use_rand=True, 
                         gpuid=synth_gpu, debug=False, add_outside=oshapes)
    if 0:
        vgen = gens.synth_gen(mri_segs_recoded[len(mri_segs_recoded)-val_size:], gen_model, None, labels_in, 
                              batch_size=batch_size, use_rand=True, 
                              gpuid=synth_gpu, debug=False, add_outside=oshapes)


initial_epoch = 3955
initial_epoch = 0

write_cb = nes.callbacks.WriteHist(name+'.txt', mode='w' if initial_epoch == 0 else 'a')

mc_cb = tf.keras.callbacks.ModelCheckpoint(name+'.checkpoint.h5', save_best_only=True)
lr_cb = nes.tf.callbacks.ReduceLRWithModelCheckpointAndRecovery(name+'.h5', monitor='loss',
                                                                    verbose=2, cooldown=cooldown, 
                                                                    recovery_decrease_factor=1,
                                                                    factor=.8, patience=patience, 
                                                                    thresh_increase_factor=1.2,
                                                                    thresh=thresh,
                                                                    save_weights_only=True, 
                                                                    burn_in=50,
                                                                    min_lr=1e-7,
                                                                    restart=initial_epoch>0,
                                                                    nloss=5)

callbacks = [lr_cb, write_cb, ne.callbacks.LRLog(), mc_cb]

with tf.device(model_device):
    nes.utils.check_and_compile(model, gen, optimizer=keras.optimizers.Adam(learning_rate=lr), 
                                loss=[lfunc], check_layers=False, run_eagerly=True)
    print(f"{'saving' if dofit else 'loading'} fit results to {lr_cb.fname} and hist to {write_cb.fname}")
    if dofit:
        # model.load_weights('aseg.h5')
        fhist = model.fit(gen, epochs=int(10000), steps_per_epoch=50, 
                          initial_epoch=initial_epoch, callbacks=callbacks, 
                          validation_data=vgen, validation_steps=5)
    else:
        model.load_weights(lr_cb.fname)
    

if save_model:
    aseg_fname = 'aseg.h5'
    print(f'saving results to {aseg_fname}')
    # m.save(aseg_atlas_fname)

ntest = 15
vgen = gens.real_gen(mri_man_segs_recoded, mri_norms, vxm_model, norm_atlas, 
                     None, labels_in, 
                     batch_size=1, use_rand=True, 
                     gpuid=synth_gpu, debug=False, add_outside=oshapes)
gen = gens.synth_gen(mri_segs_recoded, gen_model, vxm_model, norm_atlas, 
                     None, labels_in, batch_size=1, use_rand=True, gpuid=synth_gpu, debug=False, 
                     add_outside=oshapes)

ilist = []
olist = []
plist = []
dlist = []
choroid_label = target_lut.search('Left-Choroid')[0]
mask = np.ones((nlabels_small,))
mask[lesion_label] = 0
mask[choroid_label] = 0
mask[0] = 0
lfunc_dice = ne.losses.Dice(nb_labels=nlabels_small, weights=None, check_input_limits=False).loss
for n in tqdm(range(ntest)):
    inb, outb = next(vgen)
    pred = model.predict(inb)
    d = model.evaluate(inb, outb, verbose=0)
    d = lfunc_dice(tf.convert_to_tensor(outb, tf.float32), tf.convert_to_tensor(pred, tf.float32))
    d = (d.numpy() * mask).sum() / mask.sum()
    dlist.append(d)
    ilist.append(inb[0].squeeze().copy())
    olist.append(np.argmax(outb[0].squeeze(), axis=-1).copy())
    plist.append(np.argmax(pred[0].squeeze(), axis=-1).copy())


print(f'real dice {np.array(dlist).mean()}')
print(f'{dlist}')


imgs = np.array(ilist)
tseg = np.array(olist)
pseg = np.array(plist)
fv = fs.Freeview(swap_batch_dim=True)
fv.vol(imgs, name='img', opts=':locked=1:linked=1')
fv.vol(tseg, name='true seg', opts=':colormap=lut:visible=0:linked=1', lut=target_lut)
fv.vol(pseg, name='pred seg', opts=':colormap=lut:visible=1:linked=1', lut=target_lut)
fv.show()
pfc(write_cb.fname, keys=['loss', 'val_loss'], close_all=True, smooth=15, remove_outlier_thresh=2,
        outlier_whalf=4, plot_block=False)

test_mseg = False
if test_mseg:
    aseg_atlas_fname = 'aseg_atlas.h5'
    print(f'reading aseg atlas model from {aseg_atlas_fname}')
    atlas_model = tf.keras.models.load_model(aseg_atlas_fname, custom_objects=ld.layer_dict)
    mapping = fs.lookups.nonlateral_aseg_recoder()
    target_lut = mapping.target_lut
    lut_name = 'nonlat.txt'
    adir = '/autofs/cluster/freesurfer/subjects/atlases/aseg_atlas'
    mname = 'seg_edited.mgz'
    vname = 'norm.mgz'
    sfile = os.path.join(adir, 'scripts', 'subjects.txt')
    with open(sfile, 'r') as f:
        subjects = f.read().split('\n')[0:-1]

    mri_man_segs = []  # manual segs
    mri_norms = []  # mri vols
    mri_norms_orig = []
    mri_man_segs_orig = []
    for s in tqdm(subjects):
        mri_seg_orig = fs.Volume.read(os.path.join(adir, s, 'mri', mname))
        mri_man_segs_orig.append(mri_seg_orig)
        mri_seg = mri_seg_orig.fit_to_shape(target_shape, center='bbox')
        mri_man_segs.append(mri_seg)
        mri_norm_orig = fs.Volume.read(os.path.join(adir, s, 'mri', vname))
        mri_norm = mri_norm_orig.resample_like(mri_seg)
        mri_norms.append(mri_norm)
        mri_norms_orig.append(mri_norm_orig)

    mri_man_segs_recoded = [fs.label.recode(mri, mapping) for mri in tqdm(mri_man_segs)]

    mri_seg_atlas = fs.Volume.read("aseg_atlas.mgz")
    hard_seg = np.argmax(mri_seg_atlas.data, axis=-1)
    mri_hard_seg = mri_seg_atlas.copy(hard_seg).fit_to_shape(target_shape, center='bbox')
    mri_norm_atlas = fs.Volume.read("norm_atlas.mgz").resample_like(mri_hard_seg)
    mri_seg_atlas = mri_seg_atlas.resample_like(mri_hard_seg)
    norm_atlas = (mri_norm_atlas.data / mri_norm_atlas.data.max())[np.newaxis, ..., np.newaxis]

    # psize = (norm_atlas.shape[1] - mri_norms[0].shape[0]) // 2
    # pad = ((0,0), (psize,psize), (psize, psize), (psize, psize), (0, 0))
    dice_list = []
    elist = []
    elist_in_atlas = []
    alist_in_atlas = []
    nlist_in_atlas = []
    mlist_in_atlas = []
    for sno, s in enumerate(tqdm(subjects)):
        mseg_onehot = np.eye(nlabels_small)[mri_man_segs_recoded[sno].data]
        norm = (mri_norms[sno].data / mri_norms[sno].data.max())[np.newaxis, ..., np.newaxis]
        pred = model.predict(norm)
        transform = gens.vxm_model.predict([l, norm, norm_atlas])
        # transform = transform[:, psize:-psize, psize:-psize, psize:-psize, :]
        dice = lfunc(tf.convert_to_tensor(mseg_onehot[np.newaxis], tf.float32), 
                     tf.convert_to_tensor(pred, tf.float32))
        dice_list.append(dice.numpy())
        ev = (mseg_onehot - pred[0])
        evol = (ev**2).sum(axis=-1)
        elist.append(evol)
        evol_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([evol[np.newaxis, ..., np.newaxis], transform])
        norm_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([norm, transform])
        aseg_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([pred[0][np.newaxis, ...], transform])
        mseg_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([mseg_onehot[np.newaxis, ...], transform])
        elist_in_atlas.append(evol_in_atlas.numpy().squeeze())
        nlist_in_atlas.append(norm_in_atlas.numpy().squeeze())
        #alist_in_atlas.append(np.argmax(aseg_in_atlas.numpy(), axis=-1).squeeze())
        #mlist_in_atlas.append(np.argmax(mseg_in_atlas.numpy(), axis=-1).squeeze())
        alist_in_atlas.append(aseg_in_atlas.numpy().squeeze())
        mlist_in_atlas.append(mseg_in_atlas.numpy().squeeze())

    patch_size = 16
    evol_avg = np.array(elist_in_atlas).mean(axis=0)
    nvol_avg = np.array(nlist_in_atlas).mean(axis=0)
    nbhd_img = scipy.ndimage.convolve(evol_avg, np.ones((patch_size,)*3)/(patch_size**3), mode='constant')
    mseg_avg = np.argmax(np.array(mlist_in_atlas).mean(axis=0), axis=-1)
    aseg_avg = np.argmax(np.array(alist_in_atlas).mean(axis=0), axis=-1)
    max_ind = np.argmax(nbhd_img)
    print('computing index occurence volumes')
    aseg_inds = np.argmax(np.array(alist_in_atlas), axis=-1)
    mseg_inds = np.argmax(np.array(mlist_in_atlas), axis=-1)
    max_inds = 10
    mseg_ind_vol = np.zeros(target_shape + (max_inds,))
    aseg_ind_vol = np.zeros(target_shape + (max_inds,))
    mseg_num_inds = np.zeros(target_shape)
    aseg_num_inds = np.zeros(target_shape)
    for x in tqdm(range(aseg_inds.shape[1])):
        for y in range(aseg_inds.shape[2]):
            for z in range(aseg_inds.shape[3]):
                ind_list = -1 * np.ones((max_inds,))
                u = np.unique(mseg_inds[:, x, y, z])
                for lno, l in enumerate(u):
                    if lno >= max_ind:
                        break
                    mseg_ind_vol[x, y, z, lno] = l
                
                mseg_num_inds[x, y, z] = len(u)
                ind_list = -1 * np.ones((max_inds,))
                u = np.unique(aseg_inds[:, x, y, z])
                for lno, l in enumerate(u):
                    aseg_ind_vol[x, y, z, lno] = l

                aseg_num_inds[x, y, z] = len(u)

    subs = np.unravel_index(max_ind, nbhd_img.shape)
    labels_mseg = np.unique(np.argmax(np.array(mlist_in_atlas), axis=-1)[:, subs[0], subs[1], subs[2]])
    labels_aseg = np.unique(np.argmax(np.array(alist_in_atlas), axis=-1)[:, subs[0], subs[1], subs[2]])
    print(subs)

    fv = fs.Freeview()
    fv.vol(mri_segs[0].copy(nvol_avg), name='norm avg')
    fv.vol(mri_segs[0].copy(mseg_avg), name='mseg', opts=':colormap=lut:lut=nonlat')
    fv.vol(mri_segs[0].copy(aseg_avg), name='aseg', opts=':colormap=lut:lut=nonlat')
    nvols = 3
    mvols = np.argmax(np.array(mlist_in_atlas[0:nvols]), axis=-1)
    avols = np.argmax(np.array(alist_in_atlas[0:nvols]), axis=-1)
    fv.vol(mri_segs[0].copy(np.transpose(np.array(nlist_in_atlas[0:nvols]), (1, 2, 3, 0))), name='norm vols')
    fv.vol(mri_segs[0].copy(np.transpose(mvols, (1, 2, 3, 0))), name='man segs', opts=':colormap=lut:lut=nonlat')
    fv.vol(mri_segs[0].copy(np.transpose(avols, (1, 2, 3, 0))), name='synth segs', opts=':colormap=lut:lut=nonlat')
    fv.vol(mri_segs[0].copy(nbhd_img), name='err avg', opts=':colormap=heat:heatscale=.25,.5')
    fv.show(opts=f'-slice {subs[0]} {subs[1]} {subs[2]}', verbose=True)

