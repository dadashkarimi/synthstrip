import os
import nibabel
import numpy as np
from scipy import ndimage
import sys
from skimage.exposure._adapthist import interpolate
sys.path.append('./Demic')
from image_io.file_read_write import *
from util.image_process import *
from util.dice_evaluation import get_largest_component

def get_bounding_box(volume, margin = (3,8,8)):
    [d_idxes, h_idxes, w_idxes] = np.nonzero(volume>0)
    [D, H, W] = volume.shape
    print(D, H, W, margin, len(d_idxes))
    mind = max(d_idxes.min() - margin[0], 0)
    maxd = min(d_idxes.max() + margin[0], D)
    minh = max(h_idxes.min() - margin[1], 0)
    maxh = min(h_idxes.max() + margin[1], H)
    minw = max(w_idxes.min() - margin[2], 0)
    maxw = min(w_idxes.max() + margin[2], W)
    return [mind, maxd, minh, maxh, minw, maxw]

def crop_volume(volume, roi):
    return volume[np.ix_(range(roi[0], roi[1]),
                         range(roi[2], roi[3]),
                         range(roi[4], roi[5]))]  
                      
def get_patient_names(input_folder):
    file_names = os.listdir(input_folder)
    patient_names = []
    num_p = 0
    num_n = 0
    for x in file_names:
        if("Image.nii.gz" in x):
            patient_name = "_".join(x.split("_")[:-1])
            patient_names.append(patient_name)
            print(patient_name)
            if(patient_name[:3] == "17_"):
                num_p = num_p + 1
            else:
                num_n = num_n + 1
    print("patients volume:", num_p)
    print("normal volume:", num_n)

def get_bbox_and_crop():
    img_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain0'
    seg_folder = '/Users/guotaiwang/Documents/workspace/tf_project/fetal_brain_seg/result/pnet-s10-ml-esb'
    save_folder = '/Users/guotaiwang/Dropbox/FetalBrain/auto_seg'
    file_names = os.listdir(seg_folder)
    file_names = [name for name in file_names if name[:3]=='17_' and 'Seg' in name]
    bbox = []
    for lab_name in file_names:
        patient_name = '_'.join(lab_name.split('_')[:-1])
        img_name =  patient_name + '_Image.nii.gz'
        img_full_name = os.path.join(img_folder, img_name)
        lab_full_name = os.path.join(seg_folder, lab_name)
        img_obj = nibabel.load(img_full_name)
        img = img_obj.get_data()
        lab = nibabel.load(lab_full_name).get_data()
        margin = [40, 40, 40]
        [idx_min, idx_max] = get_ND_bounding_box(lab, margin)
        bbox.append(idx_min + idx_max)
        img_sub = crop_ND_volume_with_bounding_box(img, idx_min, idx_max)
        lab_sub = crop_ND_volume_with_bounding_box(lab, idx_min, idx_max)
        img_save_name = os.path.join(save_folder, img_name)
        lab_save_name = os.path.join(save_folder, lab_name)
        
        img_sub_obj = nibabel.Nifti1Image(img_sub, img_obj.affine, img_obj.header)
        lab_sub_obj = nibabel.Nifti1Image(lab_sub, img_obj.affine, img_obj.header)
        nibabel.save(img_sub_obj, img_save_name)
        nibabel.save(lab_sub_obj, lab_save_name)
        print(name, img.shape)
    bbox = np.asarray(bbox)
    np.savetxt(save_folder + '/../bbox.txt', bbox)

def resample_and_crop():
    img_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain'
    save_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain_bb'
    file_names = os.listdir(img_folder)
    file_names = [name for name in file_names if 'Image' in name]
    for img_name in file_names:
        patient_name = '_'.join(img_name.split('_')[:-1])
        img_name = patient_name + '_Image.nii.gz'
        lab_name = patient_name + '_Label.nii.gz'
        img_full_name = os.path.join(img_folder, img_name)
        lab_full_name = os.path.join(img_folder, lab_name)
        
        img_obj = nibabel.load(img_full_name)
        img     = img_obj.get_data()
        lab     = nibabel.load(lab_full_name).get_data()
        spacing = img_obj.header.get_zooms()
        print('img size', img.shape)
        
        scale = [spacing[0], spacing[1], 1.0]
        img_resample = ndimage.interpolation.zoom(img, scale, order = 1)
        lab_resample = ndimage.interpolation.zoom(lab, scale, order = 0)
        margin = [20, 20, 10]
        [idx_min, idx_max] = get_ND_bounding_box(lab_resample, margin)
        img_sub = crop_ND_volume_with_bounding_box(img_resample, idx_min, idx_max)
        lab_sub = crop_ND_volume_with_bounding_box(lab_resample, idx_min, idx_max)

        img_sub_obj = nibabel.Nifti1Image(img_sub, img_obj.affine, img_obj.header)
        lab_sub_obj = nibabel.Nifti1Image(lab_sub, img_obj.affine, img_obj.header)

        img_save_name = os.path.join(save_folder, img_name)
        lab_save_name = os.path.join(save_folder, lab_name)
        nibabel.save(img_sub_obj, img_save_name)
        nibabel.save(lab_sub_obj, lab_save_name)

def fetal_brain_preprocess(input_folder, output_folder, file_names, \
    img_postfix, lab_postfix, wht_postfix, crop = True):
    with open(file_names) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    patient_names = ['a22_05']
    for patient_name in patient_names:
        print(patient_name)
        lab_name = os.path.join(input_folder, "{0:}_{1:}.nii.gz".format(patient_name, lab_postfix))
        lab_img = nibabel.load(lab_name)
        lab = lab_img.get_data()
        lab_unique = np.unique(lab)
        assert(len(lab_unique) >=2)
        if(len(lab_unique) > 2):
            lab = lab == 2
        lab = np.asarray(lab > 0, np.uint8)
        assert(lab.sum() > 0)
        bb  = get_bounding_box(lab, (8,8,3))
        [W, H, D] = lab.shape
        roi = [0, W, 0, H] + [bb[4], bb[5]]
        for mod_idx in range(len(img_postfix)):
            mod = img_postfix[mod_idx]
            img_name = os.path.join(input_folder, "{0:}_{1:}.nii.gz".format(patient_name, mod))
            img = nibabel.load(img_name).get_data()
            threshold = 40
            strct = ndimage.generate_binary_structure(3, 5)
            weight = img <= threshold
            weight = ndimage.binary_opening(weight, strct)
            weight = np.asarray(1 - weight, np.float32)
            img_norm = itensity_normalize_one_volume(img, weight)
            if(crop):
                img_norm = crop_volume(img_norm, roi)
            img_norm = nibabel.Nifti1Image(img_norm, lab_img.affine, lab_img.header)
            img_name = os.path.join(output_folder, "{0:}_{1:}.nii.gz".format(patient_name, mod))
            nibabel.save(img_norm, img_name)
            if(mod_idx ==0):
                wht_name = os.path.join(output_folder, "{0:}_{1:}.nii.gz".format(patient_name, wht_postfix))
                if(crop):
                    weight = crop_volume(weight, roi)
                weight = nibabel.Nifti1Image(weight, lab_img.affine, lab_img.header)
                nibabel.save(weight, wht_name)
        if(crop):
            lab = crop_volume(lab, roi)
        lab = nibabel.Nifti1Image(lab, lab_img.affine, lab_img.header)
        lab_name = os.path.join(output_folder, "{0:}_{1:}.nii.gz".format(patient_name, lab_postfix))
        nibabel.save(lab, lab_name)

def resize_2d_image_to_fixed_width(img, width, order = 1):
    [H, W] = img.shape
    scale  = float(width)/W
    resized_img = ndimage.interpolation.zoom(img, scale, order = order)
    return resized_img

def resize_2d_image_to_fixed_height(img, height, order = 1):
    [H, W] = img.shape
    scale  = float(height)/H
    resized_img = ndimage.interpolation.zoom(img, scale, order = order)
    return resized_img

def scale_itensity_for_visualize(img):
    minv = img.min()
    maxv = img.max()
    output = (img - minv)*255/(maxv-minv)
    output = np.uint8(output)
    return output

## for 3D images
def get_largest_component(img): # 2D or 3D
    if(img.sum()==0):
        print('the largest component is null')
        return img
    if(len(img.shape) == 3):
        s = ndimage.generate_binary_structure(3,1) # iterate structure
    elif(len(img.shape) == 2):
        s = ndimage.generate_binary_structure(2,1) # iterate structure
    else:
        raise ValueError("the dimension number shoud be 2 or 3")
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    return np.asarray(labeled_array == max_label, np.uint8)

def get_detection_binary_bounding_box(img, margin, spacing = [1,1,1], mode = 0):
    """
    mode = 0: get bounding box based on extent
    mode = 1: fit a bounding box based on cube
    mode = 2: ransac fiting
    """
    strt = ndimage.generate_binary_structure(3,2) # iterate structure
    post = padded_binary_closing(img, strt)
    post = get_largest_component(post)
    if(mode == 0):
        bb_min, bb_max = get_ND_bounding_box(post, margin)
    elif(mode == 1):
        bb_min, bb_max = get_robust_3d_bounding_box(post, margin, spacing)
    else:
        points = get_segmentation_surface(post, spacing, sample_ratio = 0.25)
        bb_min,bb_max = ransac_elipsoid_fitting(points, iter = 3)
        bb_min = [max(0, int(bb_min[i]/spacing[i]) - margin[i]) for i in range(3)]
        bb_max = [min(img.shape[i]-1, int(bb_max[i]/spacing[i]) + margin[i]) for i in range(3)]

    out = np.zeros_like(img)
    out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = 1
    return out

def padded_binary_closing(img, strt):
    [D, H, W] = img.shape
    temp_img = np.zeros([D+2, H+2, W+2])
    temp_img[1:D+1, 1:H+1, 1:W+1] = img
    temp_img = ndimage.morphology.binary_closing(temp_img, strt)
    return temp_img[1:D+1, 1:H+1, 1:W+1]

def get_robust_3d_bounding_box(label, margin, spacing):
    # 1, get central point
    d_list, h_list, w_list = np.nonzero(label)
    d_c = int(d_list.mean())
    
    # 2, infer bounding box size
    [D, H, W] = label.shape
    sub_volume = label[max(0, d_c -2) : min(d_c + 2, D-1)]
    mean_slice = np.asarray(np.mean(sub_volume, axis = 0), np.uint8)
    [idx2d_min, idx2d_max] = get_ND_bounding_box(mean_slice, [margin[1], margin[2]])
    roi_h = idx2d_max[0] - idx2d_min[0]
    roi_w = idx2d_max[1] - idx2d_min[1]
    roi_d = int(math.sqrt(roi_h * roi_w * spacing[1] * spacing[2])/spacing[0])
    half_roi_d = roi_d/2
    
    print('bbox size', roi_d, roi_h, roi_w)
    # adjust bouding box along z axis
    sub_volume = crop_ND_volume_with_bounding_box(label, [0] + idx2d_min, [D-1] + idx2d_max)
    slice_size = np.sum(sub_volume, axis = (1,2))
    
    center_d_list = []
    size_sum = 0
    for temp_d in range(D):
        temp_d_min = max(0, temp_d - half_roi_d)
        temp_d_max = min(temp_d_min + roi_d, D-1)
        temp_size  = np.sum(slice_size[temp_d_min:temp_d_max + 1])
        if(temp_size > size_sum):
            size_sum = temp_size
            center_d_list = [temp_d]
        elif(temp_size == size_sum):
            center_d_list.append(temp_d)
    center_d = int(np.mean(center_d_list))
    d_min = max(0, center_d - half_roi_d)
    d_max = min(d_min + roi_d, D-1)
    
    bb_min = [d_min] + idx2d_min
    bb_max = [d_max] + idx2d_max
    return bb_min, bb_max

def get_segmentation_surface(img, spacing, sample_ratio = 0.2):
    points = []
    [D, H, W] = img.shape
    temp_img = np.zeros([D+2, H+2, W+2])
    temp_img[1:D+1, 1:H+1, 1:W+1] = img
    doff = [-1, 1, 0, 0, 0, 0]
    hoff = [0,  0, -1, 1, 0, 0]
    woff = [0, 0, 0, 0, -1, 1]
    for i in range(D):
        for j in range(H):
            for k in range(W):
                r = random.random()
                if(r > sample_ratio):
                    continue
                if(temp_img[i+1][j+1][k+1] > 0):
                    edge = False
                    for idx in range(6):
                        if(temp_img[i+1+doff[idx]][j+1+hoff[idx]][k+1+woff[idx]] == 0):
                            edge = True
                            break
                    if(edge):
                        temp_point = [i*spacing[0], j*spacing[1], k*spacing[2]]
                        points.append(temp_point)
    return points

def ransac_elipsoid_fitting(points, iter = 3):
    distance = np.zeros((len(points), 1), np.float32)
    for i in range(iter):
        
        if(i == 0):
            inlier = random.sample(points, int(len(points)* 0.1))
        else:
            inlier_flag = np.asarray(np.abs(distance - distance.mean()) < 2*distance.std(), np.float32)
            inlier = [points[idx] for idx in range(len(points)) if inlier_flag[idx][0]>0]
            inlier = random.sample(inlier, int(len(inlier)* 0.2))
        inlier = np.asarray(inlier)
        
        c, r, evecs, v = ellipsoid_fit(inlier)
        
        for idx in range(len(points)):
            p = points[idx]
            dis = (p[0] - c[0])*(p[0] - c[0])/r[0]/r[0] + \
                  (p[1] - c[1])*(p[1] - c[1])/r[1]/r[1] + \
                  (p[2] - c[2])*(p[2] - c[2])/r[2]/r[2] - 1.0
            distance[idx, 0] = dis
    inlier_flag = np.asarray(np.abs(distance - distance.mean()) < 2*distance.std(), np.float32)
    inlier = [points[idx] for idx in range(len(points)) if inlier_flag[idx][0]>0]
    inlier = np.asarray(inlier)
    idx_min = inlier.min(axis = 0)
    idx_max = inlier.max(axis = 0)
    return idx_min, idx_max


## for ND images
def itensity_normalize_one_volume(volume, mask = None, replace = False):
    """
        normalize a volume image with mean and std of the mask region
        """
    if(mask is None):
        mask = volume > 0
    pixels = volume[mask>0]
    mean = pixels.mean()
    std  = pixels.std()
    if(std/mean > 0.8): 
        idx = (mask > 0) * (volume < 3*mean)
        pixels = volume[idx>0]
        mean = pixels.mean()
        std = pixels.std()
    out = (volume - mean)/std
    if(replace):
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[mask==0] = out_random[mask==0]
    return out

def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    return ndimage.interpolation.zoom(volume, scale, order = order)

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of an ND binary volume
    """
    input_shape = label.shape
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def pad_ND_volume_to_desired_shape(volume, desired_shape, mode = 'reflect'):
    """ Pad a volume to desired shape
        if the input size is larger than output shape, then reture then input volume
    """
    input_shape = volume.shape
    output_shape = [max(input_shape[i], desired_shape[i]) for i in range(len(input_shape))]
    pad_width = []
    pad_flag  = False
    for i in range(len(input_shape)):
        pad_lr = output_shape[i]-input_shape[i]
        if(pad_lr > 0):
            pad_flag = True
        pad_l  = int(pad_lr/2)
        pad_r  = pad_lr - pad_l
        pad_width.append((pad_l, pad_r))
    if(pad_flag):
        volume = np.pad(volume, pad_width, mode = mode)
    return volume

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

def extract_roi_from_nd_volume(volume, roi_center, roi_shape, fill = 'random'):
    '''Extract an roi from a nD volume
        volume      : input nD numpy array
        roi_center  : center of roi with position
        output_shape: shape of roi
        '''
    input_shape = volume.shape
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = roi_shape)
    else:
        output = np.zeros(roi_shape)
    r0max = [int(x/2) for x in roi_shape]
    r1max = [roi_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - roi_center[i]) for i in range(len(r0max))]
    out_center = r0max
    if(len(roi_center)==3):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                    range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                    range(roi_center[2] - r0[2], roi_center[2] + r1[2]))]
    elif(len(roi_center)==4):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]),
                      range(out_center[3] - r0[3], out_center[3] + r1[3]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                      range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                      range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                      range(roi_center[3] - r0[3], roi_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output

def set_roi_to_nd_volume(volume, roi_center, sub_volume):
    '''Set an roi of an ND volume with a sub volume
        volume: an ND numpy array
        roi_center: center of roi
        sub_volume: the sub volume that will be copied
        '''
    volume_shape = volume.shape
    patch_shape  = sub_volume.shape
    output_volume = volume
    for i in range(len(roi_center)):
        if(roi_center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - roi_center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(roi_center) == 3):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(roi_center) == 4):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                             range(roi_center[3] - r0[3], roi_center[3] + r1[3]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                          range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume

def convert_label(label, source_list, target_list):
    """
    convert a label map based a source list and a target list of labels
    label: nd array 
    source_list: a list of labels that will be converted, e.g. [0, 1, 2, 4]
    target_list: a list of target labels, e.g. [0, 1, 2, 3]
    """
    assert(len(source_list) == len(target_list))
    label_converted = np.zeros_like(label)
    for i in range(len(source_list)):
        label_temp = np.asarray(label == source_list[i], np.int32)
        label_temp = label_temp * target_list[i]
        label_converted = label_converted + label_temp
    return label_converted
    
