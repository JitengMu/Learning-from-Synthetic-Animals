from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
from pose.utils.evaluation  import final_preds
import pose.models as models

from scipy.io import loadmat
import glob

import scipy.misc
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage

# bbox using keypoints
def cal_bbox(kpts_list):
    bbox_list = []
    for kpts in kpts_list:
        pts = np.load(kpts).astype(np.float16)
        y_min = np.min(pts[:,1])
        y_max = np.max(pts[:,1])
        x_min = np.min(pts[:,0])
        x_max = np.max(pts[:,0])
        bbox_list.append([x_min, x_max, y_min, y_max])
    return bbox_list


def load_animal(data_dir='./', animal='horse'):
    """
    Output:
    img_list: Nx3   # each image is associated with a shot-id and a shot-id frame_id,
                    # e.g. ('***.jpg', 100, 2) means the second frame in shot 100.
    anno_list: Nx3  # (x, y, visiblity)
    """

    # img_list contains all image paths
    img_list = glob.glob(os.path.join(data_dir, 'synthetic_animal', animal+'_combineds5r5_texture', '*img.png'))
    img_list = sorted(img_list)
    # anno_list contains all anno lists
    #seg_list = []
    kpts_list = []
    for img_path in img_list:
        #seg_list.append(img_path[:-7]+'seg.png')
        kpts_list.append(img_path[:-7]+'kpts.npy')

    bbox_list = cal_bbox(kpts_list)
    return img_list, kpts_list, bbox_list

## bbox using segmentation masks
#def cal_bbox(seg_list):
#    bbox_list = []
#    for seg in seg_list:
#        seg_img = plt.imread(seg)
#        y, x = np.where(seg_img==1)
#        y_min = np.min(y)
#        y_max = np.max(y)
#        x_min = np.min(x)
#        x_max = np.max(x)
#        bbox_list.append([x_min, x_max, y_min, y_max])
#    return bbox_list
#
#
#def load_animal(data_dir='./', animal='horse'):
#    """
#    Output:
#    img_list: Nx3   # each image is associated with a shot-id and a shot-id frame_id,
#                    # e.g. ('***.jpg', 100, 2) means the second frame in shot 100.
#    anno_list: Nx3  # (x, y, visiblity)
#    """
#
#    # img_list contains all image paths
#    img_list = glob.glob(os.path.join(data_dir, 'synthetic_animal', animal , '*img.png'))
#    img_list = sorted(img_list)
#    # anno_list contains all anno lists
#    seg_list = []
#    kpts_list = []
#    for img_path in img_list:
#        seg_list.append(img_path[:-7]+'seg.png')
#        kpts_list.append(img_path[:-7]+'kpts.npy')
#
#    bbox_list = cal_bbox(seg_list)
#    return img_list, kpts_list, bbox_list
#

def train_test_split(img_list, animal):
    if isfile('./data/synthetic_animal/' +animal+'_combineds5r5_texture' + '/train_idxs.npy'):
        train_idxs = np.load('./data/synthetic_animal/' +animal+'_combineds5r5_texture'+ '/train_idxs.npy')
        valid_idxs = np.load('./data/synthetic_animal/' +animal+'_combineds5r5_texture'+ '/valid_idxs.npy')
    else:
    #    np.save('../data/synthetic_animal/' +animal+'_combineds5r5_texture'+ '/train_idxs.npy', train_idxs)
    #    np.save('../data/synthetic_animal/' +animal+'_combineds5r5_texture'+ '/valid_idxs.npy', valid_idxs)
        raise Exception('no train/val split exists')

    return train_idxs, valid_idxs

class Synthetic_Animal_SP(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        print()
        print('==> synthetic_animal_sp')
        self.animal = kwargs['animal']
        self.nParts = 18
        if self.animal=='horse':
            self.idxs = np.array([1718,1684,1271,1634,1650,1643,1659,925,392,564,993,726,1585,1556,427,1548,967,877])
            self.idxs_mask = np.zeros(3299) # for horse
        elif self.animal=='tiger':
            self.idxs = np.array([2753, 2679, 2032, 1451, 1287, 3085, 1632, 229, 1441, 1280, 2201, 1662, 266, 158, 270, 152, 219, 129 ])
            self.idxs_mask = np.zeros(3299)
        else:
            raise Exception('animal should be horse/tiger')
        self.idxs_mask[self.idxs] = 1 # adjusting which keypoints to compute los
        self.img_folder = kwargs['image_path'] # root image folders
#         self.jsonfile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.train_with_occlusion = True

        # create train/val split
        self.img_list, self.anno_list, self.bbox_list = load_animal(data_dir=self.img_folder, animal=self.animal)
        #print("total number of images:", len(self.img_list))
        self.train_list, self.valid_list = train_test_split(self.img_list, self.animal)
        #print("train images:", len(self.train_list))
        #print("test images:", len(self.valid_list))
        self.mean, self.std = self._compute_mean(self.animal)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
                       [
                        sometimes(iaa.Affine(
                        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-30, 30), # rotate by -45 to +45 degrees
                        shear=(-20, 20), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    sometimes(iaa.AdditiveGaussianNoise(scale=0.5*255, per_channel=0.5)),
                    sometimes(iaa.GaussianBlur(sigma=(1.0,5.0))),
                    sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
                    #sometimes(iaa.CoarseDropout((0.2, 0.25), size_percent=0.01)),
                        ],
                random_order=True
            )

    def _compute_mean(self, animal):
        meanstd_file = './data/synthetic_animal/' + animal+'_combineds5r5_texture' + '/mean.pth.tar'
        #print('load from mean file:', meanstd_file)
        if isfile(meanstd_file):
            #print("load mean file")
            meanstd = torch.load(meanstd_file)
        else:
            print("generate mean file")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train_list:
                a = self.img_list[index]
                img_path = os.path.join(self.img_folder, 'synthetic_animal', self.animal, a)
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train_list)
            std /= len(self.train_list)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        #if self.is_train:
        #    print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        #    print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        
        if self.is_train:
            x_min, x_max, y_min, y_max = self.bbox_list[self.train_list[index]]
            pts = np.load(self.anno_list[self.train_list[index]]).astype(np.float16)
            img_path = self.img_list[self.train_list[index]]
        else:
            x_min, x_max, y_min, y_max = self.bbox_list[self.valid_list[index]]
            pts = np.load(self.anno_list[self.valid_list[index]]).astype(np.float16)
            img_path = self.img_list[self.valid_list[index]]
        
        # update keypoints visibility for different number of keypoints
        if self.train_with_occlusion:
            pts[self.idxs,2] = 1
        else:
            pts *= pts[:,2].reshape(-1,1)

        pts = pts[self.idxs]
        pts_aug = pts[:,:2].copy()

        # center and scale
        x_min = np.clip(x_min, 0,640)
        y_min = np.clip(y_min, 0,480)
        x_max = np.clip(x_max, 0,640)
        y_max = np.clip(y_max, 0,480)

        c = torch.Tensor(( (x_min+x_max)/2.0, (y_min+y_max)/2.0 ))
        s = max(x_max-x_min, y_max-y_min)/200.0*1.25

        # Adjust center/scale slightly to avoid cropping limbs
#         if c[0] != -1:
#             c[1] = c[1] + 15 * s
#             s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.shape[0]
        img = np.array(imageio.imread(img_path))[:,:,:3]
        img_aug = np.expand_dims(img, axis=0)
        pts_aug = np.expand_dims(pts_aug, axis=0)

        r = 0
        if self.is_train:
            img_aug, pts_aug = self.seq(images=img_aug, keypoints=pts_aug)

        img = img_aug.squeeze(0)
        img = im_to_torch(img)
        pts[:,:2] = pts_aug
        pts = torch.Tensor(pts)

        for j in range(self.idxs.shape[0]):
            if pts[j][0]<0 or pts[j][1]<0 or pts[j][0]>640 or pts[j][1]>480:
                pts[j] = 0

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
#             if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis
        tpts[:,2] = target_weight.view(-1)

        # Meta info
        meta = {'index' : index, 'center' : c, 'scale' : s,
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}
        #print(inp.size(), target.size(), meta)
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)


def synthetic_animal_sp(**kwargs):
    return Synthetic_Animal_SP(**kwargs)

#synthetic_animal_sp.njoints = 3299  # ugly but works
synthetic_animal_sp.njoints = 18
