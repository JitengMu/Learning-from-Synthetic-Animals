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


def load_animal(data_dir, animal,  trainlist, validlist, label_dir):
    """
    Output:
    img_list: Nx3   # each image is associated with a shot-id and a shot-id frame_id,
                    # e.g. ('***.jpg', 100, 2) means the second frame in shot 100.
    anno_list: Nx3  # (x, y, visiblity)
    """

    # img_list contains all image paths
    img_list = glob.glob(os.path.join(data_dir, 'real_animal_crop', 'real_'+animal+'_crop', '*.jpg'))
    img_list = sorted(img_list)
    # anno_list contains all anno lists
    #seg_list = []
    kpts_list = []
    #for img_path in img_list:
    #    #seg_list.append(img_path[:-7]+'seg.png')
    #    kpts_list.append(img_path[:-7]+'kpts.npy')

    # modify anno_list for ssl    
    if isfile(label_dir + "ssl_labels/ssl_labels_train.npy"):
        print("==> load ssl labels")
        ssl_labels_list = np.load(label_dir + "ssl_labels/ssl_labels_train.npy", allow_pickle=True)
        ssl_labels = ssl_labels_list.item()
        for j in range(len(img_list)):
            if j in trainlist:
                idx = int(np.where(trainlist==j)[0])
                kpts_list.append(ssl_labels.get(idx))
            else:
                kpts_list.append(None)
    else:
        print("==> no ssl_labels")
        for img_path in img_list:
            kpts_list.append(None)
    
    #bbox_list = cal_bbox(kpts_list)
    bbox_list = None

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

def train_test_split_by_video(animal):
    """
    Split datasets by videos
    """

    if isfile('./data/real_animal/' +animal+ '/train_idxs_by_video.npy'):
        train_idxs = np.load('./data/real_animal/' +animal+ '/train_idxs_by_video.npy')
        valid_idxs = np.load('./data/real_animal/' +animal+ '/valid_idxs_by_video.npy')
    else:
        raise Exception('no train/val split exists')

    #print('split-by-video number of training images: ', len(train_idxs))
    #print('split-by-video number of testing images: ', len(valid_idxs))

    return train_idxs, valid_idxs


class Real_Animal_Crop(data.Dataset):
    def __init__(self, is_train = True, is_aug = True, **kwargs):
        print()
        print("==> load real_animal_sp_crop")
        self.animal = kwargs['animal']
        self.nParts = 18
        if self.animal=='horse':
            self.idxs = np.arange(18)
            self.idxs_mask = np.zeros(18)
        elif self.animal=='tiger':
            self.idxs = np.arange(18)
            self.idxs_mask = np.zeros(18)
        else:
            raise Exception('animal should be tiger/horse')
        self.idxs_mask[self.idxs] = 1 # adjusting which keypoints to compute los
        self.img_folder = kwargs['image_path'] # root image folders
#         self.jsonfile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.is_aug     = is_aug
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        #self.train_with_occlusion = False

        # create train/val split
        self.train_list, self.valid_list = train_test_split_by_video(self.animal)
        self.img_list, self.anno_list, self.bbox_list = load_animal(data_dir=self.img_folder, animal=self.animal, \
                                                                    trainlist=self.train_list, validlist=self.valid_list, label_dir=kwargs['checkpoint'])
        #print("total number of images:", len(self.img_list))
        print("train images:", len(self.train_list))
        #print("test images:", len(self.valid_list))
        self.mean, self.std = self._compute_mean(self.animal)

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
                img_path = os.path.join(self.img_folder, 'real_animal', self.animal, a)
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
        
        #print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        #print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor

        if self.is_train:
            #x_min, x_max, y_min, y_max = self.bbox_list[self.train_list[index]]
            pts = self.anno_list[self.train_list[index]]
            img_path = self.img_list[self.train_list[index]]
        else:
            #x_min, x_max, y_min, y_max = self.bbox_list[self.valid_list[index]]
            pts = self.anno_list[self.valid_list[index]]
            img_path = self.img_list[self.valid_list[index]]

        if pts is not None:        
            # update keypoints visibility for different number of keypoints
            #if self.train_with_occlusion:
            #    pts[self.idxs,2] = 1
            #else:
            #    pts *= pts[:,2].reshape(-1,1i)
            pts = pts.astype(np.float16)
            pts = pts[self.idxs]
            pts_aug = pts[:,:2].copy()
    
            ## center and scale
            #x_min = np.clip(x_min, 0,640)
            #y_min = np.clip(y_min, 0,480)
            #x_max = np.clip(x_max, 0,640)
            #y_max = np.clip(y_max, 0,480)
    
            #c = torch.Tensor(( (x_min+x_max)/2.0, (y_min+y_max)/2.0 ))
            #s = max(x_max-x_min, y_max-y_min)/200.0
    
            c = torch.Tensor((128,128))
            s = 256.0/200.0
            r = 0
 
            # Adjust center/scale slightly to avoid cropping limbs
    #         if c[0] != -1:
    #             c[1] = c[1] + 15 * s
    #             s = s * 1.25
    
            # For single-person pose estimation with a centered/scaled figure
            nparts = pts.shape[0]
            img = np.array(imageio.imread(img_path))[:,:,:3]
            assert img.shape[0]==256
            img = im_to_torch(img)
            pts = torch.Tensor(pts)

            if self.is_aug:
                s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
                r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
  
                if random.random() <= 0.5:
                    img = torch.from_numpy(fliplr(img.numpy())).float()
                    pts = shufflelr(pts, width=img.size(2), dataset='real_animal')
                    c[0] = img.size(2) - c[0]
                # Color
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1) 
  
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
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, [self.out_res, self.out_res], rot=r))
                    target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                    target_weight[i, 0] *= vis
            tpts[:,2] = target_weight.view(-1)

        else:
            img = np.array(imageio.imread(img_path))[:,:,:3]
            assert img.shape[0]==256
            img = im_to_torch(img)
            c = torch.Tensor((128,128))
            s = 256.0/200.0
            r = 0
            inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
            inp = color_normalize(inp, self.mean, self.std)
            pts = torch.Tensor(0)
            tpts = torch.Tensor(0)
            target_weight = torch.Tensor(0)
            target = torch.Tensor(0)

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


def real_animal_crop(**kwargs):
    return Real_Animal_Crop(**kwargs)

#synthetic_animal_sp.njoints = 3299  # ugly but works
real_animal_crop.njoints = 18
