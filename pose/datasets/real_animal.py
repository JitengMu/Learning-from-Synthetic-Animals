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

from scipy.io import loadmat

def load_animal(data_dir='./', animal='horse'):
    """
    Output:
    img_list: Nx3   # each image is associated with a shot-id and a shot-id frame_id,
                    # e.g. ('***.jpg', 100, 2) means the second frame in shot 100.
    anno_list: Nx3  # (x, y, visiblity)
    """

    range_path = os.path.join(data_dir, 'behaviorDiscovery2.0/ranges', animal, 'ranges.mat')
    landmark_path = os.path.join(data_dir, 'behaviorDiscovery2.0/landmarks', animal)

    img_list = []  # img_list contains all image paths
    anno_list = [] # anno_list contains all anno lists
    range_file = loadmat(range_path)

    for video in range_file['ranges']:
        # range_file['ranges'] is a numpy array [Nx3]: shot_id, start_frame, end_frame
        shot_id = video[0]
        landmark_path_video = os.path.join(landmark_path, str(shot_id)+'.mat')

        if not os.path.isfile(landmark_path_video):
            continue
        landmark_file = loadmat(landmark_path_video)

        for frame in range(video[1], video[2]+1): # ??? video[2]+1
            frame_id = frame - video[1]
            img_name = '0'*(8-len(str(frame))) + str(frame) + '.jpg'
            img_list.append([img_name, shot_id, frame_id])
            
            coord = landmark_file['landmarks'][frame_id][0][0][0][0]
            vis = landmark_file['landmarks'][frame_id][0][0][0][1]
            landmark = np.hstack((coord, vis))
            anno_list.append(landmark[:18,:])
            
    return img_list, anno_list

def dataset_filter(anno_list):
    """
    output:
    idxs: valid_idxs after filtering
    """
    num_kpts = anno_list[0].shape[0]
    idxs = []
    for i in range(len(anno_list)):
        s = sum(anno_list[i][:,2])
        if s>num_kpts//2:
            idxs.append(i)
    return idxs


def train_test_split_by_video(idxs, img_list, animal):
    """
    Split datasets by videos
    """
    video_idxs = []
    for i, item in enumerate(img_list):
        if i in idxs:
            video_idxs.append(item[1])
    video_idxs = list(set(video_idxs))

    num_videos = len(video_idxs)

    valid_video_idxs = np.random.choice(video_idxs, num_videos//5, replace=False)
    train_idxs = []
    valid_idxs = []
    for i, item in enumerate(img_list):
        if i in idxs:
            if item[1] in valid_video_idxs:
                valid_idxs.append(i)
            else:
                train_idxs.append(i)

    if isfile('./data/real_animal/' +animal + '/train_idxs_by_video.npy'):
        train_idxs = np.load('./data/real_animal/' +animal+ '/train_idxs_by_video.npy')
        valid_idxs = np.load('./data/real_animal/' +animal+ '/valid_idxs_by_video.npy')
    else:
        np.save('./data/real_animal/' +animal+ '/train_idxs_by_video.npy', train_idxs)
        np.save('./data/real_animal/' +animal+ '/valid_idxs_by_video.npy', valid_idxs)

    print('split-by-video number of training images: ', len(train_idxs))
    print('split-by-video number of testing images: ', len(valid_idxs))

    return train_idxs, valid_idxs

class Real_Animal(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        print("init real animal stacked hourglass augmentation")
        self.img_folder = kwargs['image_path'] # root image folders
#         self.jsonfile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.animal = kwargs['animal']
        
        # create train/val split
        
        self.img_list, self.anno_list = load_animal(data_dir=self.img_folder, animal=self.animal)
        idxs = dataset_filter(self.anno_list)
        #self.train_list, self.valid_list = train_test_split1(idxs, self.img_list, self.animal)
        self.train_list, self.valid_list = train_test_split_by_video(idxs, self.img_list, self.animal)
        self.mean, self.std = self._compute_mean(self.animal)

    def _compute_mean(self, animal):
        meanstd_file = './data/real_animal/' + animal + '/mean_by_video.pth.tar'
        #meanstd_file = './data/synthetic_animal/' + animal+'_random_texture' + '/mean.pth.tar'
        if isfile(meanstd_file):
            print("load mean file")
            meanstd = torch.load(meanstd_file)
        else:
            print("generate mean file")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train_list:
                a = self.img_list[index][0]
                img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', self.animal, a)
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
        print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        
        if self.is_train:
            a = self.img_list[self.train_list[index]][0]
            img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', self.animal, a)
            pts = torch.Tensor(self.anno_list[self.train_list[index]].astype(np.float32))
            x_min = float(np.min(self.anno_list[self.train_list[index]][:,0] \
                                 [self.anno_list[self.train_list[index]][:,0]>0]))
            x_max = float(np.max(self.anno_list[self.train_list[index]][:,0] \
                                 [self.anno_list[self.train_list[index]][:,0]>0]))
            y_min = float(np.min(self.anno_list[self.train_list[index]][:,1] \
                                 [self.anno_list[self.train_list[index]][:,1]>0]))
            y_max = float(np.max(self.anno_list[self.train_list[index]][:,1] \
                                 [self.anno_list[self.train_list[index]][:,1]>0]))
        else:
            a = self.img_list[self.valid_list[index]][0]
            img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', self.animal, a)
            pts = torch.Tensor(self.anno_list[self.valid_list[index]].astype(np.float32))
            x_min = float(np.min(self.anno_list[self.valid_list[index]][:,0] \
                                 [self.anno_list[self.valid_list[index]][:,0]>0]))
            x_max = float(np.max(self.anno_list[self.valid_list[index]][:,0] \
                                 [self.anno_list[self.valid_list[index]][:,0]>0]))
            y_min = float(np.min(self.anno_list[self.valid_list[index]][:,1] \
                                 [self.anno_list[self.valid_list[index]][:,1]>0]))
            y_max = float(np.max(self.anno_list[self.valid_list[index]][:,1] \
                                 [self.anno_list[self.valid_list[index]][:,1]>0]))


        c = torch.Tensor(( (x_min+x_max)/2.0, (y_min+y_max)/2.0 ))
        s = max(x_max-x_min, y_max-y_min)/200.0 * 1.5


        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
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
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index' : index, 'center' : c, 'scale' : s,
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)


def real_animal(**kwargs):
    return Real_Animal(**kwargs)

real_animal.njoints = 18  # ugly but works
