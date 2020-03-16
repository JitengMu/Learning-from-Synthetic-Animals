import os, time, glob, itertools, random, imageio, re
import numpy as np 
from tqdm import tqdm
import cv2
from unrealcv.util import read_png, read_npy

import argparse
import unrealdb as udb 
import unrealdb.asset
from unrealdb.asset import skeleton_mapping
from unrealdb.asset.activity import ACT_SKEL_GIRL, ACT_SKEL_RP

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Action Dataset')
    parser.add_argument('--out_path', type=str, help="Root path to store the generated data")
    parser.add_argument('--var', type=str, help="Choose in ['az', 'el', 'dist']")
    parser.add_argument('--char', type=str, help="Girl or RP")
    args = parser.parse_args()
    return args


def azimuth_render_params(character_pack):

    opt = dict(
        mesh = character_pack['mesh'],
        anim = character_pack['anim'],
        ratio = np.arange(0, 1, 0.05),  # 20 frames for each push-up
        dist = [180],
        az = np.arange(0, 360, 1),   # 360 az angles
        el = [0]
    )

    render_params = itertools.product(opt['mesh'], opt['anim'], opt['ratio'],
        opt['dist'], opt['az'], opt['el'])
    render_params = list(render_params)
    return render_params


def elevation_render_params(character_pack):

    opt = dict(
        mesh = character_pack['mesh'],
        anim = character_pack['anim'],
        ratio = np.arange(0, 1, 0.05),  # 20 frames for each push-up
        dist = [180],
        az = [180],
        el = np.arange(-30, 60, 0.25)
    )

    render_params = itertools.product(opt['mesh'], opt['anim'], opt['ratio'],
        opt['dist'], opt['az'], opt['el'])
    render_params = list(render_params)
    return render_params


def distance_render_params(character_pack):

    opt = dict(
        mesh = character_pack['mesh'],
        anim = character_pack['anim'],
        ratio = np.arange(0, 1, 0.05),  # 20 frames for each push-up
        dist = np.arange(100, 460, 1),
        az = [180],
        el = np.arange(0)
    )
    
    render_params = itertools.product(opt['mesh'], opt['anim'], opt['ratio'],
        opt['dist'], opt['az'], opt['el'])
    render_params = list(render_params)
    return render_params


def azimuth_load_data(root, animal, render_params):

    for i, param in enumerate(tqdm(render_params)):
        mesh, anim, ratio, dist, az, el = param
        az = int(az)
        vid = mesh.strip("'").split('.')[-1] + "_az_" + "{:03d}".format(az)
        dir = os.path.join(root, vid)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        animal.set_mesh(mesh)
        animal.set_animation(anim, ratio)
        animal.set_tracking_camera(dist, az, el)
        img = animal.get_img()

        for repeat in range(0, 5):
            img_idx = int(ratio*20) + repeat*20
            filename = os.path.join(dir, "img_{:05d}.png".format(img_idx))
            #imageio.imwrite(filename, img)
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))


def elevation_load_data(root, animal, render_params):
    for i, param in enumerate(tqdm(render_params)):
        mesh, anim, ratio, dist, az, el = param
        el_pat = int((el+30)*4)
        vid = mesh.strip("'").split('.')[-1] + "_el_" + "{:03d}".format(el_pat)
        dir = os.path.join(root, vid)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        animal.set_mesh(mesh)
        animal.set_animation(anim, ratio)
        animal.set_tracking_camera(dist, az, el)
        img = animal.get_img()

        for repeat in range(0, 5):
            img_idx = int(ratio*20) + repeat*20
            filename = os.path.join(dir, "img_{:05d}.png".format(img_idx))
            #imageio.imwrite(filename, img)
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))


def distance_load_data(root, animal, render_params):
    for i, param in enumerate(tqdm(render_params)):
        mesh, anim, ratio, dist, az, el = param
        dist_pat = int(dist-100)
        vid = mesh.strip("'").split('.')[-1] + "_dist_" + "{:03d}".format(dist)
        dir = os.path.join(root, vid)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        animal.set_mesh(mesh)
        animal.set_animation(anim, ratio)
        animal.set_tracking_camera(dist, az, el)
        img = animal.get_img()

        for repeat in range(0, 5):
            img_idx = int(ratio*20) + repeat*20
            filename = os.path.join(dir, "img_{:05d}.png".format(img_idx))
            #imageio.imwrite(filename, img)
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))


def main(args):
    udb.connect('localhost', 9000)

    # reset the program
    map_name = 'EmptyPlane'
    udb.client.request('vset /action/game/level {map_name}'.format(**locals()))

    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    obj_id = 'tiger'
    animal = udb.CvCharacter(obj_id)
    animal.spawn()
    env = udb.CvEnv()
   
    SKEL_GIRL = "Skeleton'/Game/Girl_01/meshes/girl_01_Skeleton.girl_01_Skeleton'"
    SKEL_RP = "Skeleton'/Game/RP_Character/00_rp_master/UE4_Mannequin_Skeleton.UE4_Mannequin_Skeleton'"
    skel_Girl = skeleton_mapping[SKEL_GIRL]
    skel_RP = skeleton_mapping[SKEL_RP]

    rp_mesh = skel_RP
    rp_anim = [ACT_SKEL_RP[0]]
    rp_pack = dict(
        mesh = rp_mesh,
        anim = rp_anim
    )

    girl_mesh = [skel_Girl[3], skel_Girl[4], skel_Girl[5], skel_Girl[7]]
    girl_anim = [ACT_SKEL_GIRL[0]]
    girl_pack = dict(
    	mesh = girl_mesh,
    	anim = girl_anim
    )
    
    if args.char == 'Girl':
    	character_pack = girl_pack
    elif args.char == 'RP':
    	character_pack = rp_pack

    var = args.var
    assert var in ['az', 'el', 'dist']
    if var == 'az':
        render_params = azimuth_render_params(character_pack)
        azimuth_load_data(out_path, animal, render_params)
    elif var == 'el':
        render_params = elevation_render_params(character_pack)
        elevation_load_data(out_path, animal, render_params)
    elif var == 'dist':
        render_params = distance_render_params(character_pack)
        distance_load_data(out_path, animal, render_params)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    

    








        








    
