# Make a scene with car and human and draw bounding box.

import os, time, glob, itertools, random, imageio, re
import numpy as np
from tqdm import tqdm
from unrealcv.util import read_png, read_npy
import cv2

import unrealdb as udb
import unrealdb.asset
from unrealdb.asset.diva import anim_paths, MESH
from unrealdb import CvCharacter, CvCar

def make_frames(act_opt, frame_count, car_type):
    door_states = {
        'RR': [0, 0, 0, 1, 0, 0],
        'RF': [0, 1, 0, 0, 0, 0],
        'LR': [0, 0, -1, 0, 0, 0],
        'LF': [-1, 0, 0, 0, 0, 0],
        'Trunk': [0, 0, 0, 0, 0, 1]
    }
    
    mesh_path, anim_path, door_name, human_pose, act_label, flip = act_opt
    if car_type == 'DefHybrid' and ('LR' == door_name or 'RR' == door_name):
        return []

    # Return animation data according to activity type
    frames = []
    max_angle = random.randint(50, 75)


    for frame_id in range(frame_count):
        ratio = float(frame_id) / frame_count

        if act_label in ['Exiting', 'Open_Trunk']:
            door_angles = np.array(door_states[door_name]) * ratio * max_angle
        if act_label in ['Entering', 'Closing_Trunk']:
            door_angles = np.array(door_states[door_name]) * (1 - ratio) * max_angle
        if act_label in ['OpenEnterClose', 'OpenExitClose']:
            if ratio < 0.5:
                door_angles = np.array(door_states[door_name]) * ratio * 2 * max_angle
            else:
                door_angles = np.array(door_states[door_name]) * (1 - ratio) * 2 * max_angle

        frame = dict()
        frame['car'] = dict()
        frame['human'] = dict()
        frame['car']['door_angles'] = door_angles 
        frame['human']['animation_path'] = anim_path
        frame['human']['ratio'] = ratio
        frame['human']['pose'] = human_pose
        frame['human']['mesh_path'] = mesh_path
        frame['act_label'] = act_label
        frames.append(frame)

    return frames

def main():
    udb.connect('localhost', 9000)

    # reset the program
    map_name = 'EmptyPlane'
    udb.client.request('vset /action/game/level {map_name}'.format(**locals()))
    udb.client.request('vset /camera/0/location 400 0 300')
    udb.client.request('vset /camera/0/rotation 0 180 0')

    human_mask_color = [0, 128, 0]
    human = CvCharacter('human')
    human.spawn()
    human.set_mesh(MESH.MAN)
    print(udb.client.request('vget /objects'))

    car = CvCar('car')
    car.spawn()
    car_type = 'suv'
    car.set_mesh(car_type)


    dist = 350
    az = 0
    el = -5
    human.set_tracking_camera(dist, az, el)
    img = human.get_img()
    seg = human.get_seg()

    act_opt = anim_paths[0]
    frame_count = 64
    frames = make_frames(act_opt, frame_count, car_type)

    for frame_id, frame in enumerate(tqdm(frames)):
        act_label = frame['act_label']

        door_angles = frame['car']['door_angles']
        animation_path = frame['human']['animation_path']
        ratio = frame['human']['ratio']
        human_pose = frame['human']['pose'] # location, rotation
        mesh_path = frame['human']['mesh_path']
        x, y, z, pitch, yaw, roll = human_pose

        fl, fr, bl, br, hood, trunk = door_angles
        car.set_part_angles(fl, fr, bl, br, hood, trunk)

        # Set car and human location using asset data
        human.set_animation(animation_path, ratio)
        human.set_loc(x, y, z)
        human.set_rot(pitch, yaw, roll)

    # Get 3d bounding box of an object.
    res = udb.client.request('vget /object/human/bounds')
    print(res)
    res = udb.client.request('vget /object/car/bounds')
    print(res)

    cv2.imwrite('bb.png', img[:,:,2::-1])

if __name__ == '__main__':
    main()