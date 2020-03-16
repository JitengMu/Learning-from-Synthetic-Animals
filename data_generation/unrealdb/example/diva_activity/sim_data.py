import os, random, time, glob, pickle
import pdb

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import unrealcv
from tqdm import tqdm
from unrealcv.util import read_npy, read_png
from unrealdb.asset.diva import *
from unrealdb import CvCharacter, CvEnv, CvCar, CvCamera
import unrealdb as udb

# texture_images = glob.glob("C:/qiuwch/code/AnimalParsing/shapenet_dev/DescribableTextures/images/*/*.jpg")
texture_images = glob.glob("C:/qiuwch/code/AnimalParsing/textures/val2017/*.jpg")

# class CvEnv(CvObject):
#     def __init__(self, id):
#         super().__init__(id)
        
#     def set_floor_color(self, R, G, B):
#         actor_name = self.id
#         self.request('vset /env/floor/color {R} {G} {B}'.format(**locals()))

#     def set_floor_texture(self, texture_file):
#         actor_name = self.id
#         self.request('vset /env/floor/texture {texture_file}'.format(**locals()))\


def get_bb(mask, color, margin):
    if mask is None:
        return None
    r, g, b = color
    obj_mask = (mask[:,:,0] == r) & (mask[:,:,1] == g) & (mask[:,:,2] == b)
    ys, xs = np.where(obj_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    bb = [(xmin + xmax)/2, (ymin + ymax)/2, w, h]
    # bb = [xmin - w * margin, ymin - h * margin, xmax + w * margin, ymax + h * margin]
    bb = [int(v) for v in bb]
    return bb

def get_crop(mask, color, margin):
    if mask is None:
        return None
    r1, g1, b1 = color[0]
    r2, g2, b2 = color[1]
    obj_mask = (mask[:,:,0] == r1) & (mask[:,:,1] == g1) & (mask[:,:,2] == b1) \
                | (mask[:,:,0] == r2) & (mask[:,:,1] == g2) & (mask[:,:,2] == b2)
    ys, xs = np.where(obj_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    bb = [xmin - w * margin, ymin - h * margin, xmax + w * margin, ymax + h * margin]
    bb = [int(v) for v in bb]
    return bb

def make_filename(root_dir, obj_type, act_label, cam_id, seq_id, modality, frame_id):
    return os.path.join(root_dir, obj_type, act_label, 
        'cam_%s' % str(cam_id), 
        'seq_%s' % str(seq_id), 
        modality, 
        '%04d.png' % frame_id)

door_states = {
    'RR': [0, 0, 0, 1, 0, 0],
    'RF': [0, 1, 0, 0, 0, 0],
    'LR': [0, 0, -1, 0, 0, 0],
    'LF': [-1, 0, 0, 0, 0, 0],
    'Trunk': [0, 0, 0, 0, 0, 1]
}

def make_frames(act_opt, frame_count, car_type):
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


def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.isdir(folder):
        os.makedirs(folder)

def save_info(img, mask, root_dir, act_label, cam_id, seq_id, frame_id, human_mask_color, car_mask_color):
    filename = make_filename(root_dir, 'imgs', act_label, cam_id, seq_id, 'rgb', frame_id)
    mkdir(filename)
    cv2.imwrite(filename, img[:,:,[2,1,0]])
    filename = make_filename(root_dir, 'imgs', act_label, cam_id, seq_id, 'mask', frame_id)
    mkdir(filename)
    cv2.imwrite(filename, mask[:,:,[2,1,0]])

    # Crop human and car patch from raw image
    bb = get_bb(mask, car_mask_color, 0.15)
    if bb:
        x, y, w, h = bb
    else:
        x, y, w, h = 0, 0, 0, 0
    filename = make_filename(root_dir, 'imgs', act_label, cam_id, seq_id, 'car', frame_id)
    filename = filename.replace('png', 'txt')
    mkdir(filename)
    f = open(filename, 'w')
    f.write('{} {} {} {}'.format(x, y, w, h))
    f.close()

    bb = get_bb(mask, human_mask_color, 0.25)
    if bb:
        x, y, w, h = bb
    else:
        x, y, w, h = 0, 0, 0, 0
    filename = make_filename(root_dir, 'imgs', act_label, cam_id, seq_id, 'human', frame_id)
    filename = filename.replace('png', 'txt')
    mkdir(filename)
    f = open(filename, 'w')
    f.write('{} {} {} {}'.format(x, y, w, h))
    f.close()

def main():
    udb.connect('localhost', 9000)

    opt = ObjectView(dict())
    opt.DEBUG = False
    opt.capture = True

    env = CvEnv() #

    human_mask_color = [0, 128, 0]
    human = CvCharacter('human')
    human.DEBUG = opt.DEBUG
    human.spawn()
    human.set_mesh(MESH.MAN)
    r, g, b = human_mask_color
    human.set_mask_color(r, g, b)

    car_mask_color = [128, 0, 0]
    car = CvCar('car')
    car.DEBUG = opt.DEBUG
    car.spawn()
    car_type = 'suv'
    car.set_mesh(car_type)
    car.set_loc(-200,-200,30)
    r, g, b = car_mask_color
    car.set_mask_color(r, g, b)

    cam = CvCamera('cam')
    # traj = pickle.load(open('cam_traj_1004.p','rb'))
    traj = {0: (183.022, -521.394, 896.025, -60, 500.0, 0.0)}

    occluder1 = CvCar('occ1')
    occluder1.spawn()
    car_type = 'hatchback'
    occluder1.set_mesh(car_type)
    occluder1.set_loc(np.random.randint(75,125), np.random.randint(-250,-150), 30)

    occluder2 = CvCar('occ2')
    occluder2.spawn()
    car_type = 'sedan2door'
    occluder2.set_mesh(car_type)
    occluder2.set_loc(np.random.randint(-525, -475), np.random.randint(-250, -150), 30)

    root_dir = '.'

    for seq_id, act_opt in enumerate(anim_paths):
        for cam_id in range(len(traj)):
            cam = CvCamera('1')
            cam.DEBUG = True
            cam.spawn()
            cam.set_loc(traj[cam_id][0], traj[cam_id][1], traj[cam_id][2])
            cam.set_rot(traj[cam_id][3], traj[cam_id][4], traj[cam_id][5])
            cam.DEBUG = opt.DEBUG
            cam.spawn() 


            frame_count = random.randint(32, 64)
            frames = make_frames(act_opt, frame_count, car_type)

            frame = frames[0]
            mesh_path = frame['human']['mesh_path']
            human.set_mesh(mesh_path)
            if act_opt[0] == MESH.GIRL1:
                #         meshes = ["SkeletalMesh'/Game/Girl_01/meshes/girl_01_a.girl_01_a'", "SkeletalMesh'/Game/Girl_01/meshes/girl_01_b.girl_01_b'", \
                #                 "SkeletalMesh'/Game/Girl_01/meshes/girl_01_c.girl_01_c'", "SkeletalMesh'/Game/Girl_01/meshes/girl_01_e.girl_01_e'", \
                #                 "SkeletalMesh'/Game/Girl_01/meshes/girl_01_f.girl_01_f'", "SkeletalMesh'/Game/Girl_01/meshes/girl_01_h.girl_01_h'", \
                #                 "SkeletalMesh'/Game/Girl_01/meshes/girl_01_i.girl_01_i'"]
                #         mesh = random.choice(meshes)
                #         actor_name = self.id
                #         self.request('vset /human/{actor_name}/mesh {mesh}'.format(**locals()))
                # human.set_girl_random_texture()
                pass

            # material_filename = np.random.choice(texture_images)
            # car.set_texture(material_filename)
            # material_filename = np.random.choice(texture_images)
            # occluder1.set_texture(material_filename)
            # material_filename = np.random.choice(texture_images)
            # occluder2.set_texture(material_filename)
            # material_filename = np.random.choice(texture_images)
            # env.set_floor_texture(material_filename)

            # flip the action
            if act_opt[5]:
                if act_opt[1] in [ANIM.OpenEnterClose3, ANIM.ManOpenExitClose3, ANIM.ManOpenEnterClose3]: # these actions flip by y axis
                    human.set_scale(1, -1, 1)
                else: # these actions flip by x axis
                    human.set_scale(-1, 1, 1)
            else:
                human.set_scale(1, 1, 1)

            Xmin, Ymin, Xmax, Ymax = None, None, None, None
            crop_imgs = []
            crop_masks = []

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
                human.set_animation(animation_path, ratio)
                human.set_loc(x, y, z)
                human.set_rot(pitch, yaw, roll)

                # Get images
                img = cam.get_rgb()
                mask = cam.get_mask()
                bb = get_crop(mask, [human_mask_color,car_mask_color], 0.15)
                if bb:
                    xmin, ymin, xmax, ymax = bb
                    crop_imgs.append(img[ymin:ymax, xmin:xmax])
                    crop_masks.append(mask[ymin:ymax, xmin:xmax])
                    if Xmin == None or Xmin > xmin:
                        Xmin = xmin
                    if Ymin == None or Ymin > ymin:
                        Ymin = ymin
                    if Xmax == None or Xmax < xmax:
                        Xmax = xmax
                    if Ymax == None or Ymax < ymax:
                        Ymax = ymax

            for frame_id in range(len(crop_imgs)):
                # img = crop_imgs[frame_id]
                # mask = crop_masks[frame_id]
                img = cv2.resize(crop_imgs[frame_id], (Xmax-Xmin, Ymax-Ymin),interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(crop_masks[frame_id], (Xmax-Xmin, Ymax-Ymin),interpolation=cv2.INTER_CUBIC)
                if frame_id / len(frames) < 0.3 and act_label in ['OpenExitClose', 'OpenEnterClose']:
                    save_info(img, mask, root_dir, 'Opening', cam_id, seq_id, frame_id, human_mask_color, car_mask_color)
                if frame_id / len(frames) > 0.6 and act_label in ['OpenExitClose', 'OpenEnterClose']:
                    save_info(img, mask, root_dir, 'Closing', cam_id, seq_id, frame_id, human_mask_color, car_mask_color)     
                if frame_id / len(frames) < 0.5 and act_label == 'Exiting':
                    save_info(img, mask, root_dir, 'Opening', cam_id, seq_id, frame_id, human_mask_color, car_mask_color)
                if frame_id / len(frames) > 0.5 and act_label == 'Entering':
                    save_info(img, mask, root_dir, 'Closing', cam_id, seq_id, frame_id, human_mask_color, car_mask_color)   

                if act_label == 'OpenExitClose':
                    if frame_id / len(frames) >= 0.3 and frame_id / len(frames) <= 0.6:
                        save_info(img, mask, root_dir, 'Exiting', cam_id, seq_id, frame_id, human_mask_color, car_mask_color)  
                elif act_label == 'OpenEnterClose':
                    if frame_id / len(frames) >= 0.3 and frame_id / len(frames) <= 0.6:
                        save_info(img, mask, root_dir, 'Entering', cam_id, seq_id, frame_id, human_mask_color, car_mask_color)  
                else:
                    save_info(img, mask, root_dir, act_label, cam_id, seq_id, frame_id, human_mask_color, car_mask_color)   

if __name__ == '__main__':
    main()
