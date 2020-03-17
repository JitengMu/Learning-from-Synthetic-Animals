import os, time, glob, itertools, random, imageio, re
import numpy as np
from tqdm import tqdm
from unrealcv.util import read_png, read_npy

import unrealdb as udb
import unrealdb.asset
import unrealdb.asset.animal

from unrealdb import d3
from PIL import Image
import pdb
import argparse

def make_filename(num_img, mesh, anim, time, dist, az, el):
    def get_mesh_name(mesh_path):
        re_mesh = re.compile("SkeletalMesh'.*/(.*)\.(.*)'")
        match = re_mesh.match(mesh_path)
        return match.group(1)

    def get_anim_name(anim_path):
        re_anim = re.compile("AnimSequence'.*/(.*)\.(.*)'")
        match = re_anim.match(anim_path)
        return match.group(1)

    mesh_name = get_mesh_name(mesh)
    anim_name = get_anim_name(anim)
    template = '{num_img}_{mesh_name}_{anim_name}_{time:.2f}_{dist:.2f}_{az:.2f}_{el:.2f}.png'
    filename = template.format(**locals())
    return filename

def glob_images(image_folder):
    filenames = glob.glob(os.path.join(image_folder, '*.jpg'))
    filenames += glob.glob(os.path.join(image_folder, '*.png'))
    return filenames

def load_render_params(global_animal):
    if global_animal=='tiger':
        opt = dict(
            mesh = [udb.asset.MESH_TIGER],
            anim = udb.asset.tiger_animations,
            ratio = np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist = [150, 200, 250, 300, 350],
            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el = [0, 10, 20, 30, 150, 160, 170, 180]
        )

    elif global_animal=='horse':
        opt = dict(
            mesh = [udb.asset.MESH_HORSE],
            anim = udb.asset.horse_animations,
            ratio = np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist = [250, 300, 350, 400, 450],
            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el = [0, 10, 20, 30, 150, 160, 170, 180]
        )

    elif global_animal=='domestic_sheep':
        opt = dict(
            mesh = [udb.asset.MESH_DOMESTIC_SHEEP],
            anim = udb.asset.domestic_sheep_animations,
            ratio = np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist = [150, 200, 250, 300, 350],
            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el = [0, 10, 20, 30, 150, 160, 170, 180]
        )

    elif global_animal=='hellenic_hound':
        opt = dict(
            mesh = [udb.asset.MESH_HELLENIC_HOUND],
            anim = udb.asset.hellenic_hound_animations,
            ratio = np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist = [100, 125, 150, 200, 250],
            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el = [0, 10, 20, 30, 150, 160, 170, 180]
        )

    elif global_animal=='elephant':
        opt = dict(
            mesh = [udb.asset.MESH_ELEPHANT],
            anim = udb.asset.elephant_animations,
            ratio = np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist = [350, 400, 450, 500, 550],
            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el = [0, 10, 20, 30, 150, 160, 170, 180]
        )

    elif global_animal=='bat':
        opt = dict(
            mesh = [udb.asset.MESH_BAT],
            anim = udb.asset.bat_animations,
            ratio = np.arange(0.1, 0.9, 0.05),
            # ratio = [0],
            dist = [75, 100, 125, 150, 200],
            az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
                190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
            el = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        )

    render_params = itertools.product(opt['mesh'], opt['anim'], opt['ratio'],
        opt['dist'], opt['az'], opt['el'])
    render_params = list(render_params)
    return render_params

def parse_kpts(filename, offset):
    res = udb.client.request('vget /animal/tiger/vertex obj'.format(**locals()))
    data = res.strip().split('\n')
    kpts_3d_array = np.zeros((3299, 3))
    for i, line in enumerate(data):
        _, x, y, z = line.split(' ')
        kpts_3d_array[i] = float(x), float(y), float(z)+offset-80
    return kpts_3d_array

def get_camera_params():
    cam_loc = udb.client.request('vget /camera/1/location'.format(**locals()))
    cam_rot = udb.client.request('vget /camera/1/rotation'.format(**locals()))
    return cam_loc, cam_rot

def transform_kpts(cam_loc, cam_rot, kpts_3d, depth_img):
    # x, y, z = # Get camera location in world coordinate
    # pitch, yaw, roll, # camera rotation
    # width, height =  # image width 
    # f  = width / 2 # FOV = 90
    width = 640
    height = 480
    x, y, z = cam_loc
    pitch, yaw, roll = cam_rot
    cam_pose = d3.CameraPose(x, y, z, pitch, yaw, roll, width, height, width / 2)

    # points_2d = cam_pose.project_to_2d(points_3d)  # list of 2d point, x, y
    points_3d_cam = cam_pose.project_to_cam_space(kpts_3d)
    # x, y, z # z means distance to image plane.
    depth = depth_img # Get depth image from the simulator. w x h x 1 float array.
    epsilon = 15
    kpts_2d = points_3d_cam[:,:2]
    kpts_z = points_3d_cam[:,2]

    vis = np.zeros((kpts_3d.shape[0], 1))
    for i, (x, y, z) in enumerate(points_3d_cam):
        x = int(x)
        y = int(y)
        if y<0 or y>=480 or x<0 or x>=640:
            vis[i] = 0
        else:
            real_z = depth[y][x]
            if abs(real_z - z) < epsilon:
                # print(abs(real_z - z))
                vis[i] = 1
            else:
                # print(abs(real_z - z))
                vis[i] = 0 

    # points_3d = # read 3D keypoint from AnimalParsing
    kpts = np.hstack((kpts_2d, vis))
    return kpts, kpts_z

def main(args):
    udb.connect('localhost', 9900)

    global_animal=args.animal

    # reset the program
    map_name = 'AnimalDataCapture'
    udb.client.request('vset /action/game/level {map_name}'.format(**locals()))
    udb.client.request('vset /camera/0/location 500 0 300')
    udb.client.request('vset /camera/0/rotation -20 180 0')


    val2017_dir = os.path.abspath(args.random_texture_path)
    bg_path_list = glob_images(val2017_dir)
    texture_path_list = glob_images(val2017_dir)

    render_params = load_render_params(global_animal)
    random.shuffle(render_params)

    obj_id = 'tiger'
    animal = udb.CvAnimal(obj_id)
    animal.spawn()

    # acquire offset
    obj_loc = udb.client.request('vget /object/tiger/location')
    obj_loc = [float(v) for v in obj_loc.split(' ')]
    offset = obj_loc[2]

    r, g, b = 155, 168, 157
    animal.set_mask_color(r, g, b)
    if global_animal=='tiger':
        animal.set_mesh(udb.asset.MESH_TIGER)
    elif global_animal=='horse':
        animal.set_mesh(udb.asset.MESH_HORSE)
    elif global_animal=='domestic_sheep':
        animal.set_mesh(udb.asset.MESH_DOMESTIC_SHEEP)
    elif global_animal=='hellenic_hound':
        animal.set_mesh(udb.asset.MESH_HELLENIC_HOUND)
    elif global_animal=='elephant':
        animal.set_mesh(udb.asset.MESH_ELEPHANT)

    env = udb.CvEnv()

    output_dir = args.output_path
    if not os.path.isdir(output_dir): os.makedirs(output_dir)

    img_idx = 0
    for i, param in enumerate(tqdm(render_params)):
        mesh, anim, ratio, dist, az, el = param
        filename = make_filename(img_idx, mesh, anim, ratio, dist, az, el)

        sky_texture = random.choice(bg_path_list)
        floor_texture = random.choice(bg_path_list)
        animal_texture = random.choice(texture_path_list)

        # Update the scene
        env.set_random_light()
        env.set_floor(floor_texture)
        env.set_sky(sky_texture)
        if args.use_random_texture:
            animal.set_texture(animal_texture) 
        animal.set_animation(anim, ratio)

#        if global_animal=='horse':
#            # set different original textures
#            _, animal_texture = random.choice(list(udb.asset.animal.horse_material.items()))
#            _, animal_texture_fur = random.choice(list(udb.asset.animal.horse_material.items()))
#            animal.set_material(0, animal_texture)
#            animal.set_material(1, animal_texture_fur)

        # Capture data
        animal.set_tracking_camera(dist, az, el)
        img = animal.get_img()
        seg = animal.get_seg()
        depth = animal.get_depth()
        mask = udb.get_mask(seg, [r,g,b])

        # get kpts
        ## get cam_loc and cam_rot
        cam_loc, cam_rot = get_camera_params()
        cam_loc = [float(item) for item in cam_loc.split(' ')]
        cam_rot = [float(item) for item in cam_rot.split(' ')]

        ## transform keypoints
        kp_3d_array = parse_kpts(filename, offset)
        kpts, kpts_z = transform_kpts(cam_loc, cam_rot, kp_3d_array, depth)

        ## transform images and kpts
        img = Image.fromarray(img[:,:,:3])
        seg_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        seg_mask[mask==False] = 0 # tiger/horse
        seg_mask[mask==True] = 255 # tiger/horse

        # # save imgs
        if global_animal=='tiger':
            kp_18_id = [2679,2753,2032,1451,1287,3085,1632,229,1441,1280,2201,1662,266,158,270,152,219,129]
        elif global_animal=='horse':
            kp_18_id = [1718,1684,1271,1634,1650,1643,1659,925,392,564,993,726,1585,1556,427,1548,967,877]
        elif global_animal=='domestic_sheep':
            kp_18_id = [2046,1944,1267,1875,1900,1868,1894,687,173,1829,1422,821,624,580,622,575,1370,716]
        elif global_animal=='hellenic_hound':
            kp_18_id = [2028,2580,912,878,977,1541,1734,480,799,1575,1446,602,755,673,780,1580,466,631]
        elif global_animal=='elephant':
            kp_18_id = [1980,2051,1734,2122,2155,2070,2166,681,923,1442,1041,1528,78,599,25,595,171,570]

        if sum(kpts[kp_18_id,2])>=6:
            imageio.imwrite(os.path.join(output_dir, filename + '_img.png'), img)
            imageio.imwrite(os.path.join(output_dir, filename + '_seg.png'), seg_mask)
            np.save(os.path.join(output_dir, filename + '_depth.npy'), depth)
            np.save(os.path.join(output_dir, filename + '_kpts.npy'), kpts)
            np.save(os.path.join(output_dir, filename + '_kpts_z.npy'), kpts_z)

            img_idx += 1
            if img_idx>args.num_imgs-1:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic Animal Dataset Generation')
    parser.add_argument('--animal', default='horse', type=str,
                       help='horse | tiger | sheep | hound | elephant')
    parser.add_argument('--num-imgs', default=10, type=int,
                       help='output resolution (default: 10, to gen GT)')
    parser.add_argument('--random-texture-path', default='./data_generation/val2017', type=str,
                       help='coco val 2017')
    parser.add_argument('--output-path', default='./data_generation/generated_data', type=str,
                       help='output directory')
    parser.add_argument('--use-random-texture', action='store_true',
                       help='whether use random texture or not')
 
    main(parser.parse_args())
