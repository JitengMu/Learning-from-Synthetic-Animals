import os, time, glob, itertools, random, imageio, re
import numpy as np
from tqdm import tqdm
from unrealcv.util import read_png, read_npy

import unrealdb as udb
import unrealdb.asset

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

def load_render_params():
    opt = dict(
        mesh = [udb.asset.MESH_HORSE],
        anim = udb.asset.tiger_animations,
        ratio = np.arange(0.1, 0.9, 0.05),
        # ratio = [0],
        # dist = [350, 400, 450, 500, 550, 600],
        dist = [350],
        # az = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, \
        #        190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350],
        az = [0],
        # el = [-5, 0, 10, 20, 30, 150, 160, 170, 180]
        el = [-5]
    )
    render_params = itertools.product(opt['mesh'], opt['anim'], opt['ratio'],
        opt['dist'], opt['az'], opt['el'])
    render_params = list(render_params)
    return render_params


def main():
    udb.connect('localhost', 9000)

    # reset the program
    map_name = 'AnimalDataCapture'
    udb.client.request('vset /action/game/level {map_name}'.format(**locals()))
    udb.client.request('vset /camera/0/location 400 0 300')
    udb.client.request('vset /camera/0/rotation 0 180 0')


    val2017_dir = '/data/qiuwch/val2017'
    bg_path_list = glob_images(val2017_dir)
    texture_path_list = glob_images(val2017_dir)

    render_params = load_render_params()
    # random.shuffle(render_params)

    num_img = 0

    obj_id = 'tiger'
    animal = udb.CvAnimal(obj_id)
    animal.spawn()

    r, g, b = 155, 168, 157
    animal.set_mask_color(r, g, b)
    animal.set_mesh(udb.asset.MESH_TIGER)

    env = udb.CvEnv()

    # for delay in range(10):
    for delay in [0]:
        output_dir = os.path.join(str(delay), 'generated_data')
        mask_dir = os.path.join(str(delay), 'masked')
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        if not os.path.isdir(mask_dir): os.makedirs(mask_dir)

        for i, param in enumerate(tqdm(render_params)):
            mesh, anim, ratio, dist, az, el = param
            filename = make_filename(i, mesh, anim, ratio, dist, az, el)

            # sky_texture = random.choice(bg_path_list)
            # floor_texture = random.choice(bg_path_list)
            # animal_texture = random.choice(texture_path_list)

            # Update the scene
            # env.set_floor(floor_texture)
            # env.set_sky(sky_texture)
            # animal.set_texture(animal_texture) # this will crash
            animal.set_animation(anim, ratio)

            # Capture data
            animal.set_tracking_camera(dist, az, el)
            img = animal.get_img()
            seg = animal.get_seg()
            depth = animal.get_depth()
            mask = udb.get_mask(seg, [r,g,b])
            obj_img = udb.mask_img(img, mask)

            imageio.imwrite(os.path.join(output_dir, filename + '_img.png'), img)
            imageio.imwrite(os.path.join(output_dir, filename + '_seg.png'), seg)
            np.save(os.path.join(output_dir, filename + '_depth.npy'), depth)
            imageio.imwrite(os.path.join(mask_dir, filename + '_mask.png'), obj_img)

if __name__ == '__main__':
    main()
    # pass
