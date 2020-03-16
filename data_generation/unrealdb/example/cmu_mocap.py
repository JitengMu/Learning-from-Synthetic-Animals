import os, time, glob, itertools, random, imageio, re
import numpy as np
from tqdm import tqdm
from unrealcv.util import read_png, read_npy

import unrealdb as udb
import unrealdb.asset
import unrealdb.asset.cmu_mocap

def load_render_params():
    opt = dict(
        mesh = [udb.asset.MESH_GIRL_A],
        anim = udb.asset.cmu_mocap.CMU_SKEL_GIRL,
        ratio = np.arange(0.1, 0.9, 0.05),
        dist = [350],
        az = [0],
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

    output_dir = os.path.abspath('./generated_data')
    mask_dir = os.path.abspath('./masked')
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    if not os.path.isdir(mask_dir): os.mkdir(mask_dir)

    render_params = load_render_params()

    obj_id = 'tiger'
    animal = udb.CvAnimal(obj_id)
    animal.spawn()

    r, g, b = 155, 168, 157
    animal.set_mask_color(r, g, b)
    param = render_params[0]
    animal.set_mesh(param[0])

    env = udb.CvEnv()

    for i, param in enumerate(tqdm(render_params)):
        mesh, anim, ratio, dist, az, el = param
        filename = '%08d' % i

        animal.set_animation(anim, ratio)
        animal.set_tracking_camera(dist, az, el)
        img = animal.get_img()

        imageio.imwrite(os.path.join(output_dir, filename + '_img.png'), img)

if __name__ == '__main__':
    main()
