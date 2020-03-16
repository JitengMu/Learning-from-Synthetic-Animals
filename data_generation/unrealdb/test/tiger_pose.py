import unrealdb as udb
import unrealdb.asset
from tqdm import tqdm
import imageio
import numpy as np
import os
import time

params = [
    ["AnimSequence'/Game/Animal_pack_ultra_2/Animations/tiger_attackA_anim.tiger_attackA_anim'", 0.3, 'attack'],
    ["AnimSequence'/Game/Animal_pack_ultra_2/Animations/tiger_death_anim.tiger_death_anim'", 0.4, 'death'],
    ["AnimSequence'/Game/Animal_pack_ultra_2/Animations/tiger_roar_anim.tiger_roar_anim'", 0.4, 'roar'],
    ["AnimSequence'/Game/Animal_pack_ultra_2/Animations/tiger_run_anim.tiger_run_anim'", 0.5, 'run'],
]

udb.connect('localhost', 9090)

map_name = 'AnimalDataCapture'
udb.client.request('vset /action/game/level {map_name}'.format(**locals()))
udb.client.request('vset /camera/0/location 400 0 300')
udb.client.request('vset /camera/0/rotation 0 180 0')

obj_id = 'tiger'
animal = udb.CvAnimal(obj_id)
animal.spawn()

r, g, b = 155, 168, 157
animal.set_mask_color(r, g, b)
animal.set_mesh(udb.asset.MESH_TIGER)
animal.set_tracking_camera(350, 0, -5)

for i, param in enumerate(tqdm(params)):
    anim, ratio, key = param
    animal.set_animation(anim, ratio)
    # animal.set_tracking_camera(350, 0, -5)
    # time.sleep(5)
    print(animal.get_animation_frames(anim))

    img = animal.get_img()
    seg = animal.get_seg()
    depth = animal.get_depth()
    mask = udb.get_mask(seg, [r,g,b])
    obj_img = udb.mask_img(img, mask)

    imageio.imwrite('%s_im.png' % key, img)
    imageio.imwrite('%s_seg.png' % key, seg)
    np.save('%s_depth.npy' % key, depth)
    imageio.imwrite('%s_mask.png' % key, obj_img)
