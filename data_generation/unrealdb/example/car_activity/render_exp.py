import os, json, time
import numpy as np

from os.path import join
# from unrealcv import client
from unrealcv.util import read_png
import cv2

import unrealcv # pip install unrealcv
from connect_utils import Car_Manager, clear_env, request
from data_io_utils import load_meta_data
import unrealdb as udb

DEBUG = True
def render_from_file(meta_data, max_indx = 5):
    clear_env()
    car_shape, car_rot, car_trans, cam_location, cam_rotation, img_fov, img_size, kpt_dict = load_meta_data(meta_data)
    request('vset /camera/1/location {:.6f} {:.6f} {:.6f}'.format(cam_location[0], cam_location[1], cam_location[2]))
    request('vset /camera/1/rotation {:.6f} {:.6f} {:.6f}'.format(cam_rotation[0], cam_rotation[1], cam_rotation[2]))
    request('vset /camera/1/fov {:.6f}'.format(img_fov) )
    
    car_group = Car_Manager()

    # if max_indx <= 0:
    #     max_indx = max_indx + len(car_shape)

    #num_obj = min([len(car_trans), max_indx, 2])
    num_obj = 2
    car_color = [[i,0,0] for i in range(num_obj)]

    shape = car_shape[0]
    # for i in range(num_obj):
    #     car_group.add_car(shape, shape_lib="ShapenetKeypoint")
    #     #car_group.trans_car(car_trans[i], "car{}".format(i))
    #     car_group.trans_car([0, 100*i,150*i+100], "car{}".format(i))
    #     car_group.rot_car([0,0,0], "car{}".format(i))
    #     car_group.annotate_car(car_color[i], "car{}".format(i))
    
    with open("example/car_activity/render_shape.json", "r") as f:
        data_render = json.load(f)
    
    trans_model = data_render[shape]["trans"]
    
    i = 0
    car_group.add_car(shape,scale=(-1,1,1))
    print(trans_model)
    car_group.trans_car([trans_model[0]*500, trans_model[2]*500, 200 + trans_model[1]*500], "car{}".format(i))
    car_group.rot_car([0,90,0], "car{}".format(i))
    car_group.annotate_car(car_color[i], "car{}".format(i))
    i = 1
    car_group.add_car(shape, shape_lib="ShapenetKeypoint")
    car_group.trans_car([0, 0, 200], "car{}".format(i))
    car_group.rot_car([0,0,0], "car{}".format(i))
    car_group.annotate_car(car_color[i], "car{}".format(i))


    car_group.flush()

    cwd_root,_ = os.path.split(meta_data)
    base_dir = join(cwd_root, "render_res")
    if DEBUG:  print("Finished write object pose data ")
    time.sleep(0.1) 
    if DEBUG:  print("Acquiring image ...")
    img = read_png(request('vget /camera/1/lit png'))
    cv2.imwrite(join(base_dir, "car_arrangment.png"), img[:,:,2::-1])

    mask = read_png(request('vget /camera/1/object_mask png'))
    cv2.imwrite(join(base_dir, "car_mask.png"), mask[:,:,2::-1])

    png = read_png(request('vget /camera/0/lit png'))
    cv2.imwrite(join(base_dir, "overview.png"), png[:,:,2::-1])

    for i in range(len(car_color)):
        obj_mask = udb.get_mask(mask, car_color[i])
        [ys, xs] = np.where(obj_mask)
        bbox = [min(xs), max(xs), min(ys), max(ys)]
        print(bbox)
        obj_img = udb.mask_img(img, obj_mask)
        bbox_img = img[min(ys):max(ys), min(xs):max(xs), :] 
        cv2.imwrite(join(base_dir, "car%d_seg.png" % i), obj_img[:,:,2::-1])
        cv2.imwrite(join(base_dir, "car%d_bbox.png" % i), bbox_img[:,:,2::-1])


    if DEBUG:  print("Finished write image to files")

    return True

if __name__ == '__main__':
    script_folder = os.path.dirname(os.path.realpath(__file__))
    render_from_file(os.path.join(script_folder, 'meta_data.json'))
    