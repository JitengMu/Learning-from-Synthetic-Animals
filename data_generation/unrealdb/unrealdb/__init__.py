__all__ = ["d3", "asset"]

import os
import random
import unrealcv
import numpy as np
from unrealcv.util import read_png, read_npy
import time
from tqdm import tqdm

ip = 'localhost'
port = 9000
client = None

def connect(ip = 'localhost', port = 9000):
    global client
    client = unrealcv.Client((ip, port))
    client.connect()

def mask_img(im, mask):
    im = im.copy()
    for i in range(3):
        channel = im[:,:,i]
        channel[~mask] = 0
        # im[:,:,i] = channel
    return im

def get_mask(mask, color):
    if mask is None:
        return None
    r, g, b = color
    obj_mask = (mask[:,:,0] == r) & (mask[:,:,1] == g) & (mask[:,:,2] == b)
    return obj_mask

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
    bb = [xmin - w * margin, ymin - h * margin, xmax + w * margin, ymax + h * margin]
    bb = [int(v) for v in bb]
    return bb

def make_filename(root_dir, obj_type, act_label, cam_id, seq_id, modality, frame_id):
    return os.path.join(root_dir, obj_type, act_label, 
        'cam_%s' % str(cam_id), 
        'seq_%s' % str(seq_id), 
        modality, 
        '%04d.png' % frame_id)

def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.isdir(folder):
        os.makedirs(folder)

class ClientWrapper:
    ''' Wrapper for unrealcv client to benchmark performance '''
    def __init__(self, endpoint):
        self.client = unrealcv.Client(endpoint)
        self.logs = []
        self.DEBUG = False
        self.benchmark = True
    
    def connect(self):
        self.client.connect()

    def request(self, req):
        tic = time.time()
        res = self.client.request(req)
        toc = time.time()
        cmd = ' '.join(req.split(' ')[0:2])
        self.logs.append([cmd, req, toc - tic])
        return res

def print_summary(logs):
    cmds = set([v[0] for v in logs])
    time = { cmd: [] for cmd in cmds}
    for log in logs:
        cmd = log[0]
        time[cmd].append(log[2])
    
    for cmd in cmds:
        cmd_time = np.array(time[cmd])
        count = len(cmd_time)
        tmin = min(cmd_time)
        tmax = max(cmd_time)
        tmean = np.mean(cmd_time)
        sum = np.sum(cmd_time)
        print('{cmd:40s} {tmean:5.2f} {tmin:5.2f} {tmax:5.2f} {count:4d} {sum:8.2f}'.format(**locals()))

    
# client = ClientWrapper(('localhost', 9000))
# client = unrealcv.Client(('localhost', 9000))
# client.connect()

class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d

class CvObject:
    def __init__(self, id):
        self.id = id
        self.DEBUG = False
        self.client = client

    def request(self, req):
        res = self.client.request(req)
        if self.DEBUG:
            print(req)
            print(res[:100])
        return res

    def spawn(self):
        obj_id = self.id
        obj_type = self.type
        self.request('vset /objects/spawn {obj_type} {obj_id}'.format(**locals()))

    def destroy(self):
        id = self.id
        self.request('vset /object/{id}/destroy'.format(**locals()))

    def set_mask_color(self, r, g, b):
        id = self.id
        self.request('vset /object/{id}/color {r} {g} {b}'.format(**locals()))

    def set_rot(self, pitch, yaw, roll):
        id = self.id
        self.request('vset /object/{id:s}/rotation {pitch:d} {yaw:d} {roll:d}'.format(**locals()))

    def set_loc(self, x, y, z):
        id = self.id
        self.request('vset /object/{id:s}/location {x:d} {y:d} {z:d}'.format(**locals()))

    def set_scale(self, x, y, z):
        id = self.id
        self.request('vset /object/{id:s}/scale {x} {y} {z}'.format(**locals()))

    def get_rot(self):
        id = self.id
        res = client.request('vget /object/{id}/rotation'.format(**locals()))
        pitch, yaw, roll = [float(item) for item in res.split(' ')]
        return pitch, yaw, roll

    def get_loc(self):
        id = self.id
        res = client.request('vget /object/{id}/location'.format(**locals()))
        x, y, z = [float(item) for item in res.split(' ')]
        return x,y,z

class CvMesh(CvObject):
    def __init__(self, id):
        super().__init__(id)
        self.type = 'CvStaticMesh'

    def set_mesh(self, mesh_path):
        id = self.id
        self.request('vset /object/{id}/mesh {mesh_path}'.format(**locals()))
    
class CvCharacter(CvObject):
    def __init__(self, id):
        super().__init__(id)
        self.type = 'CvCharacter'

    def set_mesh(self, mesh_id):
        actor_name = self.id
        self.request('vset /human/{actor_name}/mesh {mesh_id}'.format(**locals()))

    def set_animation(self, anim_path, ratio):
        actor_name = self.id
        self.request('vset /human/{actor_name}/animation/ratio {anim_path} {ratio}'.format(**locals()))

    def get_animation_frames(self, anim_path):
        ''' Return number of frames '''
        obj_id = self.id
        res = self.request('vget /human/{obj_id}/animation/frames {anim_path}'.format(**locals()))
        frames = int(res)
        return frames

    def set_texture(self, texture):
        obj = self.id
        self.request('vset /animal/{obj}/texture {texture}'.format(**locals()))

    def set_tracking_camera(self, dist, az, el):
        obj_id = self.id
        self.request('vset /animal/{obj_id}/camera {dist} {az} {el}'.format(**locals()))

    def get_img(self):
        obj_id = self.id
        img = self.request('vget /animal/{obj_id}/image'.format(**locals()))
        img = read_png(img)
        return img

    def get_seg(self):
        obj_id = self.id
        seg = self.request('vget /animal/{obj_id}/seg'.format(**locals()))
        seg = read_png(seg)
        return seg

    def get_depth(self):
        obj_id = self.id
        depth = self.request('vget /animal/{obj_id}/depth'.format(**locals()))
        depth = read_npy(depth)
        return depth

    def get_2d_kp(self):
        obj = self.id
        res = self.request('vget /animal/{obj}/keypoint'.format(**locals())) 
        kp = json.loads(res)
        return kp

    def get_3d_kp(self):
        obj = self.id
        res = self.request('vget /animal/{obj}/3d_keypoint'.format(**locals())) 
        kp_3d = json.loads(res)
        return kp_3d

    def get_vertex(self):
        obj = self.id
        res = self.request('vget /animal')

class CvAnimal(CvCharacter):
    def set_material(self, element_index, material_path):
        obj = self.id
        res = self.request('vset /animal/{obj}/material {element_index} {material_path}'.format(**locals()))

class CvHuman(CvCharacter):
    def set_girl_random_texture(self):
        meshes = ['girl_a', 'girl_b', 'girl_c', 'girl_d', 'girl_e', 'girl_f', 'girl_g', 'girl_h', 'girl_i', 'girl_j']
        mesh = random.choice(meshes)
        actor_name = self.id
        self.request('vset /human/{actor_name}/mesh {mesh}'.format(**locals()))


class CvCar(CvObject):
    def __init__(self, id):
        super().__init__(id)
        self.type = 'CvShapenetCar'

    def set_part_angles(self, frontleft, frontright, rearleft, rearright, hood, trunk):
        id = self.id
        self.request('vset /car/{id}/door/angles {frontleft} {frontright} {rearleft} {rearright} {hood} {trunk}'.format(**locals()))

    def set_color(self, R, G, B):
        id = self.id
        self.request('vset /car/{id}/color {R} {G} {B}'.format(**locals()))

    def set_texture(self, texture_file):
        id = self.id
        self.request('vset /car/{id}/mesh/texture {texture_file}'.format(**locals()))

    def set_mesh(self, mesh_id):
        id = self.id
        self.request('vset /car/{id}/mesh/id {mesh_id}'.format(**locals()))

    def set_shapenet_folder(self, mesh_folder, meta_folder):
        id = self.id
        self.request('vset /car/{id}/mesh/folder {mesh_folder} {meta_folder}'.format(**locals()))

    def set_tracking_camera(self, dist, az, el):
        obj_id = self.id
        self.request('vset /car/{obj_id}/camera {dist} {az} {el}'.format(**locals()))

    def get_img(self):
        obj_id = self.id
        img = self.request('vget /car/{obj_id}/image'.format(**locals()))
        img = read_png(img)
        return img

    def get_seg(self):
        obj_id = self.id
        seg = self.request('vget /car/{obj_id}/seg'.format(**locals()))
        seg = read_png(seg)
        return seg

class CvEnv(CvObject):
    def __init__(self):
        super().__init__("")
        
    # def set_floor_color(self, R, G, B):
    #     actor_name = self.id
    #     self.request('vset /mesh/{actor_name}/color {R} {G} {B}'.format(**locals()))

    # def set_floor_texture(self, texture_file):
    #     actor_name = self.id
    #     self.request('vset /mesh/{actor_name}/texture {texture_file}'.format(**locals()))

    def set_floor(self, floor):
        self.request('vset /env/floor/texture {floor}'.format(**locals()))

    def set_sky(self, sky):
        self.request('vset /env/sky/texture {sky}'.format(**locals()))

    def set_random_light(self):
        self.request('vset /env/light/random')


class CvCamera(CvObject):
    def __init__(self, id):
        super().__init__(id)
        self.type = 'FusionCameraActor'

    def get_rgb(self):
        id = self.id
        res = self.request('vget /camera/{id}/lit png'.format(**locals()))
        return read_png(res)

    def get_img(self):
        return self.get_rgb()

    def get_mask(self):
        id = self.id
        res = self.request('vget /camera/{id}/object_mask png'.format(**locals()))
        return read_png(res)
    
    def get_seg(self):
        return self.get_mask()

    def get_depth(self):
        id = self.id
        res = self.request('vget /camera/{id}/depth npy'.format(**locals()))
        return read_npy(res)

    def set_rot(self, pitch, yaw, roll):
        id = self.id
        self.request('vset /camera/{id}/rotation {pitch} {yaw} {roll}'.format(**locals()))

    def set_loc(self, x, y, z):
        id = self.id
        self.request('vset /camera/{id}/location {x} {y} {z}'.format(**locals()))

    def set_size(self, width, height):
        id = self.id
        self.request('vset /camera/{id}/size {width} {height}'.format(**locals()))

    def track(self, obj_id, dist, az, el):
        id = self.id
        self.request('vset /camera/{id}/track {obj_id} {dist} {az} {el}')

class TextureSampler:
    def __init__(self, root):
        self.root = root

    def sample_random_texture(self):
        # TODO: Fix this
        if not os.path.isdir(self.root): return ''
        samples = os.listdir(self.root)
        random.shuffle(samples)
        random_sample = samples[0]
        imgs = os.listdir(os.path.join(self.root, random_sample))
        random.shuffle(imgs)
        random_img = imgs[0]
        return os.path.join(self.root,random_sample,random_img)


if __name__ == '__main__':
    main()
