import os
from os.path import join
# from unrealcv import client
from unrealcv.util import read_png
import unrealcv
#client = unrealcv.Client(('diva8.qiu.work', 9000))
client = unrealcv.Client(('localhost', 9000))
client.connect()
DEBUG = False


def request(req): 
    res = client.request(req)
    if DEBUG:
        print('Request: ', req)
        print('Response: ', res[:100])
    return res

class Async_request():
    def __init__(self):
        self.cache = []
    def request(self,req):
        if DEBUG:
            print('Request: ', req)
        if type(req) is list:
            self.cache += req
        else:
            self.cache.append(req)
        return True
    def flush(self):
        client.request(self.cache)
        self.cache = []
        return True

def clear_env(map_name='ParkingCapture', cam_loc=[-8860.254037844386, -0.0, 4788.589838486224], cam_rot=[-30, 0, 0]):
    res = request('vset /action/game/level {map_name}'.format(**locals()))
    # Set the default camera location
    cmd = [
        'vset /camera/0/location {:.7f} {:.7f} {:.7f}'.format(cam_loc[0],cam_loc[1],cam_loc[2]),
        'vset /camera/0/rotation {:.7f} {:.7f} {:.7f}'.format(cam_rot[0],cam_rot[1],cam_rot[2])
    ]
    request(cmd)

def spawn_car_cmd(shapnet_id, car_id, scale, shape_lib):

    cmds = ['vset /objects/spawn CvShapenetCar {}'.format(car_id)]
    cmds += ['vset /car/{}/mesh/folder /Game/{}/ x'.format(car_id, shape_lib)]
    cmds += ['vset /car/{}/mesh/id '.format(car_id)+ shapnet_id]
    cmds += ['vset /object/{}/rotation 0 0 0'.format(car_id)]
    cmds += ['vset /object/{}/location 0 500 100'.format(car_id)]
    cmds += ['vset /object/{}/scale {} {} {}'.format(car_id, scale[0], scale[1], scale[2])]
    return cmds

class Car_Manager(object):

    def __init__(self):
        self.num_obj = 0
        self.car_ids = []
        self.car_rot = {}
        self.car_trans = {}
        self.car_color = {}
        self.async_connect = Async_request()
        
    def add_car(self, shapnet_id, scale=(1,1,1), shape_lib = "ShapenetAutomatic"):

        car_id = 'car{}'.format(self.num_obj)
        self.async_connect.request(spawn_car_cmd(shapnet_id, car_id, scale, shape_lib))
        self.car_rot[car_id] = [0,0,0]
        self.car_trans[car_id] = [0,500,100]
        self.car_color[car_id] = [0, 0, 0]
        self.num_obj += 1

    def trans_car(self, trans, car_id):

        assert car_id in self.car_trans.keys()
        assert len(trans) == 3

        self.async_connect.request('vset /object/{}/location {:.7f} {:.7f} {:.7f}'.format(car_id, trans[0], trans[1], trans[2]) )
        self.car_trans[car_id] = trans
        
    def rot_car(self, rot, car_id):

        assert car_id in self.car_rot.keys()
        assert len(rot) == 3

        self.async_connect.request('vset /object/{}/rotation {:.7f} {:.7f} {:.7f}'.format(car_id, rot[0], rot[1], rot[2]))
        self.car_rot[car_id] = rot

    def annotate_car(self, color, car_id):
        assert car_id in self.car_color.keys()
        assert len(color) == 3

        self.async_connect.request('vset /object/{}/color {:d} {:d} {:d}'.format(car_id, color[0], color[1], color[2]))
        self.car_color[car_id] = color


    def flush(self):
        self.async_connect.flush()