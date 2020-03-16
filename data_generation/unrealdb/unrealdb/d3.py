# Weichao Qiu @ 2018
import numpy as np

def make_translation(X, Y, Z):
    ''' Make translation matrix '''
    T = np.matrix(
    [
        [1, 0, 0, X],
        [0, 1, 0, Y],
        [0, 0, 1, Z],
        [0, 0, 0, 1]
    ]
    ).transpose()
    return T

def make_location(X, Y, Z):
    T = np.matrix(
    [
        [X],
        [Y],
        [Z]
    ])
    return T

def make_rotation(pitch, yaw, roll):
    # Convert from degree to radius
    pitch = pitch / 180.0 * np.pi
    yaw = yaw / 180.0 * np.pi
    roll = roll / 180.0 * np.pi
    pitch = -pitch
    yaw = yaw # ???!!!
    roll = -roll # Seems UE4 rotation direction is different
    # from: http://planning.cs.uiuc.edu/node102.html
    ryaw = [
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    rpitch = [
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ]
    rroll = [
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ]
    T = np.matrix(ryaw) * np.matrix(rpitch) * np.matrix(rroll)
    return T.transpose()

class CameraPose:
    def __init__(self, x, y, z, pitch, yaw, roll, width, height, f):
        self.x, self.y, self.z = x, y, z
        self.pitch, self.yaw, self.roll = pitch, yaw, roll
        self.width, self.height, self.f = width, height, f

    def __repr__(self):
        print(self.__dict__)
        return 'x:{x} y:{y} z:{z} pitch:{pitch} yaw:{yaw} roll:{roll} width:{width} height:{height} f:{f}'.format(**self.__dict__)

    # The points_3d is in the canonical coordinate
    def project_to_2d(self, points_3d):
        ''' 
        points_3d: points in 3D world coordinate, n * 3
        cam_pose: camera pose
        '''
        if not points_3d.shape[1] == 3:
            print('The input shape is not n x 3, but n x %d' % points_3d.shape[1])
            # TODO: replace this with logging
            return

        npoints = points_3d.shape[0]
        points_3d = np.concatenate((points_3d, np.ones((npoints, 1))), axis = 1)

        cam_T = make_translation(self.x, self.y, self.z) 
        cam_R = make_rotation(self.pitch, self.yaw, self.roll)

        world_to_camera = (cam_R * cam_T).getI() # inverse matrix
        points_3d_camera = points_3d * world_to_camera

        # TODO: Need to fix this
        half_width = self.width / 2
        half_height = self.height / 2

        n_points = points_3d.shape[0]
        points_2d = np.zeros((n_points, 2))

        points_2d[:,0] = (points_3d_camera[:,1] / points_3d_camera[:,0] * self.f + half_width).squeeze()
        points_2d[:,1] = (- (points_3d_camera[:,2]) / points_3d_camera[:,0] * self.f + half_height).squeeze()

        return points_2d 

    def project_to_cam_space(self, points_3d):
        if not points_3d.shape[1] == 3:
            print('The input shape is not n x 3, but n x %d' % points_3d.shape[1])
            # TODO: replace this with logging
            return

        npoints = points_3d.shape[0]
        points_3d = np.concatenate((points_3d, np.ones((npoints, 1))), axis = 1)

        cam_T = make_translation(self.x, self.y, self.z) 
        cam_R = make_rotation(self.pitch, self.yaw, self.roll)

        world_to_camera = (cam_R * cam_T).getI() # inverse matrix
        points_3d_camera = points_3d * world_to_camera

        half_width = self.width / 2
        half_height = self.height / 2

        n_points = points_3d.shape[0]
        points_3d = np.zeros((n_points, 3))
        points_3d[:,0] = (points_3d_camera[:,1] / points_3d_camera[:,0] * self.f + half_width).squeeze()
        points_3d[:,1] = (- (points_3d_camera[:,2]) / points_3d_camera[:,0] * self.f + half_height).squeeze()
        points_3d[:,2] = points_3d_camera[:,0].squeeze()

        return points_3d


def test():
    # x, y, z = # Get camera location in world coordinate
    # pitch, yaw, roll, # camera rotation
    # width, height =  # image width 
    # f  = width / 2 # FOV = 90
    cam_pose = CameraPose(x, y, z, pitch, yaw, rool, width, height, width / 2)

    points_2d = cam_pose.project_to_2d(points_3d)  # list of 2d point, x, y
    points_3d_cam = cam_pose.project_to_cam_space(points_3d)
    # x, y, z # z means distance to image plane.

    depth = 0 # Get depth image from the simulator. w x h x 1 float array.
    epsilon = 10

    for x, y, z in points_3d_cam:
        x = int(x)
        y = int(y)
        real_z = depth[y][x]
        if abs(real_z - z) < epsilon:
            print('visible')
        else:
            print('invisible')


    # points_3d = # read 3D keypoint from AnimalParsing

    pass



if __name__ == '__main__':
    test()