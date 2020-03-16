# Visualization for debug purpose.
# Draw the human skeleton in 3D.
# Use matplotlib as a easy starter

# import sys
# sys.path.append('.')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mocap.bvh
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm
from vis import FigureVideoRecorder

class PltCanvas:
    def __init__(self):
        self.clear()

    def view_front(self):
        azimuth = -90
        elevation = 90
        self.ax.view_init(elevation, azimuth)
    
    def clear(self):
        # self.ax.clear()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.axis('equal')
        self.fig = fig
        self.ax = ax
        self.view_front()
    
    def set_equal(self):
        # from: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        '''
        ax = self.ax
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Provide basic plot features to draw a human skeleton, use matplotlib backend.
    def draw_cube(self):
        # ax.plot([1,1,1,1,0,0,0,0], [1,1,0,0,1,1,0,0], [1,0,1,0,1,0,1,0])
        self.ax.plot([1,1,1,1,0,0,0,0], [1,1,0,0,0,0,1,1], [1,0,0,1,1,0,0,1])
        # ax.show()

    def show(self, filename):
        if not filename:
            plt.show()
        else:
            plt.savefig(filename)
    
    def draw_point(self, p):
        # print(p)
        self.ax.plot([p[0]], [p[1]], [p[2]], '*')

    def draw_line(self, p1, p2):
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '-')

# Use matplotlib, blender, or other tools for 3D visualization.
class BlenderCanvas:
    pass


class XsensVisualizer:
    def __init__(self):
        canvas = PltCanvas()
        self.canvas = canvas
        self.mocap = self.load_bvh()
        self.bone_link = self.get_bone_link()
        self.mocap_joints = self.mocap.get_joints_names()
    
    def get_bone_info(self, mocap, frame_id, bone_name):
        offset = np.array(mocap.joint_offset(bone_name))
        rotation = mocap.frame_joint_channels(frame_id, \
            bone_name, ['Xrotation', 'Yrotation', 'Zrotation'])
        rx, ry, rz = rotation
        r = Rotation.from_euler('YXZ', [ry, rx, rz], degrees=True)
        # degrees is important
        # r = Rotation.from_euler('ZXY', [rz, rx, ry]) # TODO
        # The apply order of bvh is Y, X, Z
        # https://en.wikipedia.org/wiki/Euler_angles
        # https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
        # vR = vYXZ
        # xyz, extrinsic, original axis
        # XYZ, intrinsic, rotating 
        # bvh should be intrinsic?
        return offset, r

    def get_bone(self, mocap, frame_id, bone_name):
        # recursive to compute the joint 3D location
        offset, rotation = self.get_bone_info(mocap, frame_id, bone_name)
        offset = rotation.apply(offset)
        parent_bone = mocap.joint_parent(bone_name)
        while parent_bone:
            # print(parent_bone.name)
            # print(dir(parent_bone))
            # print(parent_bone.value)
            # parent_offset = np.array(mocap.joint_offset(parent_bone.name))
            parent_offset, parent_rotation = self.get_bone_info(mocap, frame_id, parent_bone.name)
            offset = offset + parent_offset # parent rotation should be applied to child bone
            offset = parent_rotation.apply(offset) # TODO: check this.
            parent_bone = mocap.joint_parent(parent_bone.name)

        # offset = mocap.joint_offset(bone_name)
        # rotation = mocap.frame_joint_channels(frame_id, \
        #     bone_name, ['Xrotation', 'Yrotation', 'Zrotation'])
        # rx, ry, rz = rotation
        # r = Rotation.from_euler('yxz', [ry, rx, rz])
        # r.apply(bone_offset)

        # r = Rotation.from_euler('zyx', [90, 45, 30], degrees=True)
        # Compute only offset without rotation
        # print('{bone_name} {offset} {rotation}'.format(**locals()))
        return offset

    def load_bvh(self):
        bvh_testfile = './data/bvh/abandon package.bvh'
        bvh_file = bvh_testfile
        with open(bvh_file) as f:
            data = f.read()
        bvh_data = mocap.bvh.Bvh(data)
        return bvh_data
    
    def get_frame_3d(self, frame_id):
        # Get frame data as a numpy array with N x 3
        mocap = self.mocap
        mocap_joints = self.mocap_joints
        N = len(mocap_joints)
        frame = np.zeros((N, 3))
        for i, bone_name in enumerate(mocap_joints):
            bone_offset = self.get_bone(mocap, frame_id, bone_name)
            frame[i, :] = bone_offset
        return frame
    
    # def get_frame_3d_1(self, frame_id):
    #     mocap = self.mocap
    #     mocap_joints = self.mocap_joints

    #     # rotation? save the parent

    #     for i, bone_name in enumerate(mocap_joints):
    #         bone_offset = self.get_bone(mocap, frame_id, bone_name)
    #         pxyz = frame[pi, :]
    #         xyz = pxyz + pR @ offset
    #         frame[i, :] = bone_offset 
    #     return frame

        pass
        # Iterate from the parent to child.
    
    def get_bone_link(self):
        # bone_link is a N x 2 array, which row is endpoint index
        bone_link = []
        mocap = self.mocap
        mocap_joints = mocap.get_joints_names()
        for i, bone_name in enumerate(mocap_joints):
            parent_index = mocap.joint_parent_index(bone_name)
            # children = mocap.joint_direct_children(bone_name)
            # for c in children:
            #     bone_link.append([i, c.index])
            if parent_index != -1:
                bone_link.append([parent_index, i])

        # return np.array(bone_link)
        return bone_link
    
    def draw_frame(self, frame_id):
        canvas = self.canvas
        mocap = self.mocap
        bone_link = self.bone_link
        mocap_joints = self.mocap_joints
        N = len(mocap_joints)

        frame = self.get_frame_3d(frame_id)
        # print(bone_link)
        # for bone_name in mocap_joints:
        #     bone_offset = get_bone(mocap, frame_id, bone_name)
        #     # print(bone_offset)
        #     canvas.draw_point(bone_offset)
        
        # for i in range(N):
        #     canvas.draw_point(frame[i,:])

        canvas.clear()
        for bone in bone_link:
            p1, p2 = bone
            canvas.draw_line(frame[p1,:], frame[p2,:])
        canvas.set_equal()

        # filename = None
        # filename = '%d.png' % frame_id
        # canvas.show(filename)



# The rotation should be times on all parent rotation
# How to transverse the bone tree?
