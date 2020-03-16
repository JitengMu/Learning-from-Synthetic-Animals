from math import sqrt
import numpy as np 
from scipy.spatial.transform import Rotation as Rot

def rot_matrix_from_angles(cam_angles):
    '''
    Calculate rotation matrix from UnrealCV angles
    Args:
        cam_angles: camera rotation in (YZX), z is inverse to the right hand principle
    Return:
        R_cam_w, R_w_cam: (3*3) rotation matrix
    '''
    R_w_cs = np.array([0,0,-1,1,0,0,0,-1,0]).reshape([3,3])
    # ZYX -> +Y+X+Z XYZ -> +Z+X+Y
    angel_zxy = [cam_angles[2], cam_angles[0], cam_angles[1]]
    # R_cam = Ry*Rx*Rz
    R_cam = Rot.from_euler("zxy", angel_zxy, degrees=True).as_dcm()
    #R_cam = Rot.from_euler("yxz", angel_zyx, degrees=True).as_dcm()
    R_cs_cam = R_cam 
    R_w_cam = np.dot(R_w_cs, R_cs_cam)
    R_cam_w = R_w_cam.T

    return R_cam_w, R_w_cam

def trans_cam_w_from_cam(cam_angles, cam_pos):
    '''
    Transformation from world coordinate to camera coordinate
    Args:
        cam_angles: rotation YZX order (Z is inverse to right hand order)
        cam_pos: position of camera XYZ order (X is inverse to right hand order)
    Return:
        g_w_cam: transformation from camera frame to world frame
        g_cam_w: inverse operation
    '''

    R_w_cs = np.array([0,0,-1,1,0,0,0,-1,0]).reshape([3,3])

    # ZYX -> +Y+X+Z XYZ -> +Z+X+Y
    angel_zxy = [cam_angles[2], cam_angles[0], cam_angles[1]]

    # R_cam = Ry*Rx*Rz
    R_cam = Rot.from_euler("zxy", angel_zxy, degrees=True).as_dcm()
    #R_cam = Rot.from_euler("yxz", angel_zyx, degrees=True).as_dcm()
    R_cs_cam = R_cam 
    R_w_cam = np.dot(R_w_cs, R_cs_cam)
    T_w_cam = np.array([-cam_pos[0], cam_pos[1], cam_pos[2]])/100

    g_w_cam = np.eye(4)
    g_w_cam[0:3, 0:3] = R_w_cam
    g_w_cam[0:3, 3] = T_w_cam

    g_cam_w =np.eye(4)
    g_cam_w[0:3, 0:3] = R_w_cam.T
    g_cam_w[0:3, 3] = -np.dot(R_w_cam.T, T_w_cam)

    return g_w_cam, g_cam_w

def trans_cam_w_from_plane(plane_vec, yaw_angle = 0):
    '''
    Get transformation from plane vector, if zeros_yaw, the yaw angle (y angle in camera coordinate) is zero
    Args:
        plane_vec: (4*1) vector (a,b,c,d) Equation : ax+by+cz+d = 0
    Return:
        g_w_cam, g_cam_w: 4*4 numpy array
    '''

    if plane_vec[3] < 0:
        plane_vec = -plane_vec
    
    # Estimation transformation matrix
    g_w_cam = np.eye(4)
    norm_z = np.array([0,-1,0]) # +Z in world is -Y in camera if camera in initial position
    axis_rot = np.cross(plane_vec[0:3].reshape(-1), norm_z)  # rotate from current back to the init pose
    norm_rot = sqrt(np.dot(axis_rot.T, axis_rot))   
    cosine_rot = np.dot(plane_vec[0:3], norm_z)  # dot product of the two vectors
    
    theta = np.arctan2(norm_rot, cosine_rot)
    rot_vector = axis_rot/norm_rot*theta

    Rot_plane = Rot.from_rotvec(rot_vector).as_dcm()
    R_sc = np.array([1,0,0,0,0,1,0,-1,0]).reshape((3,3))
    
    g_w_cam[0:3, 0:3] = np.dot(R_sc, Rot_plane)
    

    R_w_cs = np.array([0,0,-1,1,0,0,0,-1,0]).reshape([3,3])
    R_cs_cam = np.dot(R_w_cs.T, g_w_cam[0:3, 0:3])
    angel_zxy = Rot.from_dcm(R_cs_cam).as_euler("zxy", degrees=True)

    # Force the y axis in camera to be yaw_angle
    Ry = Rot.from_euler("y", -angel_zxy[2] + yaw_angle, degrees=True).as_dcm()
    R_cs_cam = np.dot(Ry, R_cs_cam)
    R_w_cam = np.dot(R_w_cs, R_cs_cam)

    # Update g_w_cam 
    g_w_cam[0:3, 0:3] = R_w_cam
    
    # camera optical center point to world center
    T_cam_w = np.zeros(3)
    T_cam_w[2] = -plane_vec[3]/plane_vec[2]
    T_w_cam = -np.dot(g_w_cam[0:3, 0:3], T_cam_w)
    g_w_cam[0:3, 3] = T_w_cam

    # Find transltation from world to camera
    g_cam_w = np.eye(4)
    g_cam_w[0:3, 0:3] = g_w_cam[0:3, 0:3].T
    g_cam_w[0:3, 3] = - np.dot(g_w_cam[0:3, 0:3].T, g_w_cam[0:3, 3])
    
    return g_w_cam, g_cam_w

def from_rtvecs_to_euler(rot_vec, trans):
    '''
    Transform from rvec and tvec to UnrealCV(only for camera pose)
    Args:
        rot_vec: (3,) array, (rvec from opencv)
        trans: (3,) array (tvec from opencv)
    Return:
        cam_angles, cam_pos
    '''
    g_cam_w = np.eye(4)
    g_cam_w[0:3, 0:3] = Rot.from_rotvec(rot_vec).as_dcm()
    g_cam_w[0:3, 3] = trans

    R_w_cam = g_cam_w[0:3, 0:3].T
    T_w_cam = -np.dot(g_cam_w[0:3, 0:3].T, g_cam_w[0:3, 3])

    # g_w_cam = np.eye(4)
    # g_w_cam[0:3, 0:3] = R_w_cam
    # g_w_cam[0:3, 3] = T_w_cam

    cam_pos = [0,0,0]
    cam_angles = [0,0,0]
    cam_pos[0], cam_pos[1], cam_pos[2] = tuple(100*T_w_cam)
    cam_pos[0] = -cam_pos[0]
    
    R_w_cs = np.array([0,0,-1,1,0,0,0,-1,0]).reshape([3,3])
    R_cs_cam = np.dot(R_w_cs.T, R_w_cam)
    angel_zxy = Rot.from_dcm(R_cs_cam).as_euler("zxy", degrees=True)
    cam_angles[2], cam_angles[0], cam_angles[1] = tuple(angel_zxy)

    return cam_angles, cam_pos

def convert_unrealcv_pose_to_cam_rts(car_trans, car_rot, g_cam_w):
    '''
    Convert synthetic object trans and rot in camera coordinate
    Args:
        car_trans: [num_obj*3] car translation (XYZ) order, x is inverse to the right hand coordinate 
        car_rot: [num_obj*3] car rotation in (YZX), z is inverse to the right hand principle
        g_cam_w: transformation matrix from world to camera (4*4)
    Return:
        R_out: Rotation input (num_obj*3) in rotvec form
        T_out: Translation input (num_obj*3)
    '''
    # Convert car rotation axis order to normal order (YZX -> ZYX)
    # ZYX -> -Z+Y-X
    car_rot_array = np.array(car_rot)
    axis_idx = [1,0,2]
    car_rot_array = -car_rot_array[:, axis_idx]
    car_rot_array[:, 1] = -car_rot_array[:, 1]

    # Define variables
    num_obj = len(car_trans)
    R_out = np.zeros([num_obj,3])
    T_out = np.zeros([num_obj,3])

    for i in range(num_obj):
        
        car_rot_i = car_rot_array[i,:]
        Rm_i = Rot.from_euler('zyx', car_rot_i, degrees=True).as_dcm()
        T_i = np.array(car_trans[i], dtype=float)/100
        # Convert x to right hand axis
        T_i[0] = -T_i[0]

        g_w_obj = np.eye(4)
        g_w_obj[0:3,0:3] = Rm_i
        g_w_obj[0:3,3] = T_i
        g_cam_obj = np.dot(g_cam_w, g_w_obj)

        T_out[i,:] = g_cam_obj[0:3, 3]
        R_out[i,:] = Rot.from_dcm(g_cam_obj[0:3, 0:3]).as_rotvec()
    
    return R_out, T_out

def convert_rts_to_unrealcv_pose(R_out, T_out, g_cam_w):
    '''
    Convert trans and rot in camera coordinate to UnrealCV synthetic input  
    Args:
        R_out: Rotation input (num_obj*3) in rotvec form
        T_out: Translation input (num_obj*3)
        g_cam_w: transformation matrix from world to camera (4*4)
    Return:
        car_trans: [num_obj*3] car translation (XYZ) order, x is inverse to the right hand coordinate 
        car_rot: [num_obj*3] car rotation in (YZX), z is inverse to the right hand principle
    '''
    num_obj = R_out.shape[0]
    car_trans = np.zeros([num_obj,3])
    car_rot_array = np.zeros([num_obj,3])

    g_w_cam = np.eye(4)
    g_w_cam[0:3, 0:3] = g_cam_w[0:3, 0:3].T
    g_w_cam[0:3, 3] = - np.dot(g_cam_w[0:3, 0:3].T, g_cam_w[0:3,3])

    for i in range(num_obj):
        g_cam_obj = np.eye(4)
        T_out_i = T_out[i,:]
        Rm_out_i = Rot.from_rotvec(R_out[i,:]).as_dcm()

        g_cam_obj[0:3, 0:3] = Rm_out_i
        g_cam_obj[0:3, 3] = T_out_i
    
        g_w_obj = np.dot(g_w_cam, g_cam_obj)

        car_trans[i,:] = 100*g_w_obj[0:3,3]
        car_trans[i,0] = -car_trans[i,0] # X is inverse the order
        car_rot_i = Rot.from_dcm(g_w_obj[0:3, 0:3]).as_euler('zyx', degrees=True)
        car_rot_array[i,:] = car_rot_i
    
    # convert to UnrealCV order YZX <- ZYX ( ZYX ->  YZX)
    # ZYX <- -Z+Y-X  (YZX -> +Y-Z-X)
    axis_idx = [1,0,2]
    car_rot = -car_rot_array[:, axis_idx]  # Reorder the axis
    car_rot[:, 0] = -car_rot[:, 0]   # invert Y axis
    
    car_trans = car_trans.astype(float).tolist()
    car_rot = car_rot.astype(float).tolist()
    
    return car_trans, car_rot

def cal_kpt_in_cam(R_in, T_in, kpt_mean, kpt_basis=None, shape_in=None):
    '''
    convert the objects from original coordinate to camera coordinate
    Args:
        R_in: Rotation input (num_obj*3) in rotvec form
        T_in: Translation input (num_obj*3)
        kpt_mean: Mean keypoint in 3D (num_obj*3)
        kpt_basis: num_basis*(num_kpt*3)
        shape_in: num_obj*num_basis
    Return:
        kpts_3d: keypoints in 3D space(num_obj*(num_kpt*3))
    '''
    assert  R_in.shape[0] == T_in.shape[0]
    num_obj = R_in.shape[0]
    num_kpt = kpt_mean.shape[0]
    if (kpt_basis is not None) and (shape_in is not None):
        assert shape_in.shape[0] == T_in.shape[0]
        assert kpt_basis.shape[0] == shape_in.shape[1]
        assert 3*num_kpt == kpt_basis.shape[1]
        deform = True
    else:
        deform = False
    kpts_3d = np.zeros([num_obj,num_kpt*3])
    
    for i in range(num_obj):
        Rm_i = Rot.from_rotvec(R_in[i,:]).as_dcm()
        T_i = T_in[i,:]
        if deform: kpt_o_i = kpt_mean + np.dot(shape_in[i,:].reshape([1,-1]), kpt_basis).reshape([num_kpt, 3])
        else: kpt_o_i = kpt_mean
        kpt_c_i = np.dot(Rm_i, kpt_o_i.T) + T_i.reshape([3,1])
        
        kpts_3d[i,:] = np.reshape(kpt_c_i.T, num_kpt*3)
    
    return kpts_3d

