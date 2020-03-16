import json
import numpy as np 
import os
from os.path import join

from diva_transform import convert_unrealcv_pose_to_cam_rts, trans_cam_w_from_cam, convert_rts_to_unrealcv_pose

def wirte_meta_data(shapenet_ids, car_shape, car_rot, car_trans, cam_location, cam_rotation, img_fov, img_size, kpt_dict, data_path):
    '''
    Input: shapenet_ids, car_shape, car_rot, car_trans, cam_location, cam_rotation, img_fov, img_size(w*l), kpt_dict, data_path
    '''
    data_dict = {}
    data_dict['shapenet_ids'] = shapenet_ids
    data_dict['car_shape'] = car_shape
    data_dict['car_rot'] = car_rot
    data_dict['car_trans'] = car_trans
    data_dict['cam_location'] = cam_location
    data_dict['cam_rotation'] = cam_rotation
    data_dict['kpt_dict'] = kpt_dict
    data_dict['img_fov'] = img_fov
    data_dict['img_size'] = img_size
    data_dict['kpt_dict'] = kpt_dict

    with open(data_path, 'w') as f:
        json.dump(data_dict, f)
    return True

def load_meta_data(data_path):
    '''
    Load json data from a destination
    Return:
        car_shape, car_rot, car_trans, cam_location, cam_rotation, img_fov, img_size, kpt_dict
    '''
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
        car_shape = data_dict['car_shape']  
        car_rot = data_dict['car_rot']  
        car_trans = data_dict['car_trans']  
        cam_location = data_dict['cam_location']
        cam_rotation = data_dict['cam_rotation']
        kpt_dict = data_dict['kpt_dict'] 
        img_fov = data_dict['img_fov']  
        img_size = data_dict['img_size'] 
        assert len(car_rot) == len(car_shape)
        assert len(car_rot) == len(car_trans)
        
    return car_shape, car_rot, car_trans, cam_location, cam_rotation, img_fov, img_size, kpt_dict

def load_shapenet_kpt(models_dir, shapenet_ids):
    '''
    Load all model's ID and keypoints from models DIR
    Args:
        models_dir: abolute dir
        shapenet_ids: IDs to be load, if None, load all existing subfolders 
    Return:
        kpt_dict: key is shapenet id
    '''
    kpt_dict = {}
    R_model = np.array([-1,0,0,0,0,1,0,1,0]).reshape([3,3])
    if shapenet_ids is None:
        shapenet_ids = os.listdir(models_dir)
    for i, shape in enumerate(shapenet_ids):
        with open(join(models_dir, shape, "anchors.txt"), 'r') as annot_file:
            kpt_raw =  np.loadtxt(annot_file).T
            kpt_raw = 5*np.dot(R_model, kpt_raw).T
            kpt_dict[shape] = kpt_raw.astype(float).tolist()
    return kpt_dict

def save_optmization_data(json_path, R_out, T_out, shape_coe, plane_vec, img_fov):
    '''
    Dump the results of optimization to json file
    Args:
        json_path: absolute path for json file
        R_out: (num_obj,3) numpy array      
        T_out: (num_obj,3) numpy array     
        shape_coe: (num_obj,num_basis)      
        plane_vec: (4,1) numpy array
        img_fov: field of view in image
    Return:
        retval
    '''
    assert R_out.shape[1] == 3
    assert T_out.shape[1] == 3
    assert plane_vec.shape[0] == 4
    assert R_out.shape[0] == T_out.shape[0]
    assert R_out.shape[0] == shape_coe.shape[0]
    with open(json_path,"w") as f:
        data_dict = {}
        data_dict["R_out"] = R_out.astype(float).tolist()
        data_dict["T_out"] = T_out.astype(float).tolist()
        data_dict["shape_coe"] = shape_coe.astype(float).tolist()
        data_dict["plane_vec"] = plane_vec.astype(float).tolist()
        data_dict["img_fov"] = float(img_fov)
        json.dump(data_dict, f)
    
    return True

def load_optmization_data(json_path):
    '''
    Dump the results of optimization to json file
    Args:
        json_path
    Return:
        R_out: (num_obj,3) numpy array      
        T_out: (num_obj,3) numpy array     
        shape_coe: (num_obj,num_basis)      
        plane_vec: (4,1) numpy array
    '''
    with open(json_path, "r") as f:
        data_dict = json.load(f)
        R_out = np.array(data_dict["R_out"])
        T_out = np.array(data_dict["T_out"])
        shape_coe = np.array(data_dict["shape_coe"])
        plane_vec = np.array(data_dict["plane_vec"])
        img_fov = data_dict["img_fov"]
    assert R_out.shape[1] == 3
    assert T_out.shape[1] == 3
    assert plane_vec.shape[0] == 4
    
    return R_out, T_out, shape_coe, plane_vec, img_fov

def convert_to_camera_intrinsics(img_size, img_FOV):
    '''
    Args:
        img_size: (w*l)
        img_FOV: in degree form
    Return:
        K: camera intrinsics
    '''
    K = np.eye(3)
    K[0,2], K[1,2] = img_size[0]/2, img_size[1]/2
    s = np.tan(img_FOV/2/180*np.pi)
    K[0,0] = K[0,2]/s
    K[1,1] = K[0,2]/s
    
    return K

def pca_approx_shape(car_shape, kpt_dict, kpt_mean, kpt_basis):
    '''
    Args:
        car_shape: [num_obj] shapenet ID
        kpt_dict: dict contains shapnet ID and [num_kpt*3] keypoint position 
        kpt_mean:  (108,) numpy array
        kpt_basis: (108,num_basis)
    Return:
        shape_coe: (num_obj*num_basis)
    '''

    num_obj = len(car_shape)
    shape_coe = np.zeros([num_obj, kpt_basis.shape[1]])

    for i, shape_id in enumerate(car_shape):
        kpt_i = np.array(kpt_dict[car_shape[i]]).reshape(-1)
        kpt_i_unbiased = kpt_i - np.array(kpt_mean)
        shape_coe[i,:] = np.dot(kpt_i_unbiased, kpt_basis)

    return shape_coe

def search_similar_model(shape_coe, kpt_dict, kpt_mean, kpt_basis):
    '''
    Args:
        shape_coe: num_obj*num_basis numpy array
        kpt_dict: key is shape_net IDs, [36,3] numpy array 
        kpt_mean: [(36*3),] numpy array
        kpt_basis: [(36*3),num_basis] numpy array  
    Return:
        car_shape: list, length is num_obj
    '''
    if not shape_coe.shape[1] == kpt_basis.shape[1]: # same number of basis
        kpt_basis = kpt_basis[:, 0:shape_coe.shape[1]]
    assert shape_coe.shape[1] == kpt_basis.shape[1] # same number of basis

    car_shape_keys = [i for i in kpt_dict.keys()]
    car_shape_coef = np.zeros([len(car_shape_keys), kpt_basis.shape[1]])
    
    for i, key in enumerate(car_shape_keys):
        kpt_i = np.array(kpt_dict[key]).reshape(-1)
        kpt_i_unbiased = kpt_i - kpt_mean
        shape_coef = np.dot(kpt_i_unbiased, kpt_basis)
        car_shape_coef[i] = shape_coef
    
    car_shape = []
    
    for i in range(shape_coe.shape[0]):
        shape_coe_i = shape_coe[i,:]
        dis = car_shape_coef - shape_coe_i
        dis_norm = np.dot(dis, dis.T).diagonal()
        min_indx = np.argmin(dis_norm)
        car_shape.append(car_shape_keys[min_indx])
    
    assert len(car_shape) == shape_coe.shape[0]

    return car_shape

def convert_meta_data_to_optimize_spec(car_shape, car_rot, car_trans, cam_pos, cam_rot, kpt_dict, kpt_mean, kpt_basis):
    '''
    Convert synthetic meta data to the same specification of synthetic data
    Args:
        car_shape: [num_obj] shapenet ID
        car_rot: [num_obj] car angles in UnrealCV order
        car_trans: [num_obj] car translation in UnrealCV order
        cam_pos: position of camera XYZ order (X is inverse to right hand order)
        cam_rot:rotation YZX order (Z is inverse to right hand order)
        kpt_dict: dict contains shapnet ID and [num_kpt*3] keypoint position 
        kpt_mean:([num_kpt*3],) numpy array             OR    None
        kpt_basis:([num_kpt*3],num_basis) numpy array   OR    None
    Return:
        R_out: (num_obj,3) numpy array      OR    None
        T_out: (num_obj,3) numpy array     
        shape_coe: (num_obj,num_basis)      OR    None
        plane_vec: (4,1) numpy array
    '''
    
    # From camera to world coordinate
    g_w_cam, g_cam_w = trans_cam_w_from_cam(cam_rot, cam_pos)
    if kpt_mean is None or kpt_basis is None:
        R_out = None
        shape_coe = None
        car_center_w = np.ones([4, len(car_trans)])
        car_center_w[0:3, :] = (np.array(car_trans).T)/100
        car_center_w[0,:] = -car_center_w[0,:]
        car_center_cam = np.dot(g_cam_w, car_center_w)
        T_out = car_center_cam[0:3,:].T
    else:
        # PCA approximation
        shape_coe = pca_approx_shape(car_shape, kpt_dict, kpt_mean, kpt_basis)
        # Transformation
        R_out, T_out = convert_unrealcv_pose_to_cam_rts(car_trans, car_rot, g_cam_w)
        # car_trans_re, car_rot_re = convert_rts_to_unrealcv_pose(R_out, T_out, g_cam_w)
        # print("DEBUG.....")
    norm_z = np.zeros([4,2])
    norm_z[3,:] = 1
    norm_z[2,1] = 1
    norm_z_cam = np.dot(g_cam_w, norm_z)

    plane_vec = np.zeros([4])
    plane_vec[0:3] = norm_z_cam[0:3,1] - norm_z_cam[0:3,0]
    d = np.dot(plane_vec[0:3], norm_z_cam[0:3,0])
    plane_vec[3] = -d

    return R_out, T_out, shape_coe, plane_vec