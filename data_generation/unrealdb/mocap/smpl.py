# Control SMPL model.
# Use alt-s to save file.
import sys
sys.path.append('.')
import mocap.bvh
import numpy as np

class BlenderUtil:
    def get_scene(self):
        pass

smpl_bones = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# Need to retarget from mocap to this.

# # order
# part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
#               'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
#               'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
#               'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
#               'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
#               'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

class SMPLModel:
    # TODO: Need to define some basic type, such as bone
    # Shoule I reuse what is in blender or define my own data type?
    # It depends on whether there is clear document for blender data type.
    # related data type and document can be found
    # bpy_types.PoseBone: https://docs.blender.org/api/current/bpy.types.PoseBone.html
    # bpy_types.
    def __init__(self, obj):
        self.arm_obj = obj

    def get_all_bones(self):
        bones = []
        for bone in self.arm_obj.pose.bones:
            bones.append(bone)
        return bones

    def reset_pose(self):
        print('Reset human pose')
        # For SMPL the reset pose is a T-pose model
        # For bvh, the reset pose is also T-pose
        # Iterate over all pose bones and reset the pose bone to zero.
        bones = self.get_all_bones()
        quat_default = (1, 0, 0, 0)
        for bone in bones:
            self.set_quat(bone.name, quat_default)
            # bone.rotation_mode = 'QUATERNION'
            # bone.rotation_quaternion = 

    def set_quat(self, bone_name, quat):
        bone = self.arm_obj.pose.bones[bone_name]
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = quat
    
    def set_trans(self, bone_name, trans):
        bone = self.arm_obj.pose.bones[bone_name]
        bone.location = trans

    def a_pose(self):
        print('Set human to A pose')

    def set_bvh_pose(self, bvh_data, frame_id, bone_mapping):
        # This is perframe animation
        # in order to set the pose for each frame, I need to use keyframe_insert
        # Use bone mapping to set pose.

        bvh_bone = 'Hips'
        # smpl_bone = 'm_avg_root' # Change this bone location
        smpl_bone = 'm_avg_Pelvis' # Change this bone location
        trans = bvh_data.get_local_trans(bvh_bone, frame_id)
        x, y, z = trans / 100.0
        self.set_trans(smpl_bone, [x,y,z])
        for k, v in bone_mapping.items():
            smpl_bone = 'm_avg_' + k
            # print('Set bone %s with joint info %s from bvh' % (smpl_bone, v))
            # Read data from bvh data and use it to control the SMPL model
            r = bvh_data.get_local_quat(v, frame_id)
            # print(r)
            self.set_quat(smpl_bone, r)
            # blender order is w, x, y, z
            # scipy order is x, y, z, w

    def add_keyframe(self, frame):
        for bone in self.get_all_bones():
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

# test bvh data
# data/bvh/abandon package.bvh

class BvhData:
    def __init__(self, bvh_filename):
        self.load_bvh(bvh_filename)

    def load_bvh(self, bvh_filename):
        with open(bvh_filename) as f:
            data = f.read()
        bvh_data = mocap.bvh.Bvh(data)
        self.data = bvh_data

    def get_all_bones(self):
        joint_names = self.data.get_joints_names()
        return joint_names

    def get_local_trans(self, bone_name, frame_id):
        trans = self.data.frame_joint_channels(frame_id, \
            bone_name, ['Xposition', 'Yposition', 'Zposition'])
        return np.array(trans)


    def get_local_rot(self, bone_name, frame_id):
        from scipy.spatial.transform import Rotation
        # Get local rotation without considering the hierarchy
        # offset = np.array(mocap.joint_offset(bone_name))
        # offset is not needed
        rotation = self.data.frame_joint_channels(frame_id, \
            bone_name, ['Xrotation', 'Yrotation', 'Zrotation'])
        rx, ry, rz = rotation
        scipy_rot = Rotation.from_euler('YXZ', [ry, rx, rz], degrees=True)
        return scipy_rot

    def get_local_quat(self, bone_name, frame_id):
        r = self.get_local_rot(bone_name, frame_id)
        import mathutils
        r = r.as_quat()
        # convert to blender quat format
        x, y, z, w = r
        r = mathutils.Quaternion((w, x, y, z)) # Swap the order
        return r

    def get_global_quat(self, bone_name):
        pass

    def __len__(self):
        return self.data.nframes

# Draw the 3D points of the human bvh.

# Do very simple retarget, read more generic retarget algorithm from the mblab.
# The retarget is needed for low-cost mocap as well.

# joints = mocap.frame_joint_channels(22, 'LeftCollar', ['Xrotation', 'Yrotation', 'Zrotation'])
# offset = mocap.joint_offset('LeftCollar')
# # joint offset?
# print(offset)
# print(joints)

# Visualize bvh file with matplotlib to make sure my understanding is correct.

# Use scipy to convert rotation between euler angle, etc.
