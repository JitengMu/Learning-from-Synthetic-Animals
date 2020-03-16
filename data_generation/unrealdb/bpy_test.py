# Test this script inside blender.

import importlib
import sys
import os
script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# The __file__ has been shown inside the blend file?
sys.path.append(script_dir)
os.chdir(script_dir)
print('='*30 + ' Begin ' + '='*30)
print(__file__)
print(script_dir)
import importlib
import bpy
import mocap.bvh
import mocap.smpl
import mocap.data
importlib.reload(mocap.data)
from tqdm import tqdm
importlib.reload(mocap.smpl)

bvh_filename = './data/bvh/abandon package.bvh'

# The basic pipeline in blender should be

bvh_data = mocap.smpl.BvhData(bvh_filename)
bones = bvh_data.get_all_bones()
print(bones)


obj = bpy.context.object

model = mocap.smpl.SMPLModel(obj)
# bones = model.get_all_bones()
# bones_name = [v.name for v in bones]
# print(bones_name)
# print(type(bones[0]))
# print(bones)
model.reset_pose() # For debug purpose
model.a_pose()  # For debug purpose
frame_id = 300
# Bone mapping defines the mapping which bone should be controlled by which bvh data
# key: value -> SMPL: bvh
# This can be computed automatically if a good retarget system has been implemented
# But it is currently manual

# frame_range = range(1200, 1300)
frame_range = range(1400, 1550)
# frame_range = range(len(bvh_data))
for fi, frame_id in enumerate(tqdm(frame_range)):
    model.set_bvh_pose(bvh_data, frame_id, mocap.data.smpl2bvh)
    model.add_keyframe(fi)