# Split the bone hierarchy into sections, this can make debug easier
# spine, l/r arm, l/r leg
# b_sp = ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head']
# s_sp = ['root', 'Pelvis', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head']
# b_sp = ['Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head']
# s_sp = ['root', 'Pelvis', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head']
b_sp = [ 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head']
s_sp = [ 'Pelvis', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head']

b_larm = ['LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist']
b_rarm = ['RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist']
b_lleg = ['LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe']
b_rleg = ['RightHip', 'RightKnee', 'RightAnkle', 'RightToe']


s_larm = ['L_Collar', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand']
s_rarm = ['R_Collar', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
s_lleg = ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot']
s_rleg = ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot']
# def add_prefix(bones):
#     smpl_prefix = 'm_avg_'
#     for i in range(len(bones)): bones[i] = smpl_prefix + bones[i]
# [add_prefix(v) for v in [s_sp, s_rarm, s_larm, s_lleg, s_rleg]]

smpl2bvh = dict()
smpl2bvh.update(zip(s_sp, b_sp))
smpl2bvh.update(zip(s_larm, b_larm))
smpl2bvh.update(zip(s_rarm, b_rarm))
smpl2bvh.update(zip(s_lleg, b_lleg))
smpl2bvh.update(zip(s_rleg, b_rleg))

bvh2smpl = dict()
bvh2smpl.update(zip(b_sp, s_sp))
bvh2smpl.update(zip(b_larm, s_larm))
bvh2smpl.update(zip(b_rarm, s_rarm))
bvh2smpl.update(zip(b_lleg, s_lleg))
bvh2smpl.update(zip(b_rleg, s_rleg))