import bpy
# How an algorithm can recognize a transform
# Especially a 3D tranfrom? Can we adapt it to different shapes
import numpy as np

class KeyFrame:
    # Animation data for an object
    def __init__(self, frame_index, key, value):
        self.frame_index = frame_index
        self.data_path = key
        self.value = value
    
    def __repr__(self):
        return '%d: %s %s' % (self.frame_index, self.data_path, str(self.value))

def scale(duration):
    pass

def rotate(axis, angle, duration):
    pass

def translate(start, end, duration):
    # Translate an object from point a to point b
    # Use blender to automatically interpolate
    kfs = []
    fps = 24
    kfs.append(KeyFrame(0, 'location', start))
    kfs.append(KeyFrame(duration * fps, 'location', end))
    return kfs

def animate(obj, kfs):
    # obj: object to manipulate
    # kfs: key frames
    # Generate keyframe-based animation for an object
    scene = bpy.context.scene
    for kf in kfs:
        print(kf)
        scene.frame_set(kf.frame_index)
        setattr(obj, kf.data_path, kf.value)
        obj.keyframe_insert(data_path=kf.data_path, index=-1)
    frame_end = max([v.frame_index for v in kfs])
    scene.frame_end = frame_end

    max_frame = max([kf.frame_index for kf in kfs])
    scene = bpy.data.scenes['Scene']
    scene.frame_end = max_frame

    max_frame = max([kf.frame_index for kf in kfs])
    scene = bpy.data.scenes['Scene']
    scene.frame_end = max_frame

def test():
    kfs = translate([0, 0, 0], [2, 0, 0], 1)
    animate(bpy.context.active_object, kfs)
