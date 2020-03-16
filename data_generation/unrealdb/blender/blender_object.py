# Make a scene from yaml definition
import bpy
import mathutils
import sys

class Scene:
    def __init__(self):
        self.scene = bpy.context.scene
        self.objects = dict()
        self.objects['scene'] = self
        self.frame_id = 0
        # discover objects into the scene?

    def get(self, obj_name):
        return self.objects.get(obj_name)

    def import_fbx(self, fbx_filename, scale):
        bpy.ops.import_scene.fbx(filepath=fbx_filename, global_scale=scale)

    def import_bvh(self, bvh_filename, scale):
        bpy.ops.import_anim.bvh(filepath = bvh_filename, global_scale=scale)

    def load(self, blend_filename):
        # bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)
        bpy.ops.wm.open_mainfile(filepath = blend_filename)

    def save(self, blend_filename):
        # bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
        bpy.ops.wm.save_as_mainfile(filepath = blend_filename)

    def clear(self):
        for obj in self.scene.objects:  
            self.scene.objects.unlink(obj)
        self.objects.clear()
        self.objects['scene'] = self

    def add_object(self, cls_name, obj_name):
        cls_type = globals()[cls_name]
        if not cls_type:
            print('Can not find object type %s' % cls_name)
            return

        obj = cls_type(obj_name)
        self.objects[obj_name] = obj 
        obj.scene = self

    def find_type(self, blender_type, id, obj_name):
        # blender_type can be 'CAMERA', 'MESH', 'LIGHT', 'ARMATURE'
        objs = [v for v in bpy.data.objects if v.type == blender_type]
        blender_obj = objs[id]

        obj = Object(obj_name)
        obj.obj = blender_obj
        self.objects[obj_name] = obj
        obj.scene = self

    def find_object(self, blender_name, obj_name):
        # Make a wrapper for blender object
        blender_obj = bpy.data.objects.get(blender_name)
        if not blender_obj:
            print('Can not find object %s in the scene' % scene_name)
            return
        
        # Make a wrapper
        obj = Object(obj_name)
        obj.obj = blender_obj # set the internal data
        self.objects[obj_name] = obj
        obj.scene = self

    def render_opengl(self, filename):
        bpy.context.scene.render.filepath = filename
        bpy.ops.render.opengl(write_still = True)

    def render(self, filename):
        pass

    def set_frame_current(self, frame_id):
        # bpy.data.scenes['Scene'].frame_current = frame_id
        # https://blender.stackexchange.com/questions/27579/render-specific-frames-with-opengl-via-python
        # self.scene.frame_current = frame_id
        frame_id = int(frame_id)
        self.frame_id = frame_id
        self.scene.frame_set(frame_id)
        # https://www.blender.org/forum/viewtopic.php?t=27854
        self.scene.update()
    
    def exit(self):
        bpy.ops.wm.quit_blender()
        # sys.exit()

# progressbar
# import bpy
# wm = bpy.context.window_manager

# # progress from [0 - 1000]
# tot = 1000
# wm.progress_begin(0, tot)
# for i in range(tot):
#     wm.progress_update(i)
# wm.progress_end()
# https://blender.stackexchange.com/questions/3219/how-to-show-to-the-user-a-progression-in-a-script

class Object:
    def __init__(self, obj_name):
        self.obj = None

    def set_loc(self, x, y, z):
        self.obj.location = [x, y, z]

    def get_loc(self):
        return self.obj.location

    def set_rot(self, pitch, yaw, roll):
        pass

class Camera(Object):
    # https://blenderartists.org/t/how-to-add-an-empty-and-a-camera-using-python-script/588300
    def __init__(self, obj_name):
        cam = bpy.data.cameras.new(obj_name)
        cam_obj = bpy.data.objects.new(obj_name, cam)
        bpy.context.scene.objects.link(cam_obj)
        self.obj = cam_obj

    def track_offset(self, target_name, ox, oy, oz):
        target = self.scene.get(target_name)
        self.obj.location = target.get_loc() + mathutils.Vector((ox, oy, oz))
        self.look_at(target_name)
        pass

    # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    def look_at(self, target_name):
        # Move the camera to a correct location
        scene = self.scene
        target = scene.get(target_name)
        # target_loc = target.obj.location
        target_loc = target.get_loc()
        cam_loc = self.obj.location
        direction = target_loc - cam_loc
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.obj.rotation_euler = rot_quat.to_euler()

        # loc_camera = obj_camera.matrix_world.to_translation()
        # direction = point - loc_camera
        # # point the cameras '-Z' and use its 'Y' as up
        # rot_quat = direction.to_track_quat('-Z', 'Y')
        # # assume we're using euler rotation
        # obj_camera.rotation_euler = rot_quat.to_euler()
    
    # https://blender.stackexchange.com/questions/30643/how-to-toggle-to-camera-view-via-python
    # https://blender.stackexchange.com/questions/145538/how-do-i-change-camera-views-using-python
    def view_camera(self):
        # bpy.context.active_object = self.obj
        # bpy.context.scene.objects.active = self.obj
        bpy.context.scene.camera = self.obj
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].region_3d.view_perspective = 'CAMERA'
                # override = bpy.context.copy()
                # override['area'] = area
                # bpy.ops.view3d.viewnumpad(override, type = 'CAMERA')

class Human(Object):
    def __init__(self, obj_name):
        pass

    def load_bvh(self, bvh_filename):
        pass

    def set_bvh_index(self, frame_id):
        # Use the bvh data to control the human bone.
        pass

class Mesh(Object):
    def __init__(self, obj_name):
        self.obj_name = obj_name
        self.obj = None

    def set_mesh(self, mesh):
        if mesh == 'cube':
            bpy.ops.mesh.primitive_cube_add()
            self.obj = bpy.context.active_object

    def get_rot(self):
        pass

# Track a single bone
class Bone(Object):
    def __init__(self, obj_name):
        self.obj = None

    def set_bone(self, arm_name, bone_name):
        # Wrap a pose bone in the scene
        self.arm = self.scene.get(arm_name)
        self.bone = self.arm.obj.pose.bones[bone_name]

    def get_loc(self):
        # https://blender.stackexchange.com/questions/35982/how-to-get-posebone-global-location
        # https://blender.stackexchange.com/questions/15353/get-the-location-in-world-coordinates-of-a-bones-head-and-tail-while-in-pose-mo
        # print(self.bone.location)
        # return self.bone.location
        # https://blender.stackexchange.com/questions/35982/how-to-get-posebone-global-location
        return self.arm.obj.matrix_world * self.bone.matrix * self.bone.location
        # self.bone.location is local coordinate