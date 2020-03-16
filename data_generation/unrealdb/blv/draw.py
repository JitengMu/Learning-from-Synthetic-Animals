# Draw object in the scene
import bpy

def replace():
    # replace exist object to another one.
    pass

def cube():
    bpy.ops.mesh.primitive_cube_add()
    return bpy.context.active_object

def obj():
    # Load an object file into the scene
    pass

def axis(name):
    ax = bpy.data.objects.get(name)
    if ax: return ax

    bpy.ops.object.empty_add(type='ARROWS')
    ax = bpy.context.active_object
    ax.name = name
    return ax

def ball(name):
    obj = bpy.data.objects.get(name)
    if obj: return obj

    bpy.ops.mesh.primitive_uv_sphere_add()
    obj = bpy.context.active_object
    obj.name = name
    return obj

def camera(name=None):
    cam = None
    if name: cam = bpy.data.objects.get(name)
    if cam:
        return cam
    else:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        if name: cam.name = name
        return cam

def image(name=None, img_path=None):
    if name and bpy.data.objects.get(name):
        img = bpy.data.objects[name]
        if img_path: img.data = load_image(img_path)
        return img 

    bpy.ops.object.empty_add(type='IMAGE')
    img = bpy.context.active_object
    if name: img.name = name
    if img_path: img.data = load_image(img_path)
    return img

def load_image(img_path):
    # Load all images
    # img_path = 'D:\\human36m\\processed\\S1\\Directions-1\\imageSequence-undistorted\\60457274\\img_000001.jpg'
    img = bpy.data.images.load(img_path)
    return img

def get():
    pass
    # return obj


# How to draw a skeleton in blender.