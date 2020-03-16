import bpy
import bpy_extras

camera = bpy.data.objects['Camera']
# kp_list = [[float(x) for x in line.strip().split(' ')[-3:]] for line in open(kp_file).readlines()]

def project(cam, p3):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, p3)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return (co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1], co_2d.z)