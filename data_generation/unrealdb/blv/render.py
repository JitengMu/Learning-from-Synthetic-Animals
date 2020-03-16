import os
import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bpy
import bl.d3

class Logger:
    # From: https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output
    def __init__(self):
        pass

    def off(self):
        # redirect output to log file
        logfile = 'blender_render.log'
        open(logfile, 'a').close()
        self.old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

    def on(self):
        # disable output redirection
        os.close(1)
        os.dup(self.old)
        os.close(self.old)

logger = Logger()

def depth(filename=None):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    renderLayersNode = tree.nodes.get('Render Layers')
    compositeNode = tree.nodes.get('Composite')
    tree.links.new(renderLayersNode.outputs[2], compositeNode.inputs[0])

    scene = bpy.data.scenes['Scene']
    scene.render.image_settings.file_format = 'OPEN_EXR' # set the format. the extension should be exr?
    if filename:
        depth_file = filename
    else:
        depth_file = 'depth.exr'

    render(depth_file)

    if not filename:
        depth = cv2.imread(depth_file,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return depth

def render(output):
    output = os.path.join(os.getcwd(), output)
    bpy.data.scenes['Scene'].render.filepath = output 
    logger.off()
    bpy.ops.render.render(write_still=True)
    logger.on()

def video(filename):
    bpy.data.scenes['Scene'].render.image_settings.file_format = 'FFMPEG'
    bpy.data.scenes['Scene'].render.filepath = os.path.join(os.getcwd(), filename)
    logger.off()
    bpy.ops.render.render(animation=True)
    logger.on()

def img(filename=None):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    renderLayersNode = tree.nodes.get('Render Layers')
    compositeNode = tree.nodes.get('Composite')
    tree.links.new(renderLayersNode.outputs[0], compositeNode.inputs[0])

    if filename:
        output = filename
    else:
        output = 'img.png'

    scene = bpy.data.scenes['Scene']
    scene.render.image_settings.file_format = 'PNG'
    render(output)

    if not filename:
        im = cv2.imread(output)
        return im

def seg(filename):
    pass

def vertex(obj=None):
    # Save vertex information
    if not obj:
        obj = bpy.context.active_object

    vertexs = []
    for vv in obj.data.vertices:
        vertexs.append(vv.co)
    print(vertexs)
    return vertexs

def camera(filename):
    # Save camera information
    # Camera intrinsic and extrinsic
    pass

def kp():
    # Save 2d keypoint
    # Use camera and vertex information to project 3d to 2d
    print(bl.d3.camera)
    cam = bl.d3.camera
    vertexs = vertex()
    ps = []
    for p3 in vertexs:
        print(p3)
        p2 = bl.d3.project(cam, p3) 
        ps.append(p2)
        print(p2)

    d = depth()
    i = img()

    plt.figure()
    plt.imshow(i)
    # plt.hold(True)
    for p in ps:
        x = int(p[0])
        y = int(p[1])

        vz = d[int(p[1]), int(p[0]), 0]
        rz = p[2]
        print("Depth %f, Real %f" % (vz, rz))
        if vz > 10e10:
            plt.plot(x, y, 'y*')
        else:
            if (abs(vz - rz) < 0.1):
                plt.plot(x, y, 'b*')
            else:
                plt.plot(x, y, 'r*')
    
    plt.savefig('test.png')
    
    # print(d)
    
