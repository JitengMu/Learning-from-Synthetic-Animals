

class Scene:
    def __init__(self):
        self.objects = dict()
        self.objects['scene'] = self

    def get(self, obj_name):
        return self.objects.get(obj_name)

    def import_fbx(self, fbx_filename, scale):
        print('Import %s into the scene' % fbx_filename)

    def save(self, blend_filename):
        print('Save scene to %s' % blend_filename)

    def load(self, blend_filename):
        print('Load scene from %s' % blend_filename)

class CvCamera:
    def __init__(self):
        pass

    def save(self, ):
        pass

class CvHuman:
    def __init__(self):
        pass

    def set_bvh(self, bvh_filename):
        pass

    def set_bvh_id(self, frame_id):
        pass

class CvLight:
    pass