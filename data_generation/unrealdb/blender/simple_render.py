import render
from simple_object import *
import yaml

args = render.args
scene = Scene()

data = yaml.load(open(args.seq_file), Loader=yaml.FullLoader)
print(data)
frames = list(data.keys())
# print(frames)
for k, v in data.items():
    cmds = v
    print(cmds)
    for cmd in cmds:
        print(cmd)
        # The cmd format is obj, function, args
        obj = scene.get(cmd[0])
        func = getattr(obj, cmd[1][1:])
        assert(obj is not None)
        assert(func is not None)
        func(*cmd[2:])
        # print(k, v)