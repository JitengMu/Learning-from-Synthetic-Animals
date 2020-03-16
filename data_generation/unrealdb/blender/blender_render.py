# Render scene using the seq.yaml file.


import sys
import yaml
import pdb
import traceback, sys, code
sys.path.append('.')
from blender_object import *
import render_args
from tqdm import tqdm
# https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error

DEBUG = False

def render_sequence(scene, seq_file):
    data = yaml.load(open(seq_file), Loader=yaml.FullLoader)

    tick_cmds = []
    if data.get('tick'): 
        tick_cmds = data['tick']
        data.remove('tick')
    # frames = list(data.keys())
    # frames.remove('tick')

    keys = list(data.keys())
    frames = []
    expanded_data = dict()
    for v in keys:
        if type(v) is int:
            if not expanded_data.get(v): expanded_data[v] = []
            expanded_data[v] += data[v]
        elif type(v) is str and ':' in v:
            start, end, step = v.split(':')
            start = int(start); end = int(end); step = int(step)
            for i in range(start, end+1, step):
                if not expanded_data.get(i): expanded_data[i] = []
                expanded_data[i] += data[v]
        else:
            # Defined variables
            # assert(False)
            print(v)

    del data
    frames = sorted(list(expanded_data.keys()))
    for frame_id in tqdm(frames):
        scene.frame_id = frame_id
        cmds = expanded_data[frame_id]
        cmds = cmds + tick_cmds
        if DEBUG: print(cmds)
        for cmd in cmds:
            if DEBUG: print(cmd)
            # The cmd format is obj, function, args
            obj = scene.get(cmd[0])
            func = getattr(obj, cmd[1][1:])
            assert(obj is not None)
            assert(func is not None)
            args = cmd[2:]
            args = [v.format(**obj.__dict__) if type(v) is str else v for v in args]
            func(*args)

def main():
    print('Render inside blender')
    args = render_args.args
    scene = Scene()

    for seq_file in args.seq_files:
        render_sequence(scene, seq_file)

if __name__ == '__main__':
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        # type, value, tb = sys.exc_info()
        # traceback.print_exc()
        # last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        # frame = last_frame().tb_frame
        # ns = dict(frame.f_globals)
        # ns.update(frame.f_locals)
        # code.interact(local=ns)