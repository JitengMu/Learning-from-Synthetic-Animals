# define some utility functions to work in the blender console.
import os
import sys
from pprint import pprint


def ls():
    pprint(os.listdir())

def pwd():
    pprint(os.getcwd())

def mkdir(dir):
    os.mkdir(dir)

def run(script):
    script = os.path.join(os.getcwd(), script)
    if not os.path.isfile(script):
        print('Can not find file %s' % script)
        return
    # Run script of the editor
    with open(script) as f:
        exec(f.read())

def reload(module):
    import importlib
    importlib.reload(module)