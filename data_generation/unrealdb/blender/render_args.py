# Use this code this way
# This code is used for parsing the sequence
# python simple_render.py seq.yaml 

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('seq_files', nargs='+')
# parser.add_argument('start', type=int, nargs='?', default=0)
# parser.add_argument('end', type=int, nargs='?', default=9999)

print(sys.argv)
argv = sys.argv[1:]
# Need to modify this to support inside blender.
# Remove arguments before -- to support blender
if '--' in argv:
    idx = argv.index('--')
    argv = argv[idx+1:]
args = parser.parse_args(argv)

# print(args.seq_file)
# print(args.start)
# print(args.end)

# Write a yaml reader that can be shared between render.



# obj = scene.get(obj_name)
# getattr(obj, function_name)(args)

# Check number of arguments
# https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function

# Why write the generation seq?
# 1. reproduce
# 2. version control
# 3. check the valid of the sequence
# 4. visualize the sequence with different code.
# 5. easy to make progress bar
# 6. easy to distribute the rendering and parallel the job.
# 7. easier to read and record animation than python script.
# 8. generate yaml file, not python file.
# 9. no serialization code for python code.
# 10. make the syntax simpler and less prone to error.
# 11. if I want to make the rendering parallel, I need to make sure no dependency in the scene.

if __name__ == '__main__':
    import yaml
    from pprint import pprint
    data = yaml.load(open(args.seq_file), Loader=yaml.FullLoader)

    pprint(data)