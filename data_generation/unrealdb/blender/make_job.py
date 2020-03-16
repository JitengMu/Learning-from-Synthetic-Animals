# Generate yaml job file, then use render to render these job files.

template_file = 'bvh_template1.yaml'

job_folder = './job/'

import glob
import os
import bvh

bvh_files = glob.glob('../data/bvh/*.bvh')

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

template = open(template_file).read()
for bvh_file in bvh_files:
# bvh_file = '../data/bvh/embrace.bvh'
    act_name = os.path.basename(bvh_file).replace('.bvh', '')
    print(bvh_file)
    bvh_data = bvh.Bvh(open(bvh_file).read())
    print(bvh_data.nframes)
    
    num_frame = bvh_data.nframes

    job_filename = os.path.join(job_folder, '{act_name}.yaml'.format(**locals()))
    with open(job_filename, 'w') as f:
        kv = SafeDict(locals())
        f.write(template.format_map(kv))


